"""Report phase-candidate validation without treating a privileged control as HyRA."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .artifacts import write_json


def collect(source):
    source = Path(source)
    summary = json.loads((source / "phase_matrix_summary.json").read_text(encoding="utf-8"))
    bins, hydrology = {}, {}
    for path in sorted((source / "jobs").glob("*/condition_*/phase*/phase_case.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        for pair in case["pairs"]:
            key = (case["phase_s"], case["side"], pair["delay_s"])
            bucket = bins.setdefault(key, dict(nominal_failures=0, oracle_failures=0, target_instances=0,
                unserved=dict(no_window_in_mission=0, undiscovered=0, windows_closed_before_discovery=0, window_remained_after_discovery=0)))
            bucket["nominal_failures"] += pair["nominal_summary"]["failed_service_attempts"]
            bucket["oracle_failures"] += pair["oracle_summary"]["failed_service_attempts"]
            bucket["target_instances"] += pair["nominal_summary"]["hidden_targets"]
            for target in pair["nominal_targets"]:
                if target["served_at_s"] is not None:
                    continue
                intervals, found = target["truth_safe_start_intervals_s"], target["discovered_at_s"]
                if not intervals:
                    reason = "no_window_in_mission"
                elif found is None:
                    reason = "undiscovered"
                elif not any(b >= found - 1e-7 for a, b in intervals):
                    reason = "windows_closed_before_discovery"
                else:
                    reason = "window_remained_after_discovery"
                bucket["unserved"][reason] += 1
            hkey = (case["family"]["family_id"], *key, case["condition"]["rise_m"], case["condition"]["rise_duration_s"])
            targets = pair["nominal_summary"]["hidden_targets"]
            hydrology.setdefault(hkey, []).append((pair["nominal_summary"]["successful_services"] / targets,
                pair["oracle_summary"]["successful_services"] / targets, pair["achieved_oracle_minus_nominal_pp"]))
    hydrology_family_means = [dict(family_id=key[0], phase_s=key[1], side=key[2], delay_s=key[3], rise_m=key[4], rise_duration_s=key[5],
        nominal_rate=float(np.mean([v[0] for v in values])), oracle_rate=float(np.mean([v[1] for v in values])),
        delta_pp=float(np.mean([v[2] for v in values]))) for key, values in sorted(hydrology.items())]
    groups = []
    for group in summary["groups"]:
        key = (group["phase_s"], group["side"], group["delay_s"])
        families = [r for r in summary["family_means"] if (r["phase_s"], r["side"], r["delay_s"]) == key]
        groups.append({**group, **bins[key],
            "families_with_positive_oracle_delta": sum(r["achieved_oracle_delta_pp"] > 1e-9 for r in families),
            "family_oracle_delta_min_pp": min(r["achieved_oracle_delta_pp"] for r in families),
            "family_oracle_delta_max_pp": max(r["achieved_oracle_delta_pp"] for r in families)})
    main = next((r for r in groups if (r["phase_s"], r["side"], r["delay_s"]) == (600., 50, 0.)), None)
    if summary["status"] != "completed":
        decision = "incomplete_do_not_select"
    elif main and main["achieved_oracle_delta_pp"] > 1e-9:
        decision = "phase600_has_achieved_information_opportunity_pending_numerical_and_strong_baseline_checks"
    elif main and main["nominal_rate"] < 1 - 1e-9:
        decision = "phase600_losses_not_recovered_on_average_by_this_forecast_diagnostic"
    else:
        decision = "phase600_main_condition_still_at_ceiling"
    return dict(source_run=str(source), status=summary["status"], decision=decision,
                completed_phase_cases=summary["completed_phase_cases"], completed_pairs=summary["completed_pairs"],
                completed_episodes=summary["completed_episodes"], groups=groups,
                family_means=summary["family_means"], hydrology_family_means=hydrology_family_means,
                implementation_scope=dict(prediction_used_for_discovered_task_assignment=True,
                    exploration_uses_future_water_windows=False, formal_multi_visit_planner_implemented=False),
                globally_frozen_phase=None, H_E_P_trained=False, formal_test=False,
                no_P_effect_or_confidence_interval=True)


def write_report(root, source):
    root, source = Path(root).resolve(), Path(source).resolve()
    diagnosis = collect(source)
    write_json(source / "phase_matrix_diagnosis.json", diagnosis)
    lines = ["# 450／600秒候选：完整设计集核验", "",
             f"状态：**{diagnosis['status']}**；已完成{diagnosis['completed_phase_cases']}/432个阶段场景、"
             f"{diagnosis['completed_pairs']}/1152组成对运行，共{diagnosis['completed_episodes']}/2304回合。", "",
             "## 核验内容", "",
             "6个设计家族×12种水情×3尺度×450／600秒两档水动力预运行阶段。每个任务完整运行1200秒，真值和独立名义场均实际积分至1800秒。每场景两套匹配布局／噪声，2500格另测10秒延迟。没有访问训练、验证或测试家族真值。", "",
             "名义基准使用原有未训练的名义预测适配器。理想预测诊断使用同一贪心规划器、60秒发布有效期、感知与载荷规则，仅在目标已发现后提供该位置的未来真值水情。目标位置仍需搜索，不能提前查询。理想预测具有特权信息，不是H／E／P、最优联合规划上界或训练数据。", "",
             "## 家族平均结果", "",
             "先在家族内平均水情和布局，再平均6个家族；正负收益场景均保留。Δ表示理想预测诊断减名义基准，不表示HyRA收益。", "",
             "| 阶段 | 规模／延迟 | 名义完成率 | 理想预测完成率 | Δ（百分点） | 正增量家族 | 名义满完成率回合 | 理想预测负收益回合 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in diagnosis["groups"]:
        lines.append(f"| {row['phase_s']:g}s | {row['side'] ** 2}／{row['delay_s']:g}s | {100 * row['nominal_rate']:.3f}% | "
            f"{100 * row['oracle_rate']:.3f}% | {row['achieved_oracle_delta_pp']:+.3f} | "
            f"{row['families_with_positive_oracle_delta']}/{row['families']} | {row['nominal_ceiling_pairs']}/{row['pairs']} | {row['negative_oracle_pairs']}/{row['pairs']} |")
    if (root / "docs" / "results" / "候选阶段完成率与预测诊断.png").exists():
        lines += ["", "![候选阶段完成率与预测诊断](候选阶段完成率与预测诊断.png)"]
    lines += ["", "## 未完成任务的性质", "",
              "以下为当前名义路线下的后验诊断。发现时已关窗的任务，未必能仅靠发现后的预测修正挽回；路线提前改变仍可能改变发现时间。发现时尚有窗口也不保证联合路径和载荷允许完成。", "",
              "| 阶段／规模／延迟 | 整个任务无窗口 | 未发现 | 发现前已关窗 | 发现后仍有窗口但未完成 |",
              "|---|---:|---:|---:|---:|"]
    for row in diagnosis["groups"]:
        failures = row["unserved"]
        lines.append(f"| {row['phase_s']:g}s／{row['side'] ** 2}／{row['delay_s']:g}s | {failures['no_window_in_mission']} | "
                     f"{failures['undiscovered']} | {failures['windows_closed_before_discovery']} | {failures['window_remained_after_discovery']} |")
    main = next((r for r in diagnosis["groups"] if (r["phase_s"], r["side"], r["delay_s"]) == (600., 50, 0.)), None)
    lines += ["", "## 目前可下的结论", ""]
    if diagnosis["decision"].startswith("phase600_has_achieved"):
        lines += [f"600秒候选在2500格零延迟主条件下，理想预测诊断平均多完成{main['achieved_oracle_delta_pp']:.3f}个百分点，"
                  f"{main['families_with_positive_oracle_delta']}/6家族为正；家族增量范围为{main['family_oracle_delta_min_pp']:+.3f}至{main['family_oracle_delta_max_pp']:+.3f}个百分点。"
                  "这只说明当前诊断中存在部分实际任务差异，不构成稳定收益、正式模型收益或冻结条件的依据。", ""]
    elif diagnosis["decision"].endswith("forecast_diagnostic"):
        lines += ["600秒候选虽然降低了主条件完成率，但理想预测诊断平均没有挽回任务；不能把完成率下降当成可供HyRA利用的空间。应继续定位搜索和任务时序瓶颈。", ""]
    elif diagnosis["status"] == "completed":
        lines += ["600秒主条件仍处于完成率天花板，固定后移启动阶段尚未提供所需区分能力。", ""]
    else:
        lines += ["数据尚不完整，不据此选择阶段。", ""]
    ceilings = [f"{r['phase_s']:g}s、{r['side'] ** 2}格／{r['delay_s']:g}s延迟" for r in diagnosis["groups"] if r["nominal_rate"] >= 1 - 1e-9]
    if ceilings:
        lines += ["仍完全处于天花板的组别：" + "；".join(ceilings) + "。不能将小规模或单一水情的结果泛化到所有尺度。", ""]
    missed = sum(sum(r["unserved"].values()) for r in diagnosis["groups"])
    late = sum(r["unserved"]["windows_closed_before_discovery"] for r in diagnosis["groups"])
    if missed:
        lines += [f"名义路线中，未完成目标实例共{missed}个，其中{late}个（{100 * late / missed:.1f}%）在发现前已关窗。"
                  "这说明仅改善发现后的水情判断，可能触及不到主要损失发生的环节；它不是所有其他探索路线都不可挽回的证明。", ""]
    lines += ["当前实现还有一个明确边界：预测仅用于已发现任务的窗口与指派；探索前沿的分数为新增覆盖格数除以飞行时间加常数，尚未使用未来水情窗口。规划器也仍是开发用单步贪心，而非正式多访问滚动规划器。", "",
              "因此下一步应先让窗口预测进入探索前沿的可服务性／时效性选择，并完善多访问滚动规划，再在相同场景配对回放。不能据本轮理想对照的微小或负增量，否定预测对未知区域探索的潜在价值。", ""]
    lines += ["本轮不证明HyRA收益，不说明物理联系或时间衰减有效，也没有选择正式阶段。理想诊断的贪心路线也可能少完成任务，负收益已计入均值。", "",
              "下一步需结合全条件窗口分布和数值加密结果确定候选；正式H／E强历史基线和P及两项消融尚待实现、训练和验证选模。原3024回合的水情分配及延迟方法数口径也仍需冻结。", "",
              "## 资源与完整性", "",
              f"名义规划超时共{sum(r['nominal_timeouts'] for r in diagnosis['groups'])}次；理想预测规划超时共{sum(r['oracle_timeouts'] for r in diagnosis['groups'])}次。"
              "若超时影响路线，必须结合事件记录解释，不能将性能降级归因于信息模块。", "",
              "本轮水动力网格10米、输出2秒；没有据此宣布数值收敛。理想预测增量并非工程目标中的P对B*增量，不能据此宣称达到3个百分点目标。", "",
              "## 原始证据", "",
              f"- 运行：`{source}`", "- 固定候选、代码和环境：`phase_matrix_spec.json`",
              "- 逐家族、条件与布局的正负增量：`phase_matrix_summary.json`", "- 失效类型与诊断：`phase_matrix_diagnosis.json`",
              "- 每个场景来源、目标和两种路线：`jobs/*/condition_*/phase*/phase_case.json`",
              "- 每回合轨迹、服务、原始观测与理想查询：各 `nominal/` 和 `truth_diagnostic/` 目录。", ""]
    audit_path = source / "phase_matrix_integrity_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines += ["## 数据与评分核查", "",
                  f"核查状态：{'通过' if audit['passed'] else '存在错误'}。核对{audit['unique_physical_fields']}份物理场、"
                  f"{audit['episodes']}回合、{audit['legal_arrived_observations']:,}条到达观测与{audit['discovered_target_oracle_queries']:,}次理想查询；"
                  "未发现提前查询未发现目标的情况。", "",
                  f"按保存的真值曲线独立重算{audit['service_intervals_independently_rescored']:,}次完整服务。"
                  f"严格浮点比较差异{audit['strict_saved_depth_scoring_mismatches']}次，"
                  f"阈值±{audit['numerical_boundary_band_m']:g}米内的事件共{audit['boundary_band_events']}次。"
                  "若存在阈值邻近的浮点差异，应与原始记录一并核对；本核查没有宣布空间数值收敛。", ""]
    path = root / "docs" / "results" / "20261018_450与600秒候选完整核验.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "results" / "latest_phase_candidate_diagnosis.json", dict(run=str(source), report=str(path), **diagnosis))
    if diagnosis["status"] == "completed":
        write_json(root / "NEXT_STEP.json", dict(
            stage="phase_candidate_validation_completed", evidence_run=str(source), report=str(path),
            selected_phase=None, H_E_P_trained=False, formal_test_episodes=0,
            next_engineering_action="make_window_predictions_inform_exploration_before_target_discovery_and_complete_multi_visit_rolling_planning",
            reason="Current exploration scores only coverage and travel; inspect discovery-before-window-close failures in the evidence report",
            subsequent_checks=["paired_replay_on_all_retained_scenarios", "spatial_and_temporal_numerical_sensitivity", "strong_legal_history_baseline_then_capacity_matched_H_E_P_training"],
            original_scenarios_preserved=True))
    return path


def plot(root, source):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties
    root = Path(root)
    diagnosis = collect(source)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    keys = [(32, 0.), (50, 0.), (50, 10.), (70, 0.)]
    labels = ["1024／0s", "2500／0s", "2500／10s", "4900／0s"]
    for column, phase in enumerate((450., 600.)):
        rows = [next((r for r in diagnosis["groups"] if (r["phase_s"], r["side"], r["delay_s"]) == (phase, s, d)), None) for s, d in keys]
        if any(r is None for r in rows):
            continue
        x = np.arange(4)
        axis = axes[0, column]
        axis.bar(x - .17, [100 * r["nominal_rate"] for r in rows], .32, label="名义基准", color="#98bdb4")
        axis.bar(x + .17, [100 * r["oracle_rate"] for r in rows], .32, label="理想预测诊断", color="#2b6f62")
        axis.set_title(f"水动力预运行{phase:g}秒", fontproperties=font)
        axis.set_ylim(0, 105)
        axis.set_ylabel("完成率（%）", fontproperties=font)
        axis.set_xticks(x, labels, fontproperties=font)
        axis.legend(prop=font, fontsize=9)
        axis.grid(axis="y", alpha=.18)
        axis = axes[1, column]
        axis.axhline(0, color="#89817b", linestyle="--", linewidth=.8)
        for i, (side, delay) in enumerate(keys):
            values = [r["achieved_oracle_delta_pp"] for r in diagnosis["family_means"] if (r["phase_s"], r["side"], r["delay_s"]) == (phase, side, delay)]
            axis.scatter(i + np.linspace(-.12, .12, len(values)), values, s=24, color="#4f9f90", alpha=.7)
            axis.scatter(i, rows[i]["achieved_oracle_delta_pp"], marker="D", s=36, color="#1f4038", zorder=4)
        axis.set_xticks(x, labels, fontproperties=font)
        axis.set_ylabel("理想预测−名义（百分点）", fontproperties=font)
        axis.grid(axis="y", alpha=.18)
    figure.suptitle("候选阶段的完成率与预测诊断", fontproperties=font, fontsize=16)
    figure.supxlabel("下排圆点为每个设计家族的均值；菱形为家族平均，非置信区间", fontproperties=font, fontsize=10)
    destination = root / "docs" / "results" / "候选阶段完成率与预测诊断.png"
    figure.savefig(destination, dpi=180, facecolor="white")
    plt.close(figure)
    return destination
