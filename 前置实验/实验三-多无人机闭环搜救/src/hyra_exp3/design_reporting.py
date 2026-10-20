"""Evidence-linked design-set report, kept separate from formal model evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .artifacts import write_json


def write_design_report(root, source_run):
    root, source_run = Path(root).resolve(), Path(source_run).resolve()
    if not source_run.is_relative_to(root / "results" / "runs"):
        raise ValueError("Expected a design-check run inside this experiment")
    summary = json.loads((source_run / "design_summary.json").read_text(encoding="utf-8"))
    spec = json.loads((source_run / "design_spec.json").read_text(encoding="utf-8"))
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((source_run / "jobs").glob("*/condition_*/case_summary.json"))]
    completed, total_targets = 0, 0
    per_group = {}
    for case in cases:
        for episode in case["episodes"]:
            score = episode["summary"]
            completed += score["successful_services"]
            total_targets += score["hidden_targets"]
            bucket = per_group.setdefault((case["side"], episode["delay_s"]), dict(margins=[], served=[], discovered=[], censored=0))
            for target in episode["targets"]:
                if target["served_at_s"] is not None:
                    bucket["served"].append(target["served_at_s"])
                if target["discovered_at_s"] is not None:
                    bucket["discovered"].append(target["discovered_at_s"])
                if target["safe_start_margin_after_actual_service_s"] is not None:
                    bucket["margins"].append(target["safe_start_margin_after_actual_service_s"])
                bucket["censored"] += target["closing_time_censored_by_episode_end"]
    def stats(values):
        return {str(q): float(np.quantile(values, q)) for q in (0, .1, .5, .9, 1)} if values else None
    diagnosis = dict(source_run=str(source_run), status=summary["status"],
                     completed_cases=summary["completed_cases"], completed_episodes=summary["completed_episodes"],
                     successful_target_instances=completed, total_target_instances=total_targets,
                     grouped_diagnostics=[dict(side=s, delay_s=d, service_finish_s=stats(v["served"]),
                                                discovered_s=stats(v["discovered"]), safe_start_margin_s=stats(v["margins"]),
                                                censored_target_instances=v["censored"])
                                          for (s, d), v in sorted(per_group.items())],
                     decision="completion_metric_at_ceiling_in_entire_checked_design_set" if summary["status"] == "completed" and summary["all_completed_episodes_at_ceiling"]
                              else "inspect_remaining_opportunity_and_failures_before_freeze",
                     actual_test_episodes=0, numerical_convergence_confirmed=False,
                     current_generator_only=True, no_model_effect_estimate=True)
    write_json(source_run / "opportunity_diagnosis.json", diagnosis)
    wall = json.loads((source_run / "state.json").read_text(encoding="utf-8"))
    lines = ["# 实验三：独立设计集机会空间检查", "",
             f"运行状态：**{summary['status']}**。实际完成 {summary['completed_cases']}/{spec['planned_cases']} 个水情场景、{summary['completed_episodes']}/{spec['planned_episodes']} 个闭环回合。", ""]
    if diagnosis["decision"] == "completion_metric_at_ceiling_in_entire_checked_design_set":
        lines += ["**结论：当前生成器下的整个设计集出现完成率天花板，不宜直接进入五组正式收益比较。**", "",
                  "名义预测加开发用贪心规划已在全部检查回合完成全部隐藏目标，主指标没有正提升空间。这个发现支持先修改并重新核验实验条件；它不支持宣称HyRA有效或无效。", ""]
    else:
        lines += ["当前记录显示仍有未完成检查或未完成任务，需要先分析其原因，再决定是否冻结场景。", ""]
    lines += ["## 检查范围", "",
              "使用此前登记的6个设计家族；每家族完整交叉0.6／0.9米涨幅、600／900秒涨水历时、0／0.1／0.2米局地扰动和1024／2500／4900格三种规模，共216个水情场景。每场景两套预先固定的布局／观测噪声配置，2500格另测10秒延迟，共576回合。", "",
              "所有目标均在水情评估前按几何可达位置抽样；全部隐藏目标计入分母。未根据任何方法收益删除或调整场景。仅使用设计家族，未生成训练、验证或测试家族真值。当前没有训练模型，因此不重复三个模型训练种子。", "",
              "水动力任务格10米、计算网格10米；ANUGA真值与名义场独立求解。每个家族／尺度／涨幅／历时的名义场仅计算一次，跨三档真值扰动共享；共216份真值场和72份名义场。", "",
              "## 完成率与可提升上限", "",
              "下表先在家族内平均12种水情和2套布局，再平均家族。完成率上限余量为100%减当前名义基准完成率，不是P相对B*的效果估计。", "",
              "| 规模 | 延迟 | 家族 | 回合 | 满完成率回合 | 家族平均完成率 | 完成率上限余量 | 规划超时次数 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group in summary["by_scale_delay"]:
        lines.append(f"| {group['side'] ** 2} | {group['delay_s']:g}s | {group['families']} | {group['episodes']} | "
                     f"{group['ceiling_episodes']} | {100 * group['family_mean_success_rate']:.3f}% | {group['mean_headroom_pp']:.3f}个百分点 | {group['planning_timeouts']} |")
    lines += ["", f"合计完成 {completed}/{total_targets} 个目标实例。同一家族、条件和布局的重复实例不作为独立家族。", "",
              "## 为什么看起来过于宽松", "",
              "以下时间量为实际执行后的真值诊断，未提供给规划器。关窗余量指：目标所在真值曲线中，包含实际服务开始时刻的完整12秒服务窗口，距离其最后合法开始时间还有多久。若整个回合都安全，则最后合法开始时间由回合终点截断。", "",
              "| 规模／延迟 | 发现时间中位数 | 最晚完成时间 | 服务开始后关窗余量（最小／中位数） |",
              "|---|---:|---:|---:|"]
    for group in diagnosis["grouped_diagnostics"]:
        found, served, margin = group["discovered_s"], group["service_finish_s"], group["safe_start_margin_s"]
        lines.append(f"| {group['side'] ** 2}／{group['delay_s']:g}s | "
                     f"{found['0.5']:.1f}s | {served['1']:.1f}s | {margin['0']:.1f}s／{margin['0.5']:.1f}s |")
    lines += ["", "这组时间差可判断任务是否在水情成为约束前就已完成。目标实例、格子和时间点的分位数仅描述分布，不作为独立样本进行显著性检验。", "",
              "![设计集实际服务时序与真值窗口](设计集服务时序.png)", "",
              "## 下一步决策", "",
              "保留本轮全部结果和当前生成器版本。根据真值窗口与任务时序，另立开发版本评估任务启动时水情阶段、地形／初始水位和任务可用时刻等因素；预先固定候选条件，再在设计家族上完整检查。选择依据应为数值稳定、几何可达、存在可完成任务及非平凡的行动窗口，不能依据P是否胜出。", "",
              "确认设计具有区分能力后，冻结场景和规划器，再开展H／E／P及消融训练、固定记录回放和验证集B*选取。独立测试继续封存。", "",
              "## 证据边界", "",
              "本轮使用未训练的名义预测适配器和开发用单步贪心规划，不是正式强历史基线H／E，也没有P效果或配对置信区间。结论限定当前6家族生成器、时序和条件；不能推出所有洪涝场景均无收益。全设计集的空间加密及输出间峰值核验尚未完成，当前不宣布数值收敛。", "",
              "## 原始记录", "",
              f"- 运行目录：`{source_run}`", "- 固定执行清单与代码指纹：`design_spec.json`", "- 家族平均及逐回合统计：`design_summary.json`",
              "- 时序与余量诊断：`opportunity_diagnosis.json`", "- 每场景窗口、目标与名义场来源：`jobs/*/condition_*/case_summary.json`",
              "- 每回合轨迹、观测及服务事件：`jobs/*/condition_*/layout*_delay*/`",
              "- 求解器日志、进度与错误：各job下的 `solver.log`、`progress.json` 与可选 `job_error.json`。", "",
              f"主进程运行记录耗时：{wall.get('elapsed_wall_s', 0) / 60:.2f}分钟。实际是否完成以 `state.json` 与设计汇总为准。", ""]
    phase_pointer = root / "results" / "latest_phase_probe.json"
    if phase_pointer.exists():
        phase = json.loads(phase_pointer.read_text(encoding="utf-8"))
        lines += ["## 已追加的启动阶段候选诊断", "",
                  "另立未冻结的开发变体，在一个设计家族、2500格、0.9米／600秒／0.1米扰动下，预先固定7个水动力先行阶段，实际完成28回合。每个任务仍完整运行1200秒，未修改本轮原方案。", "",
                  "水动力预运行450秒时，零延迟名义基准平均完成29.5／30个；预运行600秒时完成26／30个，而30个目标均各自具有几何可达窗口。这里的可达上界忽略搜索和联合路径，不保证能多完成4个。预运行900秒时目标均已没有完整窗口，完成率下降没有挽回价值。", "",
                  "因此，600秒值得作为下一版设计的优先复核候选，450秒可作为较温和的对照；仍需扩展到全部设计家族、水情和尺度。此判断依据窗口与任务的重叠，不依据HyRA是否胜出。", "",
                  "全部7档与延迟条件见[启动阶段候选报告](20261018_任务启动阶段候选诊断.md)，没有选择正式阶段，也没有启动测试集。", ""]
    audit_path = source_run / "design_integrity_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        lines += ["## 数据完整性核查", "",
                  f"核查状态：{'通过' if audit['passed'] else '存在错误'}。核对216个场景、576个回合、288份物理场校验值、"
                  f"{audit['legal_arrived_observations']:,}条合法到达观测，以及目标抽样、分母、完整服务时长、唯一成功和载荷记账。", "",
                  "本核查没有重新积分水动力或逐次重算真值服务水深峰值；数值收敛门禁仍未通过。完整记录为 `design_integrity_audit.json`。", ""]
    path = root / "docs" / "results" / "20261018_设计集机会空间检查.md"
    latest_candidate = root / "results" / "latest_phase_candidate_diagnosis.json"
    if latest_candidate.exists():
        candidate = json.loads(latest_candidate.read_text(encoding="utf-8"))
        if candidate["status"] == "completed":
            lines.insert(2, "**后续更新：450／600秒候选已完成完整设计集复核，主规模仍未出现稳定的理想预测增量。下一步重点已转向让预测参与探索决策，详见[最新核验报告](20261018_450与600秒候选完整核验.md)。以下保留本轮原设计集及小样本候选的阶段记录。**\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    write_json(root / "results" / "latest_design_check.json", dict(run=str(source_run), report=str(path), **diagnosis))
    return path
