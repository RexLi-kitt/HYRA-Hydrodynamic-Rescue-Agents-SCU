"""Summarize actual completed development runs, including incomplete prerequisites."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import write_json


def progress_report(root):
    root = Path(root)
    completed, failed = [], []
    for run in sorted((root / "results" / "runs").iterdir()):
        if not (run / "state.json").exists():
            continue
        state = json.loads((run / "state.json").read_text(encoding="utf-8"))
        args = json.loads((run / "arguments.json").read_text(encoding="utf-8"))
        item = dict(run=str(run), state=state, arguments=args)
        if state["status"] == "completed":
            completed.append(item)
        elif state["status"] == "failed":
            failed.append(item)
    episodes, physical_runs = {}, {}
    numerical = None
    design_check = None
    phase_validation = None
    planner_replay = None
    matched_control = None
    tests = None
    for item in completed:
        run = Path(item["run"])
        if (run / "preflight_summary.json").exists():
            physical_runs[item["arguments"]["side"]] = (item, json.loads((run / "preflight_summary.json").read_text(encoding="utf-8")))
        if (run / "episode.json").exists():
            data = json.loads((run / "episode.json").read_text(encoding="utf-8"))
            config = data["execution_record"]["config"]
            episodes[(config["map_shape"][0], config["observation_delay_s"])] = (item, data)
        if (run / "numerical_comparison.json").exists():
            numerical = (item, json.loads((run / "numerical_comparison.json").read_text(encoding="utf-8")))
        if (run / "test_result.json").exists():
            tests = item
        if (run / "design_summary.json").exists():
            design_check = (item, json.loads((run / "design_summary.json").read_text(encoding="utf-8")))
        if (run / "phase_matrix_summary.json").exists():
            phase_validation = (item, json.loads((run / "phase_matrix_summary.json").read_text(encoding="utf-8")))
        if (run / "planner_replay_summary.json").exists():
            planner_replay = (item, json.loads((run / "planner_replay_summary.json").read_text(encoding="utf-8")))
        if (run / "matched_control_summary.json").exists():
            matched_control = (item, json.loads((run / "matched_control_summary.json").read_text(encoding="utf-8")))
    record = dict(
        phase="forecast_exploration_multi_visit_replay" if planner_replay else ("phase_candidate_validation" if phase_validation else ("design_opportunity_check" if design_check else "engineering_and_initial_numerical_preflight")), formal_test=False,
        completed_runs=len(completed), failed_runs=len(failed),
        benchmarks=[dict(run=i["run"], summary=d["summary"], config=d["execution_record"]["config"])
                    for i, d in episodes.values()],
        latest_numerical_run=numerical[0]["run"] if numerical else None,
        latest_tests_run=tests["run"] if tests else None,
        full_design_matrix_completed=bool(design_check and design_check[1]["status"] == "completed"),
        design_opportunity_run=design_check[0]["run"] if design_check else None,
        phase_candidate_validation_run=phase_validation[0]["run"] if phase_validation else None,
        planner_replay_run=planner_replay[0]["run"] if planner_replay else None,
        forecast_exploration_and_multi_visit_implemented=bool(planner_replay),
        same_frontier_candidate_control_completed=bool(matched_control),
        matched_frontier_control_run=matched_control[0]["run"] if matched_control else None,
        selected_mission_phase=None,
        full_numerical_design_gate_completed=False, trained_H_E_P=False, fixed_route_comparison_completed=False,
        validation_B_star_selected=False, configuration_frozen=False, independent_test_episodes=0,
        historical_failed_runs=failed,
    )
    write_json(root / "results" / "progress.json", record)
    cost = None
    if all((side, 0.) in episodes and side in physical_runs for side in (32, 50, 70)):
        bytes_by_side = {}
        for side in (32, 50, 70):
            item, data = episodes[(side, 0.)]
            path = Path(item["run"])
            if (path / "episode_storage.json").exists():
                bytes_by_side[side] = json.loads((path / "episode_storage.json").read_text(encoding="utf-8"))["total_episode_log_bytes"]
            else:
                bytes_by_side[side] = (path / "episode.json").stat().st_size
        physical_pair_bytes = sum(data["truth"]["field_bytes"] + data["nominal"]["field_bytes"] for _, data in physical_runs.values())
        physical_pair_seconds = sum(data["truth"]["elapsed_wall_s"] + data["nominal"]["elapsed_wall_s"] for _, data in physical_runs.values())
        episode_total = 216 * (3 * bytes_by_side[32] + 8 * bytes_by_side[50] + 3 * bytes_by_side[70])
        cost = dict(
            assumptions=["One truth/nominal field pair shared by methods/layouts/training seeds",
                         "All 78 families cross 12 hydrology conditions for a conservative field estimate",
                         "Compressed episode logs at measured single-design-family sizes",
                         "Excludes training features, neural checkpoints and extra refinement meshes",
                         "Sequential solver timing extrapolation, not a runtime guarantee"],
            measured_episode_log_bytes_by_side=bytes_by_side,
            projected_3024_episode_log_bytes=episode_total,
            projected_36288_episode_log_bytes=12 * episode_total,
            projected_all_78_families_full_hydrology_field_bytes=78 * 12 * physical_pair_bytes,
            projected_six_design_families_field_generation_wall_hours=6 * 12 * physical_pair_seconds / 3600,
            projected_all_78_families_field_generation_wall_hours=78 * 12 * physical_pair_seconds / 3600,
        )
        write_json(root / "results" / "storage_and_time_estimate.json", cost)
    lines = ["# 实验三启动与预检记录（2026-10-18）", "",
             "实验三已建立独立代码包并开始执行。当前完成的是工程与初次数值预检；五组训练、验证选模和独立测试尚未执行。", "",
             "## 实际运行", "",
             "以下均使用独立 ANUGA 真值与名义场、10米任务格、1200秒回合、5米/秒连续飞行和完整12秒服务。信息模块仅为名义预测适配器，规划器为开发用单步贪心分配。", "",
             "| 任务格／机群 | 观测延迟 | 实际完成／隐藏目标 | 探索覆盖 | 规划中位／P95（ms） | 超时 | 闭环进程峰值内存 |", "|---|---:|---:|---:|---:|---:|---:|"]
    if design_check:
        info = design_check[1]
        lines[2] += (f" 后续已完成{info['completed_cases']}个场景、{info['completed_episodes']}个回合的独立设计集机会空间检查，"
                     "见[设计集报告](20261018_设计集机会空间检查.md)。")
    if phase_validation:
        info = phase_validation[1]
        lines[2] += (f" 随后完成450／600秒候选的{info['completed_phase_cases']}个阶段场景、{info['completed_episodes']}回合核验，"
                     "见[候选完整核验](20261018_450与600秒候选完整核验.md)；尚未选择正式阶段。")
    if planner_replay:
        info = planner_replay[1]
        lines[2] += (f" 随后接通预测探索与多访问规划，并在全部{info['pairs']}组保留场景新增{info['episodes']}回合，"
                     "见[规划改动回放](20261018_预测探索与多访问规划回放.md)；该证据不构成H／E／P效果结论。")
    if matched_control:
        lines[2] += f" 同候选、同路线规划的去前沿水情权重对照另完成{matched_control[1]['episodes']}回合。"
    for (side, delay), (item, data) in sorted(episodes.items()):
        score = data["summary"]
        memory = item["state"].get("peak_process_memory_bytes")
        lines.append(f"| {side * side}格／{data['execution_record']['config']['n_drones']}机 | {delay:g}s | "
                     f"{score['successful_services']}/{score['hidden_targets']} | {100 * score['coverage_fraction']:.1f}% | "
                     f"{score['planning_latency_ms']['median']:.2f}／{score['planning_latency_ms']['p95']:.2f} | "
                     f"{score['planning_timeouts']} | {memory / 2**20:.1f} MiB |")
    lines += ["", "各回合的目标、轨迹、服务起止、预测版本和规划事件保存在对应 `episode.json`。新版运行将原始观测及到达时间无损保存为 `observations.npz`，由JSON中的路径和校验值关联；早期记录仍在JSON内。所有隐藏目标均计入分母，失败回合也保留。", ""]
    if physical_runs:
        lines += ["水动力生成与闭环分进程实测。10米水动力网格将每个方形计算单元分为4个三角形：", "",
                  "| 任务格 | 水动力三角形 | 场生成耗时（真值／名义） | 两场压缩存储 | 生成＋闭环进程峰值内存 |",
                  "|---|---:|---:|---:|---:|"]
        for side, (item, data) in sorted(physical_runs.items()):
            true, nominal = data["truth"], data["nominal"]
            peak = item["state"]["peak_process_memory_bytes"]
            lines.append(f"| {side * side} | {true['manifest']['hydraulic_triangles']} | "
                         f"{true['elapsed_wall_s']:.2f}s／{nominal['elapsed_wall_s']:.2f}s | "
                         f"{(true['field_bytes'] + nominal['field_bytes']) / 2**20:.2f} MiB | {peak / 2**20:.1f} MiB |")
        lines.append("")
    ceiling = [f"{side * side}格、{delay:g}秒延迟" for (side, delay), (_, data) in sorted(episodes.items())
               if data["summary"]["success_rate"] == 1.]
    if ceiling:
        lines += ["## 设计集需要继续检查的现象", "",
                  "在已运行的" + "、".join(ceiling) + "条件中，名义预测已经完成全部目标。这些回合的完成率提升上限为零。不能把它们称为 HyRA 正收益证据，也不能据此删除场景或只保留有利样本。", "",
                  ("后续完整设计集检查已执行，具体覆盖与结论见[设计集报告](20261018_设计集机会空间检查.md)；独立测试仍未开启。"
                   if design_check else "下一步需按预先列出的6个设计家族与12种水情条件，报告窗口分布、发现时间、服务完成时间及剩余改进空间，再决定整个设计能否回答研究问题。当前单个家族不能代表36个测试家族。"), ""]
    if numerical:
        metrics = numerical[1]
        space, clock = metrics["spatial_10m_vs_5m"], metrics["temporal_2s_vs_1s"]
        lines += ["## 初次数值核验", "",
                  f"2500格、一个设计家族：10米相对5米水动力网格的水深 MAE 为 {space['depth_mae_m'] * 1000:.3f} mm，最大差 {space['depth_max_abs_m'] * 1000:.3f} mm；完整12秒窗口标签差异为 {100 * space['complete_window_disagreement_fraction']:.4f}%。", "",
                  f"2秒相对1秒输出的窗口标签差异为 {100 * clock['complete_window_disagreement_fraction']:.4f}%；比较方式：`{metrics.get('temporal_comparison', 'subsampled_same_1s_solve')}`。", "",
                  "这只是描述性设计诊断；尚未完成全部家族、扰动、涨水条件和更细网格的收敛核验，也未冻结误差容限。", ""]
    if cost:
        lines += ["## 算力与存储外推", "",
                  f"按本机实测、各方法共享场文件，3024回合的闭环日志约 {cost['projected_3024_episode_log_bytes'] / 2**30:.2f} GiB；36288回合约 {cost['projected_36288_episode_log_bytes'] / 2**30:.2f} GiB。", "",
                  f"若全部78家族均交叉12种水情，10米网格的真值／名义采样场约 {cost['projected_all_78_families_full_hydrology_field_bytes'] / 2**30:.2f} GiB；六设计家族全条件串行生成约 {cost['projected_six_design_families_field_generation_wall_hours']:.2f} 小时。", "",
                  "这些是当前一个家族的粗外推，未包括模型训练、特征缓存、检查点、更细网格和额外重跑；详见 `results/storage_and_time_estimate.json`。", ""]
    lines += ["## 协议审计", "",
              "已登记全新的6设计／24训练／12验证／36测试家族；全部尺度、扰动、目标布局继承家族划分。仅生成划分清单，没有读取或生成测试真值。", "",
              "方案中的3024回合对应每个家族一种预分配水情，且10秒延迟扩展仅跑H／E／P。若每个家族交叉4种涨水与3档扰动，则主测试28512回合、三方法延迟扩展7776回合，共36288回合；延迟五方法则共41472回合。条件分配与延迟方法数仍标为待定，未暗中冻结。", "",
              "## 当前实现与未完成项", "",
              "已实现：ANUGA有限涨水及局地边界扰动、独立名义场、可变规模任务格、稀疏覆盖计算、连续到达事件、实际位置采样、观测延迟、完整服务评分、版本与有效期接口、规划超时回退和逐次运行留痕。" + (" 另实现预测前沿探索与有限候选路线的中央多访问滚动分配，并完成全部保留场景配对回放。" if planner_replay else ""), "",
              "尚未实现或完成：容量与输入匹配的H／E／P及两项消融训练、经训练验证的稀疏联系、固定路线信息比较、正式多访问滚动规划器、改派收益的配对反事实归因、验证集B*选择、全设计集门禁与独立测试。因此当前没有P相对B*增量、置信区间或正式实验结论。", "",
              "开发用预测有效期60秒；地形、局地边界脉冲和初始水位的具体参数均已留痕，但仍属开发选择。真值评分按已保存ANUGA输出的分段线性插值执行，不能声称输出间隔内未解析的峰值已被完全排除。", "",
              "## 原始证据", ""]
    for (side, delay), (item, _) in sorted(episodes.items()):
        lines.append(f"- {side * side}格／{delay:g}秒延迟：`{Path(item['run']).relative_to(root).as_posix()}`")
    for side, (item, _) in sorted(physical_runs.items()):
        lines.append(f"- {side * side}格水动力场：`{Path(item['run']).relative_to(root).as_posix()}`")
    if numerical:
        lines.append(f"- 数值诊断：`{Path(numerical[0]['run']).relative_to(root).as_posix()}`")
    if tests:
        import xml.etree.ElementTree as ET
        suite = ET.parse(Path(tests["run"]) / "tests.xml").getroot().find("testsuite")
        lines.append(f"- 回归测试：{suite.attrib.get('tests')}项，失败{suite.attrib.get('failures')}项，错误{suite.attrib.get('errors')}项；`{Path(tests['run']).relative_to(root).as_posix()}`")
    lines += [f"- 历史失败运行：{len(failed)}次，原错误与现场保留在 `results/progress.json` 所列目录。", "",
              "每个运行目录保存源代码快照及SHA-256、配置、环境、完成状态与原始统计。当前目录未发现Git元数据，未提交或发布。", ""]
    path = root / "docs" / "results" / "20261018_启动与预检记录.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Progress report: {path}", flush=True)
    return record
