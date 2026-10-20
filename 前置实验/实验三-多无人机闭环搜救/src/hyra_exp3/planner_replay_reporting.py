"""Report the retained engineering replay without selecting favorable cases."""
from pathlib import Path
import numpy as np

from .artifacts import write_json
from .protocol import load_protocol


def report_replay(root, source):
    root, source = Path(root), Path(source)
    summary = load_protocol(source / "planner_replay_summary.json")
    if summary["status"] != "completed":
        raise ValueError("A completed full-matrix replay is required")
    jobs = [load_protocol(p) for p in sorted((source / "jobs").glob("*/job_result.json"))]
    rows = [row for job in jobs for row in job["records"]]
    control, control_source = None, None
    pointer = root / "results/latest_matched_frontier_control.json"
    if pointer.exists():
        candidate = Path(load_protocol(pointer)["run"])
        if Path(load_protocol(candidate / "control_spec.json")["source"]).resolve() == source.resolve():
            control, control_source = load_protocol(candidate / "matched_control_summary.json"), candidate
    report = root / "docs/results/20261018_预测探索与多访问规划回放.md"
    labels = {"exploration": "仅探索改动", "routes": "仅多访问", "combined": "两项合用", "combined_truth_diagnostic": "合用＋真值诊断"}
    lines = ["# 预测探索与多访问滚动规划：完整设计场景回放", "",
        "已把窗口预测接入目标发现之前的探索，并实现中央多访问滚动分配。全部保留的450／600秒候选场景均已配对回放，没有按完成率或预测收益删选场景。结果仍是设计阶段的工程证据；H／E／P尚未训练，正式测试回合为0，任务启动阶段未冻结。", "",
        "## 实际实现", "",
        "- 探索：从公共网格、已到达覆盖和当前无人机位置生成每机最多16个空间分散的前沿候选；查询中心及足迹四角的完整服务窗口，考虑估计的到达、下一次采样、通信延迟和局部飞行，再按可服务性与真实关窗紧迫度调整覆盖收益。超过60秒发布有效期按未知处理，不把截断有效期当成真实关窗。没有向规划器传入隐藏目标位置或未到达观测。",
        "- 多访问：每机最多按剩余载荷生成4次访问、60秒内可完成的路线。有限宽度搜索产生候选，中央二元优化选择互不冲突的路线，优先规划完成数，其次路线末端覆盖代理量和飞行时间。只执行首段，下一规划时刻重算。候选路线不穷尽，不能称为全局最优；覆盖目标仍使用代理量。",
        "- 保留12秒完整服务、失败耗时与成功扣载荷、连续飞行、5秒沿途采样、10秒周期及发现／失败事件重规划、100毫秒规划预算。当前名义适配器仍不融合观测，后续强历史基准和神经模型尚未实现。", "",
        "## 回放范围与对照", "",
        f"6设计家族×12水情×3尺度×2启动阶段×2布局，加上主尺度10秒延迟：共{summary['pairs']}组固定配对。复用原始独立ANUGA场与旧名义控制器记录，新增{summary['episodes']}回合：仅预测探索、仅多访问、两项合用、两项合用的真值预测诊断。所有隐藏目标均计入分母。", "",
        "真值诊断允许在公共前沿查询未来水深，与新名义控制器使用相同位置查询接口和60秒有效期；任务位置仍须通过到达的传感观测发现。它具有特权未来水情，不能作为P或训练数据。上一轮诊断仅能查询已发现目标，因此两轮真值诊断的信息权限不同；下面只把同一新控制器内的真值与名义差解释为本轮预测信息诊断。", "",
        "归因限制：探索改动还包含16个空间分散候选的筛选，而旧控制器在全部网格上直接评分。即使前沿水情完全相同，筛选也可能改变路线。因此探索改动／两项合用相对旧控制器的差值是整项工程改动的效果，不能单独归因于水情预测。正式信息模块对比必须共用候选生成与规划器。" + ("本次已追加全部1152组的同候选去权重对照，见下表。" if control else "尚需同候选去权重对照。"), "",
        "## 完成率", "",
        "各项先在家族内平均，再对6家族平均；下表单位为百分比，括号为相对旧名义控制器的百分点变化。", "",
        "| 启动阶段 | 格数／延迟 | 旧控制器 | 仅探索改动 | 仅多访问 | 两项合用 | 合用＋真值诊断 | 真值−名义（合用） |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for g in summary["groups"]:
        values = [f"{100*g['variants'][name]['rate']:.2f}（{g['variants'][name]['delta_vs_baseline_pp']:+.2f}）" for name in labels]
        lines.append(f"| {g['phase_s']:g}s | {int(g['side'])**2}／{g['delay_s']:g}s | {100*g['baseline_rate']:.2f} | " + " | ".join(values) + f" | {g['matched_combined_truth_minus_nominal_pp']:+.2f} |")
    if control:
        lines += ["", "## 同候选、同路线规划的前沿水情权重对照", "",
                  "额外1152回合仅把前沿已发布支持内的水情权重设为1、支持外仍使用相同未知权重；已发现目标的名义窗口、16候选生成、路线优化、布局、采样、延迟和100毫秒预算均不变。合计新增5760回合。控制用合成中性曲线不是水动力预测，明确排除训练用途。", "",
                  "| 阶段 | 格数／延迟 | 移除前沿水情权重的完成率 | 原名义水情权重的完成率 | 名义权重增量（百分点） | 增加／减少配对 | 改善家族 |", "|---:|---|---:|---:|---:|---|---:|"]
        for g in control["groups"]:
            lines.append(f"| {g['phase_s']:g}s | {int(g['side'])**2}／{g['delay_s']:g}s | {100*g['neutral_rate']:.2f} | {100*g['nominal_weighted_rate']:.2f} | {g['nominal_water_weight_gain_pp']:+.2f} | {g['positive_pairs']}／{g['negative_pairs']} | {g['positive_families']}/6 |")
        lines += ["", "这个差值仅隔离当前名义水情权重在新控制器中的作用；名义场低估涨水，可能给出错误的探索优先级。真值诊断−名义预测还同时改变已发现目标的窗口，不能再单独拆成前沿贡献。"]
    lines += ["", "![全部场景完成率](预测探索与多访问回放.png)", "", "## 正负结果与家族稳定性", "",
              "每组144对。完整保留下降、持平和上升的配对，避免把总体均值当作每种水情均有效。", "",
              "| 阶段 | 格数／延迟 | 合用优于／差于／持平旧控制器 | 合用改善家族 | 合用−旧的家族配对描述性95%区间（百分点） | 真值诊断−合用名义的区间 |",
              "|---:|---|---|---:|---:|---:|"]
    rng = np.random.default_rng(202610181236)
    intervals = []
    for g in summary["groups"]:
        subset = [r for r in rows if (r["phase_s"], r["side"], r["delay_s"]) == (g["phase_s"], g["side"], g["delay_s"])]
        families = sorted({r["family_id"] for r in subset})
        differences = []
        for family in families:
            selected = [r for r in subset if r["family_id"] == family]
            differences.append([np.mean([100 * (r["variants"]["combined"]["summary"]["successful_services"] - r["baseline"]["successful_services"]) / r["targets"] for r in selected]),
                                np.mean([100 * (r["variants"]["combined_truth_diagnostic"]["summary"]["successful_services"] - r["variants"]["combined"]["summary"]["successful_services"]) / r["targets"] for r in selected])])
        draws = rng.integers(0, len(families), (10000, len(families)))
        bounds = np.quantile(np.asarray(differences)[draws].mean(axis=1), [.025, .975], axis=0)
        v = g["variants"]["combined"]
        lines.append(f"| {g['phase_s']:g}s | {int(g['side'])**2}／{g['delay_s']:g}s | {v['positive_pairs']}／{v['negative_pairs']}／{v['unchanged_pairs']} | {v['positive_families']}/6 | [{bounds[0,0]:+.2f}, {bounds[1,0]:+.2f}] | [{bounds[0,1]:+.2f}, {bounds[1,1]:+.2f}] |")
        intervals.append(dict(phase_s=g["phase_s"], side=g["side"], delay_s=g["delay_s"], family_differences_pp=differences, bootstrap_95_pp=bounds.tolist()))
    lines += ["", "区间按家族整组重采样10000次计算，仅有6个设计家族，用于描述开发结果，不能替代独立验证或确认性检验。", "", "## 漏救机制与运行约束", "",
              "仅在回放结束后使用真实完整窗口归因；这些标签未提供给规划器。", "",
              "| 控制器 | 无完整窗口 | 未发现 | 发现时已关窗 | 发现时仍有窗口 | 规划超时次数 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in ("baseline", *labels):
        missed = [r["baseline_missed"] if name == "baseline" else r["variants"][name]["missed"] for r in rows]
        counts = [sum(m[k] for m in missed) for k in ("no_complete_window", "undiscovered", "discovered_after_close", "window_remained_at_discovery")]
        timeouts = sum((r["baseline"] if name == "baseline" else r["variants"][name]["summary"])["planning_timeouts"] for r in rows)
        lines.append(f"| {labels.get(name, '旧名义控制器')} | " + " | ".join(map(str, counts)) + f" | {timeouts} |")
    a = summary["audit"]
    lines += ["", f"逐回合审计通过：{a['observations']:,}条已到达观测、{a['queries']:,}次查询（其中公共前沿{a['frontier_queries']:,}次）、{a['planned_visits_checked']:,}次计划访问的已发现条件／飞行时刻／载荷／版本／窗口有效期，另独立重算{a['complete_services_rescored']:,}段完整服务。真实服务评分差异为0，任务提前可见和查询越界违规为0。每回合原始观测、查询角色、发布版本和服务事件均已保存及哈希登记。", "",
        "审计针对保存场的分段线性曲线；输出之间未解析的水深峰值仍需数值加密检查。六进程下的真实超时按原100毫秒预算处理并计入结果，没有放宽预算或删除超时场景。", "",
        "## 下一步", "",
        ("同候选去权重对照已完成。" if control else "先补同候选、去除前沿水情权重的对照。") + "接下来检查规划时限与候选路线截断的敏感性，并对代表性的关窗与服务临界场景完成空间／时间加密；再实现只使用合法历史观测的强基准和容量匹配H／E／P。两档启动阶段与全部失败场景继续保留，使用验证集作最终选择，独立测试保持关闭。", "",
        "这次回放只能检验决策链是否接通以及工程控制器如何变化，不能证明HyRA优于强基准。", "",
        f"数据运行：`{source}`。", ""]
    report.parent.mkdir(parents=True, exist_ok=True)
    solver_audit = source / "solver_completion_audit.json"
    if solver_audit.exists():
        solver = load_protocol(solver_audit)
        lines += [f"另检查每次候选池优化状态：仅多访问对照有{solver['uncertified_candidate_pool_events']}次未证候选池最优，采用可行分配，仍计入结果；合用名义、合用真值诊断和同候选中性对照均未出现此状态。候选集合本身不穷尽，因此即使集合内最优也不代表全局最优。", ""]
    if control:
        focus = next(g for g in summary["groups"] if (g["phase_s"], g["side"], g["delay_s"]) == (600, 50, 10))
        matched = next(g for g in control["groups"] if (g["phase_s"], g["side"], g["delay_s"]) == (600, 50, 10))
        lines[2] += (f" 主尺度10秒延迟、600秒预运行下，旧名义控制器{100*focus['baseline_rate']:.2f}%，两项合用{100*focus['variants']['combined']['rate']:.2f}%；"
                     f"同候选对照中名义前沿水情权重的增量为{matched['nominal_water_weight_gain_pp']:+.2f}个百分点，"
                     f"仅{matched['positive_families']}/6家族改善；同一新控制器的真值预测诊断再增{focus['matched_combined_truth_minus_nominal_pp']:+.2f}个百分点。"
                     " 小尺度600秒条件下，名义前沿水情权重反而降低1.22个百分点，当前没有跨尺度稳定收益证据。")
        main_audit_path = source / "serialized_integrity_audit.json"
        if main_audit_path.exists():
            ma = load_protocol(main_audit_path)
            lines += [f"主回放的落盘审计通过：{ma['episodes']}回合、{ma['queries']:,}次查询、{ma['multi_visit_routes']:,}条实际多访问计划、{ma['checked_trajectory_segments']:,}段采样轨迹；全部源码快照与历史配对记录的哈希通过。", ""]
        audit_path = control_source / "serialized_integrity_audit.json"
        if audit_path.exists():
            ca = load_protocol(audit_path)
            lines += [f"同候选对照的落盘审计亦通过：{ca['episodes']}回合，{ca['queries']:,}次查询，{ca['multi_visit_routes']:,}条多访问计划；其在线审计另独立重算{ca['original_online_audit']['complete_services_rescored']:,}段完整服务。源码、配对、原始记录哈希和飞行时刻检查均通过。控制运行：`{control_source}`。", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    write_json(source / "family_descriptive_intervals.json", dict(bootstrap_seed=202610181236, replicates=10000, groups=intervals, confirmatory=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharey=True)
    methods = ["baseline", *labels]
    colors = ["#747c89", "#159a9c", "#c4912c", "#345ca8", "#983b71"]
    for ax, phase in zip(axes, (450., 600.)):
        groups = [g for g in summary["groups"] if g["phase_s"] == phase]
        x = np.arange(len(groups))
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = [100 * (g["baseline_rate"] if method == "baseline" else g["variants"][method]["rate"]) for g in groups]
            ax.bar(x + (i - 2) * .15, values, width=.14, label=labels.get(method, "旧名义控制器"), color=color)
        ax.set_xticks(x, [f"{int(g['side'])**2}\n延迟{g['delay_s']:g}s" for g in groups], fontproperties=font)
        ax.set_title(f"预运行{phase:g}秒；各组144对", fontproperties=font)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("完成率（%）", fontproperties=font)
    axes[1].legend(prop=font, fontsize=8, loc="lower left")
    fig.tight_layout()
    figure = report.with_name("预测探索与多访问回放.png")
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    write_json(root / "NEXT_STEP.json", dict(stage="forecast_exploration_multi_visit_replay_completed", evidence_run=str(source), report=str(report),
        selected_phase=None, H_E_P_trained=False, formal_test_episodes=0, original_scenarios_preserved=True,
        matched_frontier_control_completed=bool(control), matched_frontier_control_run=str(control_source) if control else None,
        next_engineering_action="numerical_and_route_pool_sensitivity_then_strong_history_baseline" if control else "same_frontier_candidates_without_forecast_weights_control",
        subsequent_checks=["space_time_refinement_on_closing_windows", "bounded_route_pool_and_real_time_budget_sensitivity", "legal_history_baseline_and_capacity_matched_H_E_P_training"],
        caveat="engineering_replay_not_formal_HyRA_effect"))
    return report
