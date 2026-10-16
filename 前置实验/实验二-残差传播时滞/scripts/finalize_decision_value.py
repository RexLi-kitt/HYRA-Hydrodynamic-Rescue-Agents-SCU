"""Summarize the finite two-order diagnostic without pseudo-replication.

The independent unit is a flood family.  R5 phases are averaged inside each
family, separately for every communication delay, before averaging families.
Nothing in this report is an estimate of mutual information or Bayesian VoI.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


METHODS = ("forecast_only", "raw_self", "raw_full", "recent", "physical", "uniform", "wrong")
CONTROLS = ("forecast_only", "raw_full", "recent", "uniform", "wrong")
GROUP = ("scope", "revisit", "delay", "split", "condition")
SPLITS = {"train": 0, "validation": 1, "test": 2}


def read(path, optional=False):
    if optional and not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rows(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return value["rows"]
    raise ValueError("Expected a JSON list or an object with a rows list")


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def avg(values):
    values = [float(v) for v in values if v is not None]
    if any(not np.isfinite(v) for v in values):
        raise ValueError("Nonfinite score in input")
    return float(np.mean(values)) if values else None


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def scope(row):
    revisit = float(row["revisit"])
    if revisit == 5:
        return "R5_four_phases"
    if revisit in (10, 40):
        return f"R{revisit:g}_phase0"
    raise ValueError(f"Unspecified revisit interval: {revisit}")


def key(row, fields):
    return tuple(row[k] for k in fields)


def phase_average(source, kind):
    """Return one row per family/method/condition/delay; require full phases."""
    groups = defaultdict(list)
    fields = (*GROUP, "family", "method")
    for original in source:
        row = dict(original, scope=scope(original))
        groups[key(row, fields)].append(row)
    output = []
    for k, rr in sorted(groups.items()):
        phases = sorted(float(r["phase"]) for r in rr)
        expected = [0., .25, .5, .75] if k[1] == 5 else [0.]
        if phases != expected:
            raise ValueError(f"Missing/duplicate phases for {k}: {phases}; expected {expected}")
        item = dict(zip(fields, k))
        item.update(phase_count=len(rr), phases=phases)
        if kind == "task":
            metrics = ("mean_yield", "mean_delta_nominal", "mean_regret", "mean_baseline_regret")
            for metric in metrics:
                item[metric] = avg(r[metric] for r in rr)
            for metric in ("positive", "negative", "changed"):
                item[f"{metric}_rate"] = avg(ratio(r[metric], r["cases"]) for r in rr)
                item[f"mean_{metric}_cases_per_phase"] = avg(r[metric] for r in rr)
            count_name = "cases"
        elif kind == "information":
            item["mae_m"] = avg(r["mae_m"] for r in rr)
            item["rmse_m"] = avg(r["rmse_m"] for r in rr)
            count_name = "samples"
        elif kind == "window":
            for metric in ("correctable_cases", "captured_cases", "initial_gain_cases", "late_only_gain_cases"):
                item[f"mean_{metric}_per_phase"] = avg(r[metric] for r in rr)
                item[metric.removesuffix("_cases") + "_rate_all"] = avg(ratio(r[metric], r["cases"]) for r in rr)
            item["capture_rate_correctable"] = avg(ratio(r["captured_cases"], r["correctable_cases"]) for r in rr)
            item["phases_with_correctable_cases"] = sum(r["correctable_cases"] > 0 for r in rr)
            count_name = "cases"
        else:
            raise ValueError(kind)
        item[f"mean_{count_name}_per_phase"] = avg(r[count_name] for r in rr)
        item[f"{count_name}_min_per_phase"] = min(r[count_name] for r in rr)
        item[f"{count_name}_max_per_phase"] = max(r[count_name] for r in rr)
        output.append(item)
    return output


def family_summary(source):
    groups = defaultdict(list)
    fields = (*GROUP, "method")
    for row in source:
        groups[key(row, fields)].append(row)
    output = []
    excluded = {*fields, "family", "phases", "phase_count"}
    for k, rr in sorted(groups.items()):
        if len({r["family"] for r in rr}) != len(rr):
            raise ValueError(f"Duplicate family in {k}")
        item = dict(zip(fields, k))
        item.update(independent_families=len(rr), families=sorted(r["family"] for r in rr),
                    phase_count=rr[0]["phase_count"])
        for metric in rr[0]:
            if metric not in excluded:
                item[metric] = avg(r[metric] for r in rr)
        output.append(item)
    return output


def paired_comparisons(source, metric, lower_is_better=False):
    groups = defaultdict(list)
    for row in source:
        groups[key(row, GROUP)].append(row)
    output = []
    for k, rr in sorted(groups.items()):
        proposed = {r["family"]: r for r in rr if r["method"] == "physical"}
        for control in CONTROLS:
            baseline = {r["family"]: r for r in rr if r["method"] == control}
            if not baseline:
                continue
            if set(proposed) != set(baseline):
                raise ValueError(f"Unpaired families for {k}, {control}")
            values = []
            for family in sorted(proposed):
                a, b = proposed[family][metric], baseline[family][metric]
                values.append(dict(family=family, physical=a, control=b, delta=a-b))
            delta = [r["delta"] for r in values]
            item = dict(zip(GROUP, k))
            item.update(proposed="physical", control=control, metric=metric,
                        delta_definition="physical minus control", lower_is_better=lower_is_better,
                        independent_families=len(values), physical_mean=avg(v["physical"] for v in values),
                        control_mean=avg(v["control"] for v in values), mean_delta=avg(delta),
                        favorable_families=sum(d < -1e-12 if lower_is_better else d > 1e-12 for d in delta),
                        adverse_families=sum(d > 1e-12 if lower_is_better else d < -1e-12 for d in delta),
                        tied_families=sum(abs(d) <= 1e-12 for d in delta), family_values=values)
            output.append(item)
    return output


def opportunity_summary(source):
    grouped = defaultdict(list)
    fields = ("kind", "split", "condition")
    for row in source:
        grouped[key(row, fields)].append(row)
    output = []
    for k, rr in sorted(grouped.items()):
        if len(set(r["family"] for r in rr)) != len(rr):
            raise ValueError(f"Duplicate opportunity family: {k}")
        item = dict(zip(fields, k))
        item.update(independent_families=len(rr), cases=sum(r["cases"] for r in rr),
                    excluded=sum(r["excluded"] for r in rr), mean_gap=avg(r["mean_gap"] for r in rr),
                    positive_gap_rate=avg(ratio(r["positive_gap"], r["cases"]) for r in rr),
                    order_sensitive_rate=avg(ratio(r["order_sensitive"], r["cases"]) for r in rr))
        output.append(item)
    return output


def recommendation(task, comparisons, opportunity, audited):
    """Describe evidence, never promote this reused exploratory sample to a gate."""
    primary = [r for r in comparisons if r["split"] == "test" and r["condition"].startswith("C1")
               and r["revisit"] == 5]
    forecasts = [r for r in primary if r["control"] == "forecast_only"]
    controls = [r for r in primary if r["control"] in ("raw_full", "recent", "uniform", "wrong")]
    residual_positive = bool(forecasts) and all(r["mean_delta"] > 1e-12 and r["adverse_families"] == 0 for r in forecasts)
    unique_positive = bool(controls) and all(r["mean_delta"] > 1e-12 and r["adverse_families"] == 0 for r in controls)
    general = [r for r in opportunity if r["split"] == "test" and r["kind"] == "general"]
    opportunity_exists = any(r["positive_gap_rate"] and r["positive_gap_rate"] > 0 for r in general)
    if not forecasts:
        status = "attribution_control_missing"
        decision = "缺少 forecast_only 对照，尚不能把名义基线之上的收益归因于历史观测或残差。"
    elif not residual_positive:
        status = "residual_increment_not_consistent"
        decision = "在本轮 C1、R5 的延迟条件与家族内，物理组相对 forecast_only 尚未形成方向一致的额外任务收益。"
    elif not unique_positive:
        status = "physical_unique_increment_not_consistent"
        decision = "物理组相对 forecast_only 有探索性正增量，但相对原始历史及匹配核对照尚无一致的独立增益。"
    else:
        status = "exploratory_increment_requires_new_families"
        decision = "本轮出现方向一致的探索性增量；下一步需冻结方案并在新家族和更广布局中复核。"
    return dict(status=status, decision=decision, formal_test=False, qmix_ready=False,
                independent_test_families=sorted({f for r in task if r["split"] == "test" for f in r["families"]}),
                general_opportunity_observed=opportunity_exists,
                consistent_increment_vs_forecast_only=residual_positive,
                consistent_increment_vs_all_matched_controls=unique_positive,
                implementation_audit_present=audited,
                interpretation="内部描述性诊断；一致仅表示本轮各延迟均值为正且无反向家族，不是显著性检验或预注册门槛。",
                next_step=("优先分解 forecast_only 已解释的收益与新增观测的增量，核对改序事件；拓展50秒内可行的截面布局后冻结新实验。"
                           if not unique_positive else "固定观测、模型、评价和选择规则，新增独立家族及一般任务布局；通过后再考虑多机策略。"))


def figure(run, task, comparisons):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    colors = {0: "#207f8c", 10: "#b76839"}
    for delay in sorted({r["delay"] for r in task if r["revisit"] == 5}):
        selected = [r for r in task if r["revisit"] == 5 and r["delay"] == delay and r["split"] == "test" and r["condition"].startswith("C1")]
        values = {r["method"]: r["mean_delta_nominal"] for r in selected}
        available = [m for m in METHODS if m in values]
        axes[0].plot([METHODS.index(m) for m in available], [values[m] for m in available], marker="o", label=f"delay {delay:g} s", color=colors.get(delay))
        cc = [r for r in comparisons if r["revisit"] == 5 and r["delay"] == delay and r["split"] == "test" and r["condition"].startswith("C1")]
        offset = -.09 if delay == 0 else .09
        for row in cc:
            x = CONTROLS.index(row["control"]) + offset
            axes[1].scatter([x]*len(row["family_values"]), [v["delta"] for v in row["family_values"]], color=colors.get(delay), alpha=.6, s=35)
            axes[1].scatter(x, row["mean_delta"], color=colors.get(delay), marker="_", s=260)
    axes[0].set(xticks=range(len(METHODS)), xticklabels=METHODS, ylabel="Mean completed services above nominal", title="C1 / R5: equal family means after phase averaging")
    axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(len(CONTROLS)), xticklabels=CONTROLS, ylabel="Physical minus control (completed services)", title="Three reused diagnostic families; dots = families")
    for ax in axes:
        ax.axhline(0, color="#777", lw=1)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
        ax.grid(axis="y", alpha=.18)
    fig.savefig(run / "decision_value_summary.png", dpi=170)
    plt.close(fig)


def fmt(v, precision=4):
    return "—" if v is None else f"{v:+.{precision}f}"


def pct(v):
    return "—" if v is None else f"{100*v:.2f}%"


def report(run, task, comparisons, opportunity, windows, info, rec, audit):
    lines = ["# 残差到有限救援选择：V1 内部诊断", "", rec["decision"], "",
        "本报告复用已查看的探索家族。train、validation、test 分开呈现；test 仅含3个已查看家族，称为内部诊断，不属于盲测。重复决策时刻、相位、方法和通信条件均不是新的独立样本。", "",
        "## 评价范围与归因限制", "",
        "- 每个初始状态只比较两个目标的两种固定救援顺序，完整飞行与服务均须落在当前已发布预测的50秒范围内。任务读出只覆盖物理图已校准目标截面；一般布局仅用于任务机会诊断。",
        "- 原始未审计运行中，32个截面布局仅3个通过50秒条件；71,928行方法/采样重复记录对应999个唯一 event/layout/time 状态。本轮新增 forecast_only 后重新执行，独立性仍按家族计算；不能把行数用作独立样本数。",
        "- R5 对每个通信延迟分别在同一家族内平均4相位，再对家族等权平均。R10、R40只有零相位，单列展示；它们的差异同时包含采样频率与相位覆盖差异。C0/C1/C2互不混合。",
        "- 固定监测站按合成计划采样，由已有洪水轨迹插值得到观测；更短再访间隔增加观测/通信次数。没有支付真实无人机飞行、等待或服务机会成本，收益不是净主动观测收益。任务读出使用12个计划，时间/核扫描另覆盖包括R2/R20在内的40种日程，见decision_timing_completion.json。",
        "- 残差采用实际存在且发布时间不晚于观测有效时刻减20秒的最近预测版本。实际提前量为20秒至小于30秒，这是兼容密采样的新契约，不能与旧严格20秒匹配实验直接混为同一结果。60秒历史跨度固定，以6个年龄槽保留最近因果可见记录。",
        "- forecast_only 仅含当前预测曲线、时刻、坐标，其历史96槽与汇总9槽归零；这是看过初始结果后新增的归因对照。相对名义顺序的提升可能来自学习预测偏差，不能自动归因于残差。",
        "- 各组均分配134个输入槽，但有效秩和岭正则作用不同；相同槽数不意味着相同有效容量。物理/新近性/均匀/错位组共享原始历史，仅增加不同组织方式的摘要。",
        "- 这里测量预测误差、有限顺序选择和实现收益；不宣称互信息、贝叶斯VoI、主动观测最优性或多机策略验证。旧信息门槛失败结论没有被本轮替换。", "",
        "## 任务中是否存在可挽回损失", "",
        "参考真值只用于离线评分可执行的两种顺序。mean_gap 是有限动作集最优值减名义选择值。布局不按真值收益筛选。表中比例和均值先在家族内计算，再家族等权；cases/excluded 为本条件下记录数，不是独立样本数。", "",
        "|布局|划分|条件|家族数|可评分状态|排除状态|存在正差距|顺序影响收益|平均差距|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in opportunity:
        lines.append(f"|{row['kind']}|{row['split']}|{row['condition']}|{row['independent_families']}|{row['cases']}|{row['excluded']}|{pct(row['positive_gap_rate'])}|{pct(row['order_sensitive_rate'])}|{fmt(row['mean_gap'])}|")
    lines += ["", "## 初始决策的实际收益", "", "下表为平均完成服务数相对名义顺序的差值。正数为增加，负数为损失；每个初始情境只评分一次。完整的绝对完成数、遗憾、改序率及正负收益比例保存在 task_summary.json。", "",
              "|再访/相位|延迟|划分|条件|家族数|" + "|".join(METHODS) + "|",
              "|---|---:|---|---|---:|" + "|".join(["---:"]*len(METHODS)) + "|"]
    grouped = defaultdict(list)
    for row in task:
        grouped[key(row, GROUP)].append(row)
    for k, rr in sorted(grouped.items(), key=lambda kv: (kv[0][1], kv[0][2], SPLITS.get(kv[0][3], 9), kv[0][4])):
        by_method = {r["method"]: r for r in rr}
        lines.append(f"|{k[0]}|{k[2]:g}|{k[3]}|{k[4]}|{rr[0]['independent_families']}|" + "|".join(fmt(by_method[m]["mean_delta_nominal"]) if m in by_method else "—" for m in METHODS) + "|")
    lines += ["", "## 物理组是否有独立增量", "", "以下展示R5内部诊断家族的全部C0/C1/C2配对结果，差值始终为物理组减对照。每个家族值已在四相位内平均。小样本不计算显著性；完整 train/validation/test 及 R10/R40 配对记录保存在 task_comparisons.json。", "",
              "|延迟|条件|对照|平均差值|有利/相同/不利家族|家族配对差值|",
              "|---:|---|---|---:|---|---|"]
    for row in comparisons:
        if row["revisit"] != 5 or row["split"] != "test":
            continue
        pairs = "; ".join(f"{v['family']}: {fmt(v['delta'])}" for v in row["family_values"])
        lines.append(f"|{row['delay']:g}|{row['condition']}|{row['control']}|{fmt(row['mean_delta'])}|{row['favorable_families']}/{row['tied_families']}/{row['adverse_families']}|{pairs}|")
    lines += ["", f"![初始决策收益与家族配对差值]({(run/'decision_value_summary.png').as_posix()})", "", "## 预测与有限改序窗口", "",
              "information_summary.json 分别报告各划分、条件、采样计划和方法的MAE与RMSE；RMSE也是先对相位的RMSE等权平均，再对家族等权，并非所有记录混合计算的RMSE。预测改善与实际服务收益分别呈现。", "",
              "改序窗口只枚举继续初始名义顺序所经过的有限路径前缀，比较在该前缀切换顺序与继续原顺序。captured 表示这些前缀中至少一次抓住收益，不代表闭环策略实际执行，不把多个前缀收益累加。late_only 表示初始时没有收益而在后续前缀存在收益。零可挽回情境的捕获率记为缺失，不能写成0%。", "",
              "|再访/相位|延迟|条件|方法|存在可挽回窗口|捕获率（可挽回情境）|初始获益|仅后续前缀获益|",
              "|---|---:|---|---|---:|---:|---:|---:|"]
    for row in windows:
        if row["split"] != "test" or row["method"] not in ("forecast_only", "raw_full", "physical"):
            continue
        lines.append(f"|{row['scope']}|{row['delay']:g}|{row['condition']}|{row['method']}|{pct(row['correctable_rate_all'])}|{pct(row['capture_rate_correctable'])}|{pct(row['initial_gain_rate_all'])}|{pct(row['late_only_gain_rate_all'])}|")
    lines += ["", "train/validation 的同口径窗口结果及所有方法均保存在 window_summary.json。各家族的原始配对值、相位数和每相位案例数保存在 *_family_phase_averages.json，避免重复相位被误计为独立证据。", "", "## 实现核查与下一步", "",
              ("implementation_audit.json 已提供；测试、旧实现复现和输入哈希核查的具体状态以该文件为准。审计文件存在本身不代表所有检查通过。" if audit is not None else "implementation_audit.json 尚未提供，当前报告的实现核查状态为待补充。"), "",
              rec["next_step"], "", "当前结论仅用于决定下一项诊断。尚不启动正式盲测结论或QMIX阶段。"]
    optional_files = ["implementation_audit.json", "decision_timing_completion.json", "verification.json", "verification_tests.xml", "task_summary.json", "task_comparisons.json", "information_summary.json", "information_comparisons.json", "window_summary.json", "recommendation.json"]
    lines += ["", "## 可复核产物", ""]
    for name in optional_files:
        if (run / name).exists():
            lines.append(f"- [{name}]({(run/name).as_posix()})")
    (run / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finalize(run):
    run = Path(run).resolve()
    sources = {"task": "family_task_scores.json", "information": "family_information_scores.json", "window": "window_family_scores.json"}
    summaries = {}
    comparisons = {}
    for kind, filename in sources.items():
        family = phase_average(rows(read(run / filename)), kind)
        summary = family_summary(family)
        write(run / f"{kind}_family_phase_averages.json", family)
        write(run / f"{kind}_summary.json", summary)
        summaries[kind] = summary
        if kind in ("task", "information"):
            metric = "mean_yield" if kind == "task" else "mae_m"
            comparisons[kind] = paired_comparisons(family, metric, kind == "information")
            write(run / f"{kind}_comparisons.json", comparisons[kind])
    opportunity = opportunity_summary(rows(read(run / "opportunity_summary.json")))
    write(run / "opportunity_aggregate.json", opportunity)
    audit = read(run / "implementation_audit.json", optional=True)
    rec = recommendation(summaries["task"], comparisons["task"], opportunity, audit is not None)
    write(run / "recommendation.json", rec)
    figure(run, summaries["task"], comparisons["task"])
    report(run, summaries["task"], comparisons["task"], opportunity,
           summaries["window"], summaries["information"], rec, audit)
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run), ensure_ascii=False, indent=2))
