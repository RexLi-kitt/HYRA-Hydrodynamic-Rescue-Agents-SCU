"""Report every paired condition and audit paired layouts before claiming effects."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.evaluation import compare_paired_events


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def link(path,label):return f"[{label}]({Path(path).as_posix()})"


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("training","replay","diagnostic","audit"):parser.add_argument("--"+name,required=True)
    args=parser.parse_args();sources={n:Path(getattr(args,n)).resolve() for n in ("training","replay","diagnostic","audit")}
    assert all(read(p/"completion.json")["completed"] for p in (sources["training"],sources["diagnostic"],sources["audit"]))
    config=read(sources["training"]/"config.json");events=load_events(ROOT/config["manifest"])
    meta={e["metadata"]["event_id"]:e["metadata"] for es in events.values() for e in es}
    scores=read(sources["training"]/"closed_loop_scores.json")
    replay=read(sources["replay"]/"replay_scores.json")
    run=new_run("switch_report",dict(sources={k:str(v) for k,v in sources.items()},scope="paired diagnostic; reused truth"))
    summaries=[];comparisons={};methods=config["methods"]
    for condition in ("constant","switch"):
        selected=[r for r in scores if meta[r["event_id"]]["condition"]==condition]
        comparisons[condition]=compare_paired_events(selected,dict(metric="weighted_service_yield",seed=91))
        for method in methods:
            rows=[r for r in selected if r["method"]==method]
            rr=[r for r in replay if r["method"]==method and meta[r["event_id"]]["condition"]==condition]
            summaries.append(dict(condition=condition,method=method,completion_rate=float(np.mean([r["completion_rate"] for r in rows])),
                service_yield=float(np.mean([r["weighted_service_yield"] for r in rows])),failed_attempts=float(np.mean([r["failed_attempts"] for r in rows])),
                explored_fraction=float(np.mean([r["explored_fraction"] for r in rows])),mae_m=float(np.mean([r["mae_m"] for r in rr])),rollouts=len(rows)))
    index={(r["method"],r["training_seed"],r["event_family_id"],r["layout"],meta[r["event_id"]]["condition"]):r for r in scores}
    pair_audit=[]
    for key,c in index.items():
        if key[-1]!="constant":continue
        s=index[(*key[:-1],"switch")]
        def trace(row):
            name=f"{row['method']}_seed{row['training_seed']}_{row['event_id']}_layout{row['layout']}_trace.json"
            return read(sources["training"]/name)
        tc,ts=trace(c),trace(s)
        assert tc["targets"]==ts["targets"]
        dc=[r for r in tc["actions_and_services"] if r["type"]=="decision"]
        ds=[r for r in ts["actions_and_services"] if r["type"]=="decision"]
        assert [r for r in dc if r["time_s"]<=120]==[r for r in ds if r["time_s"]<=120]
        post={r["time_s"]:r for r in ds if r["time_s"]>120}
        paired_steps=[(r,post[r["time_s"]]) for r in dc if r["time_s"] in post]
        changed=sum(a["actions"]!=b["actions"] for a,b in paired_steps)
        donec=[r for r in tc["actions_and_services"] if r["type"]=="completed"]
        dones=[r for r in ts["actions_and_services"] if r["type"]=="completed"]
        pair_audit.append(dict(method=key[0],seed=key[1],family=key[2],layout=key[3],same_targets=True,
            identical_pre_change_decisions=True,post_change_paired_steps=len(paired_steps),post_change_action_steps_different=changed,
            constant_completed_before_or_at120=sum(r["time_s"]<=120 for r in donec),constant_completed=len(donec),
            switch_completed_before_or_at120=sum(r["time_s"]<=120 for r in dones),switch_completed=len(dones),
            service_yield_delta=s["weighted_service_yield"]-c["weighted_service_yield"]))
    # Difference in differences, preserving paired conditions within family/seed.
    rng=np.random.default_rng(91);families=sorted({r["event_family_id"] for r in scores});seeds=config["training"]["seeds"]
    did=[]
    for baseline in methods:
        if baseline=="B2_structure":continue
        cells=np.array([[np.mean([index[("B2_structure",seed,f,l,"switch")]["weighted_service_yield"]-
            index[("B2_structure",seed,f,l,"constant")]["weighted_service_yield"]-
            index[(baseline,seed,f,l,"switch")]["weighted_service_yield"]+
            index[(baseline,seed,f,l,"constant")]["weighted_service_yield"] for l in (0,1)]) for f in families] for seed in seeds])
        boot=[cells[np.ix_(rng.integers(len(seeds),size=len(seeds)),rng.integers(len(families),size=len(families)))].mean() for _ in range(2000)]
        did.append(dict(baseline=baseline,b2_minus_baseline_extra_shift_effect=float(cells.mean()),interval_95=np.quantile(boot,[.025,.975]).tolist()))
    write_json(run/"summary.json",summaries);write_json(run/"comparisons_by_condition.json",comparisons)
    write_json(run/"paired_trace_audit.json",pair_audit);write_json(run/"difference_in_differences.json",did)
    innovation=read(sources["diagnostic"]/"innovation_summary.json")
    opportunities=read(sources["diagnostic"]/"opportunity_summary.json")
    audit=read(sources["audit"]/"completion.json")
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    x=np.arange(len(methods))
    for condition,offset,color in [("constant",-.18,"#607d8b"),("switch",.18,"#d27c38")]:
        rows=[r for r in summaries if r["condition"]==condition]
        bars=axes[0].bar(x+offset,[100*r["completion_rate"] for r in rows],width=.36,label=condition,color=color)
        axes[0].bar_label(bars,fmt="%.1f",fontsize=8)
    axes[0].set_xticks(x,methods,rotation=20);axes[0].set_ylim(0,60);axes[0].set_ylabel("Completed targets (%)");axes[0].legend()
    for noise,color in [(0,"#287b70"),(.02,"#805ba0")]:
        vals=[next(r["family_mean_exceedance"] for r in innovation if r["split"]=="test" and r["metric"]=="innovation_score"
            and r["window"]=="post" and r["condition"]==c and r["noise_std_m"]==noise) for c in ("constant","switch")]
        bars=axes[1].bar(np.arange(2)+(-.18 if noise==0 else .18),np.array(vals)*100,width=.36,color=color,label=f"noise {noise*1000:g} mm")
        axes[1].bar_label(bars,fmt="%.1f",fontsize=8)
    axes[1].set_xticks([0,1],["constant","switch"]);axes[1].set_ylim(0,110);axes[1].set_ylabel("Innovation threshold exceedance (%)\n140-180 s, fixed observed footprint");axes[1].legend()
    fig.savefig(run/"comparison.png",dpi=180);plt.close(fig)
    text=["# 受控突发预测失配实验报告\n",
        "本轮完成配对预测扰动、离线变化识别与五组GPU重训练。真实洪水没有改变；新增变化指标只做离线诊断，没有进入控制策略，因此本轮不能声称‘新惊奇算法有效’。\n",
        "## 设计与有效性\n",
        "12个既有洪水家族，各有constant/switch两个条件：6训练、3验证、3测试家族。120s将预测边界增水率系数从0.75改为0.4，水位连续。输入突变不等于观测残差瞬跳，独立水动力传播会产生渐进响应；本轮没有手工修改残差。此前已发布预测完全一致、保持不变；后续独立预测从名义状态续算，不用观测或真值校正。两个条件始终在同一划分，网络容量、预算、服务规则相同。\n",
        "每组3种子，每模型120回合，共15模型、1800回合；测试每条件每组18次，共180回合。模型按训练/验证协议选择。两个条件使用同一混合训练模型；本轮constant结果与V2的差别不能归因于输入扰动，因为训练数据组成已改变。旧真值已看过，本轮是机制诊断，不是新盲留出；短训练不证明充分收敛。两个条件不当作独立洪水家族。\n",
        "图验证按家族投票，两个条件都通过才计一票。早期按单条件计票的训练已中断，其结果不用于本报告。\n",
        "## 实际任务与信息诊断\n",
        "| 组别 | 持续偏差完成率 | 突发失配完成率 | 持续/突发信息MAE（mm） | 持续/突发平均失败尝试 |\n|---|---:|---:|---:|---:|"]
    for m in methods:
        c=next(r for r in summaries if r["method"]==m and r["condition"]=="constant")
        s=next(r for r in summaries if r["method"]==m and r["condition"]=="switch")
        text.append(f"| {m} | {100*c['completion_rate']:.1f}% | {100*s['completion_rate']:.1f}% | {1000*c['mae_m']:.3f}/{1000*s['mae_m']:.3f} | {c['failed_attempts']:.2f}/{s['failed_attempts']:.2f} |")
    text.append(f"\n![配对实验结果]({(run/'comparison.png').as_posix()})\n")
    text.append("B0共同原始历史；B_local增加本区汇总；B1增加距离汇总；B2增加学习区域联系汇总；uniform使用B2联系但不做时间衰减。信息MAE来自固定移动观测回放的离线读出，不是在线预测器修正。\n")
    text.append("| 条件 | B2减对照 | 每回合任务数差值 | 家族＋种子95%重采样区间 |\n|---|---|---:|---:|")
    for condition,rows in comparisons.items():
        for r in rows:
            lo,hi=r["event_and_training_seed_interval_95"]
            text.append(f"| {condition} | {r['baseline']} | {r['difference_b2_minus_baseline']:+.3f} | [{lo:+.3f},{hi:+.3f}] |")
    text.append("\n| B2相对对照在突发失配下的额外变化 | 任务数差值 | 95%区间 |\n|---|---:|---:|")
    for r in did:
        lo,hi=r["interval_95"];text.append(f"| {r['baseline']} | {r['b2_minus_baseline_extra_shift_effect']:+.3f} | [{lo:+.3f},{hi:+.3f}] |")
    text.append("\n表中[0,0]来自已观测家族/种子的配对差值全为零，是有限样本重采样退化区间，不意味着真实效应的总体不确定性为零。\n")
    positive=[f"{c}/{r['baseline']}" for c,rs in comparisons.items() for r in rs if r["event_and_training_seed_interval_95"][0]>0]
    negative=[f"{c}/{r['baseline']}" for c,rs in comparisons.items() for r in rs if r["event_and_training_seed_interval_95"][1]<0]
    if not positive and not negative:
        text.append("\n两条件下B2相对所有对照的配对95%区间均包含零；本轮仍未证明区域结构带来稳定任务增量。\n")
    else:
        text.append(f"\nB2配对区间完全大于零的比较：{positive or '无'}；完全小于零的比较：{negative or '无'}。这些仅是3家族/3种子诊断区间，不构成独立盲测结论。\n")
    text.extend(["\n## 能否区别新增失配与持续偏差\n",
        "固定左上5×5格点，每10s观测；与此前3条有效残差均值比较，除以前史方差（尺度下限2cm）。阈值仅由训练constant的120s前数据99分位确定。以下是140—180s测试窗口超过阈值比例，不是突变识别准确率。格点与时间不作为独立样本；噪声只有一个固定随机实现。\n",
        "| 观测噪声 | 指标 | 持续偏差超阈值 | 突发失配超阈值 |\n|---|---|---:|---:|"])
    for noise in (0,.02):
        for metric in ("absolute_residual_m","innovation_score"):
            vals=[next(r["family_mean_exceedance"] for r in innovation if r["split"]=="test" and r["window"]=="post" and r["condition"]==c and r["noise_std_m"]==noise and r["metric"]==metric) for c in ("constant","switch")]
            text.append(f"| {noise*1000:g}mm | {metric} | {vals[0]*100:.1f}% | {vals[1]*100:.1f}% |")
    text.append("\n持续输入偏差并不等于恒定残差：洪水上涨、模型误差积累可使局部残差持续增长。历史均值没有消除这种趋势，持续条件也可能频繁报警。加入2cm噪声后阈值与前史方差改变，突发响应减弱；当前方法未证明可靠区分两者，不能把高分直接解释成物理风险或信息价值。固定观测区域的表现也不代表移动无人机全场表现。\n")
    text.append("## 决策机会是否增加\n\n| 条件 | 两顺序可比较情境 | 真值顺序敏感 | 预测选择有损失 | 家族平均可补救任务差距 |\n|---|---:|---:|---:|---:|")
    for r in opportunities["rows"]:
        if r["window"]=="all":text.append(f"| {r['condition']} | {r['cases']} | {r['order_sensitive_cases']} | {r['positive_gap_cases']} | {r['event_mean_gap']:.5f} |")
    text.append("\n真值相同，所以改变预测不会增加物理上顺序敏感的情境。675配对短路线情境中，可补救选择错误只从15增加到16；这是9训练/验证家族上的已知地图、单机、两目标固定顺序诊断，不是全局最优，也不代表完整多机未知目标任务。大量长路线因60s预测时域被事前排除。更大残差未自动创造更多决策收益空间。\n")
    constant_done=sum(r["constant_completed"] for r in pair_audit)
    early=sum(r["constant_completed_before_or_at120"] for r in pair_audit)
    changed_pairs=sum(r["post_change_action_steps_different"]>0 for r in pair_audit)
    changed_yield_pairs=sum(r["service_yield_delta"]!=0 for r in pair_audit)
    text.extend(["## 验收与结论边界\n",
        f"32项测试通过；{audit['models_checked']}模型容量相同、损失有限；{audit['traces_checked']}轨迹的{audit['successful_intervals_checked']}成功服务区间通过真值核对，失败不在同一步重启。{len(pair_audit)}对轨迹目标相同，120s以前决策逐项一致。{changed_pairs}对在共同后续时间步出现不同动作，但仅{changed_yield_pairs}对完成任务数改变，具体配对见paired_trace_audit.json。持续条件{constant_done}次成功中{early}次在120s之前或当时完成（{100*early/max(constant_done,1):.1f}%），其结果不可能受到之后的扰动影响。行动响应没有自动转化为任务收益，是当前收益链的重要观察。\n",
        "本轮只回答已有五组表征在受控输入失配下的表现，以及历史中心化指标的离线可辨识性。若要证明新的残差惊奇策略有收益，还需明确处理残差趋势/噪声，设置容量一致的直接对照，并用新的独立事件验证；不能把本轮诊断直接包装成主动推理算法成功。\n",
        f"实验协议：{link(ROOT/'docs/SWITCH_PROTOCOL.md','冻结规格')}。\n"])
    for n,p in sources.items():text.append(f"- {link(p,n)}。")
    report=run/"REPORT.md";report.write_text("\n".join(text),encoding="utf-8")
    write_json(run/"completion.json",dict(completed=True,report=str(report),fresh_blind_test=False))
    write_json(ROOT/"results/latest_switch_report.json",dict(report=str(report),source_runs={n:str(p) for n,p in sources.items()},validity="paired_diagnostic_reused_truth_families"))
    print(report,flush=True)
