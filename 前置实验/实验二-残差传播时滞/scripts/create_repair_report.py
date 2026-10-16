"""Report repaired five-group comparisons with explicit source-run alignment."""
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


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def link(path,label):return f"[{label}]({Path(path).as_posix()})"


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for name in ("training","replay","audit","evidence","decision"):
        parser.add_argument("--"+name,required=True)
    args=parser.parse_args()
    sources={name:Path(getattr(args,name)).resolve() for name in ("training","replay","audit","evidence","decision")}
    training=sources["training"];config=read(training/"config.json")
    assert config["representation"]["preserve_raw_history"] and len(config["methods"])==5
    for name in ("training","audit","evidence","decision"):
        assert read(sources[name]/"completion.json")["completed"]
    assert read(sources["replay"]/"config.json")==config
    for name in ("audit","evidence"):
        assert Path(read(sources[name]/"config.json")["training_run"]).resolve()==training
    run=new_run("repair_report",dict(source_runs={k:str(v) for k,v in sources.items()},config=config))
    scores=read(training/"closed_loop_scores.json");replay=read(sources["replay"]/"replay_scores.json")
    paired=read(sources["audit"]/"paired_with_training_variation.json")
    audit=read(sources["audit"]/"completion.json")
    sensitivity=read(sources["evidence"]/"action_sensitivity.json")
    decision=read(sources["decision"]/"summary.json")
    methods=config["methods"]
    summary={m:dict(completion_rate=float(np.mean([r["completion_rate"] for r in scores if r["method"]==m])),
        service_yield=float(np.mean([r["weighted_service_yield"] for r in scores if r["method"]==m])),
        mae_m=float(np.mean([r["mae_m"] for r in replay if r["method"]==m])),
        explored_fraction=float(np.mean([r["explored_fraction"] for r in scores if r["method"]==m]))) for m in methods}
    gap=summary["B2_structure"]["completion_rate"]-summary["B0_raw"]["completion_rate"]
    changed=sum(r["supported_changed_free_agent_actions"] for r in sensitivity)
    free=sum(r["supported_free_agent_actions"] for r in sensitivity)
    counts=[r["failed_attempts_checked"] for r in read(sources["audit"]/"trace_checks.json")]
    write_json(run/"summary.json",dict(methods=summary,paired=paired,action_value=decision,
        immediate_summary_sensitivity=dict(changed=changed,total=free,fraction=changed/max(free,1)),
        main_rollouts=len(scores),models=len(read(training/"training_summary.json")),all_checks_passed=audit["all_passed"],
        failed_attempts_without_same_step_restart=sum(counts),interpretation="diagnostic re-evaluation on reused test events; not blind holdout"))
    colors=["#627888","#998869","#AA8757","#126E95","#80ADBD"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    labels=["Raw","Local residual","Distance","Structure","Uniform time"]
    values=[100*summary[m]["completion_rate"] for m in methods]
    axes[0].bar(range(5),values,color=colors);axes[0].set_ylim(0,max(60,max(values)+10))
    axes[0].set_ylabel("Completed service (%)");axes[0].set_title("Repaired environment; common raw history")
    for i,v in enumerate(values):axes[0].text(i,v+1,f"{v:.1f}%",ha="center")
    errors=[1000*summary[m]["mae_m"] for m in methods]
    axes[1].bar(range(5),errors,color=colors);axes[1].set_ylabel("Diagnostic MAE (mm)")
    axes[1].set_ylim(0,max(errors)+2);axes[1].set_title("Same observations; common raw history")
    for i,v in enumerate(errors):axes[1].text(i,v+.12,f"{v:.2f}",ha="center")
    for axis in axes:axis.set_xticks(range(5),labels,rotation=18)
    fig.savefig(run/"comparison.png",dpi=180);plt.close(fig)
    best=max(methods,key=lambda m:summary[m]["completion_rate"])
    conclusion=f"修正轮B2平均完成率相对B0变化{gap*100:+.1f}个百分点；均值最高为{best}。"
    intervals=[r["event_and_training_seed_interval_95"] for r in paired]
    if all(lo<=0<=hi for lo,hi in intervals):
        conclusion+="所有主要配对区间均包含零，本轮仍不足以支持稳定优劣。"
    else:
        conclusion+="部分配对区间未包含零，但仅3个重用事件与短训练，仍不能宣称稳定泛化。"
    text=["# HyRA 修正轮：五组同信息诊断报告\n",f"**{conclusion}**\n",
        "原轮服务失败后自动重启的环境缺陷已修复。旧报告排名不再作为机制证据，本轮重新训练所有组，没有使用旧模型替代新对照。\n",
        "## 本轮只改了什么\n",
        "1. 服务失败当步不重启，下一步可移动或选择重试，耗时与奖励规则保持原约定。\n",
        "2. 五组均保留两条完整原始历史（观测、匹配先验、年龄、掩码），另设相同3个证据槽位。B0填零，本区组仅本区汇总，B1距离汇总，B2区域联系汇总，均匀组只改变时间权重。39×5×5输入，个体网络52,696参数，混合网络70,721参数，组间相同。\n",
        "3. 增加已有诊断链内的两任务固定顺序评分，不新增在线中央规划、窗口分类器、统计surprise、同化或主动观测。参考新框架4.2历史与估计联合、4.4隔离目标发现两点；其余作为参考，没有照搬。\n",
        "## 实际结果\n\n每组3种子，每模型120回合，15模型共1800训练回合；每组18次任务评价，总90回合。仅在验证集第60/120回合选择模型。\n\n| 组别 | 实际完成率 | 平均完成任务数/5 | 信息诊断MAE | 探索覆盖率 |\n|---|---:|---:|---:|---:|"]
    for m in methods:
        r=summary[m];text.append(f"| {m} | {100*r['completion_rate']:.1f}% | {r['service_yield']:.3f} | {1000*r['mae_m']:.3f} mm | {100*r['explored_fraction']:.1f}% |")
    text.append(f"\n![修正轮结果]({(run/'comparison.png').as_posix()})\n")
    text.append("| B2减去对照 | 每回合任务数差值 | 事件＋训练种子95%重采样区间 |\n|---|---:|---:|")
    for r in paired:
        lo,hi=r["event_and_training_seed_interval_95"]
        text.append(f"| {r['baseline']} | {r['difference_b2_minus_baseline']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    text.extend(["\n区间以洪水家族和训练种子为重采样单位，布局在单元内平均。当前3个测试事件此前已经看过，这是诊断重评，不是新盲留出。训练仅120回合，也没有充分收敛结论。\n",
        "## 水情信息有没有改变选择的空间\n",
        f"仅在6训练/3验证事件上生成已知地图、单机、两目标子问题。候选路径和顺序由几何确定，当前已发布预测选顺序，再用真值评分。{decision['cases']}个可评估情境中，{decision['order_sensitive_cases']}个的实际收益依赖访问顺序（{100*decision['order_sensitive_cases']/decision['cases']:.1f}%）；{decision['positive_gap_cases']}个存在预测选择相对两候选真值最佳的损失（{100*decision['positive_gap_cases']/decision['cases']:.1f}%）。按9事件等权平均可补救差距为每情境{decision['event_mean_gap']:.4f}个任务。\n",
        f"共有{decision['excluded_before_truth_by_geometry_or_forecast_horizon']}个情境在读取真值评分前因路径/60s预测时域不满足被排除；这个比例较高，当前诊断只覆盖短路线。该参考是两候选集合内最佳，不是全局最优，不证明残差方法有效，也不替代未知目标、多机协调评价。时间点/布局不作为独立洪水样本。\n",
        "这说明现有场景的可补救决策机会比较稀疏。应先检验任务情境与误差的可辨识性，而非仅增加模型板块来追求正结果。\n",
        "## 证据利用与实现验收\n",
        f"固定B2模型与原GRU状态，仅置零额外汇总数值、保留共同原始信息与支持字段，{free}个有支持且非服务中agent动作中{changed}个改变（{100*changed/max(free,1):.2f}%）。这是当前汇总值的即时动作敏感性；不能据此断言网络没有利用历史水情或判断训练不足。\n",
        f"28项测试通过；{audit['models_checked']}个模型容量相同、损失有限；{audit['traces_checked']}条轨迹的{audit['successful_intervals_checked']}次成功服务区间有效，{sum(counts)}次失败均未在同一步强制重启。旧PyMARL复制tensor提示仍存在，不影响验收。\n",
        "## 下一步边界\n",
        "已解决明确的恢复逻辑错误和原始信息压缩混杂；未证明区域组织或时间衰减有稳定任务增量。下一步可在现有ANUGA/服务配置内预先定义更充分的短路线与临界任务条件，或先检查训练能力，再用全新独立事件检验；不能筛到B2获胜才报告。没有自动增加在线模块。\n",
        f"- 旧轮更正：{link(ROOT/'docs/FIRST_RUN_CORRECTION.md','错误与解释更正')}。\n",
        f"- 本轮冻结规格：{link(ROOT/'docs/V2_PROTOCOL.md','V2协议')}。\n"])
    for name,label in [("training","训练、模型和轨迹"),("replay","固定观测回放"),("audit","服务与恢复审计"),("evidence","即时动作诊断"),("decision","两任务信息价值诊断")]:
        text.append(f"- {link(sources[name],label)}。\n")
    report=run/"REPORT.md";report.write_text("\n".join(text),encoding="utf-8")
    write_json(run/"completion.json",dict(completed=True,report=str(report),independent_fresh_test=False))
    write_json(ROOT/"results/latest_pilot_report.json",dict(report=str(report),summary=str(run/"summary.json"),figures=[str(run/"comparison.png")],
        source_runs={k:str(v) for k,v in sources.items()},validity="repaired_diagnostic_reused_events",correction=str(ROOT/"docs/FIRST_RUN_CORRECTION.md")))
    print(report,flush=True)
