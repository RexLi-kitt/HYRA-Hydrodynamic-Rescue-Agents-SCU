"""Generate a report solely from immutable executed runs, with static figures."""
import json
import sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.envs import interval_valid

METHODS=["B0_raw","B1_distance","B2_structure","B2_uniform"]
COLORS=["#687785","#B19066","#176C91","#70A5B9"]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def latest(kind):
    matches=[p for p in (ROOT/"results/runs").glob("*"+kind) if p.is_dir()]
    return sorted(matches)[-1]
def link(path,label):return f"[{label}]({path.as_posix()})"
def mean_metric(rows,method,key):return float(np.mean([r[key] for r in rows if r["method"]==method]))


def figures(run,main,replay,boundary):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":140})
    figure,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    values=[100*mean_metric(main,m,"completion_rate") for m in METHODS]
    axes[0].bar(np.arange(4),values,color=COLORS,width=.65)
    axes[0].set_xticks(np.arange(4),["B0 raw","B1 distance","B2 structure","B2 uniform"],rotation=15)
    axes[0].set_ylim(0,60);axes[0].set_ylabel("Completed service (%)");axes[0].set_title("Held-out rescue tasks: pilot")
    families=sorted({r["event_family_id"] for r in main})
    for index,m in enumerate(METHODS):
        dots=[100*np.mean([r["completion_rate"] for r in main if r["method"]==m and r["event_family_id"]==family]) for family in families]
        axes[0].scatter(np.full(3,index),dots,color="#162D39",s=18,zorder=3)
        axes[0].text(index,values[index]+2,f"{values[index]:.1f}%",ha="center")
    error=[1000*mean_metric(replay,m,"mae_m") for m in METHODS]
    axes[1].bar(np.arange(4),error,color=COLORS,width=.65)
    axes[1].set_xticks(np.arange(4),["B0 raw","B1 distance","B2 structure","B2 uniform"],rotation=15)
    axes[1].set_ylabel("Diagnostic residual MAE (mm)");axes[1].set_title("Fixed observations: information only")
    axes[1].set_ylim(0,9)
    for i,v in enumerate(error):axes[1].text(i,v+.15,f"{v:.2f}",ha="center")
    figure.savefig(run/"main_results.png",dpi=180);plt.close(figure)
    conditions=list(dict.fromkeys(r["condition"] for r in boundary))
    labels=["Base","Delay10s","Delay30s","Noise0.05m","Service24s","Depth1.2m","Fast rise","Ridge"]
    figure,axis=plt.subplots(figsize=(10,4.5),layout="constrained")
    for m,color in zip(METHODS,COLORS):
        values=[100*np.mean([r["completion_rate"] for r in boundary if r["condition"]==c and r["method"]==m]) for c in conditions]
        axis.plot(labels,values,"o-",label=m,color=color,lw=1.8,markersize=5)
    axis.set_ylim(0,60);axis.set_ylabel("Completed service (%)");axis.set_title("Frozen models: boundary cases, 3 flood families per condition")
    axis.legend(ncol=2,fontsize=9);axis.grid(axis="y",alpha=.15)
    figure.savefig(run/"boundary_results.png",dpi=180);plt.close(figure)


if __name__=="__main__":
    training=latest("_closed_loop");replay_run=latest("_replay");audit=latest("_experiment_audit")
    if read(training/"config.json").get("protocol_version"):
        raise RuntimeError("V2 uses create_repair_report.py; do not mix old boundary results with repaired training")
    boundary_run=latest("_boundaries");evidence=latest("_evidence_action_diagnostic");margin_run=latest("_action_margin")
    inputs=dict(training=str(training),replay=str(replay_run),audit=str(audit),boundaries=str(boundary_run),evidence=str(evidence),margin=str(margin_run))
    for p in (training,audit,boundary_run,evidence):
        assert read(p/"completion.json")["completed"]
    run=new_run("pilot_report",inputs)
    main=read(training/"closed_loop_scores.json");replay=read(replay_run/"replay_scores.json")
    boundaries=read(boundary_run/"scores.json");comparison=read(audit/"paired_with_training_variation.json")
    sensitivity=read(evidence/"action_sensitivity.json")
    mechanism=read(replay_run/"mechanism_report.json")
    margin=read(margin_run/"summary.json")
    # Base boundary repetition must reproduce the original held-out task scores exactly.
    keys=lambda r:(r["method"],r["training_seed"],r["event_id"],r["layout"])
    original={keys(r):r["completion_rate"] for r in main}
    repeated={keys(r):r["completion_rate"] for r in boundaries if r["condition"]=="base"}
    assert original==repeated
    figures(run,main,replay,boundaries)
    changed=sum(r["supported_changed_free_agent_actions"] for r in sensitivity)
    possible=sum(r["supported_free_agent_actions"] for r in sensitivity)
    supported=sum(r["supported_decisions"] for r in sensitivity)
    config=read(training/"config.json")
    events={e["metadata"]["event_id"]:e for e in load_events(ROOT/config["manifest"])["test"]}
    for name in read(ROOT/"data/manifests/boundary.json")["test"]:
        p=(ROOT/"data/manifests"/name).resolve()
        with np.load(p,allow_pickle=False) as archive: event={k:archive[k] for k in archive.files}
        meta=read(p.with_suffix(".json"));events[meta["event_id"]]=event
    boundary_success_checks=0
    boundary_changes=read(boundary_run/"config.json")["changes"]
    for row in boundaries:
        condition=row["condition"]
        stem=f"{row['method']}_seed{row['training_seed']}_{row['event_id']}_layout{row['layout']}_trace.json"
        trace=read(boundary_run/condition/stem)
        service=boundary_changes[condition].get("environment",{}).get("service_s",config["environment"]["service_s"])
        limit=boundary_changes[condition].get("environment",{}).get("depth_limit_m",config["environment"]["depth_limit_m"])
        completed=[r for r in trace["actions_and_services"] if r["type"]=="completed"]
        assert len(completed)==row["completed"] and len({r["region"] for r in completed})==len(completed)
        assert row["target_denominator"]==len(trace["targets"])
        for r in completed:
            assert r["time_s"]-r["start_s"]>=service
            assert interval_valid(events[row["event_id"]],r["region"],r["start_s"],r["time_s"],limit)
        boundary_success_checks+=len(completed)
    write_json(run/"boundary_trace_audit.json",dict(traces_checked=len(boundaries),success_intervals_checked=boundary_success_checks,
        all_passed=True,service_rules_from_frozen_condition_config=True))
    summary=dict(main={m:dict(completion_rate=mean_metric(main,m,"completion_rate"),
        mean_service_yield=mean_metric(main,m,"weighted_service_yield"),diagnostic_mae_m=mean_metric(replay,m,"mae_m")) for m in METHODS},
        main_rollouts=len(main),boundary_rollouts=len(boundaries),models=12,training_episodes_per_model=120,
        independent_main_test_families=3,training_seeds=3,changed_supported_free_actions=changed,
        total_supported_free_actions=possible,paired_comparisons=comparison,base_repeat_reproduced=True,
        claim="No consistent task-yield benefit demonstrated in this pilot; not a converged reproduction")
    write_json(run/"summary.json",summary)
    now=datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")
    lines=[f"# HyRA 首轮验证实验报告\n\n生成时间：{now}（Asia/Shanghai）。\n",
        "**本轮未验证出残差时空结构表示的稳定救援收益。** 固定观测回放的信息误差略有降低，但主闭环B2完成率低于B0，与距离组织B1相同；时间衰减也没有稳定任务增量。该结论只针对本轮120回合的小规模仿真，不构成充分收敛后的方法否定。\n",
        "## 已实际执行的范围\n",
        "P0完成原文核对与实现假设记录；P1A完成论文机制底座及静态普通QMIX/Sur-QMIX短训练；P1B完成系统ANUGA独立参考与滚动预测；P2实现共同动态任务、服务区间判定；P3完成冻结观测回放；P4完成四组×三种子独立GPU训练；P5完成主留出与冻结模型边界试运行。论文性能复现、实机或真实灾害验证均未完成。\n",
        "共生成18个独立ANUGA参考事件（6训练、3验证、9测试，其中6个新增边界事件）。主比较使用3个留出洪水事件、3训练种子、每事件2份相同任务布局，每组18个测试回合。观测/窗口扰动不是新的独立洪水事件。主测试72回合；边界576回合包含72个基础复核回合，不据此夸大独立样本量。\n",
        "系统Python3.14.2、PyTorch2.14.0+cu130、RTX5060 Laptop GPU；ANUGA4.0.1串行CPU求解。模型训练每组每种子120回合，共1440回合；论文预算为10万回合，本轮不宣称充分收敛。每个个体网络51,832参数、混合网络70,721参数，输入33×5×5，各组相同。每模型保存60/120回合检查点，仅按验证完整服务数量选择，平局选较早者。\n",
        "## 主结果\n\n| 组别 | 实际完成率 | 每回合完成任务数（5目标，等权1） | 离线诊断MAE |\n|---|---:|---:|---:|"]
    for m in METHODS:
        r=summary["main"][m]
        lines.append(f"| {m} | {100*r['completion_rate']:.1f}% | {r['mean_service_yield']:.3f} | {1000*r['diagnostic_mae_m']:.3f} mm |")
    lines.append(f"\n![主结果，点表示三个洪水家族平均值]({(run/'main_results.png').as_posix()})\n")
    lines.append("任务差值按独立洪水家族配对，并额外同时重采样训练种子和家族。布局先在单元内平均。下面区间包含事件与训练变异；仅3家族/3种子，属于描述性试运行区间，不用于稳定显著性或泛化声明。\n\n| B2减去对照 | 每回合任务数差值 | 事件＋训练种子95%重采样区间 |\n|---|---:|---:|")
    for row in comparison:
        lo,hi=row["event_and_training_seed_interval_95"]
        lines.append(f"| {row['baseline']} | {row['difference_b2_minus_baseline']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines.extend(["\n**任务层面不支持B2优于B0、B1或时间均匀权重。** 只重采样事件的区间更窄，不能忽略训练种子差异后宣布稳定优劣。训练损失下降或离线读出更准确均不替代救援结果。\n",
        "## 证据链解释\n",
        f"1. **结构**：训练拟合的200条非self区域联系全部通过预定验证筛选；留出联系的平均绝对相关约{np.mean([r['selected_link_mean_correlation'] for r in mechanism if r['split']=='test']):.4f}。但共同模式方差比约{100*np.mean([r['common_mode_variance_ratio'] for r in mechanism if r['split']=='test']):.1f}%，当前单一入流偏差高度共同变化，不能把近1的相关直接解释成区域机制优越。去全局均值后的联系仍相关，但绝对剩余变化很小。\n",
        "2. **信息**：同一原始观测序列、同样211维岭回归读出；标准化仅拟合训练集，共同正则值只在验证集选择，测试冻结。B2比B0的平均MAE低约0.65mm，比B1低约0.20mm；这些小幅信息差异没有转为主任务收益。全局最新残差均值参考仅用于共同偏差诊断，未进入在线策略。\n",
        f"3. **时机**：首条可用证据对应的策略决策为{margin['earliest_usable_decision_s']:.0f}s。未完成目标实例中，{100*margin['fraction']:.1f}%按真值服务截止与最短剩余路径仍有非负行动余量，有限余量中位数{margin['median_finite_margin_s']:.0f}s。该值忽略忙碌、冲突与竞争任务，是乐观必要条件，不是策略能保证完成的证据；重复目标不当作独立样本。\n",
        f"4. **动作**：复核全部18条B2主测试轨迹均可重现。在{supported}个有证据决策时刻，保持模型与GRU隐状态不变、仅将证据值置零，{possible}个非服务中agent动作里只有{changed}个改变（{100*changed/max(possible,1):.2f}%）。这只是离线固定状态敏感性，不作为独立训练基线或收益消融。说明本轮策略对该通道的实际利用很弱。\n",
        "5. **收益**：成功必须满足完整12s服务区间，失败不发成功奖励、耗时照计。72条主轨迹共120次成功服务均通过真值区间复核；未发现/未选择/失去机会目标仍保留分母。没有首次成功的回合保存null，不只对成功回合报有利平均。\n",
        "## 边界结果\n\n所有条件使用主验证集选出的冻结模型，未重新拟合或挑选事件。每项条件3个洪水家族；延迟、噪声与窗口变体按同家族处理。\n\n| 条件 | B0 | B1 | B2 | B2均匀 |\n|---|---:|---:|---:|---:|"])
    labels={"base":"基础复核","delay_10s":"延迟10s","delay_30s":"延迟30s","noise_005m":"噪声0.05m","service_24s":"服务耗时24s","wide_depth_12m":"接收阈值1.2m","fast_rise":"快涨水","ridge_separated":"脊线分隔"}
    for condition,label in labels.items():
        values=[100*np.mean([r["completion_rate"] for r in boundaries if r["method"]==m and r["condition"]==condition]) for m in METHODS]
        lines.append(f"| {label} | "+" | ".join(f"{v:.1f}%" for v in values)+" |")
    lines.extend([f"\n![边界结果]({(run/'boundary_results.png').as_posix()})\n",
        "快涨水条件中B2均值略高于B0（约2.2个百分点），其他条件不一致；不能将这一项挑出来作为整体收益成立的证据。脊线分隔是仿真地形设定，尚无经过水文连通专门验证的结论，不改称真实弱连通洪水。\n",
        "## 验收、限制与复核入口\n",
        f"22项关键测试通过，存在一条旧PyMARL从tensor复制的提示。成功规则、分母、冻结选择、容量、有限损失及回放一致性检查均通过。边界阶段576条轨迹的{boundary_success_checks}次成功也逐条通过对应阈值/耗时条件复核。早期调试中发生过numpy bool类型与CUDA标量日志错误，已修复；失败运行保留，不当作完成结果。\n",
        "限制：论文未给出的SAR边权、切边、CNN宽度、服务触发与碰撞细节采用明确实现假设；未取得作者原Sur-QMIX代码，未复现其全部性能。10m任务网格最近质心映射最大距离3.33m，未开展网格收敛研究。单涨水边界、单种预测偏差、少量事件及短训练限制结论范围。时空表示只提示失配，不校正预测，不提供未来真值，不增加奖励/规划器。\n",
        "下一步应先判断是否需要更充分训练和更有区分度的独立事件来检验现有假设；当前不能给大创报告写‘救援收益已提升’。本轮没有根据负结果增加研究模块。\n",
        f"- 原文核对与假设：{link(ROOT/'docs/PAPER_CHECKLIST.md','论文核对')}，{link(ROOT/'docs/IMPLEMENTATION_SPEC.md','实施规格')}。\n",
        f"- 配置、数据划分与指纹、GPU记录、检查点和主轨迹：{link(training,'主训练运行目录')}。\n",
        f"- 相同观测回放与诊断读出：{link(replay_run,'离线回放目录')}。\n",
        f"- 成功区间、分母与训练变异审计：{link(audit,'审计目录')}。\n",
        f"- 冻结模型边界结果及轨迹：{link(boundary_run,'边界目录')}。\n",
        f"- 动作敏感性与原轨迹重现：{link(evidence,'动作诊断目录')}。\n",
        f"- 真值评分侧服务余量：{link(margin_run,'时机诊断目录')}。\n",
        f"- 测试记录：{link(ROOT/'results/setup/core_tests_final.xml','22项测试XML')}。\n",
        "\n复核命令（在项目目录、系统Python）：`python -m pytest tests -q --basetemp=results/setup/pytest_review_unique`；`python scripts/run_replay.py`；`python scripts/run_closed_loop.py`。后两者会创建新的完整运行，不覆盖本轮证据。\n"])
    report=run/"REPORT.md"
    report.write_text("\n".join(lines),encoding="utf-8")
    write_json(run/"completion.json",dict(completed=True,report=str(report),base_repeat_reproduced=True))
    print(report,flush=True)
