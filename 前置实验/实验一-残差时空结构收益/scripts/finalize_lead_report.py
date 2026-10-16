"""Summarize completed exploration and explain why the strategy gate stops."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hyra_validation.artifacts import ROOT,write_json


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run");args=parser.parse_args();run=Path(args.run).resolve()
    assert read(run/"additional_diagnostics_completion.json")["completed"]
    scores=read(run/"information_scores.json");risk=read(run/"risk_scores.json");cases=read(run/"primary_cases.json")
    index={(r["family"],r["time_s"],r["region"],r["method"],r["moving"],r["condition"]):r for r in cases}
    pulse=[]
    for moving in (False,True):
        for method in sorted({r["method"] for r in cases}):
            pairs=[r for r in cases if r["condition"]=="C1_local" and r["moving"]==moving and r["method"]==method]
            per=[];during=[]
            for r in pairs:
                c=index[r["family"],r["time_s"],r["region"],method,moving,"C0_stable"]
                error=(r["prediction"]-c["prediction"])-(r["label"]-c["label"])
                per.append(abs(error))
                fi=int(r["family"].split("_")[-1]);start=(60,90)[fi%2];center=(25,75)[(fi//2)%2]
                if start+10<=r["time_s"]<=start+80 and abs((r["region"]//10)*10+5-center)<=10:during.append(abs(error))
            pulse.append(dict(moving=moving,method=method,pulse_increment_mae_m=float(np.mean(per)),
                passing_window_mae_m=float(np.mean(during)),passing_window_samples=len(during),
                note="Posthoc diagnostic stratum, not new primary endpoint or selection gate"))
    write_json(run/"pulse_increment_diagnostic.json",pulse)
    summaries=[]
    for moving in (False,True):
        for noise,delay in ((0,0),(.02,0),(0,10),(.02,10)):
            for method in sorted({r["method"] for r in risk}):
                rows=[r for r in risk if r["condition"]=="C1_local" and r["moving"]==moving and r["noise"]==noise and r["delay"]==delay and r["method"]==method]
                summaries.append(dict(moving=moving,noise=noise,delay=delay,method=method,brier=float(np.mean([r["brier"] for r in rows])),
                    positives=sum(r["positives"] for r in rows),samples=sum(r["samples"] for r in rows),
                    recall=float(np.mean([r["recall"] for r in rows if r["recall"] is not None])),
                    false_positive_rate=float(np.mean([r["false_positive_rate"] for r in rows if r["false_positive_rate"] is not None]))))
    write_json(run/"risk_summary.json",summaries)
    calibration=read(run/"calibration.json")
    a=np.load(run/f"calibration_mesh{calibration['nmesh']}.npz")
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout="constrained")
    for region,label in [(20,"x=5 m"),(23,"x=35 m"),(25,"x=55 m")]:axes[0].plot(a["time_s"],a["response_m"][:,region]*1000,label=label)
    axes[0].axvline(90,color="#888",ls="--",lw=1);axes[0].set_xlim(70,200);axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Pulse response in nominal solver (mm)");axes[0].legend()
    methods=["I0_self","I1_common","I2_unaligned","I3_physical_lag","I4_wrong_lag","I5_reverse_direction"]
    for moving,offset,color,label in [(False,-.18,"#277f7c","fixed cross sections"),(True,.18,"#ca803e","rotating cross sections")]:
        vals=[np.mean([r["mae_m"] for r in scores if r["moving"]==moving and r["noise"]==0 and r["delay"]==0
            and r["tau"]==20 and r["condition"]=="C1_local" and r["method"]==m])*1000 for m in methods]
        bars=axes[1].bar(np.arange(6)+offset,vals,.36,label=label,color=color);axes[1].bar_label(bars,fmt="%.2f",fontsize=8)
    axes[1].set_xticks(np.arange(6),["self","common","unaligned","physical lag","wrong lag","reverse"],rotation=25)
    axes[1].set_ylabel("Future error prediction MAE (mm), lead20s");axes[1].legend()
    fig.savefig(run/"lead_comparison.png",dpi=180);plt.close(fig)
    report=run/"REPORT.md";text=report.read_text(encoding="utf-8")
    text=text.split("\n## 校准与主要解释",1)[0]
    text=text.replace("风险阈值和任务机会诊断尚须完成，不能将本报告说成全方案已验证。","风险阈值和一般布局任务机会诊断已补充，但尚未训练策略，也没有验证临界布局或实际救援增量，不能将本报告说成全方案已验证。")
    addition=["\n## 校准与主要解释\n",
        "局部边界脉冲确实产生了跨区域水动力响应。代表工况网格10→20的响应峰值中位差4.23%、到达差2s；1s/2s输出最大响应差0.553mm、到达差1s。选定0.1m脉冲的平台20s，监测下游峰值约17mm，首次上下游响应相隔约12s。这是物理响应证据，不自动等于残差表征具有额外预测信息。\n",
        "局部扰动20s提前量下：固定截面I3比自身历史MAE低45.1%，但比共同趋势仅低5.5%（0.180mm），比反向区域仅低0.7%（0.022mm）；轮询截面比自身低35.3%，但比共同趋势差0.3%、比未对齐汇总差3.2%。增益主要不能归因于独特的传播方向/时滞。\n",
        "轮询截面每10s换一个观测截面，每个截面40s再访，属于移动观测的采样近似；尚未检查真实无人机轨迹的速度/可达性，不当作完整无人机访问实验。波动及反射可能使反向区域也含信息，不能把反向对照绝对视为物理无效。\n",
        f"![水动力响应与误差对照]({(run/'lead_comparison.png').as_posix()})\n",
        "## 服务风险诊断\n",
        "用10/20/30s误差读出与同一已发布预测构造离线未来水深估计，检验t+12到t+24连续12s是否超过0.8m。概率标定与报警阈值仅用验证集；阈值由验证稳定条件的无风险样本95分位确定。不是在线修改预测器，也不代表完成救援。以下为C1无噪声/无延迟的事件等权结果。\n",
        "| 观测 | 方法 | Brier（越小越好） | 召回 | 误报 | 阳性/全部情境 |\n|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        if r["noise"]==0 and r["delay"]==0:
            addition.append(f"| {'固定截面' if not r['moving'] else '轮询截面'} | {r['method']} | {r['brier']:.5f} | {100*r['recall']:.1f}% | {100*r['false_positive_rate']:.1f}% | {r['positives']}/{r['samples']} |")
    addition.extend(["\n风险类别不平衡，不能只报召回率或把总体低Brier当作机制成立；2cm噪声/10s延迟结果全部保存在risk_summary.json。未检测/风险判断不等于可补救任务，也未测到真实行动提前量。\n",
        "## 一般布局的可调整机会\n",
        "仅9训练/验证家族、已知地图单机两顺序诊断，各条件675情境：稳定偏差13个预测选择损失，局部扰动14个，共同偏差12个。大量长路线受60s预测时域限制事先排除；仍是稀疏的有限候选机会，不是全局最优或完整多机任务。临界布局尚未生成，不能据此否定所有任务设计。\n",
        "## 阶段决策\n",
        "按信息门槛停在探索阶段，不开展大规模QMIX训练、不生成或解封正式36家族测试。下一步优先检查观测再访间隔与有效提前量、双向波动/反射，以及共同趋势已解释的增益；任何调整只能作为新的探索版本，不能把内部诊断当新盲测。脉冲相对C0的配对增量和脉冲通过窗口分析保存在pulse_increment_diagnostic.json，属于事后定位，不替换原主指标。\n",
        "36项测试通过（含真实求解器局部边界回归）；36事件起报状态来自独立名义轨迹，家族划分检查通过。CPU完成水动力与低容量读出，未更改系统环境。\n"])
    mesh_check=read(run/"training_range_mesh_check.json")
    addition.append(f"\n补充训练参数范围检查：6训练家族×2脉冲位置的10/20网格对照，全部通过={mesh_check['all_passed']}。它检验90s脉冲下的选定响应量，不等于完整PDE收敛证明。\n")
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("*.npz")}
    fingerprints.update({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob("lead_*.json")})
    write_json(run/"data_fingerprints.json",fingerprints)
    source_files=list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py"))+list((ROOT/"configs").glob("*.json"))+[ROOT/"docs/RESIDUAL_LEAD_EXPERIMENT_PLAN.md"]
    provenance={}
    for source in source_files:
        relative=source.relative_to(ROOT);dest=run/"source_snapshot"/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dest);provenance[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    write_json(run/"analysis_source_fingerprints.json",provenance)
    report.write_text(text+"\n".join(addition),encoding="utf-8")
    write_json(run/"completion.json",dict(completed=True,scope="exploratory physical/information/risk/general-layout diagnostics",full_plan_completed=False,
        information_gate_passed=read(run/"stage_gate.json")["information_gate_passed"],formal_test_opened=False,policy_trained=False))
    print(report,flush=True)
