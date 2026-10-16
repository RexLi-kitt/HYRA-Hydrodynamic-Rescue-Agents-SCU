"""Report all matched-controls results without promoting exploratory horizons."""
import argparse
import hashlib
import itertools
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hyra_validation.artifacts import ROOT,write_json
from audit_lead_direction import classify_offline_window


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def window_scores(run, direction):
    rows=[]
    for path in sorted(run.glob('predictions_*.npz')):
        name,tau_text=path.stem.removeprefix('predictions_').rsplit('_t',1)
        tau=int(tau_text)
        with np.load(path,allow_pickle=False) as a:
            labels=a['label']; predictions=a['prediction']; methods=a['methods']
            flags=[]
            for time,start,region,center in zip(a['time_s'],a['pulse_start_s'],a['region'],a['pulse_center_y_m']):
                flag=classify_offline_window(float(time+tau),float(start),int(region),direction['training_windows'])
                same_branch=abs((int(region)//10)*10+5-float(center))<=10
                flags.append({k:bool(v and same_branch) for k,v in flag.items()})
            for family in np.unique(a['family']):
                base=(a['family']==family)&(a['condition']=='C1_local')
                for stratum in ('early_response','pre_later_sensor','later_response'):
                    mask=base & np.array([r[stratum] for r in flags])
                    if not mask.any():continue
                    for method,p in zip(methods,predictions):
                        rows.append(dict(name=name,tau=tau,family=str(family),method=str(method),stratum=stratum,
                            samples=int(mask.sum()),mae_m=float(np.mean(abs(p[mask]-labels[mask])))))
    write_json(run/'window_scores.json',dict(rows=rows,
        definition='C1, same pulse branch only; classify LABEL valid time now+tau using training nominal windows shifted by simulated event onset',
        online_feature=False,used_to_change_gate=False))
    comparisons=[]
    for scope in ('fixed_n0_d0_p0','rotating_n0_d0_p0','rotating_phase_average'):
        for tau in (10,20,30):
            for stratum in ('early_response','pre_later_sensor','later_response'):
                group=[r for r in rows if r['tau']==tau and r['stratum']==stratum and
                       (r['name']==scope if scope!='rotating_phase_average' else
                        r['name'] in [f'rotating_n0_d0_p{p}' for p in range(4)])]
                for baseline in ('I1_common','N3_uniform_norm','N2_recent_norm','N4_wrong_norm'):
                    a=[];b=[];counts=[]
                    for family in sorted({r['family'] for r in group}):
                        aa=[r for r in group if r['family']==family and r['method']=='N1_physical_norm']
                        bb=[r for r in group if r['family']==family and r['method']==baseline]
                        if aa and bb:
                            a.append(np.mean([r['mae_m'] for r in aa]));b.append(np.mean([r['mae_m'] for r in bb]))
                            counts.append(dict(family=family,samples=sum(r['samples'] for r in aa)))
                    if not a:continue
                    a=np.array(a);b=np.array(b);d=a-b
                    boot=np.array([np.mean(v) for v in itertools.product(d,repeat=len(d))])
                    comparisons.append(dict(name=scope,tau=tau,stratum=stratum,baseline=baseline,
                        proposed='N1_physical_norm',relative_reduction=1-a.mean()/b.mean(),
                        mae_delta_m=d.mean(),interval_95=np.quantile(boot,[.025,.975]),
                        favorable_families=int(sum(d<0)),independent_families=len(d),counts=counts))
    write_json(run/'window_comparisons.json',comparisons)
    return comparisons


def figure(run, comparisons, timing):
    fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    colors={'fixed_n0_d0_p0':'#177f83','rotating_n0_d0_p0':'#cc7938','rotating_phase_average':'#865eae'}
    for name,label in [('fixed_n0_d0_p0','Fixed / delay 0 s'),('rotating_n0_d0_p0','Rotating / delay 0 s'),('rotating_n0_d10_p0','Rotating / delay 10 s')]:
        yy=[next(r['horizon_target_rate']*100 for r in timing if r['name']==name and r['tau']==tau and r['wait_s']==6) for tau in (10,20,30)]
        axes[0].plot([10,20,30],yy,marker='o',label=label)
    axes[0].set(xlabel='Forecast horizon (s)',ylabel='Nominal timing coverage (%)',ylim=(-2,105),xticks=[10,20,30])
    axes[0].legend(fontsize=8)
    for name,label in [('fixed_n0_d0_p0','Fixed'),('rotating_n0_d0_p0','Rotating / original phase'),('rotating_phase_average','Rotating / mean of 4 phases')]:
        rr=[next(r for r in comparisons if r['name']==name and r['tau']==tau and r['condition']=='C1_local' and r['stratum']=='all'
             and r['proposed']=='N1_physical_norm' and r['baseline']=='N2_recent_norm') for tau in (10,20,30)]
        axes[1].plot([10,20,30],[r['relative_reduction']*100 for r in rr],marker='o',label=label,color=colors[name])
    axes[1].axhline(0,color='#777',lw=1);axes[1].axhline(5,color='#aaa',ls='--',lw=1)
    axes[1].set(xlabel='Forecast horizon (s)',ylabel='MAE reduction vs matched recency (%)',xticks=[10,20,30]);axes[1].legend(fontsize=8)
    for baseline,label in [('I1_common','vs common trend'),('N3_uniform_norm','vs uniform history'),('N2_recent_norm','vs matched recency')]:
        rr=[next(r for r in comparisons if r['name']==f'rotating_n0_d0_p{p}' and r['tau']==20 and r['condition']=='C1_local'
            and r['stratum']=='all' and r['proposed']=='N1_physical_norm' and r['baseline']==baseline) for p in range(4)]
        axes[2].plot(range(4),[r['relative_reduction']*100 for r in rr],marker='o',label=label)
    axes[2].axhline(0,color='#777',lw=1)
    axes[2].set(xlabel='Rotating sampling phase',ylabel='MAE reduction at horizon 20 s (%)',xticks=range(4));axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(run/'timing_diagnostic.png',dpi=180);plt.close(fig)


def finalize(run):
    info=read(run/'information_completion.json'); assert info['completed']
    direction=read(run/'direction_audit.json'); gate=read(run/'diagnostic_gate.json')
    comps=read(run/'comparisons.json'); timing=read(run/'timing_summary.json')
    windows=window_scores(run,direction); figure(run,comps,timing)
    source=read(run/'input_source.json'); source_run=Path(source['run'])
    changed=[name for name,digest in source['files'].items() if hashlib.sha256((source_run/name).read_bytes()).hexdigest()!=digest]
    assert not changed, f'Original run changed: {changed}'
    old_gate=read(source_run/'stage_gate.json'); assert not old_gate['information_gate_passed']
    rep=read(run/'legacy_reproduction.json')
    def get(name,tau,proposed,baseline):
        return next(r for r in comps if r['name']==name and r['tau']==tau and r['condition']=='C1_local'
                    and r['stratum']=='all' and r['proposed']==proposed and r['baseline']==baseline)
    labels={'fixed_n0_d0_p0':'固定截面','rotating_n0_d0_p0':'原轮询相位','rotating_phase_average':'四相位轮询均值'}
    methods={'I1_common':'共同趋势','I2_unaligned':'原均匀历史','N0_recent':'原字段零时滞核',
        'N3_uniform_norm':'归一化均匀历史','N2_recent_norm':'匹配新近性核','N4_wrong_norm':'匹配错位核'}
    lines=['# 有效提前量与新近性对照：探索诊断结果','',
        '本轮复用已查看过的12个探索家族（6训练/3验证/3内部诊断），不属于新盲测。20秒仍为主指标；10/30秒及时间窗口只用于诊断，不替换原阶段失败结论。',
        '',f"完成{info['model_count']}个CPU线性读出（11个观测条件×3个提前量×8组），12次训练名义脉冲响应。相位在家族内平均，独立内部诊断家族始终只有3个。",'',
        '## 阶段判断','',
        f"新增机制诊断门槛：**{'通过' if gate['passed'] else '未通过'}**。原实验门槛保持未通过。" ]
    if not gate['passed']:
        lines.append('按照预先写入PROTOCOL.md的条件，本轮停在信息诊断，未进入两目标临界施救评价，未开启正式36家族测试或QMIX训练。未测量本轮救援完成率与可改顺序情境比例。')
    else:
        lines.append('仅支持进入有限两目标临界施救诊断；本信息报告尚不包含任务收益验证。')
    fixed=get('fixed_n0_d0_p0',20,'N1_physical_norm','N2_recent_norm')
    rotating=get('rotating_phase_average',20,'N1_physical_norm','N2_recent_norm')
    lines += ['',f"最关键结果：统一归一化、支持度与年龄字段后，20秒物理时滞核相对匹配新近性核，固定截面MAE变化为下降{fixed['relative_reduction']*100:+.3f}%（负值表示更差），四相位轮询为下降{rotating['relative_reduction']*100:+.4f}%，后者接近零。原字段仍有小幅差异，但不足以证明独立的物理时滞增益；也不能把跨归一化方式的变化完全归因于新近性。",
        '', '原轮询无延迟时仅14.7%的目标—时刻组合具备20秒名义时间余量；10秒延迟时为0。说明该采样条件下可利用的短时先行余量有限，但此项几何统计没有证明误差预测或实际救援必然失败。']
    lines += ['', '## 时间尺度与可辨识性','',
        '冻结图36条边的名义时滞7—23秒。20秒提前量下21条边（58.3%）的核中心被截到0，其余15条仅为3秒；核宽10秒，因而20秒条件下物理核与偏重新近性的核很相似。30秒下全部严格相同：运行时已检查特征/预测等价。这个退化限制了当前表征的辨识力，不等于水动力传播不存在。',
        '', '观测年龄=当前时间−观测有效时间，已包含通信延迟。名义提前量L=T−年龄；行动余量L−等待；相对固定预测目标的余量L−τ分别保存。下表覆盖指L≥τ且L>6秒等待、目标至少一条候选边满足；这是名义时间余量覆盖率，不是实际报警成功率或信息可预测性的严格上界。',
        '', '| 观测条件 | 10秒目标覆盖 | 20秒目标覆盖 | 30秒目标覆盖 | 有正行动余量 |',
        '|---|---:|---:|---:|---:|']
    for name,label in [('fixed_n0_d0_p0','固定，延迟0秒'),('fixed_n0_d10_p0','固定，延迟10秒'),
                       ('rotating_n0_d0_p0','原轮询，延迟0秒'),('rotating_n0_d10_p0','原轮询，延迟10秒')]:
        rr=[next(r for r in timing if r['name']==name and r['tau']==tau and r['wait_s']==6) for tau in (10,20,30)]
        lines.append(f"| {label} | {rr[0]['horizon_target_rate']:.1%} | {rr[1]['horizon_target_rate']:.1%} | {rr[2]['horizon_target_rate']:.1%} | {rr[0]['positive_target_rate']:.1%} |")
    lines += ['', '当前40秒轮询仅是采样近似，未验证真实飞行速度/路径。覆盖数据按唯一的时间/边/目标统计，未重复计作12个家族的独立证据；固定截面在本轮决策时间恰好有新样本，是较理想的同步条件。',
        '', '## 原结果是否来自偏重近期信息','',
        '下表保留原I3聚合及原支持字段，与同宽高斯核、中心为0的新近性组比较；所有其它字段与记录相同。正值表示物理核误差更低。',
        '', '| 观测 | 10秒：I3相对新近性 | 20秒 | 30秒 |','|---|---:|---:|---:|']
    for name in labels:
        rr=[get(name,t,'I3_physical_lag','N0_recent') for t in (10,20,30)]
        lines.append(f"| {labels[name]} | {rr[0]['relative_reduction']*100:+.2f}% | {rr[1]['relative_reduction']*100:+.2f}% | {rr[2]['relative_reduction']*100:+.2f}% |")
    lines += ['', '## 归一化、相同支持的机制对照','',
        '每条边的历史核先各自归一化，再以固定边权汇总；所有归一化组的支持度/平均年龄完全相同，避免核幅度或空间源权重同时改变。下表为C1、无噪声/无延迟、20秒主指标。区间按家族配对，仅供探索。',
        '', '| 观测 | 对照 | 物理核MAE下降 | 差值(mm) | 家族95%区间(mm) | 有利家族 |',
        '|---|---|---:|---:|---:|---:|']
    for name in labels:
        for baseline in ('I1_common','N3_uniform_norm','N2_recent_norm','N4_wrong_norm'):
            r=get(name,20,'N1_physical_norm',baseline);lo,hi=r['interval_95']
            lines.append(f"| {labels[name]} | {methods[baseline]} | {r['relative_reduction']*100:+.2f}% | {r['mae_delta_m']*1000:+.4f} | [{lo*1000:+.4f},{hi*1000:+.4f}] | {r['favorable_families']}/3 |")
    lines += ['', f"![时间覆盖与机制对照]({(run/'timing_diagnostic.png').as_posix()})",'',
        '原I3与新的归一化组同时改变了归一化和支持字段，不能把跨这两类组的差异归因为单一时滞。机制判断应使用各自匹配的新近性对照。',
        '', '## 相位、噪声与延迟','',
        '四轮询相位保持相同观测总数；各相位单独训练匹配读出，再在同一家族内平均误差。相位不是额外独立家族。以下均为20秒主指标，正值表示归一化物理核比匹配新近性核更好。',
        '', '| 观测条件 | MAE下降 | 有利家族 |','|---|---:|---:|']
    names=[r['name'] for r in read(run/'config.json')['scenarios']]+['rotating_phase_average']
    for name in names:
        r=get(name,20,'N1_physical_norm','N2_recent_norm')
        lines.append(f"| {name} | {r['relative_reduction']*100:+.2f}% | {r['favorable_families']}/3 |")
    lines += ['', '噪声仅有原固定种子的一次实现，不构成充分的噪声鲁棒性证明。所有C0/C1/C2、10/20/30秒及可用/过期分层保存在information_scores.json和comparisons.json，未按显著性筛选。',
        '', '## 方向与早期窗口','',
        '6个训练家族×2个脉冲中心的名义响应：上游x5在脉冲后6—12秒首次响应，目标x35/x55在18—30秒，原反向源x75在38—42秒。216个配对中x75均比目标晚10—24秒；216个配对仍只来自6个独立家族。',
        '', 'x75属于同一水动力响应系统，且所有含共同趋势的组仍间接使用上游信息。当前反向组不是严格无信息负对照；本审计未分解入射/反射波，不把较晚响应直接归因为反射。',
        '', '训练名义响应定义“目标首次响应至x75首次响应”窗口；以下只在同脉冲分支、未来标签有效时刻落入该窗口的样本中比较。事件起点仅用于事后分层，不作为读出输入。此分层不改变门槛，也不保证无反射。',
        '', '| 观测 | 提前量 | 相对匹配新近性MAE下降 | 有利家族 | 样本数 |','|---|---:|---:|---:|---:|']
    for r in windows:
        if r['stratum']=='pre_later_sensor' and r['baseline']=='N2_recent_norm':
            lines.append(f"| {labels[r['name']]} | {r['tau']}秒 | {r['relative_reduction']*100:+.2f}% | {r['favorable_families']}/{r['independent_families']} | {sum(c['samples'] for c in r['counts'])} |")
    lines += ['', '完整早期、较晚窗口均在window_scores.json和window_comparisons.json；相位均值行的样本包含同一家族/事件的重复采样设计，不能当作新增独立样本。',
        '', '## 验证与复现','',
        f"48项测试通过。原I1/I2/I3全部兼容条件逐家族复现，最大MAE差{rep['max_mae_difference_m']:.3g}m；旧运行输入文件指纹全部保持不变。未来观测/真值/预测发布、缺失、核归一化、固定空间权重、延迟不重复扣减、30秒核等价均有测试。测试结果见tests.xml。",
        '', '协议PROTOCOL.md在运行前保存；依赖版本、源码/测试快照、输入指纹、读出参数和验证正则选择均已归档。系统本地环境保持原配置。',
        '', '本轮的合理结论限定为：当前时间尺度和表征是否提供独立增量信息。不能从核退化推出物理先行关系不存在，也不能从误差改善直接推出救援收益。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sources=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'configs').glob('*.json'))
    sources += [ROOT/'docs/LEAD_TIMING_DIAGNOSTIC.md',ROOT/'docs/RESIDUAL_LEAD_EXPERIMENT_PLAN.md']
    fingerprints={}
    for path in sources:
        relative=path.relative_to(ROOT);dst=run/'source_snapshot'/relative;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dst);fingerprints[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(run/'source_fingerprints.json',fingerprints)
    write_json(run/'artifact_fingerprints.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.iterdir() if p.suffix in ('.npz','.json','.md','.xml','.png') and p.name!='artifact_fingerprints.json'})
    write_json(run/'completion.json',dict(completed=True,scope='exploratory timing, matched recency, phase and nominal response audit',
        diagnostic_gate_passed=gate['passed'],original_information_gate_passed=False,full_research_plan_complete=False,
        formal_test_opened=False,policy_trained=False,critical_rescue_tested=False,source_data_unchanged=True,
        model_count=info['model_count'],independent_diagnostic_families=3))
    write_json(ROOT/'results/latest_timing_diagnostic.json',dict(run=str(run),report=str(run/'REPORT.md'),status='completed',
        diagnostic_gate_passed=gate['passed'],formal_test_opened=False))
    print(run/'REPORT.md',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();finalize(Path(args.run).resolve())
