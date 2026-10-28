"""Produce study report only after all four independent audit families pass.

Run with the bundled Python3.12 plotting runtime, not the PyTorch3.14 runtime.
The plot package path is scoped to this process only.
"""
from collections import defaultdict
import csv
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
plot_packages=ROOT/'.runtime/plot-packages'
if plot_packages.exists():sys.path.append(str(plot_packages))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS=['QMIX','Sur-QMIX','Sur-QMIX+Data','Sur-QMIX+Physics']
COLORS=['#666d80','#3d6db5','#bb8542','#258570']


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def digest(path):return sha256(path.read_bytes()).hexdigest()
def mean(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None
def text_number(value,digits=3):return 'NA' if value is None else f'{value:.{digits}f}'
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def plots(root,summary,events,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':160})
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    groups=METHODS+['R0','R1','R2']
    quality=[summary['main_quality'][m]*100 if m in METHODS else summary['rule_quality'][m]*100 for m in groups]
    axes[0].bar(range(7),quality,color=COLORS+['#b2b2b2']*3)
    axes[0].set(xticks=range(7),xticklabels=['QMIX','Sur','Sur+Data','Sur+Physics','R0','R1','R2'],ylabel='Current correct coverage (%)',title='Closed-loop outcome:9 crossed events')
    axes[0].tick_params(axis='x',rotation=35)
    paired=summary['gate']['paired'];x=np.arange(len(paired))
    axes[1].axhline(0,color='#666',linewidth=.8)
    axes[1].plot(x,[r['delta_Data']*100 for r in paired],'o-',label='Physics minus Data',color=COLORS[3])
    axes[1].plot(x,[r['delta_Sur']*100 for r in paired],'s--',label='Physics minus Sur',color=COLORS[1])
    axes[1].set(xlabel='Crossed holdout event (3 parents x3 forcing templates)',ylabel='Paired difference (percentage points)',title='Each event averaged over3 training seeds')
    axes[1].legend();fig.savefig(out/'closed-loop.png');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for method,color in zip(METHODS,COLORS):
        histories=[read(root/'formal'/(method.replace('+','-')+f'-s{seed}')/'history.json') for seed in (11,29,47)]
        for j,rows in enumerate(histories):
            values=np.asarray([r['metrics']['time_average_correct_coverage'] for r in rows])
            axes[0,0].plot(np.arange(100,2001),np.convolve(values,np.ones(100)/100,mode='valid')*100,color=color,alpha=.5,label=method if j==0 else None)
            vals=read(root/'formal'/(method.replace('+','-')+f'-s{(11,29,47)[j]}')/'validation.json')
            axes[0,1].plot([r['episode'] for r in vals],[r['mean_return'] for r in vals],color=color,alpha=.6)
            update=[r['update'] for r in rows if r['update'] is not None]
            axes[1,0].plot(np.arange(1,len(update)+1),[max(r['td_loss'],1e-12) for r in update],color=color,alpha=.25)
            if method in METHODS[2:]:
                axes[1,1].plot(np.arange(1,len(update)+1),[max(r['auxiliary_data_loss'],1e-12) for r in update],color=color,alpha=.5,label=method if j==0 else None)
    axes[0,0].set(title='Training:100-episode moving average',xlabel='Episode',ylabel='Current correct coverage (%)');axes[0,0].legend(fontsize=8)
    axes[0,1].set(title='Frozen selection schedule:3 events',xlabel='Episode',ylabel='Mean actual task return')
    axes[1,0].set(title='All seeds:TD objective',xlabel='Update',ylabel='TD loss',yscale='log')
    axes[1,1].set(title='Same auxiliary supervision',xlabel='Update',ylabel='Data loss',yscale='log');axes[1,1].legend(fontsize=8)
    fig.savefig(out/'training.png');plt.close(fig)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve();evidence=ROOT/'docs/evidence/20261025-sciml-development'
    required=['training-audit-final.json','transition-audit-final.json','trajectory-audit-final.json','forecast-audit-final.json','data-audit-final.json','statistics-audit-final.json','targets-audit-final.json']
    audits={name:read(evidence/name) for name in required}
    assert all(r['status']=='passed' for r in audits.values())
    assert audits['transition-audit-final.json']['training_transitions']==1440000
    assert audits['trajectory-audit-final.json']['policy_episodes']==108 and audits['forecast-audit-final.json']['rows']==9720
    summary=read(root/'confirmation-summary.json');assert summary['status']=='completed'
    plan=read(root/'formal-plan.json');jobs=read(root/'training-summary.json')['jobs'];assert len(jobs)==12
    results=audits['trajectory-audit-final.json']['results'];assert len(results)==135
    events=[]
    for case in sorted({r['case'] for r in results}):
        for method in METHODS+['R0','R1','R2']:
            rows=[r for r in results if r['case']==case and r['method']==method]
            assert len(rows)==(3 if method in METHODS else 1)
            event=dict(case=case,method=method,family=rows[0]['family'],forcing_family=rows[0]['forcing_family'])
            for field in ('time_average_correct_coverage','distance_m','turn_s','collision_events','motion_cost','return_'):
                event[field]=mean([r['metrics'] for r in rows],field)
            for field in ('change_count','undiscovered_change_fraction','restricted_change_delay_s','time_average_unknown_fraction'):
                event[field]=mean([r['secondary'] for r in rows],field)
            event['quality_seed_std']=float(np.std([r['metrics']['time_average_correct_coverage'] for r in rows],ddof=1)) if len(rows)>1 else None
            events.append(event)
    for method in METHODS:
        assert abs(mean([r for r in events if r['method']==method],'time_average_correct_coverage')-summary['main_quality'][method])<1e-12
    parent_effects=[]
    for family in sorted({r['family'] for r in events}):
        rows=[r for r in summary['gate']['paired'] if r['family']==family]
        parent_effects.append(dict(family=family,delta_Data=mean(rows,'delta_Data'),delta_Sur=mean(rows,'delta_Sur'),events=len(rows)))
    resources=[]
    for method in METHODS:
        rows=[j for j in jobs if j['method']==method]
        resources.append(dict(method=method,elapsed_s=sum(s['elapsed_s'] for r in rows for s in r['segments']),
            cuda_peak_reserved_bytes=max(s['memory'].get('cuda_peak_reserved_bytes',0) for r in rows for s in r['segments']),
            process_peak_bytes=max(s['memory']['process_peak_working_set_bytes'] for r in rows for s in r['segments'])))
    out=ROOT/'docs/evidence/20261025-sciml-final';out.mkdir(parents=True,exist_ok=True)
    with (out/'closed-loop-events.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(events[0]));writer.writeheader();writer.writerows(events)
    write(out/'analysis.json',dict(events=events,parent_effects=parent_effects,resources=resources,gate=summary['gate'],
        formal_plan_sha256=digest(root/'formal-plan.json'),source_reporter_sha256=digest(Path(__file__)),
        confirmation_summary_sha256=digest(root/'confirmation-summary.json'),audit_sha256={name:digest(evidence/name) for name in required},
        generated_utc=datetime.now(timezone.utc).isoformat()))
    plots(root,summary,events,out)
    lines=['# 独立物理辅助世界模型首轮：最终实验结果','',
        '本轮达到预登记开发门槛。' if summary['gate']['passed'] else '本轮未达到预登记闭环收益门槛。','',
        '四组×三个种子×2000回合已按冻结预算完成。新留出为三个母地形与三种来水模板交叉的九事件。全部144万训练转移、108条模型与27条规则回放、9720条共同路线预测及21个水情事件标签已通过独立审计。结论仅适用于本轮合成设置。','',
        '| 方法 | 全程当前正确覆盖 % | 平均距离 m | 平均转向 s | 平均碰撞 | 未发现变化比例 | 截断变化延迟 s |','|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS+['R0','R1','R2']:
        rows=[r for r in events if r['method']==method]
        lines.append(f"| {method} | {mean(rows,'time_average_correct_coverage')*100:.3f} | {mean(rows,'distance_m'):.1f} | {mean(rows,'turn_s'):.1f} | {mean(rows,'collision_events'):.3f} | {text_number(mean(rows,'undiscovered_change_fraction'))} | {text_number(mean(rows,'restricted_change_delay_s'),1)} |")
    gate=summary['gate'];delta=gate['mean_deltas']
    lines+=['',f"物理组相对 Data、Sur、QMIX 的主指标均值差分别为 {delta['Data']*100:+.3f}、{delta['Sur']*100:+.3f}、{delta['QMIX']*100:+.3f} 个百分点。同时优于 Data 与 Sur 的事件为 {gate['positive_events']}/9。未通过项：{', '.join(gate['failed_criteria']) or '无'}。",'',
        '![闭环与逐事件差异](evidence/20261025-sciml-final/closed-loop.png)','',
        '| 母地形 | Physics−Data pp | Physics−Sur pp |','|---|---:|---:|']
    lines += [f"| {r['family']} | {r['delta_Data']*100:+.3f} | {r['delta_Sur']*100:+.3f} |" for r in parent_effects]
    if max(summary['rule_quality'].values())>summary['main_quality']['Sur-QMIX+Physics']:
        lines+=['','物理辅助策略的平均主指标仍低于本轮最好的规则基线；不能据本轮结果将它视作当前最佳部署策略。']
    lines+=['','## 预测和物理一致性','',
        '以下对相同规则轨迹先做时间平均，再做路线、训练种子、事件平均。未知/陈旧子集为空时保留NA，不填零。物理残差及预测误差改善不能替代上表中的实际任务收益。','',
        '| 辅助组 | 10秒深度 MAE m | 未知区 MAE m | 陈旧区 MAE m | 湿干均衡错误 | 动量 MAE m²/s | 局部方程率损失 | 全域体积率差绝对值 m³/s |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS[2:]:
        rows=[r for r in summary['predictions'] if r['method']==method]
        values=[mean(rows,k) for k in ('future_h_mae_m','future_h_unknown_mae_m','future_h_stale_mae_m','future_wet_classbalanced_error','future_momentum_mae_m2_s','physical_rate_loss','absolute_volume_rate_residual_m3_s')]
        lines.append('| '+method+' | '+' | '.join(text_number(v,6) for v in values)+' |')
    lines+=['','体积率差为局部深度率差先作空间有符号求和后取绝对值，局部误差可相互抵消；不等于逐格绝对残差之和。局部方程率损失另含水深与两个动量分量。算子已单独对实际求解器验证。','',
        '## 训练与计算开销','', '![训练和选型曲线](evidence/20261025-sciml-final/training.png)','',
        f"十二个任务中 {sum(r['late_drift_flag'] for r in summary['late_training_drift'])} 个触发登记的后段漂移描述阈值。2000回合或曲线趋平本身都不证明收敛；弱策略/不稳定学习限制机制归因。详见机器可读逐任务漂移。",'',
        '| 方法 | 三种子运行段合计 s | CUDA峰值 reserved MiB | 进程峰值 MiB |','|---|---:|---:|---:|']
    lines += [f"| {r['method']} | {r['elapsed_s']:.1f} | {r['cuda_peak_reserved_bytes']/2**20:.1f} | {r['process_peak_bytes']/2**20:.1f} |" for r in resources]
    lines+=['','运行段时间包含检查点与选型，启动/加载另有开销；本机同时进行只读审计，不能将观测到的墙钟差解释为严格隔离的速度排名。原生进程内存峰值为同一顺序训练进程的累计高水位，后续方法可能继承前组峰值，不能据此比较各方法独立RAM开销；CUDA峰值按任务重置。actor/mixer容量各组一致，两辅助组head结构和标签相同，物理组额外计算方程约束。','',
        '## 边界与交付','',
        '本轮只有三个确认母地形和三个来水模板；九事件存在交叉依赖，三个训练种子用于随机性检查，不能增加独立地形样本数。本轮门槛不是显著性检验，也不证明真实洪水适用性。保留负结果与不确定性，不追加机制轮次。Sur-QMIX是明确记录差异的适配版，不能称完整复现原论文性能。','',
        '- [登记与信息权限](SCIML_PROTOCOL_20261025.md)','- [全部逐事件分析与资源](evidence/20261025-sciml-final/analysis.json)',
        '- [训练预算与初始化审计](evidence/20261025-sciml-development/training-audit-final.json)',
        '- [全部训练转移重建](evidence/20261025-sciml-development/transition-audit-final.json)',
        '- [全部实际闭环轨迹审计](evidence/20261025-sciml-development/trajectory-audit-final.json)',
        '- [全部预测重算](evidence/20261025-sciml-development/forecast-audit-final.json)',
        '- [全部水情标签审计](evidence/20261025-sciml-development/data-audit-final.json)',
        '- [运行和复现命令](REPRODUCE.md)','',
        '本报告完成科学结果整理；完整源码/日志/模型交付索引、图像人工核验和逐项目标验收仍须独立完成，不能只凭本报告标记整体目标完成。','']
    (ROOT/'docs/SCIML_FINAL_20261025.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote report and figures; visual inspection and final goal acceptance still required')


if __name__=='__main__':main()
