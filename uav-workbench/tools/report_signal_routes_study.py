"""Postprocess actual frozen confirmation evidence; never fit or select a model."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import os
import signal_routes
import numpy as np
from signal_routes import METHODS
from signal_routes.common import DEFAULT,ROOT,read,write,digest,sources


def restricted_time(value,duration):
    return float(duration if value is None else min(duration,value))


def mechanism_contrasts(rows):
    parents=sorted({r['parent'] for r in rows})
    if len(parents)!=8: raise ValueError('Expected eight independent confirmation parents')
    bootstrap=np.random.default_rng(20261027).integers(0,8,(10000,8))
    results=[]
    for condition in ('seen_single','unseen_double'):
        part=[r for r in rows if (r['mechanism']=='double')==(condition=='unseen_double')]
        for i,method in enumerate(('R1','R2','R3'),1):
            for baseline in ('Bpost',f'C{i}'):
                delta=[]
                for parent in parents:
                    scores=[]
                    for name in (method,baseline):
                        values=[r['metrics']['rescue_fraction'] for r in part if r['method']==name and r['parent']==parent]
                        if not values: raise ValueError('Missing paired mechanism result')
                        scores.append(float(np.mean(values)))
                    delta.append(scores[0]-scores[1])
                delta=np.asarray(delta)
                results.append(dict(condition=condition,method=method,baseline=baseline,
                    delta=float(delta.mean()),ci95=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975]).tolist(),
                    paired_parent_deltas=delta.tolist(),scope='descriptive mechanism subgroup; not a new confirmatory test'))
    return results


def supplemental_report(study=DEFAULT):
    study=Path(study)
    if read(study/'status.json')['status']!='completed': raise ValueError('Cannot report an incomplete study')
    result=read(study/'confirmation/results.json'); rows=result['rows']
    if len(rows)!=1944: raise ValueError('Confirmation matrix incomplete')
    params=read(study/'parameters.json'); nulls={m:bool(next(r for r in params['optimization'] if r['method']==m)['null_selected'])
                                               for m in ('R1','R2','R3','C1','C2','C3')}
    enriched=[]
    for row in rows:
        metrics=dict(row['metrics']); duration=read(study/'task-selection.json')['duration_s']
        record={k:row[k] for k in ('method','training_seed','scenario_seed','parent','mechanism')}
        record.update(metrics)
        event_times_supported=all(key in metrics for key in ('first_discovery_s','first_rescue_s'))
        record['event_times_supported']=event_times_supported
        record['first_discovery_restricted_s']=restricted_time(metrics['first_discovery_s'],duration) if event_times_supported else None
        record['first_rescue_restricted_s']=restricted_time(metrics['first_rescue_s'],duration) if event_times_supported else None
        record['first_discovery_censored']=metrics['first_discovery_s'] is None if event_times_supported else None
        record['first_rescue_censored']=metrics['first_rescue_s'] is None if event_times_supported else None
        if row['method'] in METHODS:
            replay=read(row['replay']); events=replay['events']
            discovered={tuple(e['cell']) for e in events if e['type']=='discovered'}
            rescued={tuple(e['cell']) for e in events if e['type']=='rescued'}
            interrupted={tuple(e['cell']) for e in events if e['type']=='service_interrupted'}
            if len(rescued)!=metrics['rescued'] or not rescued<=discovered: raise AssertionError('Event/metric service count mismatch')
            record['failure_undiscovered']=5-len(discovered)
            record['failure_discovered_interrupted']=len((discovered-rescued)&interrupted)
            record['failure_discovered_other_unfinished']=len((discovered-rescued)-interrupted)
            record['rescued_after_interruption']=len(rescued&interrupted)
            record['all_rescued_s']=max((e['time_s'] for e in events if e['type']=='rescued'),default=None) if len(rescued)==5 else None
            record['all_rescued_censored']=len(rescued)!=5
            if sum(record[k] for k in ('failure_undiscovered','failure_discovered_interrupted','failure_discovered_other_unfinished'))!=5-metrics['rescued']:
                raise AssertionError('Failure categories are not exhaustive/disjoint')
        enriched.append(record)
    fields=list(dict.fromkeys(k for r in enriched for k in r))
    with (study/'per-episode-expanded.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(enriched)
    parents=sorted({r['parent'] for r in enriched}); summary=[]
    keys=('rescue_fraction','first_coverage_auc','current_correct_coverage_auc','distance_m','collision_events',
          'first_discovery_restricted_s','first_rescue_restricted_s','failure_undiscovered',
          'failure_discovered_interrupted','failure_discovered_other_unfinished','inference_max_s')
    for method in (*METHODS,'frontier','sweep_refresh','water_aware'):
        entry=dict(method=method)
        for key in keys:
            pv=[]
            for parent in parents:
                values=[r[key] for r in enriched if r['method']==method and r['parent']==parent and r.get(key) is not None]
                if values: pv.append(float(np.mean(values)))
            entry[key]=float(np.mean(pv)) if pv else None
        summary.append(entry)
    contrasts=mechanism_contrasts(rows)
    write(study/'supplemental-results.json',dict(schema='hyra-signal-routes-supplement-v1',
        audit_tool_sha256=digest(__file__),null_fits=nulls,parent_equal_weighted=summary,
        mechanism_paired_contrasts=contrasts,
        failure_categories='undiscovered / discovered unfinished with recorded interruption / discovered unfinished without recorded interruption',
        time_metrics='restricted first-event times include censored runs at 900s; all-rescued time retained nullable per episode',
        score_scope='frozen 8-parent synthetic confirmation; no postconfirmation tuning'))
    figures=plots(study,summary)
    lines=['## 完整训练与结果边界','',
        '三条路线均完成了登记的尝试。主环境中R1与R3的训练集拟合选择零耦合，因此它们退化为共同后验基线；这些组的比较不能作为非零方向传播或跨层耦合收益证据。',
        'R2与其置乱控制均允许非零均值修正；只有优于共同后验且优于对应置乱控制的独立结果，才支持水情配准具有增量价值。',
        '均值路线采用公开底床加合法观测水深的水面坡度，按实际米制边长差分；四项特征的归一化统计仅由训练记录确定。',
        '', '| 方法 | 未发现人数/回合 | 发现后中断且未完成人数/回合 | 其他发现后未完成人数/回合 | 首次施救受限时间/秒 |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        if r['method'] not in METHODS: continue
        lines.append(f"| {r['method']} | {r['failure_undiscovered']:.3f} | {r['failure_discovered_interrupted']:.3f} | {r['failure_discovered_other_unfinished']:.3f} | {r['first_rescue_restricted_s']:.1f} |")
    lines += ['', '未发生首次施救的回合按截止900秒计入受限时间；全部施救完成时间及删失标记保存在扩展逐回合CSV，未只对成功回合求均值。',
              '“其他发现后未完成”仅表示没有记录服务中断，不推断人员死亡、不可救或飞行安全原因。',
              '规则评价没有保存首次事件时间，相应扩展字段标为不支持，不将缺失字段解释为未发生事件。',
              '', '## 未见水动力条件下的物理消融','',
              '下表按8个父地形配对，报告施救比例差值及父地形重采样95%区间，单位为百分点。分机制结果作描述性解释，主检验仍为前述三项Holm校正比较。',
              '', '| 路线−基线 | 已见单峰差值 [95%区间] | 未见双峰差值 [95%区间] |',
              '|---|---|---|']
    for i,method in enumerate(('R1','R2','R3'),1):
        for baseline in ('Bpost',f'C{i}'):
            cells=[]
            for condition in ('seen_single','unseen_double'):
                entry=next(r for r in contrasts if (r['condition'],r['method'],r['baseline'])==(condition,method,baseline))
                cells.append(f"{100*entry['delta']:.2f} [{100*entry['ci95'][0]:.2f}, {100*entry['ci95'][1]:.2f}]")
            lines.append(f'| {method}−{baseline} | '+ ' | '.join(cells)+' |')
    lines += ['', '## 方向性检验','',
              '开发集独立方向诊断覆盖3个父地形、每条件18个快照，共684条评分。主任务R1、R3均零退化；人工方向响应条件下R1仍零退化，R3耦合非零但方向参数κ=0。正向、反向及各向同性的预测一致，因此本轮未通过“学得上游→下游非对称影响”的检验。',
              '非零参数工程样例能响应流向翻转，证明实现可表达方向变化；这不构成本轮拟合已学得方向性的证据。人工方向条件下R3相对共同基线NLL略好而RMSE略差，也不能归因为流向作用。',
              'R2采用速度大小和水面坡度幅值，未纳入带方向的速度分量。独立开发集检查对R2及C2各18个快照同时翻转u/v，非零拟合参数下的后验均值和方差仍逐位一致。因此，本轮R2检验的是标量水情协变量关联，不能支持上游→下游非对称影响。对应证据见 `mean-direction-diagnostic.json`。',
              '', '## 覆盖、任务成本与推断耗时','',
              '| 方法 | 首次覆盖AUC | 正确水情覆盖AUC | 航程/米 | 冲突事件/回合 | 后验计算回合内峰值的平均/秒 |',
              '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        def cell(key): return '—' if r[key] is None else f'{r[key]:.4f}'
        lines.append('| '+r['method']+' | '+' | '.join(cell(k) for k in ('first_coverage_auc','current_correct_coverage_auc','distance_m','collision_events','inference_max_s'))+' |')
    lines += ['', '推断耗时仅覆盖信号后验计算，受本机并行负载影响；规则及不支持的字段以“—”表示。它不等于策略总耗时或部署时限保证。',
              '', '## 图表','']
    for name,path in figures:
        lines.extend([f'![{name}]({path.as_posix()})',''])
    lines += ['## 可重放材料','',
        '- `registration.json`：冻结研究设计；`parameters.json`：训练集拟合的冻结估计器。',
        '- `formal/`：24项1024回合训练、1021次更新、初始权重、历史及三个选模点。',
        '- `sealed-models.json`：72个开发候选及24个选定模型。',
        '- `confirmation/results.json`：1728模型回放及216规则评价；每个模型回放保存实际观测、事件与重建核对。',
        '- `independent-preflight.json`、`completion-audit.json`：独立的前置和完整验收证据。',
        '- `independent-direction-diagnostics.json`：开发集冻结参数下的方向、反向和无向诊断。',
        '- `per-episode-expanded.csv`：完整失败类别、删失时间与任务成本。',
        '', '初版输出保留在 `signal-routes-20261027/`，已标为superseded；它的训练权重未混入本轮结果。当前数据复用其尚未评分的同批新物理事件，来源和摘要见数据清单。']
    target=study/'report.md'; text=target.read_text(encoding='utf-8'); marker='\n## 完整训练与结果边界\n'
    if marker in text: text=text.split(marker)[0].rstrip()+'\n'
    target.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    return dict(expanded_rows=len(enriched),figures=[str(p) for _,p in figures],null_fits=nulls)


def plots(study,summary):
    os.environ.setdefault('MPLCONFIGDIR',str(study/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','axes.spines.top':False,
                         'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':180})
    out=study/'figures'; out.mkdir(exist_ok=True); files=[]
    colors={'Braw':'#777777','Bpost':'#243b53','R1':'#207f91','R2':'#b34d2a','R3':'#7162a6',
            'C1':'#8eb8bf','C2':'#dbad99','C3':'#b7b0cf'}
    primary=read(study/'report.json')['primary_holm']
    fig,ax=plt.subplots(figsize=(7,3.8),layout='constrained')
    for i,method in enumerate(('R1','R2','R3')):
        row=next(r for r in primary if r['method']==method); values=np.array(row['paired_parent_deltas'])*100
        ax.scatter(values,i+np.linspace(-.11,.11,len(values)),s=24,color=colors[method],alpha=.7)
        low,high=np.array(row['ci95'])*100; center=100*row['delta']
        ax.errorbar(center,i,xerr=[[max(0,center-low)],[max(0,high-center)]],fmt='D',color=colors[method],capsize=4,markersize=6)
    ax.axvline(0,color='#777777',linewidth=.8); ax.set_yticks(range(3),('R1','R2','R3'))
    ax.set_xlabel('Rescue difference vs Bpost (percentage points)')
    ax.set_title('Eight parent-level paired effects and cluster bootstrap intervals')
    if max(abs(r['delta']) for r in primary)<1e-12: ax.set_xlim(-1,1)
    path=out/'paired-effects.png'; fig.savefig(path); fig.savefig(out/'paired-effects.svg'); plt.close(fig)
    files.append(('按父地形聚合的配对施救差异',path))
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True,layout='constrained')
    for ax,methods in zip(axes.ravel(),(('Braw','Bpost'),('Bpost','R1','C1'),('Bpost','R2','C2'),('Bpost','R3','C3'))):
        for method in methods:
            values=np.array([[r['metrics']['rescue_fraction'] for r in read(study/'formal'/f'{method}-s{seed}'/'history.json')]
                             for seed in (11,29,47)])
            mean=values.mean(axis=0); smooth=np.convolve(mean,np.ones(32)/32,mode='valid')*100
            ax.plot(np.arange(32,1025),smooth,label=method,color=colors[method],linewidth=1.4,
                    linestyle='--' if method=='Bpost' and len(methods)>2 else '-')
        ax.set_title(' / '.join(methods)); ax.legend(frameon=False,loc='upper left'); ax.set_ylim(0,100)
        ax.set_xlabel('Training episode'); ax.set_ylabel('Training rescue fraction (%)')
    fig.suptitle('Three-seed mean, 32-episode moving average; not convergence evidence')
    path=out/'training-curves.png'; fig.savefig(path); fig.savefig(out/'training-curves.svg'); plt.close(fig)
    files.append(('正式训练曲线，不能替代独立任务评价',path))
    return files


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,default=DEFAULT)
    print(supplemental_report(p.parse_args().study))
