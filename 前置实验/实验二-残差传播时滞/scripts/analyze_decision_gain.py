"""Locate realized rescue gains and replay matched pulse evidence on fixed truth."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import (OFFSETS,issued_curve,scheduled_records,predict_reader,
    order_scores,choose,binned_history)
from run_residual_decision_value import all_features,digest

CHAIN=('nominal','forecast_only','raw_self','raw_full','physical')
LAYERS=('learned_forecast_correction','local_history','upstream_history','physical_representation')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def initial_decisions(path):
    output={}
    with gzip.open(path,'rt',encoding='utf-8') as handle:
        for line in handle:
            row=json.loads(line)
            if row['main_outcome']:
                output.setdefault(row['case_id'],{})[row['method']]=row
    return output


def decompose_case(methods):
    values={'nominal':next(iter(methods.values()))['baseline_true_yield']}
    values.update({m:r['selected_true_yield'] for m,r in methods.items()})
    result={label:values[b]-values[a] for label,a,b in zip(LAYERS,CHAIN[:-1],CHAIN[1:])}
    if sum(result.values()) != values['physical']-values['nominal']:
        raise ValueError('Non-telescoping gains')
    return result


def pulse_contrast(control,treatment):
    """C1 truth stays fixed; only selected action is taken from C0 or C1 input."""
    if control['targets']!=treatment['targets'] or control['orders']!=treatment['orders']:
        raise ValueError('Unmatched geometry')
    a,b=control['selected_order'],treatment['selected_order']
    y=treatment['true_yields']
    return dict(delta=y[b]-y[a],changed=a!=b,action_with_C0_input=a,action_with_C1_input=b,
                C1_truth_yields=y,control_input_yield=y[a],actual_input_yield=y[b])


def aggregate(rows,kind):
    fields=('schedule','revisit','delay','phase','split','family','condition',
            'layer' if kind=='layers' else 'method')
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in fields)].append(r)
    family=[]
    for key,rr in grouped.items():
        values=np.array([r['delta'] for r in rr])
        family.append(dict(zip(fields,key),cases=len(rr),mean_delta=float(values.mean()),
            positive=int(sum(values>0)),negative=int(sum(values<0)),changed=sum(r.get('changed',False) for r in rr)))
    merged=defaultdict(list)
    for r in family:
        scope='R5_four_phases' if r['revisit']==5 else f"R{r['revisit']}_phase0"
        key=(scope,r['revisit'],r['delay'],r['split'],r['family'],r['condition'],r[fields[-1]])
        merged[key].append(r)
    within=[]
    for key,rr in merged.items():
        expected=[0,.25,.5,.75] if key[1]==5 else [0]
        if sorted(r['phase'] for r in rr)!=expected:raise ValueError('Incomplete phases')
        within.append(dict(zip(('scope','revisit','delay','split','family','condition',fields[-1]),key),
            mean_delta=float(np.mean([r['mean_delta'] for r in rr])),
            positive_per_phase=float(np.mean([r['positive'] for r in rr])),
            negative_per_phase=float(np.mean([r['negative'] for r in rr])),
            changed_per_phase=float(np.mean([r['changed'] for r in rr])),
            cases_per_phase=float(np.mean([r['cases'] for r in rr]))))
    pooled=defaultdict(list)
    for r in within:
        key=tuple(r[k] for k in ('scope','revisit','delay','split','condition',fields[-1]))
        pooled[key].append(r)
    summary=[]
    for key,rr in pooled.items():
        summary.append(dict(zip(('scope','revisit','delay','split','condition',fields[-1]),key),
            mean_delta=float(np.mean([r['mean_delta'] for r in rr])),
            family_values=[dict(family=r['family'],mean_delta=r['mean_delta']) for r in sorted(rr,key=lambda z:z['family'])],
            positive_per_phase=sum(r['positive_per_phase'] for r in rr),
            negative_per_phase=sum(r['negative_per_phase'] for r in rr),
            changed_per_phase=sum(r['changed_per_phase'] for r in rr),
            cases_per_phase=sum(r['cases_per_phase'] for r in rr),independent_families=len(rr)))
    return family,within,summary


def write_csv(path,rows):
    if not rows:return
    keys=list(rows[0])
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def detail(parent,run,item,event,graph,number):
    schedule=item['schedule'];directory=parent/schedule;method=item['after']
    model_file=np.load(directory/'models_and_internal_predictions.npz')
    now=item['now_s']; f,forecast=issued_curve(event,now)
    records=scheduled_records(event,item['revisit'],item['delay'],item['phase'])
    curves={'nominal':forecast};prediction_scores={}
    source={}
    for m in CHAIN[1:]:
        model={k:model_file[f'{m}_{k}'] for k in ('mean','std','weights')}
        corrected=forecast.copy()
        for target in item['targets']:
            x=all_features(event,records,graph,target,now,forecast)[m]
            corrected[:,target]=np.maximum(0,forecast[:,target]+predict_reader(model,x[None,:])[0])
        curves[m]=corrected
    for m,curve in curves.items():
        prediction_scores[m]=order_scores(now+OFFSETS,curve,item['orders'],now)
    selected={m:choose(ss) for m,ss in prediction_scores.items()}
    for m in CHAIN[1:]:
        if selected[m] != item['selected_orders'][m]:raise ValueError('Stored model action mismatch')
    for target in item['targets']:
        source[str(target)]={str(s):[r for r in binned_history(records,s,now) if r]
            for s in [target]+[l['source'] for l in graph[str(target)]]}
    true_curves=np.column_stack([np.interp(now+OFFSETS,event['time_s'],event['truth_depth_m'][:,r]) for r in range(100)])
    curve_rows=[]
    for target in item['targets']:
        for i,off in enumerate(OFFSETS):
            curve_rows.append(dict(target=target,time_s=now+off,reference_m=true_curves[i,target],
                **{m:float(curve[i,target]) for m,curve in curves.items()}))
    stem=f'case_{number:02d}_{item["family"]}_t{now}'
    write_csv(run/f'{stem}_curves.csv',curve_rows)
    record=dict(item,forecast_release_s=float(event['release_s'][f]),prediction_scores=prediction_scores,
        input_records=source,curve_csv=f'{stem}_curves.csv')
    write_json(run/f'{stem}.json',record)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for ax,target in zip(axes,item['targets']):
        ax.plot(now+OFFSETS,true_curves[:,target],color='black',lw=2,label='reference')
        for m in ('nominal','forecast_only','raw_self','raw_full','physical'):
            ax.plot(now+OFFSETS,curves[m][:,target],lw=1.2,label=m)
        ax.axhline(.8,color='#b43836',ls='--',lw=1)
        for oi,trace in enumerate(item['truth_traces']):
            row=next(r for r in trace['trace'] if r['region']==target)
            ax.axvspan(row['start_s'],row['start_s']+12,color=('#417e91','#c17c44')[oi],alpha=.12)
        ax.set(title=f'Target {target}; shaded = order 0/1 service intervals',xlabel='Time (s)',ylabel='Water depth (m)')
        ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(f'{schedule} / {item["family"]} / {item["layer"]} / delta {item["delta"]:+d}')
    fig.savefig(run/f'{stem}.png',dpi=160);plt.close(fig)
    return dict(file=f'{stem}.json',figure=f'{stem}.png',family=item['family'],time_s=now,layer=item['layer'],delta=item['delta'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('parent',type=Path);args=parser.parse_args()
    parent=args.parent.resolve();config=read(parent/'config.json');events=load_events(config['manifest'])
    by_event={e['metadata']['event_id']:e for seq in events.values() for e in seq}
    event_run=Path(config['manifest']).parent;graph=read(event_run/'physical_graph.json')['graph']
    all_layers=[];all_pulses=[];nonzero=[];pulse_changed=[];checks=[]
    for setting in config['schedules']:
        schedule=setting['schedule'];print('Attribution',schedule,flush=True)
        cases=initial_decisions(parent/schedule/'decisions.jsonl.gz')
        paired={}
        for cid,methods in cases.items():
            row=next(iter(methods.values()));base={k:row[k] for k in ('split','family','condition','layout','now_s','targets','orders','truth_traces','last_correctable_s','response_period')}
            selected={m:r['selected_order'] for m,r in methods.items()}
            for label,before,after in zip(LAYERS,CHAIN[:-1],CHAIN[1:]):
                delta=decompose_case(methods)[label]
                before_action=row['baseline_order'] if before=='nominal' else methods[before]['selected_order']
                item=dict(setting,**base,case_id=cid,layer=label,before=before,after=after,delta=delta,
                    changed=before_action!=methods[after]['selected_order'],selected_orders=selected,
                    baseline_order=row['baseline_order'],true_yields=row['true_yields'])
                all_layers.append(item)
                if delta:nonzero.append(item)
            paired[row['family'],row['condition'],row['layout'],row['now_s']]=methods
        for (family,condition,layout,now),methods in paired.items():
            if condition!='C1_local':continue
            zero=paired[family,'C0_stable',layout,now]
            for m,row in methods.items():
                q=zero[m];values=pulse_contrast(q,row)
                item=dict(setting,split=row['split'],family=family,condition=condition,layout=layout,now_s=now,
                    targets=row['targets'],method=m,response_period=row['response_period'],**values)
                all_pulses.append(item)
                if values['changed']:pulse_changed.append(item)
        checks.append(dict(schedule=schedule,cases=len(cases),telescoping=True))
    for family in sorted({e['metadata']['event_family_id'] for e in by_event.values()}):
        a=by_event[f'{family}_C0_stable'];b=by_event[f'{family}_C1_local']
        if not np.array_equal(a['forecast_depth_m'],b['forecast_depth_m']):raise ValueError('C0/C1 forecasts differ')
    if any(r['changed'] or r['delta'] for r in all_pulses if r['method']=='forecast_only'):
        raise ValueError('Forecast-only negative control changed under identical forecast')
    run=new_run('decision_gain_attribution',dict(parent=str(parent),objective='locate incremental gains and inspect mechanism',
        chain=CHAIN,posthoc=True,prior_results_seen=True,paired_intervention='C0/C1 evidence, fixed C1 truth, same trained model'))
    for name,rows in [('layers',all_layers),('pulse_contrast',all_pulses)]:
        family,phase,summary=aggregate(rows,'layers' if name=='layers' else 'pulse')
        write_json(run/f'{name}_families.json',family);write_json(run/f'{name}_phase_averages.json',phase);write_json(run/f'{name}_summary.json',summary)
    write_csv(run/'nonzero_increment_cases.csv',nonzero)
    write_csv(run/'pulse_changed_cases.csv',pulse_changed)
    write_json(run/'checks.json',dict(telescoping=checks,C0_C1_forecasts_equal=True,
        forecast_only_negative_control_passed=True,input_fingerprints={str(p):digest(p) for p in [parent/'family_task_scores.json',parent/'config.json']}))
    # Deterministic representatives, not chosen by effect magnitude.
    representatives=[];chosen=set()
    for item in sorted(nonzero,key=lambda r:(r['layer'],r['delay'],r['phase'],r['family'],r['now_s'],r['layout'])):
        if item['split']!='test' or item['condition']!='C1_local' or item['revisit']!=5 or item['delay']!=0 or item['phase']!=0:continue
        group=(item['layer'],np.sign(item['delta']))
        if group in chosen:continue
        chosen.add(group);representatives.append(detail(parent,run,item,by_event[item['case_id'].split('/')[0]],graph,len(representatives)+1))
    write_json(run/'representatives.json',representatives)
    lines=['# 收益产生在哪一层：配对归因诊断','','相同情境中依次比较名义顺序、只学习预测偏差、加入自身历史、加入上游历史、加入物理汇总。层间差逐情境精确相加为物理组相对名义顺序的总差。这是不同模型的增量分解，不是唯一的因果贡献分配；顺序改变也会改变分解结果。','','## C1：R5四相位先家族内平均，再家族等权','','单位为每个初始情境多完成的服务数，内部独立家族仅3个。','','|延迟|收益层|平均增量|每相位正/负案例数|家族增量|','|---:|---|---:|---|---|']
    for r in read(run/'layers_summary.json'):
        if r['split']=='test' and r['condition']=='C1_local' and r['revisit']==5:
            lines.append(f"|{r['delay']}|{r['layer']}|{r['mean_delta']:+.6f}|{r['positive_per_phase']:.2f}/{r['negative_per_phase']:.2f}|"+'; '.join(f"{v['family']}: {v['mean_delta']:+.6f}" for v in r['family_values'])+'|')
    lines+=['','## 脉冲观测是否带来了实际额外收益','','对每个同家族同布局同时刻，保持C1真实水情不变，将同一个已训练模型的C1观测输入换成C0观测输入，比较选序在C1中的收益。两事件发布预测逐元素相同，所以差异来自观测历史；forecast_only必须为零。这是仿真中的输入干预，不是实际闭环任务。','','|延迟|模型|固定C1真值下的平均信息效应|每相位改序/增益/损失数|','|---:|---|---:|---|']
    for r in read(run/'pulse_contrast_summary.json'):
        if r['split']=='test' and r['revisit']==5:
            lines.append(f"|{r['delay']}|{r['method']}|{r['mean_delta']:+.6f}|{r['changed_per_phase']:.2f}/{r['positive_per_phase']:.2f}/{r['negative_per_phase']:.2f}|")
    lines+=['','## 逐个服务区间核查','','曲线使用决策当时已发布版本与当时因果可见历史；阴影标出两个候选顺序在真值评分中对应的服务时段。图中0.8米线是当前服务阈值。完整历史、模型预测服务轨迹和真值服务轨迹见各案例JSON。代表案例按层、相位、家族及时间固定排序选取，所有非零层间差均在CSV中保留。','']
    for r in representatives:lines +=[f"### {r['layer']} / {r['family']} / t={r['time_s']} / {r['delta']:+d}",'',f"![案例]({(run/r['figure']).as_posix()})",'',f"[完整案例]({(run/r['file']).as_posix()})",'']
    lines+=['## 解释边界','','只有3个内部家族和3个通过时域门槛的截面布局。训练、验证和内部诊断分别保存在JSON；不能把相位、方法、时刻当新增独立样本。配对输入效应不等于物理传播时滞的独立贡献，后者仍需原始历史、新近性、均匀和错位控制。这个诊断没有新增洪水仿真或多机策略训练。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'completion.json',dict(completed=True,parent=str(parent),posthoc=True,new_policy_trained=False))
    write_json(ROOT/'results/latest_gain_attribution.json',dict(run=str(run),report=str(run/'REPORT.md')))
    print(run,flush=True)


if __name__=='__main__':main()
