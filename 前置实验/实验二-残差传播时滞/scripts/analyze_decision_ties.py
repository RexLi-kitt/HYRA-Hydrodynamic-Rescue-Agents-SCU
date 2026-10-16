"""Post-hoc tie-rule sensitivity with identical forecasts and frozen learned models."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import OFFSETS,issued_curve,order_scores
from analyze_decision_gain import initial_decisions,write_csv
from run_residual_decision_value import digest

METHODS=('nominal','forecast_only','raw_self','raw_full','recent','physical','uniform','wrong')
POLICIES=('fixed_first','fixed_reverse','uniform_tie_expectation','maximin_margin')


def maximum(times,values,begin,end):
    if begin<times[0] or end>times[-1]:raise ValueError('No margin extrapolation')
    inside=(times>=begin)&(times<=end)
    return float(max(np.r_[np.interp(begin,times,values),values[inside],np.interp(end,times,values)]))


def tie_weights(scores,margins,policy):
    counts=np.array([s['service_yield'] for s in scores])
    tied=np.flatnonzero(counts==max(counts))
    weights=np.zeros(len(scores))
    if policy=='uniform_tie_expectation':
        weights[tied]=1/len(tied)
    elif policy=='fixed_first':weights[tied[0]]=1
    elif policy=='fixed_reverse':weights[tied[-1]]=1
    elif policy=='maximin_margin':
        best=max(margins[i] for i in tied)
        weights[next(i for i in tied if margins[i]>=best-1e-12)]=1
    else:raise ValueError(policy)
    return weights


def family_average(rows):
    keys=('revisit','delay','phase','schedule','family','condition','method','policy')
    grouped=defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    family=[]
    for key,rr in grouped.items():
        family.append(dict(zip(keys,key),cases=len(rr),
            mean_yield=float(np.mean([r['true_yield'] for r in rr])),
            mean_delta_nominal=float(np.mean([r['delta_nominal'] for r in rr])),
            mean_regret=float(np.mean([r['regret'] for r in rr])),
            positive=sum(r['delta_nominal']>1e-12 for r in rr),
            negative=sum(r['delta_nominal']< -1e-12 for r in rr)))
    phase=defaultdict(list)
    for r in family:
        phase[r['revisit'],r['delay'],r['family'],r['condition'],r['method'],r['policy']].append(r)
    within=[]
    for key,rr in phase.items():
        expected=[0,.25,.5,.75] if key[0]==5 else [0]
        if sorted(r['phase'] for r in rr)!=expected:raise ValueError('Missing phases')
        within.append(dict(zip(('revisit','delay','family','condition','method','policy'),key),
            mean_yield=float(np.mean([r['mean_yield'] for r in rr])),
            mean_delta_nominal=float(np.mean([r['mean_delta_nominal'] for r in rr])),
            mean_regret=float(np.mean([r['mean_regret'] for r in rr]))))
    combined=defaultdict(list)
    for r in within:combined[r['revisit'],r['delay'],r['condition'],r['method'],r['policy']].append(r)
    summary=[]
    for key,rr in combined.items():
        summary.append(dict(zip(('revisit','delay','condition','method','policy'),key),
            mean_yield=float(np.mean([r['mean_yield'] for r in rr])),
            mean_delta_nominal=float(np.mean([r['mean_delta_nominal'] for r in rr])),
            mean_regret=float(np.mean([r['mean_regret'] for r in rr])),
            family_values=[dict(family=r['family'],mean_delta_nominal=r['mean_delta_nominal'],mean_regret=r['mean_regret']) for r in rr],
            independent_families=len(rr)))
    return family,within,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('parent',type=Path);args=parser.parse_args()
    parent=args.parent.resolve();config=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    events=load_events(config['manifest']);by_event={e['metadata']['event_id']:e for e in events['test']}
    run=new_run('decision_tie_attribution',dict(parent=str(parent),posthoc=True,
        policies=POLICIES,predicted_margin='minimum over target service windows of (0.8 minus maximum predicted depth in full12s)',
        lexicographic=True,truth_used_for_action=False,no_retraining=True,independent_families=3))
    output=[];support=[];audits=[];evidence=[]
    for setting in config['schedules']:
        name=setting['schedule'];print('Tie sensitivity',name,flush=True)
        cases=initial_decisions(parent/name/'decisions.jsonl.gz')
        cases={cid:mm for cid,mm in cases.items() if next(iter(mm.values()))['split']=='test'}
        index=json.loads((parent/name/'internal_prediction_index.json').read_text(encoding='utf-8'))
        positions={(r['event_id'],r['now_s'],r['target']):i for i,r in enumerate(index)}
        with np.load(parent/name/'models_and_internal_predictions.npz') as archive:
            predictions={m:archive[f'{m}_internal_prediction'] for m in METHODS if m!='nominal'}
        checked=0
        for cid,methods in cases.items():
            row=next(iter(methods.values()));eid=row['event_id'];event=by_event[eid];now=row['now_s']
            version,nominal=issued_curve(event,now);times=now+OFFSETS
            forecasts={'nominal':nominal}
            for method,pred in predictions.items():
                curve=nominal.copy()
                for target in row['targets']:
                    curve[:,target]=np.maximum(0,curve[:,target]+pred[positions[eid,now,target]])
                forecasts[method]=curve
            choice={};counts={};margins={};score_map={}
            for method,curve in forecasts.items():
                scores=order_scores(times,curve,row['orders'],now)
                counts[method]=[r['service_yield'] for r in scores]
                margins[method]=[min(.8-maximum(times,curve[:,r['region']],r['start_s'],r['start_s']+12)
                    for r in order['trace']) for order in scores]
                score_map[method]=scores
                for policy in POLICIES:choice[method,policy]=tie_weights(scores,margins[method],policy)
                stored=row['baseline_order'] if method=='nominal' else methods[method]['selected_order']
                if int(np.argmax(choice[method,'fixed_first']))!=stored:raise ValueError('Frozen action reproduction failed')
                checked+=1
            truth=np.array(row['true_yields'],float)
            for policy in POLICIES:
                baseline=float(choice['nominal',policy]@truth)
                for method in METHODS:
                    y=float(choice[method,policy]@truth)
                    output.append(dict(setting,case_id=cid,family=row['family'],condition=row['condition'],
                        layout=row['layout'],now_s=now,method=method,policy=policy,true_yield=y,
                        delta_nominal=y-baseline,regret=float(max(truth)-y)))
            for method in METHODS[1:]:
                gain=methods[method]['realized_delta']
                if gain!=0:
                    support.append(dict(setting,case_id=cid,method=method,delta=gain,
                        nominal_tie=len(set(counts['nominal']))==1,
                        method_tie=len(set(counts[method]))==1,
                        targets=row['targets'],now_s=now,condition=row['condition'],family=row['family']))
            # All cases where a gain disappears under same-input margin tie-breaking.
            if methods['physical']['realized_delta']>0:
                old=methods['physical']['realized_delta'];new=float((choice['physical','maximin_margin']-choice['nominal','maximin_margin'])@truth)
                evidence.append(dict(setting,case_id=cid,condition=row['condition'],family=row['family'],
                    targets=row['targets'],now_s=now,old_physical_gain=old,new_physical_gain=new,
                    true_yields=row['true_yields'],predicted_yields=counts,predicted_minimum_margins_m=margins,
                    fixed_actions={m:int(np.argmax(choice[m,'fixed_first'])) for m in METHODS},
                    margin_actions={m:int(np.argmax(choice[m,'maximin_margin'])) for m in METHODS}))
        audits.append(dict(schedule=name,checked_actions=checked,unique_cases=len(cases),reproduced=True))
    family,within,summary=family_average(output)
    write_json(run/'family_policy_scores.json',family);write_json(run/'family_phase_averages.json',within)
    write_json(run/'policy_summary.json',summary);write_json(run/'nonzero_fixed_rule_support.json',support)
    write_json(run/'gain_case_tie_evidence.json',evidence)
    write_csv(run/'gain_case_tie_evidence.csv',evidence)
    write_json(run/'checks.json',dict(action_reproductions=audits,truth_used_only_for_scoring=True,
        parent_inputs={str(parent/'config.json'):digest(parent/'config.json')}))
    lines=['# 深挖决策收益：平局规则敏感性','','这项事后诊断固定洪水、观测、模型参数和两个候选顺序，只改变预测完成数相同时的选序规则。maximin_margin先最大化预测完成数，平局时再最大化两个目标完整服务区间的最小预测水深余量；动作不使用真值。所有规则对名义预测及各学习模型一致应用。','','## C1 / R5：四相位家族内平均','','单位是每个初始情境的服务数；只有3个已查看内部诊断家族。随机平局使用解析期望，不额外抽样。','','|延迟|方法|平局规则|相对同规则名义预测的收益|剩余有限遗憾|','|---:|---|---|---:|---:|']
    for r in summary:
        if r['revisit']==5 and r['condition']=='C1_local' and r['method'] in ('nominal','forecast_only','raw_self','raw_full','physical'):
            lines.append(f"|{r['delay']}|{r['method']}|{r['policy']}|{r['mean_delta_nominal']:+.6f}|{r['mean_regret']:.6f}|")
    lines+=['','## 固定首序下的收益是否发生于平局','','下表包含原规则下非零收益的全部内部案例，不把相位和模型记录当独立洪水。','','|再访|延迟|正收益记录|其中名义预测平局|负收益记录|','|---:|---:|---:|---:|---:|']
    for revisit,delay in sorted({(r['revisit'],r['delay']) for r in support}):
        rr=[r for r in support if r['revisit']==revisit and r['delay']==delay]
        pos=[r for r in rr if r['delta']>0]
        lines.append(f"|{revisit}|{delay}|{len(pos)}|{sum(r['nominal_tie'] for r in pos)}|{sum(r['delta']<0 for r in rr)}|")
    lines+=['','## 解释边界','','原始固定首序是可复现基线，改变平局策略不能回写旧实验成功或失败。新规则检验的是收益是否依赖基线决策丢弃了连续风险余量。maximin_margin不保证真实最优，仍可能被预测误差误导；这是无额外观测、无重训练的更强比较。配对结果只覆盖当前有限任务和家族。','','所有增益案例的预测收益、连续水深余量、旧/新动作及真值收益见 gain_case_tie_evidence.json/CSV；所有冻结动作的逐条重现见checks.json。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'completion.json',dict(completed=True,new_models=0,posthoc=True))
    write_json(ROOT/'results/latest_tie_attribution.json',dict(run=str(run),report=str(run/'REPORT.md')))
    print(run,flush=True)


if __name__=='__main__':main()
