"""Check an age-aware signed-residual trend without training a new decision model."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import OFFSETS,scheduled_records,issued_curve,order_scores
from run_residual_decision_value import dataset,family_error
from analyze_decision_gain import initial_decisions
from analyze_decision_ties import maximum,tie_weights,family_average


def trend_prediction(raw,window):
    """Fit slope only to past, actually visible local residuals; no labels."""
    history=raw[:,:24].reshape(-1,6,4)
    x=-60*history[:,:,2];y=history[:,:,0]-history[:,:,1]
    mask=(history[:,:,3]>0)&(-x<=window)
    count=mask.sum(axis=1);safe=np.maximum(count,1)
    xbar=(x*mask).sum(axis=1)/safe;ybar=(y*mask).sum(axis=1)/safe
    center=x-xbar[:,None]
    denom=(center**2*mask).sum(axis=1)
    slope=np.divide((center*(y-ybar[:,None])*mask).sum(axis=1),denom,
        out=np.zeros(len(raw)),where=denom>1e-12)
    intercept=ybar-slope*xbar
    return intercept[:,None]+slope[:,None]*OFFSETS,dict(slope=slope,support=count)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('parent',type=Path);args=parser.parse_args()
    parent=args.parent.resolve();config=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    events=load_events(config['manifest']);base=Path(config['manifest']).parent
    graph=json.loads((base/'physical_graph.json').read_text(encoding='utf-8'))['graph']
    run=new_run('residual_trend_diagnostic',dict(parent=str(parent),posthoc=True,
        windows_s=[20,40,60],selection='validation-family equal-weight full-curve MAE only',
        slope='OLS of signed version-matched residual against past valid time',
        features='same six causal local age bins; actual elapsed age used in extrapolation',
        no_new_learned_readers=True,no_qmix=True))
    cache={};task=[];selection=[];information=[]
    for setting in (s for s in config['schedules'] if s['revisit']==5):
        name=setting['schedule'];print('Residual trend',name,flush=True)
        records={e['metadata']['event_id']:scheduled_records(e,5,setting['delay'],setting['phase']) for seq in events.values() for e in seq}
        data={s:dataset(events,s,graph,records,cache) for s in ('validation','test')}
        validation={w:family_error(trend_prediction(data['validation']['x']['raw_self'],w)[0],data['validation']) for w in (20,40,60)}
        selected=min(validation,key=validation.get)
        predictions={f'trend_{w}':trend_prediction(data['test']['x']['raw_self'],w)[0] for w in (20,40,60)}
        predictions['trend_selected']=predictions[f'trend_{selected}']
        selection.append(dict(setting,selected_window_s=selected,validation_mae=validation))
        fam=np.array([r['family'] for r in data['test']['meta']]);cond=np.array([r['condition'] for r in data['test']['meta']])
        for method,pred in predictions.items():
            error=pred-data['test']['y']
            for f,c in sorted(set(zip(fam,cond))):
                mask=(fam==f)&(cond==c)
                information.append(dict(setting,family=f,condition=c,method=method,mae_m=float(abs(error[mask]).mean())))
        lookup={(r['event_id'],r['now_s'],r['target']):i for i,r in enumerate(data['test']['meta'])}
        cases=initial_decisions(parent/name/'decisions.jsonl.gz')
        by_event={e['metadata']['event_id']:e for e in events['test']}
        for cid,methods in cases.items():
            row=methods['forecast_only']
            if row['split']!='test':continue
            eid=row['event_id'];event=by_event[eid];now=row['now_s'];f,basecurve=issued_curve(event,now);times=now+OFFSETS
            ss=order_scores(times,basecurve,row['orders'],now)
            mm=[min(.8-maximum(times,basecurve[:,t['region']],t['start_s'],t['start_s']+12) for t in order['trace']) for order in ss]
            truth=np.array(row['true_yields'])
            nominal={p:float(tie_weights(ss,mm,p)@truth) for p in ('fixed_first','maximin_margin')}
            for method,pred in predictions.items():
                curve=basecurve.copy()
                for target in row['targets']:
                    curve[:,target]=np.maximum(0,curve[:,target]+pred[lookup[eid,now,target]])
                scores=order_scores(times,curve,row['orders'],now)
                margins=[min(.8-maximum(times,curve[:,t['region']],t['start_s'],t['start_s']+12) for t in order['trace']) for order in scores]
                for policy in nominal:
                    y=float(tie_weights(scores,margins,policy)@truth)
                    task.append(dict(setting,family=row['family'],condition=row['condition'],method=method,policy=policy,
                        true_yield=y,delta_nominal=y-nominal[policy],regret=float(max(truth)-y)))
        np.savez_compressed(run/f'{name}_predictions.npz',**predictions)
    family,within,summary=family_average(task)
    write_json(run/'family_task_scores.json',family);write_json(run/'family_phase_averages.json',within)
    write_json(run/'task_summary.json',summary);write_json(run/'information_scores.json',information);write_json(run/'window_selection.json',selection)
    lines=['# 观测年龄能否用残差趋势补偿','','在每个决策时刻，仅使用六个年龄箱中已到达的本地有符号残差，对过去有效时间拟合直线；从观测时间外推到当前及未来0至50秒。少于两条可用记录时斜率为零。窗口候选20/40/60秒只按验证家族MAE选择；全部候选任务结果同时保留。没有用内部任务收益选择窗口。','','## C1 / R5','','各方法较同平局规则名义预测的服务数差。','','|延迟|趋势窗口|平局规则|增量|剩余遗憾|','|---:|---|---|---:|---:|']
    for r in summary:
        if r['condition']=='C1_local':
            lines.append(f"|{r['delay']}|{r['method']}|{r['policy']}|{r['mean_delta_nominal']:+.6f}|{r['mean_regret']:.6f}|")
    lines+=['','## 解释限制','','这是同一探索数据上的机制跟进，不是新独立验证。直线外推假设残差趋势短期延续，局部脉冲、反射和测量噪声可能破坏它。当前没有噪声测试。预测MAE与任务收益可不同向；不能依据某个候选在内部任务中表现好事后改用它作为已验证方法。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'completion.json',dict(completed=True,posthoc=True,selection_uses_internal_test=False))
    write_json(ROOT/'results/latest_residual_trend.json',dict(run=str(run),report=str(run/'REPORT.md')))
    print(run,flush=True)


if __name__=='__main__':main()
