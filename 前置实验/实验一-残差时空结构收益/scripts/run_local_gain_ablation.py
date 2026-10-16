"""Targeted local innovation/history ablations after locating task-level gains."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import OFFSETS,fit_reader,predict_reader,scheduled_records,issued_curve,order_scores
from run_residual_decision_value import dataset,family_error,digest
from analyze_decision_gain import initial_decisions
from analyze_decision_ties import maximum,tie_weights,family_average

VARIANTS=('self_latest','residual_latest','residual_history','full_latest')


def keep_latest(raw):
    result=raw.copy();hist=raw[:,:96].reshape(len(raw),4,6,4)
    saved=np.zeros_like(hist)
    for source in range(4):
        bins=np.argmax(hist[:,source,:,3]>0,axis=1)
        saved[np.arange(len(raw)),source,bins]=hist[np.arange(len(raw)),source,bins]
    result[:,:96]=saved.reshape(len(raw),96)
    return result


def inputs(data):
    self_latest=keep_latest(data['x']['raw_self'])
    residual_latest=self_latest.copy();h=residual_latest[:,:24].reshape(-1,6,4)
    h[:,:,0]-=h[:,:,1];h[:,:,1]=0
    residual_history=data['x']['raw_self'].copy();h=residual_history[:,:24].reshape(-1,6,4)
    h[:,:,0]-=h[:,:,1];h[:,:,1]=0
    return dict(self_latest=self_latest,residual_latest=residual_latest,residual_history=residual_history,
                full_latest=keep_latest(data['x']['raw_full']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('parent',type=Path);args=parser.parse_args()
    parent=args.parent.resolve();config=json.loads((parent/'config.json').read_text(encoding='utf-8'))
    events=load_events(config['manifest']);event_run=Path(config['manifest']).parent
    graph=json.loads((event_run/'physical_graph.json').read_text(encoding='utf-8'))['graph']
    run=new_run('local_gain_ablation',dict(parent=str(parent),posthoc=True,variants=VARIANTS,
        selected_models=32,regularization=[.1,1,10],schedules='R5 D0/10 four phases',protocol='docs/LOCAL_GAIN_ABLATION.md'))
    (run/'PROTOCOL.md').write_text((ROOT/'docs/LOCAL_GAIN_ABLATION.md').read_text(encoding='utf-8'),encoding='utf-8')
    task_rows=[];info=[];fit_info=[];cache={};replications=[]
    for setting in (s for s in config['schedules'] if s['revisit']==5):
        name=setting['schedule'];print('Local ablation',name,flush=True)
        records={e['metadata']['event_id']:scheduled_records(e,5,setting['delay'],setting['phase']) for seq in events.values() for e in seq}
        data={s:dataset(events,s,graph,records,cache) for s in events}
        xx={s:inputs(d) for s,d in data.items()}
        predictions={};arrays={}
        for method in VARIANTS:
            models={a:fit_reader(xx['train'][method],data['train']['y'],a) for a in (.1,1,10)}
            errors={a:family_error(predict_reader(model,xx['validation'][method]),data['validation']) for a,model in models.items()}
            a=min(errors,key=errors.get);model=models[a]
            fit_info.append(dict(setting,method=method,alpha=a,validation_family_mae=errors))
            for k in ('mean','std','weights'):arrays[f'{method}_{k}']=model[k]
            pred=predict_reader(model,xx['test'][method]);predictions[method]=pred;arrays[f'{method}_prediction']=pred
            fam=np.array([r['family'] for r in data['test']['meta']]);cond=np.array([r['condition'] for r in data['test']['meta']])
            err=pred-data['test']['y']
            for f,c in sorted(set(zip(fam,cond))):
                mask=(fam==f)&(cond==c)
                info.append(dict(setting,family=f,condition=c,method=method,mae_m=float(abs(err[mask]).mean())))
        # Nonlearned hold of the signed, version-matched historical residual.
        latest=xx['test']['residual_latest'][:,:24].reshape(-1,6,4)
        residual=latest[:,:,0].sum(axis=1)
        predictions['residual_persistence']=np.repeat(residual[:,None],len(OFFSETS),axis=1)
        arrays['residual_persistence_prediction']=predictions['residual_persistence']
        np.savez_compressed(run/f'{name}_models.npz',**arrays)
        lookup={(r['event_id'],r['now_s'],r['target']):i for i,r in enumerate(data['test']['meta'])}
        cases=initial_decisions(parent/name/'decisions.jsonl.gz')
        for event in events['test']:
            eid=event['metadata']['event_id']
            for cid,methods in cases.items():
                row=methods['forecast_only']
                if row['event_id']!=eid:continue
                now=row['now_s'];f,base=issued_curve(event,now);time=now+OFFSETS
                nominal_scores=order_scores(time,base,row['orders'],now)
                margins=[min(.8-maximum(time,base[:,t['region']],t['start_s'],t['start_s']+12)
                             for t in score['trace']) for score in nominal_scores]
                truth=np.array(row['true_yields'],float)
                nominal={p:float(tie_weights(nominal_scores,margins,p)@truth) for p in ('fixed_first','maximin_margin')}
                if nominal['fixed_first']!=row['baseline_true_yield']:raise ValueError('Nominal reproduction mismatch')
                for method,pred in predictions.items():
                    curve=base.copy()
                    for target in row['targets']:
                        curve[:,target]=np.maximum(0,curve[:,target]+pred[lookup[eid,now,target]])
                    scores=order_scores(time,curve,row['orders'],now)
                    margins=[min(.8-maximum(time,curve[:,t['region']],t['start_s'],t['start_s']+12)
                        for t in ss['trace']) for ss in scores]
                    for policy in ('fixed_first','maximin_margin'):
                        y=float(tie_weights(scores,margins,policy)@truth)
                        task_rows.append(dict(setting,family=row['family'],condition=row['condition'],layout=row['layout'],now_s=now,
                            method=method,policy=policy,true_yield=y,delta_nominal=y-nominal[policy],regret=float(max(truth)-y)))
        replications.append(dict(schedule=name,nominal_reproduced=True))
    family,within,summary=family_average(task_rows)
    write_json(run/'family_task_scores.json',family);write_json(run/'family_phase_averages.json',within)
    write_json(run/'task_summary.json',summary);write_json(run/'family_information_scores.json',info);write_json(run/'reader_selection.json',fit_info)
    write_json(run/'checks.json',dict(nominal_reproductions=replications,models=32,forecast_version_contract='latest published <= observation valid time -20'))
    lines=['# 本地校正中哪个成分产生任务收益','','在已经查看的探索数据上，进一步比较最近一次原始记录、最近一次有符号残差、六箱残差历史、各源最近记录；另有不训练的残差保持基线。32个新读出仅用训练家族拟合、验证家族选择正则。','','## C1 / R5：家族内四相位平均，再家族等权','','单位是每个情境较相同平局规则名义预测增加的完成服务数。','','|延迟|表示|平局规则|增量|剩余遗憾|','|---:|---|---|---:|---:|']
    for r in summary:
        if r['condition']=='C1_local':lines.append(f"|{r['delay']}|{r['method']}|{r['policy']}|{r['mean_delta_nominal']:+.6f}|{r['mean_regret']:.6f}|")
    lines+=['','完整C0/C1/C2和家族值见task_summary.json；对应全未来曲线误差见family_information_scores.json。模型参数和预测、正则选择均已保存。','','## 范围','','新增表示仍共享134个分配槽，但有效秩和岭惩罚作用不同；性能差不是唯一的因果贡献。residual_persistence直接把历史20至小于30秒提前量的残差平移到当前未来曲线，只作为近似基线。结果只覆盖3个内部家族、3个通过50秒门槛的截面布局，未检验噪声泛化或新正式家族。所有新比较都是事后机制诊断。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'completion.json',dict(completed=True,posthoc=True,models=32,new_solver_runs=0,new_qmix_runs=0))
    write_json(ROOT/'results/latest_local_gain_ablation.json',dict(run=str(run),report=str(run/'REPORT.md')))
    print(run,flush=True)


if __name__=='__main__':main()
