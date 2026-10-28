import itertools
import hashlib
import csv
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from scale100_train import rollout, config
from scale100_backend import make_actor
from scale100_development import legal_rule
from scale100_rules import water_aware
from scale100_env import ScaleEnv
from .belief import Parameters
from .environment import RouteEnv
from .data import cases
from .common import read, write, digest, sources
from . import METHODS, SCHEMA

RULES=('frontier','sweep_refresh','water_aware')


def rule_run(case,seed,duration,name):
    env=ScaleEnv(case,replace(config('E2','B3',seed),duration_s=float(duration))); env.reset(); done=False
    while not done:
        actions=water_aware(env) if name=='water_aware' else legal_rule(env,name)
        _,_,done=env.step(actions)
    metrics=env.metrics(); metrics['rescue_fraction']=metrics['rescued']/metrics['targets']
    metrics['service_interruptions']=sum(e['type']=='service_interrupted' for e in env.events)
    return metrics


def select_task(study):
    study=Path(study); output=study/'task-selection.json'
    if output.exists(): return read(output)
    rows=[]; duration_selected=None
    for duration in (900,600,450):
        for row,case in cases(study/'data/dataset.json','development'):
            for seed in (11,29,47):
                for name in RULES:
                    rows.append(dict(duration_s=duration,parent=row['parent_terrain'],mechanism=row['mechanism'],
                                     seed=seed,rule=name,metrics=rule_run(case,seed,duration,name)))
                    write(study/'task-selection-progress.json',dict(duration_s=duration,completed=len(rows),maximum=162,
                          last=dict(parent=row['parent_terrain'],mechanism=row['mechanism'],seed=seed,rule=name),confirmation_opened=False))
        best=max(np.mean([r['metrics']['rescue_fraction'] for r in rows if r['duration_s']==duration and r['rule']==name]) for name in RULES)
        interruptions=sum(r['metrics']['service_interruptions'] for r in rows if r['duration_s']==duration)
        if .2<=best<=.9 and interruptions>0:
            duration_selected=duration; break
    result=dict(schema=SCHEMA,duration_s=duration_selected or 900,selected_by='development rules only, longest eligible',
                discriminable=duration_selected is not None,scope='descriptive if no eligible duration',
                service_limits_unchanged=True,rows=rows,confirmation_opened=False)
    write(output,result); return result


def actor_run(study, actor_path, method, scenario_seed, case, save_replay=None):
    study=Path(study); meta=read(Path(actor_path).with_suffix('.json'))
    if meta['schema']!=SCHEMA or meta['sources']!=sources() or digest(actor_path)!=meta['actor_sha256']:
        raise ValueError('Actor version/source/digest mismatch')
    if digest(study/'parameters.json')!=meta['parameters_sha256']: raise ValueError('Estimator changed')
    duration=read(study/'task-selection.json')['duration_s']
    p=Parameters(**read(study/'parameters.json')['parameters'][method])
    env=RouteEnv(case,replace(config('E2','B3',scenario_seed),duration_s=float(duration)),method,p)
    actor=make_actor('aux_resilient_v4'); actor.load_state_dict(torch.load(actor_path,weights_only=True,map_location='cpu')); actor.eval()
    torch.set_num_threads(1)
    total,_,metrics=rollout(actor,env,np.random.default_rng(scenario_seed),0.,False)
    if save_replay:
        # Rebuild the entire legal run independently from the saved actions.
        check=RouteEnv(case,env.config,method,p); check.reset(); rebuilt_return=0.
        from sur_rescue.environment import ACTIONS
        for action in env.actions:
            _,reward,_=check.step([ACTIONS.index(a) for a in action['actions']]); rebuilt_return+=reward
        raw_equal=(env.observations==check.observations and env.events==check.events)
        view_equal=np.array_equal(env.view().images,check.view().images)
        if not raw_equal or not view_equal or total!=rebuilt_return: raise AssertionError('Replay audit failed')
        write(save_replay,dict(schema=SCHEMA,source_sha256=case.digest,method=method,config=meta['config']|dict(seed=scenario_seed),
            actor_sha256=digest(actor_path),parameters_sha256=meta['parameters_sha256'],
            actions=env.actions,observations=env.observations,events=env.events,
            metrics=metrics,return_=total,replay_audit=dict(raw_records_equal=raw_equal,final_actor_view_equal=view_equal,return_equal=True)))
    return metrics


def select_models(study):
    study=Path(study); output=study/'sealed-models.json'
    if output.exists():
        seal=read(output)
        if seal['sources']!=sources(): raise ValueError('Code changed after seal')
        for row in seal['models']:
            if digest(row['actor'])!=row['actor_sha256']: raise ValueError('Sealed model changed')
        return seal
    selected=[]; all_rows=[]
    dev=list(cases(study/'data/dataset.json','development'))
    for method in METHODS:
        for seed in (11,29,47):
            candidates=[]
            for episode in (256,512,1024):
                actor=study/'formal'/f'{method}-s{seed}'/f'actor-{episode:04d}.pt'
                values=[]
                for row,case in dev:
                    for scenario in (11,29,47):
                        metrics=actor_run(study,actor,method,scenario,case)
                        values.append(metrics)
                score=tuple(float(np.mean([m[k] for m in values])) for k in ('rescue_fraction','first_coverage_auc','current_correct_coverage_auc'))+(-episode,)
                entry=dict(method=method,seed=seed,episode=episode,actor=str(actor),actor_sha256=digest(actor),score=list(score))
                candidates.append(entry); all_rows.append(entry)
            selected.append(max(candidates,key=lambda r:r['score']))
            write(study/'selection-progress.json',dict(completed=len(selected),total=24,confirmation_opened=False))
    seal=dict(schema=SCHEMA,sources=sources(),parameters_sha256=digest(study/'parameters.json'),
              dataset_sha256=digest(study/'data/dataset.json'),models=selected,candidates=all_rows,
              development_only=True,confirmation_opened=False)
    write(output,seal); return seal


def confirm(study):
    study=Path(study); seal=select_models(study)
    out=study/'confirmation'; out.mkdir(exist_ok=True)
    write(out/'opened.json',dict(seal_sha256=digest(study/'sealed-models.json'),sources=sources(),
                                parameters_sha256=digest(study/'parameters.json')))
    rows=[]; held=list(cases(study/'data/dataset.json','holdout'))
    for selected in seal['models']:
        for row,case in held:
            for scenario in (11,29,47):
                name=f"{selected['method']}-s{selected['seed']}-{case.case['seed']}-{row['mechanism']}-l{scenario}"
                path=out/(name+'.json')
                if path.exists():
                    cached=read(path)
                    if cached['actor_sha256']!=selected['actor_sha256'] or cached['source_sha256']!=case.digest: raise ValueError('Cached confirmation mismatch')
                    metrics=cached['metrics']
                else: metrics=actor_run(study,selected['actor'],selected['method'],scenario,case,path)
                rows.append(dict(method=selected['method'],training_seed=selected['seed'],scenario_seed=scenario,
                                 parent=row['parent_terrain'],mechanism=row['mechanism'],metrics=metrics,replay=str(path)))
        write(out/'progress.json',dict(status='running',completed=len(rows),total=1728))
    duration=read(study/'task-selection.json')['duration_s']
    for name in RULES:
        for row,case in held:
            for scenario in (11,29,47):
                metrics=rule_run(case,scenario,duration,name)
                rows.append(dict(method=name,training_seed=None,scenario_seed=scenario,parent=row['parent_terrain'],mechanism=row['mechanism'],metrics=metrics))
    write(out/'results.json',dict(schema=SCHEMA,rows=rows,models=1728,rules=216,independent_parents=8))
    write(out/'progress.json',dict(status='completed',models=1728,rules=216))


def report(study):
    study=Path(study); rows=read(study/'confirmation/results.json')['rows']; parents=sorted({r['parent'] for r in rows})
    means={}; parent_values={}; comparisons=[]
    for method in (*METHODS,*RULES):
        pv=np.array([np.mean([r['metrics']['rescue_fraction'] for r in rows if r['method']==method and r['parent']==parent]) for parent in parents])
        parent_values[method]=pv; means[method]=float(pv.mean())
    rng=np.random.default_rng(20261027); bootstrap=rng.integers(0,len(parents),(10000,len(parents)))
    for i,method in enumerate(('R1','R2','R3'),1):
        for baseline in ('Bpost',f'C{i}','Braw'):
            delta=parent_values[method]-parent_values[baseline]; ci=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
            signs=np.asarray(list(itertools.product((-1,1),repeat=len(delta))))
            p=float(np.mean(np.abs((signs*delta).mean(axis=1))>=abs(delta.mean())-1e-15))
            comparisons.append(dict(method=method,baseline=baseline,delta=float(delta.mean()),ci95=ci.tolist(),paired_parent_deltas=delta.tolist(),exact_sign_flip_p=p))
    primary=sorted([r for r in comparisons if r['baseline']=='Bpost'],key=lambda r:r['exact_sign_flip_p'])
    previous=0.
    for rank,r in enumerate(primary):
        previous=max(previous,min(1.,(3-rank)*r['exact_sign_flip_p'])); r['holm_p']=previous
    write(study/'report.json',dict(means=means,parent_values={k:v.tolist() for k,v in parent_values.items()},comparisons=comparisons,
                                   primary_holm=primary,scope='eight synthetic parents; no real-sensor or real-flood claim'))
    fields=['method','training_seed','scenario_seed','parent','mechanism','rescue_fraction','first_coverage_auc',
            'current_correct_coverage_auc','rescued','targets','distance_m','collision_events','first_discovery_s',
            'first_rescue_s','unrescued','service_interruptions','inference_max_s','inference_max_residual']
    with (study/'per-episode.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader()
        for row in rows: writer.writerow({**row,**row['metrics']})
    lines=['# 三路线信号建模：正式比较结果','','状态：24项训练、开发选模及新确认评价已完成。',
           '', '| 方法 | 施救比例 |','|---|---:|']
    lines += [f'| {method} | {value:.4f} |' for method,value in means.items()]
    lines += ['', '## 三条路线相对共同后验基线', '', '| 路线 | 差值/百分点 | 父地形配对95%区间/百分点 | Holm p |','|---|---:|---|---:|']
    for r in primary:
        lines.append(f"| {r['method']} | {100*r['delta']:.2f} | [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}] | {r['holm_p']:.4f} |")
    lines += ['', '## 真实水情与置乱控制', '', '| 路线 | 相对对应控制的施救差值/百分点 | 95%区间/百分点 |','|---|---:|---|']
    for r in comparisons:
        if r['baseline'].startswith('C'):
            lines.append(f"| {r['method']}−{r['baseline']} | {100*r['delta']:.2f} | [{100*r['ci95'][0]:.2f}, {100*r['ci95'][1]:.2f}] |")
    lines += ['', '## 已见与未见来水机制', '', '| 方法 | 单峰确认施救比例 | 未见双峰施救比例 |','|---|---:|---:|']
    for method in (*METHODS,*RULES):
        values=[]
        for kind in ('single','double'):
            pv=[np.mean([r['metrics']['rescue_fraction'] for r in rows if r['method']==method and r['parent']==parent and (r['mechanism']=='double')==(kind=='double')]) for parent in parents]
            values.append(float(np.mean(pv)))
        lines.append(f'| {method} | {values[0]:.4f} | {values[1]:.4f} |')
    diagnostic=read(study/'offline-diagnostics.json')['rows']
    lines += ['', '## 开发集信号诊断', '', '合成条件单独用对应训练样本拟合；不属于闭环确认结果。', '', '| 条件 | 方法 | 未观测信号RMSE | 条件NLL | 95%覆盖率 |','|---|---|---:|---:|---:|']
    for condition in ('original','null','attenuation','directional'):
        for method in METHODS:
            part=[r for r in diagnostic if r['condition']==condition and r['method']==method]
            def avg(key):
                vals=[r[key] for r in part if r.get(key) is not None]
                return f'{np.mean(vals):.4f}' if vals else '—'
            lines.append(f"| {condition} | {method} | {avg('rmse')} | {avg('nll')} | {avg('coverage95')} |")
    task=read(study/'task-selection.json')
    lines += ['', f"开发规则选择时限为{task['duration_s']}秒；任务可判别门槛通过：{task['discriminable']}。若未通过，本轮按登记仅作描述性比较。"]
    lines += ['', '24项训练不是24个独立事件；统计按8个父地形聚合。区间跨零表示方向尚不确定。',
              '固定人员未发生漂流；水情可能通过初始选址规律提供位置先验。合成探测机制诊断须与本表分开解释。',
              '三路线是受限参数族比较；SAR与二次CRF存在等价情形。当前预算不构成收敛或真实洪水适用性证明。']
    (study/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
