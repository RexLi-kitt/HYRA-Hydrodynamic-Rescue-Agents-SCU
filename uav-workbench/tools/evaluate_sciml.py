"""Frozen-confirmation evaluation; never fits or selects a policy.

Register this script's hash before confirmation water is generated. All greedy
policy runs and common-route forecasts are resumable, immutable result shards.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from collections import defaultdict
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F

from hyra_uav.core import ACTIONS,Config,FloodProcess
from hyra_learning.team_env import TeamConfig,TeamEnv
from hyra_learning.observations import encode
from hyra_learning.networks import SearchAgent
from hyra_learning.world_model import WorldHead,AuxiliaryLabels,supervised_loss,field_loss,balanced_weights
from hyra_learning.physics import SWEOperator
from hyra_learning.sciml_study import actor_episode,verify
from hyra_learning.sciml_data import read,write,digest,prepare


def key(method,seed):return method.replace('+','-')+f'-s{seed}'


def register(root):
    path=root/'evaluation-plan.json'
    if path.exists():raise ValueError('Preserve the existing evaluation registration')
    if any(e['split']=='holdout' for e in read(root/'index.json')):raise ValueError('Confirmation water must not predate evaluation registration')
    design=read(root/'data-plan.json')['design']
    write(path,dict(schema='hyra-sciml-confirmation-evaluation-v1',formal_plan_sha256=digest(root/'formal-plan.json'),
        evaluator_sha256=digest(Path(__file__)),primary='event-equal physical-time class-balanced current correct coverage',
        positive_event='Physics exceeds both Data and Sur; strict greater-than zero',
        forcing_positive='each of the3 new forcing families has positive Physics-Data and Physics-Sur mean',
        gate=design['hypothesis_gate'],policy_episodes=108,rule_episodes=27,forecast_rows=9720,
        rules='fixed row-partitioned R0 scan and R1/R2 two-action greedy, coordinated by legal geometric path safety',
        forecast='selected Data/Physics models replay same3 rule trajectories per event; predict before reading labels;60 times then route then seed average',
        forecast_metrics=['10s depth all/unknown/stale MAE','10s class-balanced wet/dry error','10s momentum MAE',
                          'current depth error','normalized auxiliary supervised loss','normalized local physical rate discrepancy',
                          'absolute local volume-rate residual m3/s'],
        na='empty unknown/stale subsets remain null, report counts; no invented zeros',
        convergence_diagnostic='descriptive validation-return change: mean(1800,2000)-mean(1400,1600); abs>0.02 flagged as late drift, never a proof of convergence or reason to extend budget',
        independence='9 crossed events from3 held-out parents and3 held-out forcing families; training seeds and time samples are not new independent events',
        stop='one fixed mechanism round; no refit, reward/metric changes or holdout tuning; success criterion is not a claim of real-flood validity or superiority to rules'))


def load_model(root,row,device):
    path=root/row['path']
    if digest(path)!=row['sha256']:raise ValueError('Selected model changed')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload['method']!=row['method'] or payload['episode']!=row['episode']:raise ValueError('Selected checkpoint metadata mismatch')
    actor=SearchAgent().to(device);actor.load_state_dict(payload['actor']);actor.eval()
    world=None
    if payload['world'] is not None:
        world=WorldHead().to(device);world.load_state_dict(payload['world']);world.eval()
    return actor,world


def export_actor(root,row,payload_actor):
    folder=root/'deployment'/key(row['method'],row['seed']);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'actor.pt'
    if not path.exists():torch.save(payload_actor.state_dict(),path)
    write(folder/'actor.json',dict(method=row['method'],seed=row['seed'],selected_episode=row['episode'],actor_sha256=digest(path),
        selected_checkpoint_sha256=row['sha256'],input='16x7x7 legal local/pooled shared map,6 pose values,previous4-action onehot,step and per-agent hidden64',
        execution='clean actor only; no mixer, world head, true hydraulic fields, source labels or future forcing'))


def forecast_rows(actor,world,process,plan,entry,case_index,labels,rule_replay,row,rule):
    device=next(actor.parameters()).device
    env=TeamEnv(process,Config(**plan['config']),TeamConfig(**plan['team']))
    view,_=env.reset();hidden=actor.initial_hidden(2);previous=torch.zeros(2,4,device=device)
    records=[]
    for t,action_row in enumerate(rule_replay['actions']):
        encoded=encode(view)
        with torch.no_grad():
            _,hidden,_,_=actor(torch.as_tensor(encoded['images'],device=device),torch.as_tensor(encoded['poses'],device=device),previous,t,hidden,enhance=False)
            # Only current legal history and known static bed enter predictions.
            prediction=world(hidden.mean(dim=0)[None],torch.as_tensor(view.bed.copy(),dtype=torch.float64,device=device)[None])
            # True fields and current source are read strictly AFTER prediction,
            # for scoring. They do not become observation or policy inputs.
            target=labels.batch([case_index],[t],device)
            dt=float(target['dt'][0])
            evolved=SWEOperator().step(prediction['current'],target['bed'],dt,source=target['source'],drains=target['drains'],
                                      manning=target['manning'],solid=target['solid'],wall_x=target['wall_x'],wall_y=target['wall_y'])
            physical_rate=(evolved-prediction['current'])/dt
            physics=float(field_loss(prediction['rate'],physical_rate,balanced_weights(target)))
            mass_rate=float((prediction['rate'][:,0]-physical_rate[:,0]).sum().abs()*25)
            supervised=float(supervised_loss(prediction,target))
            future=prediction['future'][0].cpu().numpy();truth=target['future'][0].cpu().numpy()
            current=prediction['current'][0].cpu().numpy();truth_current=target['current'][0].cpu().numpy()
        known=view.agents[0].labels>=0
        age=view.agents[0].time_s-view.agents[0].observed_s
        stale=known&(age>=60)
        error=np.abs(future[0]-truth[0]);wet=truth[0]>=.05;pred_wet=future[0]>=.05
        errors=[float((pred_wet[wet==c]!=(wet[wet==c])).mean()) for c in (False,True) if np.any(wet==c)]
        records.append(dict(case=entry['id'],family=entry['family'],forcing_family=entry['forcing_family'],method=row['method'],seed=row['seed'],route=rule,time_s=t*10,
            model_sha256=row['sha256'],source_sha256=entry['source_sha256'],unknown_cells=int((~known).sum()),stale_cells=int(stale.sum()),
            future_h_mae_m=float(error.mean()),future_h_unknown_mae_m=float(error[~known].mean()) if (~known).any() else None,
            future_h_stale_mae_m=float(error[stale].mean()) if stale.any() else None,
            future_wet_classbalanced_error=float(np.mean(errors)),future_momentum_mae_m2_s=float(np.abs(future[1:]-truth[1:]).mean()),
            current_h_mae_m=float(np.abs(current[0]-truth_current[0]).mean()),supervised_loss=supervised,
            physical_rate_loss=physics,absolute_volume_rate_residual_m3_s=mass_rate))
        actions=[ACTIONS.index(a) for a in action_row['actions']]
        view,_,_,_,_=env.step(actions)
        previous=F.one_hot(torch.as_tensor(actions,device=device),4).float()
    assert len(records)==60
    return records


def average(rows,field):
    values=[r[field] for r in rows if r[field] is not None]
    return float(np.mean(values)) if values else None


def gate_from_events(events,criteria):
    paired=[]
    for case in sorted({r['case'] for r in events}):
        peers={r['method']:r for r in events if r['case']==case}
        physical=peers['Sur-QMIX+Physics'];data=peers['Sur-QMIX+Data'];sur=peers['Sur-QMIX'];base=peers['QMIX']
        paired.append(dict(case=case,family=physical['family'],forcing_family=physical['forcing_family'],
            delta_Data=physical['quality']-data['quality'],delta_Sur=physical['quality']-sur['quality'],delta_QMIX=physical['quality']-base['quality'],
            delta_collisions=physical['collisions']-data['collisions'],delta_motion_cost=physical['motion_cost']-data['motion_cost']))
    means={g:float(np.mean([r['delta_'+g] for r in paired])) for g in ('Data','Sur','QMIX','collisions','motion_cost')}
    forcing={f:{g:float(np.mean([r['delta_'+g] for r in paired if r['forcing_family']==f])) for g in ('Data','Sur')}
             for f in sorted({r['forcing_family'] for r in paired})}
    positive=sum(r['delta_Data']>0 and r['delta_Sur']>0 for r in paired)
    checks=dict(mean_Physics_Data=means['Data']*100>=criteria['physics_vs_data_mean_min_pp'],
                mean_Physics_Sur=means['Sur']*100>=criteria['physics_vs_sur_mean_min_pp'],
                simultaneous_positive_events=positive>=criteria['positive_events_min'],
                each_forcing_positive=all(d[g]>0 for d in forcing.values() for g in ('Data','Sur')),
                collision_increase=means['collisions']<=criteria['mean_collision_increase_max_per_episode'],
                motion_cost_increase=means['motion_cost']<=criteria['mean_motion_cost_increase_max'])
    return dict(paired=paired,mean_deltas=means,forcing_deltas=forcing,positive_events=positive,criteria=checks,
                passed=all(checks.values()),failed_criteria=[k for k,v in checks.items() if not v])


def summarize(root):
    registration=read(root/'evaluation-plan.json');folder=root/'evaluation'
    policy=[read(p) for p in sorted((folder/'policies').glob('*.json'))]
    rules=[read(p) for p in sorted((folder/'rules').glob('*.json'))]
    assert len(policy)==registration['policy_episodes'] and len(rules)==registration['rule_episodes']
    keys={(r['case'],r['method'],r['seed']) for r in policy};assert len(keys)==108
    held={e['id'] for e in read(root/'index.json') if e['split']=='holdout'}
    methods=read(root/'formal-plan.json')['methods']
    assert keys=={(case,method,seed) for case in held for method in methods for seed in (11,29,47)}
    assert {(r['case'],r['method']) for r in rules}=={(case,rule) for case in held for rule in ('R0','R1','R2')}
    events=[]
    for case,method in sorted({(r['case'],r['method']) for r in policy}):
        rows=[r for r in policy if r['case']==case and r['method']==method]
        assert len(rows)==3 and {r['seed'] for r in rows}=={11,29,47}
        events.append(dict(case=case,family=rows[0]['family'],forcing_family=rows[0]['forcing_family'],method=method,
            quality=float(np.mean([r['metrics']['time_average_correct_coverage'] for r in rows])),
            collisions=float(np.mean([r['metrics']['collision_events'] for r in rows])),motion_cost=float(np.mean([r['metrics']['motion_cost'] for r in rows])),
            return_=float(np.mean([r['metrics']['return_'] for r in rows])),
            training_seed_quality_std=float(np.std([r['metrics']['time_average_correct_coverage'] for r in rows],ddof=1))))
    forecasts=[]
    for path in sorted((folder/'forecasts').glob('*.jsonl')):
        forecasts.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())
    assert len(forecasts)==registration['forecast_rows']
    assert len({(r['case'],r['method'],r['seed'],r['route'],r['time_s']) for r in forecasts})==9720
    assert {(r['case'],r['method'],r['seed'],r['route'],r['time_s']) for r in forecasts}=={
        (case,method,seed,route,t) for case in held for method in ('Sur-QMIX+Data','Sur-QMIX+Physics')
        for seed in (11,29,47) for route in ('R0','R1','R2') for t in range(0,600,10)}
    fields=['future_h_mae_m','future_h_unknown_mae_m','future_h_stale_mae_m','future_wet_classbalanced_error',
            'future_momentum_mae_m2_s','current_h_mae_m','supervised_loss','physical_rate_loss','absolute_volume_rate_residual_m3_s']
    predictions=[]
    for case,method in sorted({(r['case'],r['method']) for r in forecasts}):
        seeds=[]
        for seed in (11,29,47):
            routes=[]
            for rule in ('R0','R1','R2'):
                rows=[r for r in forecasts if r['case']==case and r['method']==method and r['seed']==seed and r['route']==rule]
                assert len(rows)==60
                routes.append({field:average(rows,field) for field in fields})
            seeds.append({field:average(routes,field) for field in fields})
        predictions.append(dict(case=case,method=method,**{field:average(seeds,field) for field in fields},
            empty_unknown_frames=sum(r['unknown_cells']==0 for r in forecasts if r['case']==case and r['method']==method),
            empty_stale_frames=sum(r['stale_cells']==0 for r in forecasts if r['case']==case and r['method']==method)))
    main={m:float(np.mean([r['quality'] for r in events if r['method']==m])) for m in sorted({r['method'] for r in events})}
    rule_means={m:float(np.mean([r['metrics']['time_average_correct_coverage'] for r in rules if r['method']==m])) for m in ('R0','R1','R2')}
    gate=gate_from_events(events,registration['gate'])
    drift=[]
    for path in sorted((root/'formal').glob('*/validation.json')):
        rows=read(path);by_ep={r['episode']:r['mean_return'] for r in rows}
        assert len(by_ep)==10
        delta=.5*(by_ep[1800]+by_ep[2000]-by_ep[1400]-by_ep[1600])
        drift.append(dict(job=path.parent.name,late_validation_return_change=delta,late_drift_flag=abs(delta)>.02))
    write(root/'confirmation-summary.json',dict(status='completed',policy_episodes=108,rule_episodes=27,forecast_rows=9720,
        main_quality=main,rule_quality=rule_means,events=events,predictions=predictions,gate=gate,late_training_drift=drift,
        parents=3,forcing_families=3,hydraulic_events=9,
        limitation='One synthetic study,3 held-out parents x3 new forcing templates; seeds/routes/frames are not independent events. Gate success does not prove convergence, real-flood validity or superiority to rules.'))
    print(json.dumps(dict(main_quality=main,rule_quality=rule_means,gate=gate),ensure_ascii=False))


def evaluate(root):
    registration=read(root/'evaluation-plan.json');plan=read(root/'formal-plan.json');freeze=read(root/'model-freeze.json')
    if registration['evaluator_sha256']!=digest(Path(__file__)) or registration['formal_plan_sha256']!=digest(root/'formal-plan.json'):
        raise ValueError('Evaluation code or formal protocol changed')
    assert freeze['formal_plan_sha256']==digest(root/'formal-plan.json') and len(freeze['selected_models'])==12
    verify(plan)
    for row in freeze['selected_models']:
        assert digest(root/row['path'])==row['sha256']
    training_index=root/'training-index-frozen.json'
    if not training_index.exists():
        assert digest(root/'index.json')==freeze['training_input_sha256']
        shutil.copyfile(root/'index.json',training_index)
    assert digest(training_index)==freeze['training_input_sha256']
    prepare(root,('holdout',))
    held=[e for e in read(root/'index.json') if e['split']=='holdout'];assert len(held)==9
    processes=[FloodProcess(e['path']) for e in held];labels=AuxiliaryLabels(held)
    output=root/'evaluation'
    for name in ('rules','policies','forecasts'):(output/name).mkdir(parents=True,exist_ok=True)
    for entry,process in zip(held,processes):
        for rule in ('R0','R1','R2'):
            path=output/'rules'/f"{rule}-{entry['id']}.json"
            if path.exists():
                prior=read(path);assert prior['case']==entry['id'] and prior['method']==rule and prior['source_sha256']==entry['source_sha256']
                continue
            metrics,env=actor_episode(None,process,plan,rule)
            write(path,dict(case=entry['id'],family=entry['family'],forcing_family=entry['forcing_family'],method=rule,
                            source_sha256=entry['source_sha256'],metrics=metrics,replay=env.result()))
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for row in freeze['selected_models']:
        actor,world=load_model(root,row,plan['runtime']['device']);export_actor(root,row,actor)
        for case_index,(entry,process) in enumerate(zip(held,processes)):
            path=output/'policies'/f"{key(row['method'],row['seed'])}-{entry['id']}.json"
            if not path.exists():
                metrics,env=actor_episode(actor,process,plan)
                write(path,dict(case=entry['id'],family=entry['family'],forcing_family=entry['forcing_family'],method=row['method'],seed=row['seed'],
                    model_sha256=row['sha256'],source_sha256=entry['source_sha256'],metrics=metrics,replay=env.result()))
            else:
                previous=read(path);assert previous['model_sha256']==row['sha256'] and previous['source_sha256']==entry['source_sha256']
            if world is not None:
                for rule in ('R0','R1','R2'):
                    target=output/'forecasts'/f"{key(row['method'],row['seed'])}-{entry['id']}-{rule}.jsonl"
                    if target.exists():
                        existing=[json.loads(line) for line in target.read_text(encoding='utf-8').splitlines()]
                        assert len(existing)==60 and {r['time_s'] for r in existing}==set(range(0,600,10))
                        assert all(r['case']==entry['id'] and r['method']==row['method'] and r['seed']==row['seed']
                                   and r['route']==rule and r['model_sha256']==row['sha256'] and r['source_sha256']==entry['source_sha256'] for r in existing)
                        continue
                    replay=read(output/'rules'/f"{rule}-{entry['id']}.json")['replay']
                    records=forecast_rows(actor,world,process,plan,entry,case_index,labels,replay,row,rule)
                    partial=target.with_suffix('.partial');partial.write_text(''.join(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n' for r in records),encoding='utf-8');partial.replace(target)
        print('evaluated',row['method'],row['seed'],flush=True)
        del actor,world
        if torch.cuda.is_available():torch.cuda.empty_cache()
    summarize(root)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['register','evaluate','summarize']);parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve()
    lock=root/'evaluation.lock';token=uuid.uuid4().hex
    with lock.open('x',encoding='utf-8') as file:json.dump(dict(pid=os.getpid(),token=token,stage=args.stage),file)
    try:
        dict(register=register,evaluate=evaluate,summarize=summarize)[args.stage](root)
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()
