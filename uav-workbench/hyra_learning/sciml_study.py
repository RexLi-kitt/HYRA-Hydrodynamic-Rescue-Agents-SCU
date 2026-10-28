"""One independent four-group study: pilot, frozen training, held-out evaluation."""
import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')

from .runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F

from hyra_uav.core import Config,FloodProcess
from .team_env import TeamConfig,TeamEnv
from .observations import encode
from .training import EpisodeReplay,central_state,rollout,memory_usage
from .world_model import AuxiliaryLabels,WorldLearner
from .sciml_data import ROOT,TERRAIN,read,write,digest,prepare


def executable_sources():
    paths=list((ROOT/'hyra_learning').rglob('*.py'))+list((ROOT/'hyra_uav').glob('*.py'))
    paths += [ROOT/'learning.py',ROOT/'learning_configs/sciml_round1_design_20261025.json',
              ROOT/'hyra_learning/vendor/PYMARL_LICENSE',ROOT/'hyra_learning/vendor/SOURCE.json']
    paths += [TERRAIN/name for name in ('flood_solver.py','flood_numerics.py','flood_run.py','multilayer.py','map_sessions.py')]
    return {str(p.resolve()):digest(p) for p in sorted(paths)}


def runtime(device):
    return dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),device=device,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)


def verify(plan):
    if plan['sources']!=executable_sources():raise ValueError('Scientific sources changed after executable freeze')
    for entry in plan['inputs']:
        if digest(Path(entry['path']))!=entry['source_sha256'] or digest(Path(entry['auxiliary_path']))!=entry['auxiliary_sha256']:
            raise ValueError('Frozen hydraulic input changed')


def archive_sources(path,sources):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sources:
            p=Path(source)
            archive.write(p,str(p.relative_to(ROOT.parent)).replace('\\','/'))


def actor_episode(actor,process,plan,rule=None):
    env=TeamEnv(process,Config(**plan['config']),TeamConfig(**plan['team']))
    view,_=env.reset()
    from .team_rules import TeamRules
    policy=TeamRules(rule,view,env.config,env.team.separation_m) if rule else None
    hidden=actor.initial_hidden(2) if actor is not None else None
    previous=torch.zeros(2,4,device=hidden.device) if hidden is not None else None
    if actor is not None:actor.eval()
    total=0.
    done=False
    for t in range(60):
        if policy:
            actions=policy.choose(view)
        else:
            data=encode(view)
            with torch.no_grad():
                q,hidden,_,_=actor(torch.as_tensor(data['images'],device=hidden.device),
                                   torch.as_tensor(data['poses'],device=hidden.device),previous,t,hidden,enhance=False)
            actions=q.argmax(dim=-1).cpu().numpy()
            previous=F.one_hot(torch.as_tensor(actions,device=hidden.device),4).float()
        view,reward,_,done,_=env.step(actions)
        total+=reward
        if done:break
    return dict(return_=total,time_average_correct_coverage=env.integrated_quality/env.config.duration_s,
                distance_m=env.distance_m,turn_s=env.turn_s,collision_events=env.collision_events,
                motion_cost=env.team.distance_cost_per_m*env.distance_m+env.team.turn_cost_per_s*env.turn_s,
                final=env.snapshots[-1]['metrics']),env


def capacity(learner):
    return dict(actor=sum(p.numel() for p in learner.actor.parameters()),mixer=sum(p.numel() for p in learner.mixer.parameters()),
                auxiliary=sum(p.numel() for p in learner.world.parameters()) if learner.world is not None else 0)


def save(path,payload):
    temporary=path.with_suffix(path.suffix+'.partial');torch.save(payload,temporary);temporary.replace(path)


def checkpoint(folder,learner,replay,rng,episode,history,references,validation):
    payload=learner.checkpoint(replay,rng,episode,history)
    # Immutable episode shards avoid rewriting the entire replay buffer into
    # every recovery file. Each recovery version is immutable; only its pointer
    # changes atomically, so interruption cannot invalidate the previous pair.
    payload['replay']=[];payload['replay_refs']=references[-replay.capacity:];payload['validation']=validation
    recovery=folder/'recoveries';recovery.mkdir(exist_ok=True)
    path=recovery/f'recovery-{episode:04d}-{uuid.uuid4().hex[:8]}.pt';save(path,payload)
    write(folder/'latest.json',dict(path=str(path.relative_to(folder)),sha256=digest(path),episode=episode,updates=learner.updates))
    return payload


def restore(folder,learner,replay,rng):
    pointer=read(folder/'latest.json');path=folder/pointer['path']
    if digest(path)!=pointer['sha256']:raise ValueError('Recovery checkpoint hash mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    episodes=[]
    for reference in payload['replay_refs']:
        p=folder/reference['path']
        if digest(p)!=reference['sha256']:raise ValueError('Replay shard hash mismatch')
        episodes.append(torch.load(p,map_location='cpu',weights_only=True)['data'])
    payload['replay']=episodes
    learner.restore(payload,replay,rng)
    return payload


def run_job(root,plan,method,seed,pilot=False):
    group=method.replace('+','-')+f'-s{seed}'
    folder=root/('pilot' if pilot else 'formal')/group
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():return read(folder/'summary.json')
    fit=[e for e in plan['inputs'] if e['split']=='fit']
    selection=[e for e in plan['inputs'] if e['split']=='selection']
    labels=AuxiliaryLabels(fit)
    processes=[FloodProcess(e['path']) for e in fit]
    selection_processes=[FloodProcess(e['path']) for e in selection] if not pilot else []
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if plan['runtime']['device']=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    example=TeamEnv(processes[0],Config(**plan['config']),TeamConfig(**plan['team']));example.reset()
    learner=WorldLearner(2,len(central_state(example)),plan,plan['runtime']['device'],method,labels,seed)
    initial=folder/'initial-weights.pt'
    if not initial.exists():
        save(initial,dict(actor=learner.actor.state_dict(),mixer=learner.mixer.state_dict(),
                          world=learner.world.state_dict() if learner.world is not None else None))
        write(folder/'initial-weights.json',dict(seed=seed,method=method,path=initial.name,sha256=digest(initial)))
    elif digest(initial)!=read(folder/'initial-weights.json')['sha256']:
        raise ValueError('Initial weights changed')
    rng=np.random.default_rng(seed);replay=EpisodeReplay(plan['training']['replay_capacity_episodes'])
    history,validation,references,first=[],[],[],0
    if (folder/'latest.json').exists():
        payload=restore(folder,learner,replay,rng)
        first,history,validation=payload['episode'],payload['history'],payload['validation']
        references=payload['replay_refs']
    write(folder/'capacity.json',capacity(learner))
    episodes=plan['episodes_per_job']
    start=time.perf_counter()
    (folder/'episodes').mkdir(exist_ok=True)
    plan_identity=sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    def validate_at(episode):
        rows=[]
        for entry,process in zip(selection,selection_processes):
            metrics,_=actor_episode(learner.actor,process,plan)
            rows.append(dict(case=entry['id'],**metrics))
        val=dict(episode=episode,mean_return=float(np.mean([r['return_'] for r in rows])),rows=rows)
        validation.append(val)
        payload=checkpoint(folder,learner,replay,rng,episode,history,references,validation)
        milestone=folder/f'checkpoint-{episode:04d}.pt'
        if milestone.exists():raise ValueError('Preserve registered milestone rather than overwrite')
        save(milestone,payload)
        write(folder/'validation.json',validation)
    if not pilot and first and first%200==0 and not any(r['episode']==first for r in validation):validate_at(first)
    for episode in range(first,episodes):
        event_index=episode%len(fit)
        path=folder/'episodes'/f'episode-{episode+1:04d}.pt'
        predecessor=read(folder/'latest.json')['sha256'] if (folder/'latest.json').exists() else f'initial:{method}:{seed}:{plan_identity}'
        if path.exists():
            # If a process stops after rollout but before its optimizer/save,
            # reuse that immutable episode and restore its post-rollout RNG.
            shard=torch.load(path,map_location='cpu',weights_only=True)
            if (shard['case']!=fit[event_index]['id'] or shard['predecessor']!=predecessor
                or shard['plan_identity']!=plan_identity):raise ValueError('Pending episode provenance mismatch')
            data,metrics=shard['data'],shard['metrics'];rng.bit_generator.state=shard['post_rng']
            torch.set_rng_state(shard['torch_rng'])
            if torch.cuda.is_available():torch.cuda.set_rng_state_all(shard['cuda_rng'])
        else:
            epsilon=(1-.9*episode/11 if pilot else 1-.9*min(episode,1599)/1599)
            data,metrics,_=rollout(learner.actor,processes[event_index],plan,epsilon,rng,plan['runtime']['device'])
            data['case_index']=torch.full((61,),event_index,dtype=torch.long)
            save(path,dict(data=data,metrics=metrics,case=fit[event_index]['id'],predecessor=predecessor,plan_identity=plan_identity,post_rng=rng.bit_generator.state,
                           torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))
        replay.add(data)
        reference=dict(path=str(path.relative_to(folder)),sha256=digest(path))
        references.append(reference)
        update=learner.train_batch(replay.sample(plan['training']['batch_episodes'],rng,plan['runtime']['device'])) if len(replay.episodes)>=plan['training']['batch_episodes'] else None
        history.append(dict(episode=episode+1,event=fit[event_index]['id'],metrics=metrics,update=update))
        checkpoint(folder,learner,replay,rng,episode+1,history,references,validation)
        if not pilot and (episode+1)%200==0:validate_at(episode+1)
        if (episode+1)%10==0 or episode+1==episodes:
            write(folder/'progress.json',dict(episode=episode+1,updates=learner.updates,training_joint_steps=(episode+1)*60,
                                             segment_elapsed_s=time.perf_counter()-start))
            print(group,'episode',episode+1,'updates',learner.updates,flush=True)
    resource=memory_usage()
    if plan['runtime']['device']=='cuda':
        torch.cuda.synchronize();resource.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    segments=read(folder/'segments.json') if (folder/'segments.json').exists() else []
    segments.append(dict(first_episode=first,last_episode=episodes,elapsed_s=time.perf_counter()-start,memory=resource))
    write(folder/'segments.json',segments);write(folder/'history.json',history)
    selected=None
    if not pilot:
        best=max(validation,key=lambda r:(r['mean_return'],-r['episode']))
        p=folder/f"checkpoint-{best['episode']:04d}.pt"
        selected=dict(path=str(p.relative_to(root)),sha256=digest(p),episode=best['episode'],mean_validation_return=best['mean_return'],method=method,seed=seed)
    result=dict(status='completed',method=method,seed=seed,episodes=len(history),training_joint_steps=sum(r['metrics']['steps'] for r in history),
                updates=learner.updates,validation_opportunities=len(validation),selected=selected,capacity=capacity(learner),segments=segments,
                all_losses_finite=all(r['update'] is None or all(np.isfinite(v) for v in r['update'].values()) for r in history))
    write(folder/'summary.json',result)
    del learner,replay,labels,processes,selection_processes;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def common_plan(root,device,pilot):
    data_plan=read(root/'data-plan.json');design=data_plan['design']
    plan=dict(schema='hyra-sciml-executable-v1',design_sha256=data_plan['design_sha256'],sources=executable_sources(),
              baseline_git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent,text=True).strip(),
              runtime=runtime(device),inputs=read(root/'index.json'),config=dict(),team=asdict_team(),
              methods=design['engineering_pilot']['methods'] if pilot else design['methods'],
              seeds=[11] if pilot else design['seeds'],auxiliary=design['auxiliary'],
              training=dict(design['proposed_formal_budget']),episodes_per_job=12 if pilot else 2000,
              purpose='fit-only resource pilot' if pilot else 'one confirmatory four-group study')
    plan['config']=dict(speed_m_s=2.5,angular_speed_deg_s=9.,range_m=25.,fov_deg=90.,action_s=10.,sample_s=5.,duration_s=600.,history_length=4,age_weight=.5)
    if pilot:
        plan['training'].update(batch_episodes=8,replay_capacity_episodes=32)
    assert len(plan['inputs'])==12 and not any(e['split']=='holdout' for e in plan['inputs'])
    return plan


def asdict_team():
    from dataclasses import asdict
    return asdict(TeamConfig())


def pilot(root,device):
    audit=read(ROOT/'docs/evidence/20261025-sciml-development/data-audit.json')
    if audit['status']!='passed':raise ValueError('Auxiliary label audit required')
    path=root/'pilot-plan.json'
    if path.exists():plan=read(path);verify(plan)
    else:
        plan=common_plan(root,device,True);write(path,plan);archive_sources(root/'pilot-frozen-sources.zip',plan['sources'])
    results=[run_job(root,plan,method,11,True) for method in plan['methods']]
    passed=all(r['all_losses_finite'] and max(s['memory'].get('cuda_peak_reserved_bytes',0) for s in r['segments'])<6*2**30
               and max(s['memory']['process_peak_working_set_bytes'] for s in r['segments'])<12*2**30 for r in results)
    write(root/'pilot-summary.json',dict(status='completed',resource_gate_passed=passed,methods=results,
          limitation='Fit-only engineering pilot, no selection/holdout evaluation and no efficacy conclusion.'))


def freeze(root,device):
    if (root/'formal-plan.json').exists():raise ValueError('Formal protocol already frozen; do not overwrite')
    if not read(root/'pilot-summary.json')['resource_gate_passed']:raise ValueError('Resource gate not met')
    if shutil.disk_usage(root).free<80*2**30:raise ValueError('Formal source/checkpoint archive needs at least80GiB available')
    plan=common_plan(root,device,False)
    plan['recovery_policy']='immutable per-episode replay shards and lightweight recovery versions; selected milestones immutable; no pilot warm start'
    plan['resource_evidence']=dict(pilot_summary_sha256=digest(root/'pilot-summary.json'),
        checkpoint_engineering_amendment='After the completed pilot, recovery pointer commits and pending-episode provenance were hardened. Actor, mixer, auxiliary, optimizer, losses and budget unchanged; synthetic full-resume tests passed.',
        available_disk_bytes=shutil.disk_usage(root).free,estimated_archive_disk_bytes=55*2**30)
    write(root/'formal-plan.json',plan);archive_sources(root/'formal-frozen-sources.zip',plan['sources'])
    ledger=read(ROOT/'research/sciml_rounds.json');ledger['rounds'][0].update(state='formal_protocol_frozen',formal_plan_sha256=digest(root/'formal-plan.json'))
    write(ROOT/'research/sciml_rounds.json',ledger)


def train(root,device):
    plan=read(root/'formal-plan.json');verify(plan)
    if plan['runtime']['device']!=device:raise ValueError('Frozen device mismatch')
    results=[]
    for method in plan['methods']:
        for seed in plan['seeds']:
            write(root/'training-state.json',dict(stage='training',method=method,seed=seed,pid=os.getpid()))
            results.append(run_job(root,plan,method,seed))
    selected=[r['selected'] for r in results]
    assert len(selected)==12 and all(r['training_joint_steps']==120000 and r['validation_opportunities']==10 for r in results)
    write(root/'training-summary.json',dict(status='completed',jobs=results))
    write(root/'model-freeze.json',dict(formal_plan_sha256=digest(root/'formal-plan.json'),sources=plan['sources'],selected_models=selected,
                                      training_input_sha256=digest(root/'index.json')))
    write(root/'training-state.json',dict(stage='models_frozen',pid=os.getpid()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['pilot','freeze','train'])
    parser.add_argument('output',type=Path);parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=parser.parse_args();root=args.output.resolve()
    lock=root/'execution.lock';token=uuid.uuid4().hex
    with lock.open('x',encoding='utf-8') as file:json.dump(dict(pid=os.getpid(),token=token,stage=args.stage),file)
    try:
        if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        dict(pilot=pilot,freeze=freeze,train=train)[args.stage](root,args.device)
    finally:
        if lock.exists() and read(lock).get('token')==token:lock.unlink()


if __name__=='__main__':main()
