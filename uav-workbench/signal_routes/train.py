"""Matched B3 learners with frozen external signal inference and bounded recovery."""
import argparse
import time
from pathlib import Path
from dataclasses import replace, asdict
import numpy as np
import torch
from scale100_train import rollout, batch_episodes, config
from scale100_aux_resilient import ResilientLearner, fit_prior
from . import SCHEMA, METHODS
from .data import cases
from .belief import Parameters
from .environment import RouteEnv
from .common import read, write, digest, sources


def train(study, method, seed, episodes, phase='pilot', stop_after=None, resume=False):
    study=Path(study); dataset=study/'data/dataset.json'; params_path=study/'parameters.json'
    params=read(params_path)['parameters'][method]; parameters=Parameters(**params)
    duration=read(study/'task-selection.json')['duration_s']
    entries=list(cases(dataset,'fit')); hydros=[case for _,case in entries]
    output=study/phase/f'{method}-s{seed}'
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    cfg=replace(config('E2','B3',seed),duration_s=float(duration))
    env=RouteEnv(hydros[0],cfg,method,parameters); env.reset()
    learner=ResilientLearner(len(env.training_state()),'B3',2,fit_prior(hydros))
    learner.data_weight=0.; learner.physics_weight=0.
    spec=dict(schema=SCHEMA,observation_schema='21-channel-signal-posterior-v1',method=method,
        seed=seed,phase=phase,episodes=episodes,config=asdict(cfg),sources=sources(),
        parameters_sha256=digest(params_path),parameters=params,dataset_sha256=digest(dataset),
        task_selection_sha256=digest(study/'task-selection.json'),auxiliary_losses=False,
        optimizer='legacy resilient B3 TD-only',surprise_training_only=True)
    history=[]; replay=[]; first=0
    if resume:
        if read(output/'plan.json')!=spec: raise ValueError('Training contract changed; cannot resume')
        pointer=read(output/'latest.json'); path=output/pointer['path']
        if digest(path)!=pointer['sha256']: raise ValueError('Changed recovery checkpoint')
        state=torch.load(path,weights_only=False,map_location='cpu')
        for name in ('actor','mixer','target_actor','target_mixer'): getattr(learner,name).load_state_dict(state[name])
        learner.optimizer.load_state_dict(state['optimizer']); learner.updates=state['updates']
        replay=state['replay']; history=state['history']; first=state['episode']
        rng.bit_generator.state=state['numpy_rng']; torch.set_rng_state(state['torch_rng'])
    else:
        output.mkdir(parents=True,exist_ok=False); write(output/'plan.json',spec)
        torch.save(learner.actor.state_dict(),output/'initial-actor.pt')
    end=min(episodes,stop_after or episodes)
    if end<first: raise ValueError('Cannot rewind run')
    if end==first: return history
    start=time.perf_counter()
    for ep in range(first,end):
        idx=ep%len(hydros); cfg=replace(cfg,seed=seed+1009*ep)
        env=RouteEnv(hydros[idx],cfg,method,parameters)
        epsilon=1-.95*min(ep/max(1,int(episodes*.8)),1)
        total,data,metrics=rollout(learner.actor,env,rng,epsilon,True)
        replay.append((idx,data)); replay=replay[-32:]; update=None
        if len(replay)>=4:
            selected=[replay[j] for j in rng.choice(len(replay),4,replace=False)]
            update=learner.train(batch_episodes([d for _,d in selected]),[hydros[i] for i,_ in selected])
        # Timing is deliberately not part of deterministic history/recovery equality.
        timing={k:metrics.pop(k) for k in ('inference_max_s',)}
        history.append(dict(episode=ep+1,return_=total,metrics=metrics,epsilon=epsilon,
                            case_index=idx,scenario_seed=cfg.seed,update=update))
        progress=dict(status='running',episode=ep+1,total_episodes=episodes,updates=learner.updates,
                      elapsed_segment_s=time.perf_counter()-start,last=history[-1],timing=timing)
        write(output/'progress.json',progress)
        if (ep+1)%16==0 or ep+1==end:
            name=f'recovery-{ep+1:06d}.pt'; temp=output/(name+'.tmp')
            state={key:getattr(learner,key).state_dict() for key in ('actor','mixer','target_actor','target_mixer')}
            state.update(optimizer=learner.optimizer.state_dict(),updates=learner.updates,replay=replay,
                         history=history,episode=ep+1,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            torch.save(state,temp); temp.replace(output/name)
            write(output/'latest.json',dict(path=name,sha256=digest(output/name),episode=ep+1))
            write(output/'history.json',history)
            # Known study-owned rotating recovery files, no recursive cleanup.
            for old in sorted(output.glob('recovery-??????.pt'))[:-2]: old.unlink()
            print(f'{phase} {method} s{seed} ep {ep+1}/{episodes} ({time.perf_counter()-start:.1f}s)',flush=True)
        if ep+1 in (256,512,1024):
            actor=output/f'actor-{ep+1:04d}.pt'; torch.save(learner.actor.state_dict(),actor)
            write(actor.with_suffix('.json'),dict(**spec,trained_episodes=ep+1,updates=learner.updates,actor_sha256=digest(actor)))
    if end==episodes:
        torch.save(learner.actor.state_dict(),output/'actor.pt')
        write(output/'actor.json',dict(**spec,trained_episodes=end,updates=learner.updates,actor_sha256=digest(output/'actor.pt')))
        progress['status']='completed'; write(output/'progress.json',progress)
    return history


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--study',type=Path,required=True)
    p.add_argument('--method',choices=METHODS,required=True); p.add_argument('--seed',type=int,default=11)
    p.add_argument('--episodes',type=int,default=128); p.add_argument('--phase',default='pilot')
    p.add_argument('--stop-after',type=int); p.add_argument('--resume',action='store_true')
    a=p.parse_args(); train(a.study,a.method,a.seed,a.episodes,a.phase,a.stop_after,a.resume)
