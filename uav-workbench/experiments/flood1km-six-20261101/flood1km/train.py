"""Explicitly invoked pilot training only. Importing this module starts no work."""
import argparse,hashlib,json,os,random,time
from pathlib import Path
import numpy as np
import torch
from .env import Env,rule_action
from .learners import Learner,ALGORITHMS
from .data import sha

def sources():
    return {p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda']:torch.cuda.set_rng_state_all(s['cuda'])

def aggregate(rows):
    result={k:float(np.mean([r[k] for r in rows])) for k in ('ST','ACC','RCR','coverage','rescued')}
    for k in ('FRT','CRT'):
        values=[r[k] for r in rows if r[k] is not None]
        result[k]=dict(mean=float(np.mean(values)) if values else None,valid=len(values),total=len(rows))
    result['all_rescued_fraction']=float(np.mean([r['rescued']==20 for r in rows]))
    result['timeout_fraction']=float(np.mean([r['timed_out'] for r in rows]))
    return result

def evaluate(learner,cases,layouts=(9001,9002,9003,9004)):
    saved=rng_state();rows=[]
    try:
        for name,water in cases:
            for seed in layouts:
                env=Env(water,seed);obs=env.observe();done=False
                while not done:
                    action=rule_action(env) if learner is None else learner.act(obs,env.available(),True)[0]
                    obs,_,done,_=env.step(action)
                rows.append(dict(case=name,layout=seed,**env.metrics()))
    finally:restore_rng(saved)
    return dict(episodes=rows,summary=aggregate(rows),scope='validation only; not the sealed 50-scene final test')

def save_checkpoint(path,learner,episode,replay,steps,identity):
    tmp=path.with_suffix('.partial')
    torch.save(dict(learner=learner.checkpoint(),episode=episode,replay=replay,steps=steps,
                    rng=rng_state(),identity=identity),tmp)
    os.replace(tmp,path)

def run(args):
    args.output.mkdir(parents=True,exist_ok=args.resume)
    manifest=json.loads((args.data/'dataset.json').read_text(encoding='utf-8'))
    cases={}
    for item in manifest['cases']:
        p=Path(item['path'])
        if sha(p)!=item['sha256']:raise ValueError('Hydraulic file hash mismatch')
        water=np.load(p,mmap_mode='r',allow_pickle=False)
        if water.shape!=(301,3,100,100):raise ValueError('Wrong hydraulic grid or time support')
        cases.setdefault(item['split'],[]).append((p.name,water))
    identity=dict(algorithm=args.algorithm,seed=args.seed,episodes=args.episodes,source_hashes=sources(),
                  data_hash=sha(args.data/'dataset.json'),device=args.device,
                  protocol_hash=sha(Path(__file__).parents[1]/'protocol.json'))
    protocol=json.loads((Path(__file__).parents[1]/'protocol.json').read_text(encoding='utf-8'))
    if not args.execute:raise RuntimeError('Execution flag required; preparation alone must not launch training')
    torch.set_num_threads(2);seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently change hardware')
    first=Env(cases['train'][0][1],10001);learner=Learner(args.algorithm,first.observe().shape[-1],args.device)
    replay=[];steps=0;begin=0;checkpoint=args.output/'latest.pt';pending=[]
    if args.resume:
        # Only load checkpoints created by this local experiment, never arbitrary downloads.
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Code/config/data identity differs; no silent resume')
        learner.restore(saved['learner']);begin=saved['episode'];steps=saved['steps'];replay=saved['replay'];restore_rng(saved['rng'])
        if begin>=args.episodes:raise ValueError('This pilot is already complete; do not repeat it')
    else:
        (args.output/'identity.json').write_text(json.dumps(dict(**identity,torch=torch.__version__,parameters=sum(p.numel() for p in learner.parameters() if p.requires_grad)),indent=2),encoding='utf-8')
        (args.output/'initial-validation.json').write_text(json.dumps(evaluate(learner,cases['validation']),indent=2),encoding='utf-8')
    start=time.perf_counter();budget_stop=False
    for episode in range(begin,args.episodes):
        name,water=cases['train'][episode%len(cases['train'])]
        env=Env(water,10000+args.seed*1000+episode);obs=env.observe();done=False;returns=0.;trajectory=[];losses=[]
        while not done:
            mask=env.available();action,logs,value=learner.act(obs,mask)
            nxt,reward,done,info=env.step(action);steps+=1;returns+=reward
            row=dict(obs=obs.copy(),mask=mask.copy(),action=action.copy(),reward=reward,done=float(done),
                     next_obs=nxt.copy(),next_mask=env.available().copy(),logp=logs.copy(),value=value)
            trajectory.append(row)
            if args.algorithm in ('hasac','maac'):
                replay.append(row)
                if len(replay)>4096:replay.pop(0)
                if len(replay)>=1024 and steps%32==0:
                    for _ in range(4):losses.append(learner.update_off_policy(random.sample(replay,64)))
            obs=nxt
        if args.algorithm not in ('hasac','maac'):
            for t,row in enumerate(trajectory):row['next_action']=trajectory[t+1]['action'] if t+1<len(trajectory) else np.full(4,8,dtype=np.int64)
            pending.extend(trajectory)
            if (episode+1)%2==0 or episode+1==args.episodes:
                losses.append(learner.update_on_policy(pending));pending=[]
        record=dict(episode=episode+1,case=name,layout=10000+args.seed*1000+episode,steps=steps,
                    updates=learner.updates,return_sum=returns,elapsed_s=time.perf_counter()-start,
                    diagnostics=losses[-1] if losses else {},**env.metrics())
        with (args.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        with (args.output/'rescue-events.jsonl').open('a',encoding='utf-8') as f:
            for event in env.events:f.write(json.dumps(dict(episode=episode+1,**event))+'\n')
        print(json.dumps(record),flush=True)
        # Checkpoint at complete on-policy batches, so no partially consumed rollout is lost.
        if not pending and ((episode+1)%4==0 or episode+1==args.episodes or time.perf_counter()-start>args.max_seconds):
            save_checkpoint(checkpoint,learner,episode+1,replay,steps,identity)
        if time.perf_counter()-start>args.max_seconds and not pending:budget_stop=True;break
    final=evaluate(learner,cases['validation'])
    (args.output/'final-validation.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    if not checkpoint.exists():save_checkpoint(checkpoint,learner,episode+1,replay,steps,identity)
    status=dict(status='budget_stopped' if budget_stop else 'pilot_completed',episodes=episode+1,steps=steps,
                updates=learner.updates,checkpoint=str(checkpoint),sha256=sha(checkpoint),elapsed_s=time.perf_counter()-start,
                final_validation=final['summary'],formal_test_executed=False)
    (args.output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    if budget_stop:raise SystemExit(3)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=ALGORITHMS,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=701);p.add_argument('--episodes',type=int,default=64)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--max-seconds',type=float,default=1200)
    p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:p.error('Prepared only. Explicit --execute is required to run.')
    run(args)
