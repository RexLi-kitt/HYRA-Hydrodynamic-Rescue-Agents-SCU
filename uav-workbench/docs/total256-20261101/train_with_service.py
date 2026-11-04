"""Six-method continuation with causal singleton service masks and resume state.

Total episodes include the original 64. Coverage fallback is evaluation only,
so extra-learning effects remain separately measurable from a rule takeover.
"""
import argparse,gc,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from flood1km.env import Env
from flood1km.learners import Learner,ALGORITHMS
from flood1km.train import aggregate,seed_all,rng_state,restore_rng,sources
from flood1km.data import sha
from service_commitment import ServiceCommitment
from coverage_controller import CoverageWatchdog,service_mask

def write(path,obj):
    temp=path.with_suffix('.writing');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(temp,path)

def decision(model,c,obs,boundary,deterministic=False):
    forced,details=c.select(obs,boundary,np.full(4,8,dtype=np.int64))
    mask=service_mask(boundary,forced,details['phases'])
    action,logp,value=model.act(obs,mask,deterministic)
    controlled=np.array([x!='learned' for x in details['phases']])
    assert np.array_equal(action[controlled],forced[controlled])
    assert np.all(np.abs(logp[controlled])<1e-6)
    return action,mask,logp,value,details,int(controlled.sum())

def rollout(model,water,layout,consume):
    e=Env(water,layout);obs=e.observe();c=ServiceCommitment('commit');done=False;previous=None;ret=0.;forced_count=0
    while not done:
        action,mask,logs,value,_,count=decision(model,c,obs,e.available());forced_count+=count
        if previous is not None:
            previous['next_mask']=mask.copy();previous['next_action']=action.copy();consume(previous)
        nxt,reward,done,_=e.step(action);ret+=reward
        previous=dict(obs=obs.copy(),mask=mask.copy(),action=action.copy(),reward=reward,done=float(done),next_obs=nxt.copy(),
            next_mask=e.available().copy(),next_action=np.full(4,8,dtype=np.int64),logp=logs.copy(),value=value)
        obs=nxt
    consume(previous)
    return dict(return_sum=ret,forced_agent_steps=forced_count,**e.metrics())

def validate(model,cases,out,label,planner):
    saved=rng_state();rows=[];events=[]
    try:
        for name,water in cases:
            for layout in [9001,9002,9003,9004]:
                e=Env(water,layout);obs=e.observe();done=False;c=ServiceCommitment('commit');p=CoverageWatchdog();forced=0
                with (out/(label+'-trace.jsonl')).open('a',encoding='utf-8') as f:
                    while not done:
                        actions,mask,logs,_,detail,count=decision(model,c,obs,e.available(),True);forced+=count;planning={}
                        if planner:
                            actions,planning=p.select(obs,e.available(),actions,detail['phases'])
                            assert not(p.known & ~e.known).any()
                        obs,_,done,info=e.step(actions)
                        info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()}
                        f.write(json.dumps(dict(case=name,layout=layout,step=e.t,executed=actions.tolist(),service=detail,planning=planning,info=info))+'\n')
                rows.append(dict(case=name,layout=layout,forced_agent_steps=forced,planner_agent_steps=p.steps,**e.metrics()))
                events.extend(dict(case=name,layout=layout,**x) for x in e.events)
        result=dict(episodes=rows,events=events,summary=aggregate(rows),planner=planner)
        write(out/(label+'.json'),result);return result['summary']
    finally:restore_rng(saved)

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=args.resume);begin=time.perf_counter()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(1001)
    manifest=args.baseline/'data/dataset.json';items=json.loads(manifest.read_text(encoding='utf-8'))['cases'];cases={}
    for x in items:
        p=Path(x['path']);assert sha(p)==x['sha256'];cases.setdefault(x['split'],[]).append((p.name,np.load(p,mmap_mode='r')))
    original=args.baseline/args.algorithm/'latest.pt';base=torch.load(original,map_location='cpu',weights_only=False)
    for k,h in base['identity']['source_hashes'].items():assert sources()[k]==h
    identity=dict(algorithm=args.algorithm,base_checkpoint_hash=sha(original),data_hash=sha(manifest),seed=1001,
        source_hashes=sources(),new_code_hashes={name:sha(Path(__file__).with_name(name)) for name in ['train_with_service.py','service_commitment.py','coverage_controller.py']},
        semantics='64 original + cumulative continuation; actual actions sampled from service singleton masks; no coverage planner while training',
        replay_policy='fresh replay at original64 -> service-phase transition; save and restore replay thereafter',
        layout='920000+(total_episode-65)',validation='2 parents x 4 layouts, no final test',future_target=1024)
    model=Learner(args.algorithm,base['learner']['dim']);model.restore(base['learner'])
    episode=64;steps=0;replay=[];pending=[];updates=0;last_loss={};evaluations={}
    def save():
        assert not pending
        temp=out/'latest.partial'
        torch.save(dict(learner=model.checkpoint(),total_episode=episode,phase_steps=steps,replay=replay,rng=rng_state(),identity=identity,evaluations=evaluations),temp)
        os.replace(temp,out/'latest.pt')
    def eval_pair(total):
        pair={}
        for planner in [False,True]:
            mode='watchdog' if planner else 'service'
            pair[mode]=validate(model,cases['validation'],out,f'validation-{total:04d}-{mode}',planner)
        print(args.algorithm,'validation',total,json.dumps(pair),flush=True);return pair
    if args.resume:
        ck=torch.load(out/'latest.pt',map_location='cpu',weights_only=False)
        if ck['identity']!=identity:raise ValueError('Resume identity changed')
        model.restore(ck['learner']);episode=ck['total_episode'];steps=ck['phase_steps'];replay=ck['replay'];evaluations=ck['evaluations'];restore_rng(ck['rng'])
        if episode>=args.total:raise ValueError('Already at requested training total')
    else:
        write(out/'identity.json',identity);evaluations['64']=eval_pair(64);save()
    del base;gc.collect()
    def consume(row):
        nonlocal steps,last_loss
        assert np.asarray(row['mask'])[np.arange(4),row['action']].all()
        assert np.asarray(row['next_mask'])[np.arange(4),row['next_action']].all()
        steps+=1
        if args.algorithm in ('hasac','maac'):
            replay.append(row)
            if len(replay)>4096:replay.pop(0)
            if len(replay)>=1024 and steps%32==0:
                for _ in range(4):last_loss=model.update_off_policy(random.sample(replay,64))
        else:pending.append(row)
    while episode<args.total:
        if time.perf_counter()-begin>args.max_seconds and not pending:
            save();write(out/'status.json',dict(status='budget_stopped',total_episode=episode));return 3
        write(out/'status.json',dict(status='training',total_episode=episode,target=args.total,elapsed_s=time.perf_counter()-begin))
        name,water=cases['train'][(episode-64)%len(cases['train'])]
        record=rollout(model,water,920000+episode-64,consume);episode+=1
        if args.algorithm not in ('hasac','maac') and (episode%2==0 or episode==args.total):
            last_loss=model.update_on_policy(pending);pending=[]
        with (out/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(total_episode=episode,case=name,layout=920000+episode-65,phase_steps=steps,loss=last_loss,elapsed_s=time.perf_counter()-begin,**record))+'\n')
        if episode%32==0 or episode==args.total:
            save();print(args.algorithm,'trained',episode,'elapsed',round(time.perf_counter()-begin,1),flush=True)
            if (out.parent/'STOP_AFTER_CHECKPOINT').exists():
                write(out/'status.json',dict(status='stopped_at_checkpoint',total_episode=episode));return 4
    evaluations[str(episode)]=eval_pair(episode);save()
    assert sha(original)==identity['base_checkpoint_hash']
    write(out/'status.json',dict(status='completed',total_episode=episode,phase_steps=steps,elapsed_s=time.perf_counter()-begin,
        evaluations=evaluations,checkpoint_hash=sha(out/'latest.pt'),formal_test_executed=False,resumable_to=1024))
    return 0
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=ALGORITHMS,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--total',type=int,default=256);p.add_argument('--max-seconds',type=float,default=1800);p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.total<=64 or args.total%2:p.error('Use an even total > 64')
    try:raise SystemExit(run(args))
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
