"""Three task-decision learners, identical expert warmstart, then 64 RL episodes.
All difficulty levels remain visible. No environment tuning on learner-planner gap.
"""
from pathlib import Path
from dataclasses import replace
import argparse,json,os,random,sys,time,traceback
import numpy as np
import torch
from dispatch_env import DispatchEnv,Env,Config,anonymize_roster
from pilot import read,write,sha,seed,Learner,summary,quality
from environment import make_scenario
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-pressure-v1'))
from pressure import LEVELS,variant,paired_summary
from flood1km.learners import distribution

HERE=Path(__file__).resolve().parent
METHODS=('maac','mat','happo')

def episode(model,case,kind='dispatch',collect=False):
    water=np.load(case['field']['path'],mmap_mode='r',allow_pickle=False)
    env=(Env if kind=='original' else DispatchEnv)(water,case['scenario'])
    obs=env.observe();done=False;records=[];total=0.;free=forced=0
    def choices():return env.control()[:2] if kind=='original' else env.choices()[:2]
    mask,preferred=choices()
    while not done:
        if model is None:actions=preferred;logs=np.zeros(4);value=0.
        else:actions,logs,value=model.act(obs,mask,not collect)
        free+=int((mask.sum(1)>1).sum());forced+=int((mask.sum(1)==1).sum())
        nxt,reward,done,_=env.step(actions);nm,npref=choices();total+=reward
        if collect:records.append(dict(obs=obs,mask=mask,action=actions,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;preferred=npref
    for i,r in enumerate(records):r['next_action']=records[i+1]['action'] if i+1<len(records) else np.full(4,8,dtype=np.int64)
    metric=dict(**env.metrics(),case=case['id'],parent=case['field']['parent'],level=case['level'],
                return_sum=total,neural_choice_fraction=free/max(1,free+forced))
    metric['expired_seen_at_deadline']=metric['expired']-metric['expired_unseen']
    assert metric['supplied']+metric['expired']+metric['outside']==20
    if not collect:
        metric['persons']=[dict(id=j,fall=float(env.fall[j]),deadline=float(env.deadline[j]),
            first_seen=float(env.first_seen[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(20)]
    return metric,records

def evaluate(model,cases,kind='dispatch'):
    before=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        rows=[episode(model,c,kind)[0] for c in cases]
        return dict(summary=summary(rows),rows=rows,
            by_level={name:summary([r for r in rows if r['level']==name]) for name,_,_ in LEVELS},
            scope='development; same6 hydraulic parents; not an independent final test')
    finally:random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])

def bc_loss(model,batch):
    obs=torch.as_tensor(np.stack([r['obs'] for r in batch]),dtype=torch.float32)
    mask=torch.as_tensor(np.stack([r['mask'] for r in batch]),dtype=torch.bool)
    action=torch.as_tensor(np.stack([r['action'] for r in batch]),dtype=torch.long)
    logits=model.policy(obs,action)[0] if model.algorithm=='mat' else torch.stack([a(obs[:,i]) for i,a in enumerate(model.actors)],1)
    return -distribution(logits,mask).log_prob(action).mean()

def warmstart(model,rows,updates=128):
    params=list(model.policy.parameters()) if model.algorithm=='mat' else list(model.actors.parameters())
    optimizer=torch.optim.Adam(params,lr=3e-4);losses=[]
    for _ in range(updates):
        loss=bc_loss(model,random.sample(rows,min(64,len(rows))))
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
    return dict(updates=updates,final_loss=losses[-1],losses=losses,scope='TRAIN-only expert demonstrations; separate from64 RL episodes')

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw))
    def guard():
        if (out/'STOP_AFTER_EPISODE').exists():raise InterruptedError('User stop marker')
        if time.monotonic()-start>3600:raise TimeoutError('One-hour pilot cap; no restart')
    try:
        status('preparing');original=read(args.pilot/'protocol.json')
        for p,h in original['source_hashes'].items():assert sha(p)==h
        old=read(args.pilot/'scenarios.json');fields=[];seen=set()
        for c in old['train']:
            if c['field']['parent'] not in seen:fields.append(c['field']);seen.add(c['field']['parent'])
        assert len(fields)==16
        pressure=read(args.pressure/'scenarios.json')
        assert read(args.pressure/'status.json')['status']=='completed'
        groups={'train':[],'development':[]}
        for i in range(64):
            f=fields[i%16];water=np.load(f['path'],mmap_mode='r',allow_pickle=False)
            s=anonymize_roster(make_scenario(water,1011000+i,Config()),1021000+i)
            name,w,r=LEVELS[i%len(LEVELS)]
            c=variant(dict(id=f'dispatch-train-{i:03}',field=f,scenario=s),w,r);c['level']=name;groups['train'].append(c)
        # First12 seed IDs are registered before this pilot, never chosen by outcomes.
        for name,_,_ in LEVELS:
            for i,c0 in enumerate(pressure[name][:12]):
                import copy
                c=copy.deepcopy(c0);c['scenario']=anonymize_roster(c['scenario'],1031000+i);c['level']=name
                groups['development'].append(c)
        assert not {c['field']['parent'] for c in groups['train']} & {c['field']['parent'] for c in groups['development']}
        allfields={c['field']['path']:c['field']['sha256'] for cs in groups.values() for c in cs}
        for p,h in allfields.items():assert sha(p)==h
        hashes=dict(original['source_hashes'])
        hashes.update({str(p):sha(p) for p in HERE.glob('*.py')})
        hashes.update({str(p):sha(p) for p in (HERE.parent/'flood-drop-pressure-v1').glob('*.py')})
        protocol=dict(version='dispatch-warmstart-64-01',date='2026-11-02',methods=list(METHODS),seed=8301,
            targets_percentage_points=dict(minimum=15,satisfactory=25,aspirational=50),absolute_scene_target=[.90,.94],
            levels=[dict(name=n,window=w,sensor=r) for n,w,r in LEVELS],keep_all_levels=True,
            training_episodes_per_model=64,demonstration_episodes=16,imitation_updates_per_model=128,
            initialization='fresh task-decision networks, TRAIN-only shared coordinated-planner imitation, then RL; episode0 is after imitation, not untrained',
            baselines=['unchanged original V2 planner','observation-only coordinated task planner; same candidates/executor as learners'],
            metric='20/20 delivered before deadlines; mean of five equally weighted levels; difference to stronger full-suite planning baseline',
            decision_space='four observed-person candidates, four risk/staleness patrol regions, hold/service; learning now controls task choice',
            source_hashes=hashes,water_hashes=allfields,formal_test=False,
            observations='Same current shared measurements, discovered people, first-seen age lower bound, no actual fall times/deadlines/future water',
            roster_identity='Independent random IDs replace chronological event-order IDs for every method, to remove an unintended timing hint',
            scope='One training seed; 12 event layouts on6 development parents, repeated across5 conditions; no independent15-point claim',
            preregistered_nonadoption='No adoption from more training alone: compare selected checkpoint and64 with both planning baselines. Preserve regressions and keep V2 intact.')
        write(out/'protocol.json',protocol);write(out/'scenarios.json',groups)
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed(8301)
        baselines={}
        for kind in ('original','dispatch'):
            guard();status('planning_baseline',kind=kind)
            result=evaluate(None,groups['development'],kind);baselines[kind]=result
            write(out/f'planner-{kind}.json',result)
            print(json.dumps(dict(planner=kind,summary=result['summary'])),flush=True)
        teacher=[];teacher_metrics=[];status('demonstrations')
        for c in groups['train'][:16]:
            guard();m,rows=episode(None,c,collect=True);teacher.extend(rows);teacher_metrics.append(m)
        # Demonstration tensors are recoverable and hashed; not a paper-score claim.
        torch.save(teacher,out/'training-demonstrations.pt')
        write(out/'demonstrations.json',dict(episodes=16,transitions=len(teacher),rows=teacher_metrics,sha256=sha(out/'training-demonstrations.pt')))
        leaders={}
        for method in METHODS:
            guard();seed(8301);folder=out/method;folder.mkdir();model=Learner(method,760)
            status('imitation',method=method,updates=128)
            write(folder/'imitation.json',warmstart(model,teacher));replay=[];best=None
            for ep in range(65):
                guard()
                if ep:
                    model.train();metric,rows=episode(model,groups['train'][ep-1],collect=True)
                    if method=='maac':
                        replay.extend(rows);del replay[:-2048]
                        for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
                    else:loss=model.update_on_policy(rows)
                    metric.update(episode=ep,updates=model.updates,diagnostics=loss)
                    with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
                    status('training',method=method,episode=ep,completed_methods=list(leaders))
                if ep in (0,32,64):
                    status('development_evaluation',method=method,episode=ep,completed_methods=list(leaders))
                    model.eval();result=evaluate(model,groups['development']);write(folder/f'development-{ep:04}.json',result)
                    path=folder/f'checkpoint-{ep:04}.pt'
                    torch.save(dict(learner=model.checkpoint(),episode=ep,warmstart_updates=128,
                        protocol_sha256=sha(out/'protocol.json'),replay=replay,
                        rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),path)
                    if best is None or quality(result)>tuple(best['quality']):
                        best=dict(episode=ep,quality=quality(result),summary=result['summary'],path=str(path),sha256=sha(path),warmstart_updates=128)
                        write(folder/'best.json',best)
                    print(json.dumps(dict(method=method,episode=ep,summary=result['summary'])),flush=True)
            leaders[method]=best;write(out/'leaders.json',leaders)
        for p,h in hashes.items():assert sha(p)==h
        for p,h in allfields.items():assert sha(p)==h
        reference=max(baselines,key=lambda k:baselines[k]['summary']['scene_success'])
        paired=[dict(r,method='planner') for r in baselines[reference]['rows']]
        for method,b in leaders.items():
            paired.extend(dict(r,method=method) for r in read(out/method/f'development-{b["episode"]:04}.json')['rows'])
        result=dict(status='completed',training_episodes=192,imitation_updates=384,demonstration_episodes=16,
            development_episodes=660,stronger_planning_baseline=reference,
            baselines={k:v['summary'] for k,v in baselines.items()},learners=leaders,
            paired_differences={m:paired_summary(paired,m) for m in METHODS},
            frozen_sources_unchanged=True,formal_claim=False)
        write(out/'summary.json',result);status('completed',training_episodes=192,development_episodes=660,imitation_updates=384)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',type=Path,required=True);p.add_argument('--pressure',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('Explicit --execute required')
    run(a)
