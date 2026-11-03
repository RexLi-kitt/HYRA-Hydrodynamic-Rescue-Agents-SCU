"""Frozen V2 pilot: six methods x64, shared development-only cases, no test use."""
import argparse,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch
from environment import Config,Env,make_scenario

HERE=Path(__file__).resolve().parent
LEGACY=HERE.parent/'flood1km-six-20261101'
sys.path.insert(0,str(LEGACY))
from flood1km.learners import Learner,ALGORITHMS

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n');os.replace(tmp,p)

def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)

def summary(rows):
    return dict(n=len(rows),successes=sum(r['scene_success'] for r in rows),
                **{key:float(np.mean([r[key] for r in rows])) for key in ['scene_success','supply_fraction','supplied','expired','expired_unseen','outside','failed_after_discovery','conflicts','max_drift_m']})

def rollout(model,case,collect=False,deterministic=True):
    water=np.load(case['field']['path'],mmap_mode='r',allow_pickle=False)
    env=Env(water,case['scenario']);obs=env.observe();rows=[];done=False;total=0.;free=forced=0
    mask,preferred,_=env.control()
    while not done:
        if model is None:action=preferred;logs=np.zeros(4);value=0.
        else:action,logs,value=model.act(obs,mask,deterministic)
        free+=int((mask.sum(1)>1).sum());forced+=int((mask.sum(1)==1).sum())
        nxt,reward,done,_=env.step(action);total+=reward;nm,npref,_=env.control()
        if collect:rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;preferred=npref
    for t,row in enumerate(rows):row['next_action']=rows[t+1]['action'] if t+1<len(rows) else np.full(4,8,dtype=np.int64)
    return dict(**env.metrics(),case=case['id'],return_sum=total,neural_choice_fraction=free/max(1,free+forced)),rows

def evaluate(model,cases):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        rows=[rollout(model,c)[0] for c in cases]
        return dict(rows=rows,summary=summary(rows),scope='development only; single training seed')
    finally:random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])

def quality(result):
    s=result['summary'];return (s['scene_success'],s['supply_fraction'],-s['expired_unseen'],-s['conflicts'])

def prepare(dataset,out):
    cfg=Config();manifest=read(dataset);train=[c for c in manifest['cases'] if c['split']=='train'];dev=[c for c in manifest['cases'] if c['split']=='validation']
    if len(train)!=16 or len(dev)!=6:raise ValueError('Unexpected development split')
    for c in train+dev:
        if sha(c['path'])!=c['sha256']:raise ValueError('Hydraulic hash mismatch')
        if not c['finite_nonnegative']:raise ValueError('Invalid archived hydraulics')
        if abs(c['mass_residual_m3'])>max(1e-6,abs(c['source_volume_m3'])*1e-9):raise ValueError('Water conservation audit failed')
    groups={};start=time.monotonic()
    for split,number,fields,offset in [('train',64,train,812000),('development',12,dev,923000)]:
        group=[]
        for i in range(number):
            field=fields[i%len(fields)];water=np.load(field['path'],mmap_mode='r',allow_pickle=False)
            scenario=make_scenario(water,offset+i,cfg)
            group.append(dict(id=f'{split}-{i:03}',field=field,scenario=scenario))
        groups[split]=group
    hashes={str(p):sha(p) for p in [*HERE.glob('*.py'),LEGACY/'flood1km/learners.py']}
    protocol=dict(version='dynamic-drop-v2-pilot-01',config=asdict(cfg),algorithms=list(ALGORITHMS),episodes=64,seed=8701,
                  hydraulic_dataset_sha256=sha(dataset),source_hashes=hashes,formal_test_used=False,
                  primary='scene_success: all20 supplied within individual deadlines',target_range=[.90,.94],
                  initialization='fresh V2 learners; no inheritance of V1 scores or weights',
                  controller='shared observation-only allocator; up to3 forward actions for learner, singleton tracking during drop; pure planner control retained',
                  sensor='circular100m, instantaneous team sharing; deliberately different from V1 half-sector',
                  obstacles='synthetic permeable trapping patches for people; do not modify hydraulic field or UAV navigation',
                  adjustment_rule='After all6x64: if best development scene success<.5, compare window480 once; if >.9 compare window300 once; otherwise retain360. Regenerate valid exposure events with SAME development parent fields and seed IDs, since cutoff T-L changes. This is window+event-distribution sensitivity, not fixed-target causal proof. No auto retraining or test claim.',
                  limitations=['synthetic falling thresholds, not human instability evidence','one training seed','12 layouts on6 development parents, no 92% conclusion','conditional N fall generation, not population risk model'])
    write(out/'protocol.json',protocol);write(out/'scenarios.json',groups)
    events=[e for c in groups['train'] for e in c['scenario']['events']]
    write(out/'generation-audit.json',dict(training_events=len(events),threshold_events=sum(e['trigger']=='threshold' for e in events),
          random_events=sum(e['trigger']=='random' for e in events),fall_range=[min(e['fall'] for e in events),max(e['fall'] for e in events)],
          total_candidate_exposures=sum(c['scenario']['attempted_exposures'] for c in groups['train']),seconds=time.monotonic()-start))
    return groups,protocol

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw))
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed(8701)
        status('preparing');groups,protocol=prepare(args.dataset,out)
        status('baseline');baseline=evaluate(None,groups['development']);write(out/'planner-development.json',baseline)
        completed=[];status('training',algorithm=ALGORITHMS[0],episode=0,completed=completed)
        for algorithm in ALGORITHMS:
            seed(8701);model=Learner(algorithm,Env(np.load(groups['train'][0]['field']['path'],mmap_mode='r'),groups['train'][0]['scenario']).observe().shape[1]);replay=[]
            folder=out/algorithm;folder.mkdir();best=None
            for ep in range(65):
                if (out/'STOP_AFTER_EPISODE').exists():status('stopped',algorithm=algorithm,episode=ep,completed=completed);return
                if time.monotonic()-start>args.max_seconds:raise TimeoutError('Pilot elapsed-time cap; no silent restart')
                if ep:
                    record,rows=rollout(model,groups['train'][ep-1],collect=True,deterministic=False)
                    if algorithm in ('hasac','maac'):
                        replay.extend(rows);del replay[:-2048]
                        losses=[model.update_off_policy(random.sample(replay,min(64,len(replay)))) for _ in range(8)]
                    else:losses=[model.update_on_policy(rows)]
                    record.update(episode=ep,algorithm=algorithm,updates=model.updates,diagnostics=losses[-1])
                    with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
                    status('training',algorithm=algorithm,episode=ep,completed=completed)
                if ep in (0,32,64):
                    model.eval();result=evaluate(model,groups['development']);write(folder/f'development-{ep:04}.json',result)
                    checkpoint=folder/f'checkpoint-{ep:04}.pt'
                    torch.save(dict(learner=model.checkpoint(),episode=ep,protocol_sha256=sha(out/'protocol.json'),rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),replay=replay),checkpoint)
                    if best is None or quality(result)>tuple(best['quality']):
                        best=dict(episode=ep,path=str(checkpoint),sha256=sha(checkpoint),quality=quality(result),summary=result['summary']);write(folder/'best.json',best)
                    model.train();print(json.dumps(dict(algorithm=algorithm,episode=ep,development=result['summary'])),flush=True)
            completed.append(algorithm);write(out/'leaders.json',{a:read(out/a/'best.json') for a in completed})
        for p,h in protocol['source_hashes'].items():
            if sha(p)!=h:raise ValueError('Source changed during training: '+p)
        status('completed',completed=completed,episodes_total=384,formal_test_used=False)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=10800);p.add_argument('--execute',action='store_true');args=p.parse_args()
    if not args.execute:p.error('Explicit --execute required')
    run(args)
