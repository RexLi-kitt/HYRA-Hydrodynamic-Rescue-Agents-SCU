"""Fresh 80-person planning-residual pilot. No environment changes during training."""
from pathlib import Path
import argparse,copy,json,os,random,sys,time,traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
for name in ('flood-drop-v2','flood-drop-variable-v1'):
    sys.path.insert(0,str(HERE.parent/name))
from pilot import Learner,seed,read,write,sha
from variable_environment import Config,make_scenario
from variable_dispatch import DispatchEnv
from screen import summarize,run_case

METHODS=('maac','mat','happo')

class ResidualEnv(DispatchEnv):
    def choices(self):
        mask,expert,options,features=super().choices()
        for i,a in enumerate(expert):
            options[i][8]=copy.deepcopy(options[i][a]);features[i,8]=features[i,a].copy();mask[i,8]=True
        return mask,np.full(self.cfg.uavs,8,dtype=np.int64),options,features

def initialize(model):
    heads=[model.policy.head] if model.algorithm=='mat' else [a[-1] for a in model.actors]
    with torch.no_grad():
        for h in heads:h.weight.zero_();h.bias.zero_();h.bias[8]=5.

def quality(result):
    s=result['summary'];return (s['people_supplied'],s['all_successes'],s['at_least95_successes'])

def episode(model,case,collect=False):
    env=ResidualEnv(np.load(case['field']['path'],mmap_mode='r',allow_pickle=False),case['scenario'])
    obs=env.observe();mask,_,options,_=env.choices();done=False;rows=[];ret=0.;free=used=0
    while not done:
        action,logs,value=model.act(obs,mask,not collect)
        choices=mask.sum(1)>1;free+=int(choices.sum())
        used+=sum(bool(choices[i] and (options[i][action[i]]['person']!=options[i][8]['person'] or
            not np.allclose(options[i][action[i]]['xy'],options[i][8]['xy'],rtol=0,atol=1e-9))) for i in range(4))
        nxt,reward,done,_=env.step(action);nm,_,noptions,_=env.choices();ret+=reward
        if collect:rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;options=noptions
    for i,r in enumerate(rows):r['next_action']=rows[i+1]['action'] if i+1<len(rows) else np.full(4,8,dtype=np.int64)
    result=dict(**env.metrics(),case=case['id'],parent=case['field']['parent'],return_sum=ret,alternative_task_choice_fraction=used/max(1,free))
    assert result['supplied']+result['expired']+result['outside']==env.cfg.people
    if not collect:
        result['persons']=[dict(id=j,exposure_id=case['scenario']['events'][j]['exposure_id'],fall=int(env.fall[j]),deadline=int(env.deadline[j]),
            first_seen=float(env.first_seen[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(env.cfg.people)]
        for p in result['persons']:
            if p['status']==2:assert p['fall']+env.cfg.drop_seconds<=p['supplied_at']<=p['deadline']
    return result,rows

def evaluate(model,cases):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        rows=[episode(model,c)[0] for c in cases];return dict(rows=rows,summary=summarize(rows))
    finally:random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])

def prepare(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    cfg=Config(**read(a.candidate)['config']);assert cfg.people==80 and cfg.uavs==4
    old=read(a.previous/'scenarios.json');groups={};fields={}
    for split,n,offset in [('train',64,1831000),('development',12,1941000)]:
        pool={c['field']['parent']:c['field'] for c in old[split]};fs=list(pool.values());groups[split]=[]
        assert len(fs)==(16 if split=='train' else 6)
        for i in range(n):
            f=fs[i%len(fs)];fields[f['path']]=f
            s=make_scenario(np.load(f['path'],mmap_mode='r',allow_pickle=False),offset+i,cfg,fall_end=960,pool_size=120)
            groups[split].append(dict(id=f'load80-{split}-{i:03}',field=f,scenario=s))
    assert not {c['field']['parent'] for c in groups['train']} & {c['field']['parent'] for c in groups['development']}
    for path,f in fields.items():
        assert sha(path)==f['sha256'] and f['finite_nonnegative']
        assert abs(f['mass_residual_m3'])<=max(1e-6,abs(f['source_volume_m3'])*1e-9)
    hashes={str(p):sha(p) for name in ('flood-drop-v2','flood-drop-variable-v1','flood-drop-load80-v1','flood1km-six-20261101/flood1km') for p in (HERE.parent/name).rglob('*.py')}
    write(out/'scenarios.json',groups)
    protocol=dict(version='load80-residual-64-v1',methods=METHODS,episodes_per_method=64,seed=8301,config=read(a.candidate)['config'],
        user_correction='70% ceiling refers to ALL-PERSON SCENES, not people completion. User explicitly accepts keeping the current harder candidate for this pilot.',
        primary='People supplied; then all-person scenes; then ceil95% scenes; exact ties keep earlier checkpoint.',
        thresholds=dict(all=80,at_least95=76),generation=dict(fall_end=960,pool_size=120,train_seed=1831000,development_seed=1941000),
        scope='New layouts,16 TRAIN hydraulic parents disjoint from6 development parents; development parents previously used for calibration, not final independent test.',
        model='Fresh 1240-feature input, four agents. Planning suggestion is option8, initialization bias5. Episode0 is pure planning, not learned gain.',
        checkpoints=[0,16,32,64],reward='Existing per-person delivered/discovered/expired reward unchanged; no all-person-only reward.',
        comparison='Original and coordinated planning evaluated on identical NEW12 layouts; old67.81% must not be reused as this cohort baseline.',
        fairness='Same observed information and candidate actions. No hidden fall/deadline/future field supplied. Arrival schedule is a shared development prior; stronger anticipatory planning remains needed before superiority claims.',
        compute='CPU2 threads, one process, 192 training episodes plus24 baseline,12 identity and108 learned validation rollouts; no GPU conversion for small pilot.',
        fallback='After all64: diagnose task deviations/learning and people results. If no gain, do not assume difficulty alone. Separately preregister window240 then300 with same births and fall_end960; preserve180 results and retrain/evaluate all methods equally. No silent parameter switch or automatic relaxation in this run.',
        stop='STOP_AFTER_EPISODE or7200s; no automatic restart or1024 continuation.',
        source_hashes=hashes,water_hashes={p:f['sha256'] for p,f in fields.items()},scenario_sha256=sha(out/'scenarios.json'),candidate_sha256=sha(a.candidate))
    write(out/'protocol.json',protocol);write(out/'status.json',dict(status='prepared',training_episodes=192))

def smoke(a):
    case=read(a.output/'scenarios.json')['train'][0];water=np.load(case['field']['path'],mmap_mode='r')
    e=ResidualEnv(water,case['scenario']);assert e.observe().shape==(4,1240)
    changed=copy.deepcopy(case['scenario'])
    for x in changed['events']:x['xy']=[500.,500.]
    hidden=ResidualEnv(water,changed);assert np.array_equal(e.observe(),hidden.observe())
    assert np.array_equal(e.choices()[0],hidden.choices()[0])
    checks={}
    torch.set_num_threads(2)
    for method in METHODS:
        seed(8301);model=Learner(method,1240);initialize(model)
        env=ResidualEnv(water,case['scenario']);other=DispatchEnv(water,case['scenario'])
        rows=[];before=next(model.parameters()).detach().clone()
        for t in range(8):
            obs=env.observe();mask=env.choices()[0];act,lp,v=model.act(obs,mask,True);assert (act==8).all()
            nxt,r,done,_=env.step(act);other.step(other.choices()[1]);assert np.array_equal(env.pos,other.pos)
            rows.append(dict(obs=obs,mask=mask,action=act,reward=r,done=float(done),next_obs=nxt,next_mask=env.choices()[0],logp=lp,value=v,next_action=act))
        if method=='maac':model.update_off_policy(rows)
        else:model.update_on_policy(rows)
        model.check_finite();assert model.updates>0
        state=model.checkpoint();restored=Learner(method,1240);restored.restore(state)
        assert np.array_equal(model.act(nxt,env.choices()[0],True)[0],restored.act(nxt,env.choices()[0],True)[0])
        checks[method]=dict(finite_update=True,checkpoint_restore=True,planning_identity_steps=8)
    write(a.output/'readiness.json',dict(status='passed',input_dim=1240,future_roster_hidden_at_start=True,methods=checks))

def run(a):
    out=a.output;p=read(out/'protocol.json');cases=read(out/'scenarios.json');start=time.monotonic()
    assert read(out/'status.json')['status']=='prepared' and read(out/'readiness.json')['status']=='passed'
    assert sha(out/'scenarios.json')==p['scenario_sha256']
    def verify():
        for path,h in {**p['source_hashes'],**p['water_hashes']}.items():assert sha(path)==h,path
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw))
    def guard():
        if (out/'STOP_AFTER_EPISODE').exists():raise InterruptedError('User stop marker')
        if time.monotonic()-start>7200:raise TimeoutError('Pilot cap')
    try:
        verify();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);plans={};leaders={};identity=None
        for method in ('original','dispatch'):
            rows=[]
            for c in cases['development']:
                guard();status('planning_baseline',method=method,completed=len(rows));rows.append(run_case(c,method))
            plans[method]=dict(rows=rows,summary=summarize(rows));write(out/f'planner-{method}.json',plans[method])
        for m in METHODS:
            seed(8301);folder=out/m;folder.mkdir();model=Learner(m,1240);initialize(model);replay=[];best=None
            for ep in range(65):
                guard()
                if ep:
                    status('training',method=m,episode=ep,completed_methods=list(leaders));model.train()
                    metric,rows=episode(model,cases['train'][ep-1],True)
                    if m=='maac':
                        replay.extend(rows);del replay[:-2048]
                        for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
                    else:loss=model.update_on_policy(rows)
                    model.check_finite();metric.update(episode=ep,updates=model.updates,diagnostics=loss)
                    with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
                if ep in p['checkpoints']:
                    status('evaluation',method=m,episode=ep,completed_methods=list(leaders));model.eval()
                    if ep==0 and identity is not None:result=copy.deepcopy(identity);result['initial_reused']=True
                    else:
                        result=evaluate(model,cases['development'])
                        if ep==0:
                            for r,ref in zip(result['rows'],plans['dispatch']['rows']):
                                assert r['persons']==ref['persons']
                                for k in ('supplied','expired','outside','conflicts','coverage'):assert r[k]==ref[k]
                                assert r['alternative_task_choice_fraction']==0
                            identity=copy.deepcopy(result);write(out/'planning-identity.json',dict(status='passed',cases=12,all_person_records=True))
                    write(folder/f'development-{ep:04}.json',result)
                    ck=folder/f'checkpoint-{ep:04}.pt';torch.save(dict(learner=model.checkpoint(),episode=ep,protocol_sha256=sha(out/'protocol.json'),
                        replay=replay,rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
                    if best is None or quality(result)>tuple(best['quality']):
                        best=dict(episode=ep,quality=quality(result),summary=result['summary'],path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
                    print(json.dumps(dict(method=m,episode=ep,summary=result['summary'])),flush=True)
            leaders[m]=best;write(out/'leaders.json',leaders)
        verify();stronger=max(plans,key=lambda m:quality(plans[m]))
        write(out/'summary.json',dict(status='completed',training_episodes=192,evaluation_rollouts=144,initial_reused_rows=24,
            baselines={m:v['summary'] for m,v in plans.items()},stronger_planner=stronger,learners=leaders,
            gains={m:dict(people_pp=100*(b['summary']['people_completion']-plans[stronger]['summary']['people_completion']),
                all_scene_pp=100*(b['summary']['all_rate']-plans[stronger]['summary']['all_rate']),
                at_least95_scene_pp=100*(b['summary']['at_least95_rate']-plans[stronger]['summary']['at_least95_rate'])) for m,b in leaders.items()},
            source_and_water_hashes_unchanged=True,formal_test=False))
        status('completed',training_episodes=192,evaluation_rollouts=144)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',type=Path);p.add_argument('--previous',type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--smoke',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args();prepare(a) if a.prepare else smoke(a) if a.smoke else run(a)
