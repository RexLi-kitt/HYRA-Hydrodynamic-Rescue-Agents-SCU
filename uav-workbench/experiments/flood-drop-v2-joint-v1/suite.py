"""V2 observed-task candidates, canonical inputs and registered adaptive training."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,copy,json,os,random,time,traceback
import numpy as np
import torch
from rollout import ResidualEnv,DispatchEnv,Learner,seed,read,write,sha,episode,evaluate,quality
from variable_environment import Config,make_scenario
from screen import summarize,run_case

HERE=Path(__file__).resolve().parent
METHODS=('maac','mat','happo')
RECIPES=[dict(id='steady',bias=1.,lr=1e-4,gamma=.999),dict(id='explore',bias=.5,lr=3e-4,gamma=.995)]
PROFILES={'main40':Config(people=40,duration=1320,window=240,fall_pattern='three_bursts'),
          'stress60':Config(people=60,duration=1320,window=240,fall_pattern='three_bursts'),
          'stress80':Config(people=80,duration=1320,window=300,fall_pattern='three_bursts')}

def model_for(method,recipe):
    model=Learner(method,1240,gamma=recipe['gamma'])
    heads=[model.policy.head] if method=='mat' else [a[-1] for a in model.actors]
    with torch.no_grad():
        for h in heads:h.weight.zero_();h.bias.zero_();h.bias[8]=recipe['bias']
    for optimizer in model.optimizers:
        for group in optimizer.param_groups:group['lr']=recipe['lr']
    return model

def prepare(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    old=read(a.previous/'scenarios.json');groups={'train':[],'development':{}}
    trainfields=list({c['field']['parent']:c['field'] for c in old['train']}.values())
    devfields=list({c['field']['parent']:c['field'] for c in old['development']}.values())
    assert len(trainfields)==16 and len(devfields)==6
    assert not {f['parent'] for f in trainfields}&{f['parent'] for f in devfields}
    fields={f['path']:f for f in trainfields+devfields}
    for path,f in fields.items():
        assert sha(path)==f['sha256'] and f['finite_nonnegative']
        assert abs(f['mass_residual_m3'])<=max(1e-6,abs(f['source_volume_m3'])*1e-9)
    def case(field,cfg,number,split,profile):
        s=make_scenario(np.load(field['path'],mmap_mode='r',allow_pickle=False),number,cfg,fall_end=960,pool_size=120)
        return dict(id=f'{split}-{profile}-{number}',field=field,scenario=s)
    for i in range(128):groups['train'].append(case(trainfields[i%16],PROFILES['main40'],2051000+i,'train','main40'))
    for name,cfg in PROFILES.items():
        groups['development'][name]=[case(devfields[i%6],cfg,2161000+i,'development',name) for i in range(12 if name=='main40' else 6)]
    # Common accepted exposure pool and stable public-ID ordering across N and windows.
    for i in range(6):
        whole={e['exposure_id']:e for e in groups['development']['stress80'][i]['scenario']['events']}
        for name in ('main40','stress60'):
            for e in groups['development'][name][i]['scenario']['events']:
                assert all(e[k]==whole[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
    write(out/'scenarios.json',groups)
    dirs=['flood-drop-v2-joint-v1','flood-drop-variable-v1','flood-drop-v2','flood1km-six-20261101/flood1km']
    hashes={str(f):sha(f) for d in dirs for f in (HERE.parent/d).rglob('*.py')}
    p=dict(version='v2-joint-candidate-adaptive-v1',date='2026-11-02',methods=METHODS,recipes=RECIPES,seed=8301,
        profiles={k:asdict(v) for k,v in PROFILES.items()},primary_profile='main40',train_profile='main40',
        population='Conditioned actual falls; common120 pool, support0..960, independent public ID shuffle.',
        scope='128 prospective training layouts on16 TRAIN parents; main12 and two6-case pressure panels on6 disjoint DEV parents previously used for calibration. No final test.',
        primary_metric='Supplied people on main40; tie all40 then ceil95%=38. Pressure panels separately reported, never aggregate different N for checkpoint selection.',
        changes=['nearest2 plus2 highest observed urgency/distance utility candidates, same for stronger planning',
                 'canonical order of observed person records, padded to80; no future identity ordering',
                 'two joint optimization recipes; not a single-factor causal study'],
        constraints='4UAV,10s continuous drop,5s decisions,10m/s,100m sensor unchanged.1320s duration ensures full300s window atfall960. No added recurrent memory.',
        baseline=['original physical planner','original nearest4 coordinated planner','new balanced-candidate coordinated planner'],
        initialization='Both positive biases choose planning deterministically at0. Must prove complete person identity once and show sampled distributions for both recipes;0 is not learned gain.',
        budget='Each method/recipe32 episodes minimum:192 total. Conditional64/128; at most768. One CPU process2 threads7200s cap,STOP_AFTER_EPISODE supported.',
        extension32='At32: primary people>16 and>=0; all/95 scene counts>=16 and0; each pressure people>=0; finite checks. Then64.',
        extension64='At64: primary people>32 and>0; all/95 counts>=32 and0; each pressure people>=0. Then96/128.',
        failure_rule='No automatic difficulty changes after inspecting learner gaps. Flat or adverse recipes stop at32/64, all results retained;128 terminates.',
        fairness='Planning receives same candidate expansion and current observations; no true deadlines/future people. Strongest of all3 baselines retained. Full lookahead planner and multiple seeds remain prerequisites for broad superiority claims.',
        source_hashes=hashes,water_hashes={p:f['sha256'] for p,f in fields.items()},scenario_sha256=sha(out/'scenarios.json'))
    write(out/'protocol.json',p);write(out/'status.json',dict(status='prepared',minimum_training_episodes=192,maximum_training_episodes=768))

def panel(model,cases):return {name:evaluate(model,group) for name,group in cases.items()}

def baseline_candidate(case):
    e=DispatchEnv(np.load(case['field']['path'],mmap_mode='r'),case['scenario'])
    while True:
        _,_,done,_=e.step(e.choices()[1])
        if done:break
    r=dict(**e.metrics(),case=case['id'],parent=case['field']['parent'])
    r['persons']=[dict(id=j,exposure_id=case['scenario']['events'][j]['exposure_id'],fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
    assert r['supplied']+r['expired']+r['outside']==e.cfg.people
    return r

def eligible(now,prior,initial,strict_initial):
    s=now['main40']['summary'];p=prior['main40']['summary'];i=initial['main40']['summary']
    reasons=[]
    if s['people_supplied']<=p['people_supplied']:reasons.append('no increase over previous checkpoint')
    if s['people_supplied']<i['people_supplied'] or (strict_initial and s['people_supplied']==i['people_supplied']):reasons.append('no required gain against initial planning')
    for key in ('all_successes','at_least95_successes'):
        if s[key]<max(p[key],i[key]):reasons.append('scene regression:'+key)
    for name in ('stress60','stress80'):
        if now[name]['summary']['people_supplied']<initial[name]['summary']['people_supplied']:reasons.append('pressure regression:'+name)
    return dict(extend=not reasons,reasons=reasons)

def execute(a):
    out=a.output;p=read(out/'protocol.json');cases=read(out/'scenarios.json');start=time.monotonic();train_count=eval_count=0
    assert read(out/'status.json')['status']=='prepared' and read(out/'readiness.json')['status']=='passed'
    assert sha(out/'scenarios.json')==p['scenario_sha256']
    def verify():
        for path,h in {**p['source_hashes'],**p['water_hashes']}.items():assert sha(path)==h,path
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),elapsed_s=time.monotonic()-start,training_episodes=train_count,evaluation_rollouts=eval_count,**kw))
    def guard():
        if (out/'STOP_AFTER_EPISODE').exists():raise InterruptedError('User stop marker')
        if time.monotonic()-start>7200:raise TimeoutError('Runtime cap')
    try:
        verify();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);plans={};leaders={};identity=None
        for method in ('original','dispatch','balanced'):
            plans[method]={}
            for name,group in cases['development'].items():
                rows=[]
                for c in group:
                    guard();status('planning_baseline',method=method,profile=name,completed=len(rows))
                    rows.append(baseline_candidate(c) if method=='balanced' else run_case(c,method));eval_count+=1
                plans[method][name]=dict(rows=rows,summary=summarize(rows))
            write(out/f'planner-{method}.json',plans[method])
        for recipe in p['recipes']:
            for method in p['methods']:
                key=recipe['id']+'-'+method;folder=out/key;folder.mkdir();seed(8301)
                model=model_for(method,recipe);replay=[];best=None;results={};limit=32;ep=0;decisions=[]
                while ep<=limit:
                    guard()
                    if ep:
                        status('training',recipe=recipe['id'],method=method,episode=ep,target=limit,completed_runs=list(leaders));model.train()
                        metric,rows=episode(model,cases['train'][ep-1],True)
                        if method=='maac':
                            replay.extend(rows);del replay[:-2048]
                            for _ in range(8):loss=model.update_off_policy(random.sample(replay,min(64,len(replay))))
                        else:loss=model.update_on_policy(rows)
                        model.check_finite();train_count+=1;metric.update(episode=ep,updates=model.updates,diagnostics=loss)
                        with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(metric,allow_nan=False)+'\n')
                    if ep in (0,16,32,64,96,128):
                        status('evaluation',recipe=recipe['id'],method=method,episode=ep,target=limit,completed_runs=list(leaders));model.eval()
                        if ep==0 and identity is not None:result=copy.deepcopy(identity)
                        else:
                            result=panel(model,cases['development']);eval_count+=24
                            if ep==0:
                                for name in result:
                                    for r,b in zip(result[name]['rows'],plans['balanced'][name]['rows']):
                                        assert r['persons']==b['persons']
                                        for k in ('supplied','expired','outside','conflicts','coverage'):assert r[k]==b[k]
                                        assert r['alternative_task_choice_fraction']==0
                                identity=copy.deepcopy(result);write(out/'planning-identity.json',dict(status='passed',cases=24,all_person_records=True))
                        results[ep]=result;write(folder/f'development-{ep:04}.json',result)
                        ck=folder/f'checkpoint-{ep:04}.pt';torch.save(dict(learner=model.checkpoint(),episode=ep,recipe=recipe,replay=replay,
                            protocol_sha256=sha(out/'protocol.json'),rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())),ck)
                        q=quality(result['main40'])
                        if best is None or q>tuple(best['quality']):
                            best=dict(episode=ep,quality=q,summaries={n:x['summary'] for n,x in result.items()},path=str(ck),sha256=sha(ck));write(folder/'best.json',best)
                        if ep in (32,64):
                            d=eligible(result,results[16 if ep==32 else 32],results[0],ep==64);d.update(episode=ep);decisions.append(d)
                            if d['extend']:limit=64 if ep==32 else 128
                            write(folder/'budget-decisions.json',decisions)
                        print(json.dumps(dict(run=key,episode=ep,target=limit,summary=result['main40']['summary'])),flush=True)
                    ep+=1
                leaders[key]=dict(**best,trained_episodes=ep-1,decisions=decisions);write(out/'leaders.json',leaders)
        verify()
        strongest={n:max(plans,key=lambda m:quality(plans[m][n])) for n in cases['development']}
        write(out/'summary.json',dict(status='completed',training_episodes=train_count,evaluation_rollouts=eval_count,initial_reused_rows=120,
            baselines={m:{n:v['summary'] for n,v in by.items()} for m,by in plans.items()},strongest_planners=strongest,learners=leaders,
            gains={k:{n:dict(people_pp=100*(s['people_completion']-plans[strongest[n]][n]['summary']['people_completion']),
                all_scene_pp=100*(s['all_rate']-plans[strongest[n]][n]['summary']['all_rate']),at_least95_scene_pp=100*(s['at_least95_rate']-plans[strongest[n]][n]['summary']['at_least95_rate'])) for n,s in v['summaries'].items()} for k,v in leaders.items()},
            source_and_water_hashes_unchanged=True,formal_test=False))
        status('completed',completed_runs=list(leaders))
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args();prepare(a) if a.prepare else execute(a)
