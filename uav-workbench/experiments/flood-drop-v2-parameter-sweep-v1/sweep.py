"""Prospective difficulty screen. Select by planning burden, never by learned gap."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,json,os,sys,time,traceback
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import read,write,sha,Config,make_scenario,model_for,baseline_candidate
from coord_rollout import episode
from screen import summarize,run_case

def profiles():
    base=Config(people=60,duration=1320,window=240,fall_pattern='three_bursts')
    specs=[]
    def add(name,cfg,end=960,kind='single_factor'):
        specs.append(dict(id=name,config=asdict(cfg),fall_end=end,kind=kind))
    for n in (40,60,80):
        for w in (180,240,300):add(f'n{n}_w{w}',replace(base,people=n,window=w),kind='factorial')
    for name,values in [('trigger_depth',(.05,.20)),('trigger_speed',(.10,.40)),('random_per_minute',(.01,.15))]:
        for v in values:add(f'{name}_{v}',replace(base,**{name:v}),kind='conditional_exposure_distribution')
    add('low_triggers',replace(base,trigger_depth=.05,trigger_speed=.10),kind='joint_trigger_distribution')
    add('high_triggers',replace(base,trigger_depth=.20,trigger_speed=.40),kind='joint_trigger_distribution')
    add('time1080',replace(base,duration=1080),840,'duration_and_arrival_support')
    add('time1500',replace(base,duration=1500),1260,'duration_and_arrival_support')
    assert len(specs)==19
    return specs

def prepare(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False)
    previous=read(a.previous/'protocol.json');old=read(a.previous/'scenarios.json')['development']['main40']
    fields=list({c['field']['parent']:c['field'] for c in old}.values());assert len(fields)==6
    source={str(f):sha(f) for f in HERE.glob('*.py')}
    p=dict(version='v2-parameter-sweep19-v1',date='2026-11-02',profiles=profiles(),fields=fields,seed_base=2491000,
        methods=['original','dispatch','balanced','nearest_reserved','frozen_happo32'],
        previous=str(a.previous),frozen_checkpoint=previous['frozen_reference'],frozen_sha256=previous['frozen_reference_sha256'],
        dependency_wrapper=str(a.previous.parent/'wrapper-status.json'),
        source_hashes={**previous['source_hashes'],**source},water_hashes={f['path']:f['sha256'] for f in fields},
        scope='Prospective19-profile development screening on6 reused DEV water parents, newperson seeds. No final independent test, no new gradient training.',
        pairing='N/window profiles share120 accepted exposure pool andfall_end960; compare same-window N and same-N window. Trigger/random/time-distribution changes alter accepted population and cannot be claimed fixed-person causal effects.',
        fixed_count='N actual falls is conditioned. Threshold changes which events are accepted, not actual population count. Not human safety thresholds.',
        time='1080/1500 vary duration AND allowable arrival support840/1260, clearly labeled. All individual windows complete; no water beyond1500.',
        unchanged='4UAV,10s supply,5s decisions,10m/s,100m sensor,existing hydraulic magnitudes/units/wet threshold/obstacles. No arbitrary water rescaling.',
        primary='People supplied first; all-person scene andceil95% separate; do not aggregate differentN into one score.',
        selection='Only planning results: stronger people rate>=.90 and max all-scene rate across all4 planners<=.70. Rank by distance of stronger people rate to.95, then max all-scene rate to.50, then configured order. Frozen model results never used to choose difficulty. If none, choose none.',
        interpretation='Parameter screening can improve task scores without improving an algorithm. Report complete unfavorable cases and all baselines; no guarantee15-point gap.',
        costs='570 new evaluation rollouts, no training, CPU2 threads,one heavy process. Wait until previouswrapper completed; stop on failed dependency/STOP marker/7200s execution cap.')
    write(out/'protocol.json',p);write(out/'status.json',dict(status='prepared',profiles=19,evaluation_rollouts=570))

def run(a):
    out=a.output;p=read(out/'protocol.json');start=time.monotonic();rows=[];cases={};links={}
    assert read(out/'status.json')['status']=='prepared'
    def status(state,**kw):write(out/'status.json',dict(status=state,pid=os.getpid(),completed=len(rows),total=570,elapsed_s=time.monotonic()-start,**kw))
    def guard():
        if (out/'STOP_AFTER_CASE').exists():raise InterruptedError('User stop marker')
        if time.monotonic()-start>7200:raise TimeoutError('Sweep cap')
    try:
        dep=Path(p['dependency_wrapper'])
        assert dep.exists() and read(dep)['status']=='completed','Dependency not completed'
        for path,h in {**p['source_hashes'],**p['water_hashes']}.items():assert sha(path)==h,path
        assert sha(p['frozen_checkpoint'])==p['frozen_sha256']
        torch.set_num_threads(2);state=torch.load(p['frozen_checkpoint'],map_location='cpu',weights_only=False)
        model=model_for(state['learner']['algorithm'],state['recipe']);model.restore(state['learner']);model.eval();del state
        for spec in p['profiles']:
            cfg=Config(**spec['config']);links[spec['id']]=[]
            for i,f in enumerate(p['fields']):
                s=make_scenario(np.load(f['path'],mmap_mode='r'),p['seed_base']+i,cfg,fall_end=spec['fall_end'],pool_size=120)
                cid=f"{spec['id']}-{i}";cases[cid]=dict(id=cid,field=f,scenario=s);links[spec['id']].append(cid)
        for i in range(6):
            whole={e['exposure_id']:e for e in cases[f'n80_w300-{i}']['scenario']['events']}
            for n in (40,60,80):
                for w in (180,240,300):
                    for e in cases[f'n{n}_w{w}-{i}']['scenario']['events']:
                        assert all(e[k]==whole[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
        write(out/'scenarios.json',dict(cases=cases,links=links))
        write(out/'input-audit.json',dict(status='passed',scenarios_sha256=sha(out/'scenarios.json'),nested_people=True,paired_windows=True,complete_windows=True))
        for spec in p['profiles']:
            for cid in links[spec['id']]:
                for method in p['methods']:
                    guard();status('evaluating',profile=spec['id'],method=method,case=cid)
                    case=cases[cid]
                    if method=='frozen_happo32':r=episode(model,case)[0]
                    elif method in ('balanced','nearest_reserved'):r=baseline_candidate(case,nearest=method=='nearest_reserved')
                    else:r=run_case(case,method)
                    assert r['supplied']+r['expired']+r['outside']==r['people_total']
                    r.update(profile=spec['id'],method=method);rows.append(r)
                    with (out/'results.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(r,allow_nan=False)+'\n')
            print(json.dumps(dict(profile=spec['id'],completed=len(rows),total=570)),flush=True)
        results={};eligible=[]
        for index,spec in enumerate(p['profiles']):
            by={m:summarize([r for r in rows if r['profile']==spec['id'] and r['method']==m]) for m in p['methods']}
            planners=p['methods'][:-1]
            strong=max(planners,key=lambda m:(by[m]['people_supplied'],by[m]['all_successes'],by[m]['at_least95_successes']))
            maxall=max(by[m]['all_rate'] for m in planners)
            results[spec['id']]=dict(methods=by,stronger_planner=strong,max_planner_all_rate=maxall)
            if by[strong]['people_completion']>=.9 and maxall<=.7:
                eligible.append((abs(by[strong]['people_completion']-.95),abs(maxall-.5),index,spec['id']))
        selected=min(eligible)[-1] if eligible else None
        for path,h in {**p['source_hashes'],**p['water_hashes']}.items():assert sha(path)==h
        write(out/'summary.json',dict(profiles=results,selected_by_planning_only=selected,eligible_profiles=[x[-1] for x in sorted(eligible)],
            selected_config=next((s for s in p['profiles'] if s['id']==selected),None),evaluation_rollouts=len(rows),training_episodes=0,
            source_hashes_unchanged=True,formal_test=False))
        status('completed',selected_by_planning_only=selected)
    except BaseException as e:
        write(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));status('failed',error=repr(e));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true')
    a=p.parse_args();prepare(a) if a.prepare else run(a)
