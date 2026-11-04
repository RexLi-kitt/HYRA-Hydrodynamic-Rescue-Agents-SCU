"""Prospective workload/window interactions and one-unit local sensitivity."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,copy,time,os,json,argparse
import numpy as np
HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'flood-drop-variable-v1'
sys.path.insert(0,str(SOURCE))
from screen import read,write,sha,fingerprint,execute,CATALOG
from variable_environment import Config,make_scenario

def profiles():
    out=[]
    for n in (40,80,120):
        for w in (120,180,240):
            for pattern in ('uniform','three_bursts'):
                c=Config(people=n,duration=1200,window=w,fall_pattern=pattern)
                out.append(dict(id=f'n{n}_w{w}_{pattern}',config=asdict(c),fall_end=960,kind='factorial'))
    for duration in (900,1500):
        for pattern in ('uniform','three_bursts'):
            c=Config(people=80,duration=duration,window=180,fall_pattern=pattern)
            out.append(dict(id=f't{duration}_{pattern}',config=asdict(c),fall_end=duration-240,kind='duration_and_arrival_support'))
    for n in (40,80):
        out.append(dict(id=f'control{n}',config=asdict(Config(people=n)),fall_end=1140,kind='long_window_control'))
    center=Config(people=80,duration=1200,window=180)
    for key,values in [('people',(79,81)),('window',(179,181))]:
        for value in values:
            c=replace(center,**{key:value})
            out.append(dict(id=f'fine_{key}_{value}',config=asdict(c),fall_end=960,kind='one_unit_neighbor'))
    assert len(out)==28
    return out

def prepare(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    assert read(a.previous/'status.json')['status']=='completed'
    assert read(a.previous.parent/'completion-audit.json')['status']=='passed'
    prior=read(a.previous/'protocol.json');old=read(a.previous/'scenarios.json')['cases']
    fields={c['field']['parent']:c['field'] for c in old.values()};assert len(fields)==6
    legacy={**prior['source_hashes'],**prior['legacy_source_hashes']}
    for path,h in {**legacy,**prior['water_hashes']}.items():assert sha(path)==h
    p=dict(version='variable-load-interaction-v1',date='2026-11-02',profiles=profiles(),methods=['original','dispatch'],
        primary='People first within identical N/configuration; report all-person and ceil95% scene rates; all levels retained.',
        trigger='User explicitly requested further adjustment after full36-profile screen,2026-11-02.',
        evidence='Prior single-factor180s window gave223/240 people; N80 alone478/480. This motivates interactions, not a learned gap claim.',
        design='3N x3windows x2arrival patterns at1200s;4duration/arrival-context probes;2long-window controls;4one-unit neighbors.',
        layout_count=12,parent_count=6,seed_base=1631000,seed_stride=1,roster_pool=120,
        methods_unchanged=True,source_hashes={str(f):sha(f) for f in HERE.glob('*.py')},legacy_source_hashes=legacy,water_hashes=prior['water_hashes'],
        reference='Within each T/arrival-support context:80 people,window240,uniform falls. Context control is not a single-factor cause for all comparisons.',
        pairing='All window and N variants atT1200 sharefall_end960. Common120-event pool; prefix people and independent ID ranks. Pattern changes accepted population distribution; T probes changefall support.',
        inference='Adaptive development follow-up after prior screen,not confirmatory. New personnel layouts on same6hydraulic parents. No test or learner used.',
        fine_units='Only registered N79/80/81 andW179/180/181; not all integer values or a performance optimization.',
        unchanged_physics='4UAV,10m/s,100m sensing,10s service,10m service radius,5s decision,5m separation; current frozen hydraulic data.',
        stop='All profiles or3600s,STOP_AFTER_CASE respected; no automatic retry/retrain or discarded adverse result.',
        next_selection='No profile is chosen to maximize a learned-planner gap; report cross-context performance before any training distribution freeze.',
        old_model_scores_inherited=False)
    write(out/'protocol.json',p);write(out/'status.json',dict(status='preparing',pid=os.getpid()))
    cases={};links={};cache={};gen=('trigger_depth','trigger_speed','random_per_minute','traps','trap_radius','fall_pattern','burst_width','spatial_clusters','cluster_radius')
    def add(field,index,cfg,end):
        key=(field['sha256'],index,end,*(getattr(cfg,k) for k in gen))
        if key not in cache:
            generator=replace(Config(),people=120,window=1,drop_seconds=1,**{k:getattr(cfg,k) for k in gen})
            cache[key]=make_scenario(np.load(field['path'],mmap_mode='r'),1631000+index,generator,fall_end=end,pool_size=120)
        scenario=copy.deepcopy(cache[key]);include=set(sorted(e['exposure_id'] for e in scenario['events'])[:cfg.people])
        scenario['events']=[e for e in scenario['events'] if e['exposure_id'] in include];scenario['config']=asdict(cfg)
        for e in scenario['events']:e['deadline']=e['fall']+cfg.window
        assert len(scenario['events'])==cfg.people and all(e['deadline']<=cfg.duration for e in scenario['events'])
        cid=fingerprint(dict(parent=field['parent'],scenario=scenario));cases.setdefault(cid,dict(id=cid,field=field,scenario=scenario))
        return cid
    fs=list(fields.values())
    for spec in p['profiles']:
        cfg=Config(**spec['config']);end=spec['fall_end'];ref=Config(people=80,duration=cfg.duration,window=240)
        link=dict(candidate=[],reference=[],fall_end=end)
        for i in range(12):
            field=fs[i%6];link['candidate'].append(add(field,i,cfg,end));link['reference'].append(add(field,i,ref,end))
        links[spec['id']]=link
    # True fixed-event assertions for factorial windows/N and one-unit neighbors.
    for i in range(12):
        for pattern in ('uniform','three_bursts'):
            whole=cases[links[f'n120_w240_{pattern}']['candidate'][i]]['scenario']['events'];lookup={e['exposure_id']:e for e in whole}
            for spec in p['profiles']:
                cfg=spec['config']
                if spec['kind'] not in ('factorial','one_unit_neighbor') or cfg['fall_pattern']!=pattern:continue
                candidate=cases[links[spec['id']]['candidate'][i]]['scenario']['events']
                for e in candidate:assert all(e[k]==lookup[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
    write(out/'scenarios.json',dict(cases=cases,links=links))
    write(out/'preparation.json',dict(profile_count=len(p['profiles']),unique_cases=len(cases),rollouts=2*len(cases),elapsed_s=time.monotonic()-start,
        scenario_sha256=sha(out/'scenarios.json'),protocol_sha256=sha(out/'protocol.json')))
    write(out/'status.json',dict(status='prepared',pid=os.getpid(),rollouts=2*len(cases)))
    print(json.dumps(read(out/'preparation.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.prepare==a.execute:p.error('Exactly one mode required')
    prepare(a) if a.prepare else execute(a)
