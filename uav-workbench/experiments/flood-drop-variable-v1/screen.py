"""Preregistered variable-load diagnostic; never pick difficulty by learned gap."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,copy,hashlib,json,os,time,traceback
import numpy as np
from variable_environment import Config,Env,make_scenario
from variable_dispatch import DispatchEnv

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n');os.replace(tmp,p)
def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

CATALOG={
 'people':dict(unit='人',step=1,range=[1,200],note='N actual falls; changing N is load, not a change to task success'),
 'duration':dict(unit='秒',step=1,range=[20,1500],note='T > window; longer needs new hydraulic fields'),
 'window':dict(unit='秒',step=1,range=[1,1499],note='drop_seconds <= window < duration; never truncate late people'),
 'uavs':dict(unit='架',step=1,range=[1,8],note='All methods share team and initial perimeter positions'),
 'speed':dict(unit='米/秒',step=1,range=[1,50]),
 'sensor':dict(unit='米',step=1,range=[1,1000]),
 'drop_seconds':dict(unit='秒',step=1,range=[1,360]),
 'drop_radius':dict(unit='米',step=1,range=[1,100]),
 'step':dict(unit='秒',step=1,range=[1,30],note='Decision interval; internal drift/service remains1s'),
 'separation':dict(unit='米',step=1,range=[1,100]),
 'traps':dict(unit='处',step=1,range=[0,200],note='Synthetic permeable people-trapping patches, not urban building evidence'),
 'trap_radius':dict(unit='米',step=1,range=[1,50]),
 'trigger_depth':dict(unit='米',step=.01,range=[.01,2],note='Synthetic exposure trigger, not human safety threshold'),
 'trigger_speed':dict(unit='米/秒',step=.01,range=[.01,3],note='Synthetic exposure trigger, not human safety threshold'),
 'random_per_minute':dict(unit='概率/分钟',step=.01,range=[.01,.99],note='Given N falls, this changes accepted event distribution, not N'),
 'fall_pattern':dict(values=['uniform','three_bursts']),
 'burst_width':dict(unit='秒',step=1,range=[1,1200],note='Only active for three_bursts'),
 'spatial_clusters':dict(unit='簇',step=1,range=[0,20],note='0 means uniform wet-cell exposure'),
 'cluster_radius':dict(unit='米',step=1,range=[1,500],note='Only active when spatial_clusters>0'),
 'wet_depth':dict(unit='米',fixed=.01,note='Numerical wet/dry diagnostic, excluded from difficulty tuning'),
}

def profiles():
    base=Config(people=40)
    out=[dict(id='reference40',change={},config=asdict(base),kind='reference')]
    settings=[('people',[20,60,80]),('duration',[900,1200]),('window',[180,240,480]),
       ('uavs',[2,6]),('speed',[6.,12.]),('sensor',[50.,75.,125.]),
       ('drop_seconds',[5,15,20]),('drop_radius',[5.,15.]),('step',[1,10]),('separation',[10.]),
       ('traps',[0,24]),('trap_radius',[8.]),('trigger_depth',[.20]),('trigger_speed',[.40]),
       ('random_per_minute',[.01,.15]),('fall_pattern',['three_bursts']),('spatial_clusters',[2])]
    for key,values in settings:
        for v in values:
            c=replace(base,**{key:v});out.append(dict(id=f'{key}_{v}',change={key:v},config=asdict(c),kind='single_factor'))
    for name,changes in [('burst15',dict(fall_pattern='three_bursts',burst_width=15)),
                         ('clusters50',dict(spatial_clusters=2,cluster_radius=50.)),
                         ('integer_example',dict(people=41,duration=1499,window=359,drop_seconds=11))]:
        out.append(dict(id=name,change=changes,config=asdict(replace(base,**changes)),kind='joint_demonstration'))
    return out

def summarize(rows):
    total=sum(r['people_total'] for r in rows);people=sum(r['supplied'] for r in rows)
    return dict(cases=len(rows),people_supplied=people,people_total=total,people_completion=people/total,
        all_successes=sum(r['scene_success'] for r in rows),all_rate=sum(r['scene_success'] for r in rows)/len(rows),
        at_least95_successes=sum(r['scene95_success'] for r in rows),at_least95_rate=sum(r['scene95_success'] for r in rows)/len(rows),
        missed=total-people,expired_unseen=sum(r['expired_unseen'] for r in rows),conflicts=sum(r['conflicts'] for r in rows))

def run_case(case,method):
    cls=Env if method=='original' else DispatchEnv
    e=cls(np.load(case['field']['path'],mmap_mode='r',allow_pickle=False),case['scenario']);start=time.monotonic()
    while True:
        act=e.control()[1] if method=='original' else e.choices()[1]
        _,_,done,_=e.step(act)
        if done:break
    m=e.metrics();assert m['supplied']+m['expired']+m['outside']==m['people_total']
    persons=[dict(id=j,exposure_id=case['scenario']['events'][j]['exposure_id'],fall=int(e.fall[j]),deadline=int(e.deadline[j]),
                  first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
    for person in persons:
        if person['status']==2:assert person['fall']+e.cfg.drop_seconds<=person['supplied_at']<=person['deadline']
    return dict(**m,case=case['id'],parent=case['field']['parent'],method=method,elapsed_s=time.monotonic()-start,persons=persons)

def prepare(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);begin=time.monotonic();specs=profiles()
    previous=read(a.previous/'protocol.json');oldcases=read(a.previous/'scenarios.json')['development']
    fields={c['field']['parent']:c['field'] for c in oldcases};assert len(fields)==6
    for f in fields.values():
        assert sha(f['path'])==f['sha256'] and f['finite_nonnegative']
        assert abs(f['mass_residual_m3'])<=max(1e-6,abs(f['source_volume_m3'])*1e-9)
    oldhash={k:v for k,v in previous['source_hashes'].items()}
    for path,h in oldhash.items():assert sha(path)==h
    protocol=dict(version='variable-load-screen-v1',date='2026-11-02',purpose='Parameter implementation and planning capacity, NOT learned superiority',
        user_amendment='People can exceed20; time and other variables adjustable, mostly integer increments.',
        primary='Within identical configuration maximize people supplied; ties all-person then at-least95%. Do not rank different N by absolute totals alone.',
        threshold95='ceil(0.95*N), computed as(95*N+99)//100',catalog=CATALOG,profiles=specs,
        methods=['original','dispatch'],parents=list(fields),layouts_per_parent=1,seed_base=1521000,
        nested_population='Common80-event pool; select first N accepted exposures, independently shuffle public IDs with stable relative order.',
        pairing='For each changed T/L compare to an additional reference with SAME fall support min(T-L,1500-360); no dropped or truncated targets. Other single-factor cases share exact events unless trigger/distribution changes.',
        distribution_changes=['trigger_depth','trigger_speed','random_per_minute','fall_pattern','burst_width','spatial_clusters','cluster_radius'],
        population_risk='N actual falls is conditioned; changing trigger does not estimate natural fall incidence.',
        early_stop='At case boundaries: STOP_AFTER_CASE or3600-second execution cap; never silently drop a failed profile.',
        bounds='Frozen 1km/5s sampled hydraulic fields through1500s. Interpolation permits1s configuration; no future truth in policy. No rescaled flow or extended last-frame flooding.',
        neural='Old20-person/4-UAV networks cannot be scored at changed observation/team dimensions. No neural evaluation or training in this diagnostic; do not inherit old scores.',
        selection='No winning-difficulty selection and no automatic retraining. Report every profile and both planning baselines, using stronger fixed planner for comparisons.',
        fine_adjustment='1-person/1-second supported. This screen is preregistered coarse sampling, not exhaustive Cartesian or1-unit search.',
        limitations=['6 reused development hydraulic parents,one new layout per parent; not final test','no multi-seed learned comparison','instant perfect sensing/sharing retained; battery/weather/building navigation not newly invented'],
        source_hashes={str(f):sha(f) for f in HERE.glob('*.py')},legacy_source_hashes=oldhash,water_hashes={f['path']:f['sha256'] for f in fields.values()})
    write(out/'protocol.json',protocol);write(out/'status.json',dict(status='preparing',pid=os.getpid(),profiles=len(specs)))
    cases={};links={};cache={};gen_keys=('trigger_depth','trigger_speed','random_per_minute','traps','trap_radius','fall_pattern','burst_width','spatial_clusters','cluster_radius')
    def scenario(field,index,cfg,end):
        key=(field['sha256'],index,end,*(getattr(cfg,k) for k in gen_keys))
        if key not in cache:
            water=np.load(field['path'],mmap_mode='r',allow_pickle=False)
            generator=replace(Config(),people=80,window=1,drop_seconds=1,**{k:getattr(cfg,k) for k in gen_keys})
            cache[key]=make_scenario(water,1521000+index,generator,fall_end=end,pool_size=80)
        s=copy.deepcopy(cache[key]);chosen=set(sorted(e['exposure_id'] for e in s['events'])[:cfg.people])
        s['events']=[e for e in s['events'] if e['exposure_id'] in chosen];s['config']=asdict(cfg)
        for e in s['events']:e['deadline']=e['fall']+cfg.window
        assert len(s['events'])==cfg.people and all(e['deadline']<=cfg.duration for e in s['events'])
        return s
    def add(field,index,cfg,end):
        s=scenario(field,index,cfg,end);key=fingerprint(dict(parent=field['parent'],scenario=s))
        cases.setdefault(key,dict(id=key,field=field,scenario=s));return key
    base=Config(people=40)
    for spec in specs:
        cfg=Config(**spec['config']);end=min(cfg.duration-cfg.window,base.duration-base.window)
        links[spec['id']]=dict(candidate=[],reference=[],fall_end=end)
        for i,field in enumerate(fields.values()):
            cid=add(field,i,cfg,end);rid=add(field,i,base,end)
            links[spec['id']]['candidate'].append(cid);links[spec['id']]['reference'].append(rid)
            ce=cases[cid]['scenario']['events'];re=cases[rid]['scenario']['events']
            if not set(spec['change'])&set(protocol['distribution_changes']):
                common={e['exposure_id']:e for e in re}
                for e in ce:
                    if e['exposure_id'] in common:
                        assert all(e[k]==common[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
        write(out/'status.json',dict(status='preparing',pid=os.getpid(),profile=spec['id'],unique_cases=len(cases)))
    write(out/'scenarios.json',dict(cases=cases,links=links))
    write(out/'preparation.json',dict(elapsed_s=time.monotonic()-begin,profile_count=len(specs),unique_cases=len(cases),rollouts=2*len(cases),
          scenario_sha256=sha(out/'scenarios.json'),protocol_sha256=sha(out/'protocol.json')))
    write(out/'status.json',dict(status='prepared',pid=os.getpid(),rollouts=2*len(cases)))
    print(json.dumps(read(out/'preparation.json')),flush=True)

def execute(a):
    out=a.output;p=read(out/'protocol.json');s=read(out/'scenarios.json');prep=read(out/'preparation.json');start=time.monotonic()
    assert read(out/'status.json')['status']=='prepared' and not (out/'results.jsonl').exists()
    assert sha(out/'protocol.json')==prep['protocol_sha256'] and sha(out/'scenarios.json')==prep['scenario_sha256']
    for path,h in {**p['source_hashes'],**p['legacy_source_hashes'],**p['water_hashes']}.items():assert sha(path)==h
    rows=[]
    try:
        for cid,case in s['cases'].items():
            for method in p['methods']:
                if (out/'STOP_AFTER_CASE').exists():raise InterruptedError('User stop requested')
                if time.monotonic()-start>3600:raise TimeoutError('Screen runtime cap')
                write(out/'status.json',dict(status='running',pid=os.getpid(),completed=len(rows),total=prep['rollouts'],method=method,case=cid,elapsed_s=time.monotonic()-start))
                r=run_case(case,method);rows.append(r)
                with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                if len(rows)%24==0:print(json.dumps(dict(completed=len(rows),total=prep['rollouts'],elapsed_s=time.monotonic()-start)),flush=True)
        lookup={(r['case'],r['method']):r for r in rows};summaries={}
        for spec in p['profiles']:
            link=s['links'][spec['id']];data={}
            for side in ('candidate','reference'):
                data[side]={m:summarize([lookup[(c,m)] for c in link[side]]) for m in p['methods']}
                data[side]['stronger']=max(p['methods'],key=lambda m:(data[side][m]['people_supplied'],data[side][m]['all_successes'],data[side][m]['at_least95_successes']))
            summaries[spec['id']]=data
        for path,h in {**p['source_hashes'],**p['legacy_source_hashes'],**p['water_hashes']}.items():assert sha(path)==h
        write(out/'summary.json',dict(profiles=summaries,source_hashes_unchanged=True,neural_runs=0,formal_test=False))
        write(out/'status.json',dict(status='completed',pid=os.getpid(),completed=len(rows),total=prep['rollouts'],elapsed_s=time.monotonic()-start))
        print(json.dumps(read(out/'status.json')),flush=True)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));write(out/'status.json',dict(status='failed',completed=len(rows),error=repr(ex)));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.prepare==a.execute:p.error('Choose exactly one of prepare/execute')
    prepare(a) if a.prepare else execute(a)
