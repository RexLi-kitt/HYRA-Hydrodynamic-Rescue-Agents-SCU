"""Tune rescue window on TRAIN at shared1s; confirm on predeclared new seeds."""
from pathlib import Path
import sys,importlib.util,json,time,copy,traceback,argparse
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('flow_fog',EXP/'flood-drop-flow-fog-v1/experiment.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
sys.path.insert(0,str(EXP/'flood1km-six-20261101'))
from flood1km.data import generate
METHODS=['coverage100','flow_coverage','gru-16','gru-32_flowblend']

def run(out,weights,previous):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();done=0;rows=[];torch.set_num_threads(4)
    def status(s,**kw):f.write(out/'status.json',dict(status=s,elapsed_s=time.monotonic()-start,rollouts=done,**kw))
    def guard():
        if (out/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-start>1800:raise TimeoutError('30min ceiling')
    source={str(Path(__file__)):f.sha(__file__),**f.read(previous/'protocol.json')['source_hashes']}
    source[str(EXP/'flood1km-six-20261101/flood1km/data.py')]=f.sha(EXP/'flood1km-six-20261101/flood1km/data.py')
    source.update({str(p):f.sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    protocol=dict(version='fast-shared-window-train-tune-v1',methods=METHODS,windows_s=[180,240,300],sensor_m=60,step_s=1,duration_s=1500,people=40,
        reason='At common1s previous12newmap scenarios learned478/480 vscoverage466/480 and95%-scene+25pp; both over95% people. Reorganize taskurgency on TRAIN instead of crippling planner.',
        selection='Use same4TRAINcalibration-parent first40 rosters.48rollouts, no weightupdates. Goal85..95%people,no people regression vsboth planners,>=15pp95%-scene gap. Prefer goal met, nearest interval, higher people, higher95%gap then earlier window/method order. All raw candidates retained.',
        confirmation='Generate six new hydraulic seeds814001..814006 only after freeze; eventseeds2026110600..611,2perparent. Same generator family and physics, not realflood;48taskrollouts allfour methods, preselected learning candidate scored without test reselection.',
        fairness='Both planners same1s observation/decision; same persecond physical/service engine; same methods/weights/features/flowforecast as prior. Only common rescuewindow varied; no deleting failed people or changing truths. 10sdrop,2m/1mps triggers unchanged.',
        budget='48TRAIN+48newmapconfirmation=96rollouts,6waterfields,0new supervised/RL epochs;CPU4threads1worker,STOPfile,30min.',
        source_hashes=source,weights={str(weights/(n+'.pt')):f.sha(weights/(n+'.pt')) for n in ('gru-16','gru-32')},previous_scenarios_sha=f.sha(previous/'train-scenarios.json'))
    f.write(out/'protocol.json',protocol)
    models={}
    for name in ('gru-16','gru-32'):
        ck=torch.load(weights/(name+'.pt'),weights_only=True,map_location='cpu');m=f.Model(ck['kind'],ck['mean']);m.load_state_dict(ck['state']);m.eval();mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
        def pred(x,m=m,mu=mu,sd=sd):
            with torch.no_grad():return m(torch.tensor((x-mu)/sd)).numpy()
        models[name]=pred
    def evaluate(c,method,phase,w):
        nonlocal done
        guard();status(phase,method=method,case=c['id'],window=w)
        e=f.FlowFogEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],method,models.get(method.removesuffix('_flowblend')))
        while True:
            _,_,finished,_=e.step(e.choices()[1])
            if finished:break
        persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(40)]
        a=int((e.status==2).sum());b=int((e.status==3).sum());d=int((e.status==4).sum());assert a+b+d==40
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all()
        for p in persons:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        row=dict(phase=phase,window=w,case=c['id'],parent=c['field']['parent'],method=method,people=40,supplied=a,expired=b,outside=d,
            unseen_expired=sum(p['status']==3 and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in persons),refloat_events=int(e.refloat_counts.sum()),persons=persons)
        rows.append(row);done+=1
        with (out/'results.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(row,allow_nan=False)+'\n')
    try:
        base=[c for c in f.read(previous/'train-scenarios.json') if c['parameter']=='view60-w360'];assert len(base)==4
        train=[]
        for w in protocol['windows_s']:
            for original in base:
                c=copy.deepcopy(original);c['id']+=f'-step1-window{w}';c['window']=w;c['scenario']['config'].update(step=1,window=w)
                for event in c['scenario']['events']:event['deadline']=event['fall']+w
                train.append(c)
        f.write(out/'train-scenarios.json',train)
        for w in protocol['windows_s']:
            for method in METHODS:
                for c in [c for c in train if c['window']==w]:evaluate(c,method,'train_screen',w)
        sums={str(w):{m:f.aggregate([r for r in rows if r['window']==w and r['method']==m]) for m in METHODS} for w in protocol['windows_s']}
        candidates=[]
        for wi,w in enumerate(protocol['windows_s']):
            q=sums[str(w)];bp=max(q[m]['people_rate'] for m in METHODS[:2]);bs=max(q[m]['p95_rate'] for m in METHODS[:2])
            for mi,m in enumerate(METHODS[2:]):
                a=q[m];distance=max(.85-a['people_rate'],0,a['people_rate']-.95);gap=a['p95_rate']-bs
                met=distance==0 and a['people_rate']>=bp and gap>=.15
                candidates.append(dict(method=m,window=w,people_rate=a['people_rate'],people_gap=a['people_rate']-bp,p95_gap=gap,goal_met=met,rank=[int(met),-distance,a['people_rate'],gap,-wi,-mi]))
        chosen=max(candidates,key=lambda x:x['rank']);f.write(out/'train-summary.json',sums);f.write(out/'selection.json',dict(chosen=chosen,candidates=candidates,before_new_maps=True))
        cases=[];fields=[]
        for i,seed in enumerate(range(814001,814007)):
            guard();status('generating_water',water_seed=seed)
            field=generate(out/'fields',REPO/'terrain-workbench/terrain-lab',seed,'new-window-confirmation');fields.append(field)
            assert field['finite_nonnegative'];assert abs(field['mass_residual_m3'])<=max(1e-6,abs(field['source_volume_m3'])*1e-9)
            water=np.load(field['path'],mmap_mode='r');cfg=copy.deepcopy(base[0]['scenario']['config']);cfg.update(window=chosen['window'],step=1)
            for j in range(2):
                guard();status('generating_roster',water_seed=seed,event_seed=2026110600+2*i+j)
                scenario=f.make_scenario(water,2026110600+2*i+j,f.Config(**cfg),fall_end=960,pool_size=120);scenario['patch_fog']=False
                cases.append(dict(id=f'water{seed}-event{j}',field=field,scenario=scenario))
            f.write(out/'fields.json',fields);f.write(out/'confirmation-scenarios.json',cases)
        assert len(cases)==12 and not {c['field']['parent'] for c in cases}&{c['field']['parent'] for c in train}
        for method in METHODS:
            for c in cases:evaluate(c,method,'confirmation',chosen['window'])
        result={m:f.aggregate([r for r in rows if r['phase']=='confirmation' and r['method']==m]) for m in METHODS};a=result[chosen['method']]
        bp=max(result[m]['people_rate'] for m in METHODS[:2]);bs=max(result[m]['p95_rate'] for m in METHODS[:2]);ba=max(result[m]['all_rate'] for m in METHODS[:2])
        met=.85<=a['people_rate']<=.95 and a['people_rate']>=bp and a['p95_rate']-bs>=.15
        for p,h in {**source,**protocol['weights'],**{z['path']:z['sha256'] for z in fields}}.items():assert f.sha(p)==h,p
        f.write(out/'summary.json',dict(status='completed',selected=chosen,results=result,goal_met=met,people_gap=a['people_rate']-bp,p95_gap=a['p95_rate']-bs,all_gap=a['all_rate']-ba,
            rollouts=done,new_training_epochs=0,water_parents=6,scenes=12,source_hashes_unchanged=True))
        status('completed',goal_met=met)
    except BaseException as ex:
        f.write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.weights,a.previous)
