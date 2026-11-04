"""Frozen two-candidate confirmation on newly generated independent hydraulic seeds."""
from pathlib import Path
import sys,importlib.util,json,time,traceback,copy
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('flow_fog',EXP/'flood-drop-flow-fog-v1/experiment.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
sys.path.insert(0,str(EXP/'flood1km-six-20261101'))
from flood1km.data import generate

def run(out,weights,previous):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();done=0;rows=[];torch.set_num_threads(4)
    def status(state,**kw):f.write(out/'status.json',dict(status=state,elapsed_s=time.monotonic()-start,rollouts=done,**kw))
    def guard():
        if (out/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-start>1800:raise TimeoutError('30min ceiling')
    sources={str(Path(__file__)):f.sha(__file__),str(EXP/'flood1km-six-20261101/flood1km/data.py'):f.sha(EXP/'flood1km-six-20261101/flood1km/data.py')}
    sources.update(f.read(previous/'protocol.json')['source_hashes'])
    sources.update({str(p):f.sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    methods=['coverage100','flow_coverage','gru-32_flowblend','gru-16']
    protocol=dict(version='new-hydraulic-seeds-confirmation-v1',water_seeds=list(range(813001,813007)),event_seeds=list(range(2026110500,2026110512)),
        methods=methods,primary='gru-32_flowblend',secondary='gru-16',
        selection='Primary fixed by preceding TRAIN selection. SecondaryGRU16 observed strongest in preceding fresh-event DEV, so that cohort is now developmental for this secondary. Both frozen before generating these six new waterfields; no selection or parameter changes on this cohort.',
        config=dict(people=40,uavs=4,duration=1500,window=360,sensor=60,drop_seconds=10,trigger_depth=2,trigger_speed=1,fall_end=960,pool_size=120,patch_fog=False),
        target='Each predeclared candidate separately:85..95% people, >=15pp at-least95%-people scene advantage vs BOTH nonneural baselines, no people regression. Allpeople gap reported separately. Do not switch metric after observing results; two candidates and12cases exploratory uncertainty acknowledged.',
        independence='Six newly generated seed parents absent from previousTRAIN/DEV; same synthetic terrain/forcing generator family, not real floods or unseen generator class. 2 event seeds each, 12 scenes not independent of shared waterparent. No retraining.',
        source_hashes=sources,weights={str(weights/(n+'.pt')):f.sha(weights/(n+'.pt')) for n in ('gru-16','gru-32')},
        budget='6 new synthetic waterfields sequential then12rosters and48rollouts; CPU one worker; no additional RL/supervised epochs;STOPfile/30min cap.')
    f.write(out/'protocol.json',protocol)
    try:
        predictors={}
        for n in ('gru-16','gru-32'):
            ck=torch.load(weights/(n+'.pt'),weights_only=True,map_location='cpu');m=f.Model(ck['kind'],ck['mean']);m.load_state_dict(ck['state']);m.eval();mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
            def pred(x,m=m,mu=mu,sd=sd):
                with torch.no_grad():return m(torch.tensor((x-mu)/sd)).numpy()
            predictors[n]=pred
        prior=f.read(previous/'validation-scenarios.json')[0]['scenario']['config'];cases=[];fields=[]
        for i,seed in enumerate(protocol['water_seeds']):
            guard();status('generating_water',seed=seed)
            field=generate(out/'fields',REPO/'terrain-workbench/terrain-lab',seed,'new-map-confirmation');fields.append(field)
            assert field['finite_nonnegative'];assert abs(field['mass_residual_m3'])<=max(1e-6,abs(field['source_volume_m3'])*1e-9)
            water=np.load(field['path'],mmap_mode='r')
            for j in range(2):
                guard();status('generating_roster',water_seed=seed,event_seed=2026110500+2*i+j)
                s=f.make_scenario(water,2026110500+2*i+j,f.Config(**prior),fall_end=960,pool_size=120);s['patch_fog']=False
                cases.append(dict(id=f'water{seed}-event{j}',field=field,scenario=s))
            f.write(out/'fields.json',fields);f.write(out/'scenarios.json',cases)
        assert len(cases)==12
        assert not {c['field']['parent'] for c in cases}&{c['field']['parent'] for c in f.read(previous/'train-scenarios.json')+f.read(previous/'validation-scenarios.json')}
        for method in methods:
            for c in cases:
                guard();status('evaluating',method=method,case=c['id'])
                e=f.FlowFogEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],method,predictors.get(method.removesuffix('_flowblend')))
                while True:
                    _,_,finished,_=e.step(e.choices()[1])
                    if finished:break
                persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
                supplied=int((e.status==2).sum());expired=int((e.status==3).sum());outside=int((e.status==4).sum())
                assert supplied+expired+outside==40 and np.isfinite(e.pos).all() and np.isfinite(e.person).all()
                for p in persons:
                    if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                row=dict(method=method,case=c['id'],parent=c['field']['parent'],people=40,supplied=supplied,expired=expired,outside=outside,
                    unseen_expired=sum(p['status']==3 and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in persons),refloat_events=int(e.refloat_counts.sum()),persons=persons)
                rows.append(row);done+=1
                with (out/'results.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(row,allow_nan=False)+'\n')
        result={m:f.aggregate([r for r in rows if r['method']==m]) for m in methods};goals={}
        bp=max(result[m]['people_rate'] for m in methods[:2]);bs=max(result[m]['p95_rate'] for m in methods[:2]);ba=max(result[m]['all_rate'] for m in methods[:2])
        for m in methods[2:]:
            a=result[m];goals[m]=dict(people_gap=a['people_rate']-bp,p95_gap=a['p95_rate']-bs,all_gap=a['all_rate']-ba,
                goal_met=.85<=a['people_rate']<=.95 and a['people_rate']>=bp and a['p95_rate']-bs>=.15)
        # Paired differences and parent clustering retained; do not infer significance from12 scenes.
        paired={}
        for m in methods[2:]:
            paired[m]={b:[dict(case=c['id'],people_delta=next(r['supplied'] for r in rows if r['method']==m and r['case']==c['id'])-next(r['supplied'] for r in rows if r['method']==b and r['case']==c['id'])) for c in cases] for b in methods[:2]}
        for p,h in {**sources,**protocol['weights'],**{field['path']:field['sha256'] for field in fields}}.items():assert f.sha(p)==h,p
        f.write(out/'summary.json',dict(status='completed',results=result,goals=goals,paired=paired,water_parents=6,scenes=12,rollouts=done,new_training_epochs=0,source_hashes_unchanged=True))
        status('completed',primary_goal_met=goals['gru-32_flowblend']['goal_met'],secondary_goal_met=goals['gru-16']['goal_met'])
    except BaseException as ex:
        f.write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.weights,a.previous)
