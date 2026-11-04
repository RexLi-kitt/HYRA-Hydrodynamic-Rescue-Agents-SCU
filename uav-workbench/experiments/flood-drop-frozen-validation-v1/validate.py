"""Frozen primary validation: fresh parents, paired controls, orientation stress."""
from pathlib import Path
import sys,importlib.util,copy,json,time,traceback,argparse,contextlib,platform
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;EXP=HERE.parent;REPO=HERE.parents[2]
spec=importlib.util.spec_from_file_location('flow_fog',EXP/'flood-drop-flow-fog-v1/experiment.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
sys.path.insert(0,str(EXP/'flood1km-six-20261101'))
from flood1km.data import generate
METHODS=['coverage100','flow_coverage','gru-32_flowblend']

def xy_transform(x,kind):
    a=np.asarray(x,float);b=a.copy()
    if kind=='rotate90':b[...,0]=a[...,1];b[...,1]=1000-a[...,0]
    elif kind=='mirror_x':b[...,0]=1000-a[...,0]
    else:raise ValueError(kind)
    return b

def vector_transform(x,kind):
    a=np.asarray(x);b=a.copy()
    if kind=='rotate90':b[...,0]=a[...,1];b[...,1]=-a[...,0]
    elif kind=='mirror_x':b[...,0]=-a[...,0]
    else:raise ValueError(kind)
    return b

def transformed(water,scenario,kind):
    a=(np.rot90(water,1,axes=(-2,-1)) if kind=='rotate90' else water[...,::-1]).copy()
    uv=np.moveaxis(a[:,1:3],1,-1);a[:,1:3]=np.moveaxis(vector_transform(uv,kind),-1,1)
    s=copy.deepcopy(scenario)
    s['traps']=xy_transform(s['traps'],kind).tolist()
    for event in s['events']:
        event['xy']=xy_transform(event['xy'],kind).tolist()
        event['field'][1:]=vector_transform(event['field'][1:],kind).tolist()
    assert np.isfinite(a).all() and (a[:,0]>=0).all()
    np.testing.assert_allclose(a[:,0].sum((1,2),dtype=np.float64),water[:,0].sum((1,2),dtype=np.float64),atol=1e-8,rtol=1e-12)
    return a,s

def statistics(rows,seed=20261103):
    result={m:f.aggregate([r for r in rows if r['method']==m]) for m in METHODS}
    parents=sorted({r['parent'] for r in rows});metrics=['people_rate','all_rate','p95_rate'];values=np.zeros((len(parents),3,3))
    for i,parent in enumerate(parents):
        for j,m in enumerate(METHODS):
            rr=[r for r in rows if r['parent']==parent and r['method']==m];a=f.aggregate(rr)
            values[i,j]=[a[k] for k in metrics]
    rng=np.random.default_rng(seed);ids=rng.integers(0,len(parents),(10000,len(parents)))
    boot=values[ids].mean(1);gap=boot[:,2]-np.maximum(boot[:,0],boot[:,1]);estimate=values.mean(0)
    intervals={m:{k:np.quantile(boot[:,j,i],[.025,.975]).tolist() for i,k in enumerate(metrics)} for j,m in enumerate(METHODS)}
    gaps={k:dict(estimate=float(estimate[2,i]-max(estimate[0,i],estimate[1,i])),ci95=np.quantile(gap[:,i],[.025,.975]).tolist()) for i,k in enumerate(metrics)}
    a=result[METHODS[2]];point=.85<=a['people_rate']<=.95 and gaps['people_rate']['estimate']>=0 and gaps['p95_rate']['estimate']>=.15
    supported=intervals[METHODS[2]]['people_rate'][0]>=.85 and intervals[METHODS[2]]['people_rate'][1]<=.95 and gaps['people_rate']['ci95'][0]>=0 and gaps['p95_rate']['ci95'][0]>=.15
    return dict(results=result,parent_count=len(parents),gaps=gaps,ci95=intervals,point_goal_met=point,bootstrap_goal_supported=supported,
        uncertainty='Paired parent-cluster percentile bootstrap10000; same parent scenes resampled together, stronger of both baselines inside each draw. Approximate sampling interval for one frozen model, no training-seed uncertainty; all-zero proportions can give degenerate intervals and are not certainty.')

def leakage_audit(e):
    a=copy.copy(e);a.history=copy.deepcopy(e.history);a.prediction=None;a.predicted_at=None
    b=copy.copy(a);b.history=copy.deepcopy(a.history)
    # Keep all observable memories fixed; change hidden current/future state only.
    b.person=np.full_like(e.person,777);b.fall=e.fall+93;b.deadline=e.deadline+117;b.status=np.full_like(e.status,3)
    b.water=np.full_like(e.water,12.);b.scenario=copy.deepcopy(e.scenario);b.scenario['events']=[]
    np.testing.assert_array_equal(f.features(a),f.features(b))
    x=a.choices();y=b.choices()
    for index in (0,1,3):np.testing.assert_array_equal(x[index],y[index])
    np.testing.assert_array_equal(a.prediction,b.prediction)
    for ra,rb in zip(x[2],y[2]):
        for oa,ob in zip(ra,rb):
            assert (oa is None)==(ob is None)
            if oa is not None:
                np.testing.assert_array_equal(oa['xy'],ob['xy']);assert oa['person']==ob['person'] and oa['utility']==ob['utility']
    return dict(time_s=e.t,public_features_equal=True,predictions_equal=True,available_actions_goals_utilities_and_selected_actions_equal=True)

def run(out,previous,weights):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();done=0;rows=[];torch.set_num_threads(4)
    old=f.read(previous/'summary.json');assert old['selected']['method']=='gru-32_flowblend' and old['selected']['window']==180
    template=f.read(previous/'confirmation-scenarios.json')[0]['scenario']['config'];assert template['step']==1 and template['sensor']==60 and template['window']==180 and template['people']==40
    sources={**f.read(previous/'protocol.json')['source_hashes'],str(Path(__file__)):f.sha(__file__)}
    protocol=dict(version='frozen-expanded-validation-v1',date='2026-11-03',config=template,methods=METHODS,primary='gru-32_flowblend',
        water_seeds=list(range(915001,915025)),event_seeds=list(range(2026110800,2026110848)),
        cases='24fresh hydraulic parents x2fresh layouts=48main; first6 parents layout0 each rotate90 andmirrorX=12stress. Total60scenarios x3methods=180rollouts. Main and stress separate; no exclusion/early success stop/tuning.',
        frozen='Same selectedGRU32 weights, normalization,50:50 coverage/prediction and now/60sforecast. 60m sensor,180s window,1s step,40people4UAV,T1500,fall_end960,pool120,10sdrop,2m/1mps,patch_fogFalse. No retraining or checkpoint selection.',
        independence='New hydraulic seeds after model/config frozen; same synthetic generator family. Orientation changes rotate water vectors, people and traps consistently; start-corner set invariant. Six stress parents shared with main, not independent newmaps.',
        target='Primary: people85..95%, no people regression vs BOTH fixed planners,>=15pp in >=38/40 delivered scenes. Full40 scenes separate. Report point estimates and parent-cluster95% intervals; stability supported only when person interval inside target, paired people lower>=0 and paired95%-scene lower>=15pp. Test outcome cannot change thresholds.',
        audit='3live checkpoints0/120/480s with memories fixed and hidden persons/fall/deadline/status/wholefuturewater corrupted; compare features,predictions,candidate utilities/goals,selected actions. Not a proof of all possible leaks. Source review supplements dynamic test.',
        statistical_method='10000 paired parent-cluster bootstrap,seed20261103,2.5/97.5 percentiles. Each draw recomputes maximum of two planning means per metric. Same family/single training seed, no realworld probability guarantee.',
        budget='One sequential CPU worker,Torch4threads,60min ceiling,STOPfile; no extra training. Save every scenario and persons. Abort on internal numeric or source integrity failure.',
        source_hashes=sources,checkpoint_sha=f.sha(weights/'gru-32.pt'),prior_selection_sha=f.sha(previous/'selection.json'),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__))
    f.write(out/'protocol.json',protocol)
    def status(state,**kw):f.write(out/'status.json',dict(status=state,elapsed_s=time.monotonic()-start,rollouts=done,planned_rollouts=180,**kw))
    def guard():
        if (out/'STOP').exists():raise InterruptedError('User STOP')
        if time.monotonic()-start>3600:raise TimeoutError('60minute cap')
    ck=torch.load(weights/'gru-32.pt',map_location='cpu',weights_only=True);model=f.Model(ck['kind'],ck['mean']);model.load_state_dict(ck['state']);model.eval()
    mu=np.asarray(ck['mu'],np.float32);sd=np.asarray(ck['sd'],np.float32)
    def predict(x):
        with torch.no_grad():return model(torch.tensor((x-mu)/sd)).numpy()
    try:
        for path,h in sources.items():assert f.sha(path)==h,path
        cases=[];fields=[];known_parents={c['field']['parent'] for c in f.read(previous/'train-scenarios.json')+f.read(previous/'confirmation-scenarios.json')}
        for i,seed in enumerate(protocol['water_seeds']):
            guard();status('generating_water',water_seed=seed,parent_index=i+1)
            with (out/'generation.log').open('a',encoding='utf8') as log,contextlib.redirect_stdout(log):field=generate(out/'fields',REPO/'terrain-workbench/terrain-lab',seed,'frozen-validation')
            assert field['parent'] not in known_parents and field['finite_nonnegative'];assert abs(field['mass_residual_m3'])<=max(1e-6,abs(field['source_volume_m3'])*1e-9)
            fields.append(field);water=np.load(field['path'],mmap_mode='r')
            for j in range(2):
                guard();status('generating_roster',water_seed=seed,event_seed=2026110800+2*i+j)
                s=f.make_scenario(water,2026110800+2*i+j,f.Config(**template),fall_end=960,pool_size=120);s['patch_fog']=False
                cases.append(dict(id=f'water{seed}-event{j}',panel='main',parent=field['parent'],field=field,scenario=s))
                if i<6 and j==0:
                    for kind in ('rotate90','mirror_x'):
                        changed,ss=transformed(water,s,kind);path=out/'fields'/f'{seed}-{kind}.npy';np.save(path,changed);del changed
                        fm=dict(parent=field['parent']+'-'+kind,path=str(path.resolve()),sha256=f.sha(path),original_sha=field['sha256'],coordinate_transform=kind)
                        cases.append(dict(id=f'water{seed}-event{j}-{kind}',panel=kind,parent=field['parent'],field=fm,scenario=ss))
            f.write(out/'scenarios.json',cases);f.write(out/'fields.json',fields)
        assert len(cases)==60 and sum(c['panel']=='main' for c in cases)==48
        f.write(out/'scenario-lock.json',dict(sha256=f.sha(out/'scenarios.json'),locked_before_evaluation=True))
        audits=[]
        for method in METHODS:
            for index,c in enumerate(cases):
                guard();status('evaluating',method=method,panel=c['panel'],case=c['id'])
                e=f.FlowFogEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],method,predict if method==METHODS[2] else None)
                while True:
                    if method==METHODS[2] and index==0 and e.t in (0,120,480):audits.append(leakage_audit(e));f.write(out/'leakage-audit.json',audits)
                    _,_,finished,_=e.step(e.choices()[1])
                    if finished:break
                persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(40)]
                supplied=int((e.status==2).sum());expired=int((e.status==3).sum());outside=int((e.status==4).sum());assert supplied+expired+outside==40
                assert np.isfinite(e.pos).all() and np.isfinite(e.person).all()
                for p in persons:
                    if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                row=dict(method=method,panel=c['panel'],case=c['id'],parent=c['parent'],people=40,supplied=supplied,expired=expired,outside=outside,
                    unseen_expired=sum(p['status']==3 and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in persons),refloat_events=int(e.refloat_counts.sum()),persons=persons)
                rows.append(row);done+=1
                with (out/'results.jsonl').open('a',encoding='utf8') as fp:fp.write(json.dumps(row,allow_nan=False)+'\n')
        assert len(audits)==3
        stats={panel:statistics([r for r in rows if r['panel']==panel]) for panel in ('main','rotate90','mirror_x')}
        stats['orientation_combined']=statistics([r for r in rows if r['panel']!='main'])
        paired=[]
        for c in cases:
            matching={r['method']:r for r in rows if r['case']==c['id']}
            for baseline in METHODS[:2]:
                a=matching[METHODS[2]];b=matching[baseline]
                paired.append(dict(case=c['id'],parent=c['parent'],panel=c['panel'],baseline=baseline,people_delta=a['supplied']-b['supplied'],all_delta=int(a['supplied']==40)-int(b['supplied']==40),p95_delta=int(a['supplied']>=38)-int(b['supplied']>=38)))
        f.write(out/'paired-results.json',paired)
        for path,h in {**sources,**{c['field']['path']:c['field']['sha256'] for c in cases}}.items():assert f.sha(path)==h,path
        assert f.sha(weights/'gru-32.pt')==protocol['checkpoint_sha'];assert f.sha(out/'scenarios.json')==f.read(out/'scenario-lock.json')['sha256']
        f.write(out/'summary.json',dict(status='completed',statistics=stats,rollouts=done,new_training_epochs=0,leakage_audits_passed=3,source_checkpoint_scenario_water_hashes_unchanged=True))
        status('completed',main_point_goal_met=stats['main']['point_goal_met'],main_interval_supported=stats['main']['bootstrap_goal_supported'],orientation_point_goal_met=stats['orientation_combined']['point_goal_met'])
    except BaseException as ex:
        f.write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args();run(a.output,a.previous,a.weights)
