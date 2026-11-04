"""Frozen stage-0 calibration pilot; no training, no independent test claims."""
from pathlib import Path
import sys,json,copy,time,hashlib,traceback
import numpy as np
import torch
from adapter import ResidualEnv,FlowFogEnv
from models import DemandModel,maps
from history_model import features

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2];ROOT=REPO.parent
OLD=ROOT/'uav-convlstm20-20261104/run01'
CONV=ROOT/'uav-convlstm20-continue-20261104/run01'
OUT=ROOT/'uav-search-redesign-20261104/run01'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return [json.loads(l) for l in Path(p).read_text(encoding='utf-8-sig').splitlines() if l.strip()]
def restore(p):
    ck=torch.load(p,map_location='cpu',weights_only=True)
    m=DemandModel(ck['kind'],ck['mean']);m.load_state_dict(ck['state']);m.eval()
    mu=np.asarray(ck['mu'],np.float32);sd=np.asarray(ck['sd'],np.float32)
    def pred(x):
        with torch.inference_mode():
            z=(maps(x)-mu[None,:,None,None])/sd[None,:,None,None]
            return m(torch.from_numpy(z)[None])[0].numpy().reshape(-1)
    return pred

def selected(e):
    c=e.choices()
    return np.array([c[2][i][a]['xy'] for i,a in enumerate(c[1])])
def rotated_public(e,reindex=False):
    z=copy.copy(e)
    xy=lambda a:np.stack([a[...,1],1000-a[...,0]],axis=-1)
    vv=lambda a:np.stack([a[...,1],-a[...,0]],axis=-1)
    z.pos=xy(e.pos);z.seen_xy=xy(e.seen_xy);z.seen_v=vv(e.seen_v)
    z.known=np.rot90(e.known).copy();z.last=np.rot90(e.last).copy()
    z.measured=np.rot90(e.measured,1,(-2,-1)).copy()
    z.measured[1:]=np.stack([z.measured[2].copy(),-z.measured[1].copy()])
    if reindex:z.pos=z.pos[[1,3,0,2]]
    return z
def rotate_diagnostic(e):
    original=selected(e);expect=np.stack([original[:,1],1000-original[:,0]],axis=-1)
    a=selected(rotated_public(e));b=selected(rotated_public(e,True))
    return dict(time=e.t,kind='public-state one-decision diagnostic, not rotated full trajectory',
        fixed_ids_goal_errors_m=np.linalg.norm(a-expect,axis=1).tolist(),
        canonical_reindexed_ids_goal_errors_m=np.linalg.norm(b-expect[[1,3,0,2]],axis=1).tolist(),
        limitation='Reindex changes greedy priority as well as preferred quadrant; not isolated causal attribution.')
def leak(e):
    a=copy.copy(e);a.history=copy.deepcopy(e.history);a.prediction=None;a.predicted_at=None
    b=copy.copy(a);b.history=copy.deepcopy(a.history)
    b.person=np.full_like(e.person,888.);b.status=np.full_like(e.status,3)
    b.fall=e.fall+123;b.deadline=e.deadline+456;b.water=None
    b.scenario=copy.deepcopy(e.scenario);b.scenario['events']=[]
    np.testing.assert_array_equal(features(a),features(b))
    aa=a.choices();bb=b.choices()
    for k in (0,1,3):np.testing.assert_array_equal(aa[k],bb[k])
    np.testing.assert_array_equal(selected(a),selected(b))
    return dict(t=e.t,hidden_truth_and_water_not_read=True)

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    start=time.monotonic();done=0
    cases=read(OLD/'scenarios.json')
    models={'gru32':OLD/'gru-32.pt','conv48':CONV/'convlstm-48.pt'}
    sources={str(p):sha(p) for p in HERE.glob('*.py')}
    sources.update({str(p):sha(p) for p in (HERE.parent/'flood-drop-convlstm20-v1').glob('*.py')})
    inputs={str(p):sha(p) for p in [*models.values(),OLD/'scenarios.json',OLD/'results.jsonl',CONV/'results.jsonl']}
    fields={c['field']['path']:sha(c['field']['path']) for c in [cases['fit'][0],*cases['calibration']]}
    write(OUT/'protocol.json',dict(date='2026-11-04',stage='0 integration calibration only',
       scenario_config=read(OLD/'protocol.json')['config'],
       source_sha256=sources,input_sha256=inputs,field_sha256=fields,
       models={k:str(v) for k,v in models.items()},alphas=[.1,.25],
       cases=[c['id'] for c in cases['calibration']],
       methods='Two frozen models x2bounded amplitudes x4original calibration=16new runs. Original baseline/legacy rows reused with hashes.',
       selection='Report all variants; optional next-stage CAL nomination people first,95%-scene then loweralpha. No finaltest and no autoextension.',
       contract='One FIT scene full lockstep: originalflow vs alpha0 and constant prediction alpha.25; exact choices,physicalstate,personobservations eachstep.',
       rotation='At t0/120/480/1200 public-state decision transform, fixed IDs and canonical reindex. Not claiming fulltrajectoryequivariance.',
       leakage='Changing hiddenpeople/status/fall/deadline/futurewater must not change public features/choices.',
       stop='STOP file or1200s,1CPUjob; completed rows retained. No change to old environment/weights/defaults.'))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1200:raise TimeoutError('1200s cap')
    def status(s,**kw):write(OUT/'status.json',dict(state=s,completed=done,planned=16,elapsed_s=time.monotonic()-start,**kw))
    try:
        status('contract')
        c=cases['fit'][0];water=np.load(c['field']['path'],mmap_mode='r');scenario=c['scenario']
        a=FlowFogEnv(water,scenario,'flow_coverage')
        b=ResidualEnv(water,scenario,lambda x:np.arange(100),alpha=0.)
        n=ResidualEnv(water,scenario,lambda x:np.ones(100),alpha=.25)
        rotations=[];checks=0;h=hashlib.sha256()
        while True:
            guard()
            ac=a.choices()
            for e in (b,n):
                ec=e.choices()
                for k in (0,1,3):np.testing.assert_array_equal(ac[k],ec[k])
                np.testing.assert_array_equal(selected(a),selected(e))
            if a.t in (0,120,480,1200):rotations.append(rotate_diagnostic(a))
            da=a.step(ac[1])[2];db=b.step(b.choices()[1])[2];dn=n.step(n.choices()[1])[2]
            assert da==db==dn
            for e in (b,n):
                for field in ('pos','person','status','first_seen','delivered_at','known','last','seen_xy','seen_at','refloat_counts'):
                    np.testing.assert_array_equal(getattr(a,field),getattr(e,field))
            h.update(a.pos.tobytes());h.update(a.status.tobytes());checks+=1
            if da:break
        write(OUT/'integration-contract.json',dict(passed=True,case=c['id'],checked_steps=checks,
             zero_and_constant_exact=True,trajectory_hash=h.hexdigest(),baseline_supplied=int((a.status==2).sum())))
        write(OUT/'rotation-diagnostic.json',rotations)
        # Reuse all calibration controls, never silently mix fresh-test panels.
        original=rows(OLD/'results.jsonl');continued=rows(CONV/'results.jsonl')
        controls=[r for r in original if r['phase']=='calibration' and r['method'] in ('coverage100','flow_coverage','new_gru-32','gru-32')]
        controls += [r for r in continued if r['phase']=='calibration' and r['method']=='convlstm-48']
        write(OUT/'reused-controls.json',controls)
        predictors={k:restore(p) for k,p in models.items()};results=[];audits=[]
        for kind,predict in predictors.items():
            for alpha in (.1,.25):
                for c in cases['calibration']:
                    guard();status('evaluating',method=kind,alpha=alpha,case=c['id'])
                    w=np.load(c['field']['path'],mmap_mode='r')
                    e=ResidualEnv(w,c['scenario'],predict,alpha)
                    while True:
                        if e.t%120==0:guard()
                        if c['id']=='calibration-0-0' and alpha==.1 and e.t in (0,120,480):audits.append(dict(model=kind,**leak(e)))
                        if e.step(e.choices()[1])[2]:break
                    persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),
                        first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]))
                        for j in range(e.cfg.people)]
                    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                    for p in persons:
                        if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                    r=dict(phase='reused_calibration',case=c['id'],parent=c['field']['parent'],method=kind,alpha=alpha,
                        supplied=int((e.status==2).sum()),people=e.cfg.people,persons=persons,refloats=int(e.refloat_counts.sum()),
                        failed_not_seen_before_deadline=sum(p['status']!=2 and (p['first_seen']<0 or p['first_seen']>=p['deadline']) for p in persons))
                    results.append(r)
                    with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
                    done+=1
        write(OUT/'leakage-audit.json',audits)
        stats=[]
        for kind in models:
            for alpha in (.1,.25):
                rr=[r for r in results if r['method']==kind and r['alpha']==alpha]
                stats.append(dict(model=kind,alpha=alpha,people=sum(r['people'] for r in rr),
                    supplied=sum(r['supplied'] for r in rr),all_scenes=sum(r['supplied']==r['people'] for r in rr),
                    p95_scenes=sum(r['supplied']>=.95*r['people'] for r in rr),
                    failed_not_seen_before_deadline=sum(r['failed_not_seen_before_deadline'] for r in rr)))
        for p,v in {**sources,**inputs,**fields}.items():assert sha(p)==v,p
        write(OUT/'summary.json',dict(state='completed',statistics=stats,source_inputs_unchanged=True,
            limitation='4previously reused calibration parents only, no new learning, no generalization conclusion, no productionchange.'))
        status('completed')
        print(json.dumps(stats))
    except BaseException:
        write(OUT/'failure.json',dict(trace=traceback.format_exc()))
        status('failed');raise
if __name__=='__main__':main()

