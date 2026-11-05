"""Single-factor, training-anchor-preserving window input adapter.

The raw window stays truthful; only its extrapolation scale changes. Frozen GRU
weights did not learn variable windows, so this is a diagnostic candidate, not a
claim that re-normalizing arbitrary trained features is generally valid.
"""
import copy,json,time,traceback
import numpy as np
import torch
from train import ROOT,REPO,HERE,BASE,PREVIOUS,read,write,sha,MemoryGRU,FineEnv,predictor
OUT=ROOT/'uav-window-adapter-20261104/run01'
SCREEN=ROOT/'uav-window85-20261104/run01'

def normalized(x,mu,sd,adapted=False):
    z=(x-mu[None,:,None,None])/sd[None,:,None,None]
    if adapted:
        anchor=np.float32(420/300)
        # Retain the exact value seen during training. One raw unit means300s.
        z[:,13]=(anchor-mu[13])/sd[13]+(x[:,13]-anchor)
    return z

def load(seed,adapted):
    path=PREVIOUS/f'fine-s{seed}-e16.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
    m=MemoryGRU(ck['mean'],ck['coarse']);m.load_state_dict(ck['state']);m.eval()
    mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
    def predict(x):
        with torch.inference_mode():return m(torch.from_numpy(normalized(x,mu,sd,adapted))[None])[0].numpy().ravel()
    return predict,mu,sd

def run():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];tests=[]
    cases=read(SCREEN/'scenarios.json')['development'];lock=read(SCREEN/'input-lock.json')
    assert sha(SCREEN/'scenarios.json')==lock['scenarios_sha256']
    sources={str(p):sha(p) for folder in (HERE,BASE,HERE.parent/'flood-drop-fine-memory-v1',HERE.parent/'flood-drop-search-redesign-v1') for p in folder.glob('*.py')}
    inputs={str(p):sha(p) for p in [SCREEN/'scenarios.json',SCREEN/'results.jsonl',SCREEN/'input-lock.json',*(PREVIOUS/f'fine-s{s}-e16.pt' for s in (202611141,202611142))]}
    protocol=dict(version='window-scale-anchor-v1',date='2026-11-04',source_sha256=sources,input_sha256=inputs,
      hypothesis='Old constant-window std=.01 amplifies truthful660s input to80. Test fixed physical extrapolation scale1 while exactly preserving420 training anchor. Only channel13 changes; time normalization and all other inputs stay untouched.',
      setup='6previous DEV parents,660s window,2400s horizon,20m sensor,40people4UAV,10s service,unchanged physical environment/weights/dispatcher. Both prior fine16 seeds.',
      comparison='seed141 raw+adapted and seed142 adapted=18new full runs; seed142 raw6runs and both planners reused from exact SCREEN config after input SHA checks.',
      tests='Across24FIT parents8fixed archived frames each, exact normalized inputs and predictions at420; other63channels exact at660; same true config still passed to environment.',
      gate='Both seeds adapted >=own raw and >=207/240 strongest archived planner. Otherwise do not promote or start long training. This is DEV causality screen, not independent finaltest. Never sweep scale after results.',
      limits='Fixed weights lack variable-window training; adapter is an explicit extrapolation assumption, not retraining or proof80% success probability. It cannot establish future-field generalization.',
      budget='One CPU4 workload,18new runs,STOP/1200s guard. No new maps,training,production changes orpush.')
    write(OUT/'protocol.json',protocol)
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1200:raise TimeoutError('1200s cap')
    def status(state,**kw):write(OUT/'status.json',dict(state=state,evaluations=len(rows),elapsed_s=time.monotonic()-start,**kw))
    try:
        models={}
        for seed in (202611141,202611142):
            raw,mu,sd=load(seed,False);fixed,_,_=load(seed,True);models[seed]=(raw,fixed)
            for i in range(24):
                with np.load(PREVIOUS/f'fit-base-{i:02d}.npz') as data:
                    for j in np.linspace(0,len(data['x'])-1,8).round().astype(int):
                        x=data['x'][j];np.testing.assert_array_equal(normalized(x,mu,sd),normalized(x,mu,sd,True))
                        np.testing.assert_array_equal(raw(x),fixed(x))
                        y=x.copy();y[:,13]=660/300;a=normalized(y,mu,sd);b=normalized(y,mu,sd,True)
                        np.testing.assert_array_equal(np.delete(a,13,axis=1),np.delete(b,13,axis=1))
                        assert np.abs(b[:,13]).max()<1. and np.abs(a[:,13]).min()>79.
            tests.append(dict(seed=seed,states=192,training_anchor_exact=True,other_channels_exact=True))
        write(OUT/'readiness.json',tests)
        for c in cases:
            assert sha(c['field']['path'])==lock['water_sha256'][c['field']['path']]
            w=np.load(c['field']['path'],mmap_mode='r');sc=copy.deepcopy(c['scenario']);sc['config'].update(window=660,duration=2400)
            for event in sc['events']:event['deadline']=event['fall']+660
            for seed,adapted in ((202611141,False),(202611141,True),(202611142,True)):
                status('evaluating',case=c['id'],seed=seed,adapted=adapted);e=FineEnv(w,sc,models[seed][adapted],.25)
                while True:
                    if e.t%120==0:guard()
                    if e.step(e.choices()[1])[2]:break
                assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
                for p in persons:
                    if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                row=dict(case=c['id'],parent=c['field']['parent'],seed=seed,adapted=adapted,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=persons)
                rows.append(row)
                with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(dict(case=c['id'],evaluations=len(rows))),flush=True)
        prior=[json.loads(x) for x in (SCREEN/'results.jsonl').read_text().splitlines()]
        raw142=[r for r in prior if r['window']==660 and r['method']=='fine142-e16'];assert len(raw142)==6
        summary=[]
        for seed in (202611141,202611142):
            for adapted in (False,True):
                rr=raw142 if seed==202611142 and not adapted else [r for r in rows if r['seed']==seed and r['adapted']==adapted]
                summary.append(dict(seed=seed,adapted=adapted,supplied=sum(r['supplied'] for r in rr),people=240,all_scenes=sum(r['all_scene'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr)))
        passed=all(summary[i+1]['supplied']>=max(207,summary[i]['supplied']) for i in (0,2))
        for p,h in {**sources,**inputs}.items():assert sha(p)==h,p
        write(OUT/'summary.json',dict(results=summary,gate_passed=passed,trained=False,formal_test=False,elapsed_s=time.monotonic()-start))
        status('completed');print(json.dumps(summary),flush=True)
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':run()
