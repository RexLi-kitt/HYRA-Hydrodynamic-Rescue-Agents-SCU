"""Fixed-weight coupling change, paired development only. No test tuning."""
import copy,json,time,traceback
import numpy as np
import torch
from train import ROOT,HERE,read,write,sha,transform,MemoryGRU,predictor
from window_adapter import load
from joint_trial import aggregate
from joint_avoidance import swept_distances
from count_coupling import CountCoupledEnv,OracleCoupledEnv
from memory import public_features
OUT=ROOT/'uav-count-coupling-20261104/run01'
JOINT=ROOT/'uav-joint-avoidance-20261104/run01';PRIOR=ROOT/'uav-retrain660-20261104/run01'

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[]
    cases=read(JOINT/'confirmation-scenarios.json')[:6];models={};inputs={}
    for n in ['statistical-s202611171-e8','statistical-s202611172-e16']:
        path=PRIOR/f'{n}.pt';inputs[str(path)]=sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
        m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval();models[n]=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
    models['old142-adapter']=load(202611142,True)[0]
    old=ROOT/'uav-fine-memory-20261104/run01/fine-s202611142-e16.pt';inputs[str(old)]=sha(old)
    inputs.update({str(JOINT/n):sha(JOINT/n) for n in ('confirmation-scenarios.json','results.jsonl','summary.json')})
    sources={**read(JOINT/'protocol.json')['source_sha256'],**{str(HERE/n):sha(HERE/n) for n in ('count_coupling.py','coupling_trial.py')}}
    water={c['field']['path']:sha(c['field']['path']) for c in cases}
    write(OUT/'protocol.json',dict(source_sha256=sources,input_sha256=inputs,water_sha256=water,
        evidence='Exact100m current counts plus strongscore outperformed same residual25 in current privileged diagnostic; inspect predictor-to-search interface before more training. Detailed probe retained separately.',
        intervention='Only final count-to-score mapping changes: normalizedmedian/maxabs capped25% -> base*(1+8*expected_count). 30s caching,half current/half60s observedflow transport,history,weights and control unchanged. Factor8 inherited published local project oracle/search mode,not swept on current results. Constant map exactly neutral.',
        design='6existing first-index development parents +first3rotations,3frozen learned predictors and privileged cached-flow oracle=36new. Same repaired planners/GRU residual results reused where exact. No model training or difficulty change.',
        selection='For next step choose deployable model by min(original,rotation) peoplefraction then totalpeople then p95scenes thenname. Gate>=85% and>=both full planners in each panel, originalp95scenes>=bestplanner+1 (>=16.67pp on6), rotationp95notbelowbestplanner. All failures kept. If passes, needs freshly generated blind confirmation; never claim DEV passed is final.',
        audit='Deployable public feature and prediction insensitive to hidden persons/deadlines/future water; truecounts only explicitoracle. Actual sweptseparation>=5m;personrecords and10sservice;sourcehashes.',
        budget='OneCPU4job,1200s guard,STOP,no production/push. Privileged output never credited to model.'))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1200:raise TimeoutError('1200s')
    def record(r):
        rows.append(r)
        with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    try:
        for r in [json.loads(s) for s in (JOINT/'results.jsonl').read_text(encoding='utf-8').splitlines()]:
            if r['case'] in [c['id'] for c in cases] and (r['phase']=='confirmation' or (r['phase']=='confirmation_rotation' and r['case'] in [c['id'] for c in cases[:3]])):
                q=copy.deepcopy(r);q['reused']=True;q['method']='residual-'+r['method'] if r['method'].startswith('statistical') else r['method'];record(q)
        for phase,cc,rot in [('confirmation',cases,False),('confirmation_rotation',cases[:3],True)]:
            for c in cc:
                for method in [*models,'oracle-coupled']:
                    guard();write(OUT/'status.json',dict(state='running',case=c['id'],phase=phase,method=method,completed_new=sum(not r['reused'] for r in rows),elapsed_s=time.monotonic()-start))
                    w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
                    if rot:w,s=transform(w,s)
                    e=OracleCoupledEnv(w,s) if method=='oracle-coupled' else CountCoupledEnv(w,s,models[method],.25)
                    minimum=float('inf')
                    while True:
                        if e.t%120==0:guard()
                        if e.t in (0,480) and method!='oracle-coupled':
                            hidden=copy.copy(e);hidden.person=np.full_like(e.person,999);hidden.status=np.full_like(e.status,3);hidden.water=None;hidden.deadline=e.deadline+100
                            np.testing.assert_array_equal(public_features(e),public_features(hidden))
                        p=e.pos.copy();done=e.step(e.choices()[1])[2]
                        minimum=min(minimum,float(swept_distances(p,e.pos).min()));assert minimum>=5-1e-8
                        if done:break
                    pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
                    for p in pp:
                        if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                    r=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=pp,reused=False,privileged=method=='oracle-coupled',min_swept_separation=minimum,refloats=int(e.refloat_counts.sum()),conflicts=int(e.conflicts))
                    record(r);print(json.dumps({k:v for k,v in r.items() if k!='persons'}),flush=True)
        results={p:{m:aggregate([r for r in rows if r['phase']==p and r['method']==m]) for m in sorted({r['method'] for r in rows})} for p in ['confirmation','confirmation_rotation']}
        a,b=results.values();chosen=max(models,key=lambda n:(min(a[n]['supplied']/240,b[n]['supplied']/120),a[n]['supplied']+b[n]['supplied'],a[n]['p95_scenes']+b[n]['p95_scenes'],n))
        gate=all(d[chosen]['supplied']/d[chosen]['people']>=.85 and d[chosen]['supplied']>=max(d[n]['supplied'] for n in ('coverage100','flow_coverage')) for d in (a,b)) and a[chosen]['p95_scenes']>=max(a[n]['p95_scenes'] for n in ('coverage100','flow_coverage'))+1 and b[chosen]['p95_scenes']>=max(b[n]['p95_scenes'] for n in ('coverage100','flow_coverage'))
        for p,h in {**sources,**inputs,**water}.items():assert sha(p)==h,p
        write(OUT/'summary.json',dict(results=results,chosen=chosen,development_gate=gate,elapsed_s=time.monotonic()-start,new_rollouts=36,reused_rollouts=27,source_inputs_unchanged=True))
        write(OUT/'status.json',dict(state='completed',elapsed_s=time.monotonic()-start))
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
