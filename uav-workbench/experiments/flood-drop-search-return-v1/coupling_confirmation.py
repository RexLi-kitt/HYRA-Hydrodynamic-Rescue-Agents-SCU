"""Fresh parent-disjoint confirmation of an already frozen count-coupled model."""
import copy,json,time,traceback,contextlib
import numpy as np
import torch
from train import ROOT,REPO,HERE,read,write,sha,transform,MemoryGRU,predictor,Config,make_scenario,generate
from window_adapter import load
from joint_trial import aggregate
from joint_avoidance import SafeFlowEnv,swept_distances
from count_coupling import CountCoupledEnv
from memory import public_features
DEV=ROOT/'uav-count-coupling-20261104/run01';OUT=ROOT/'uav-count-confirmation-20261104/run01'

def main():
    dev=read(DEV/'summary.json');assert dev['development_gate'],'No development gate, no new confirmation'
    chosen=dev['chosen'];OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];cases=[];waterhash={}
    if chosen=='old142-adapter':
        pred=load(202611142,True)[0];path=ROOT/'uav-fine-memory-20261104/run01/fine-s202611142-e16.pt'
    else:
        path=ROOT/'uav-retrain660-20261104/run01'/f'{chosen}.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
        m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval();pred=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
    sources={**read(DEV/'protocol.json')['source_sha256'],str(HERE/'coupling_confirmation.py'):sha(HERE/'coupling_confirmation.py')}
    inputs={str(DEV/n):sha(DEV/n) for n in ('summary.json','protocol.json','results.jsonl')};inputs[str(path)]=sha(path)
    original=read(ROOT/'uav-joint-avoidance-20261104/run01/confirmation-scenarios.json');cfg=Config(**original[0]['scenario']['config'])
    protocol=dict(version='count-coupling-independent-confirmation-v1',source_sha256=sources,input_sha256=inputs,config=cfg.__dict__,chosen=chosen,checkpoint=str(path),
        freeze='Selected model and count coupling are fixed before any new field generation. No further selection on confirmation. Earlier20-parent failed confirmation and six-parent coupling screen remain development.',
        cases='20NEW synthetic parents1105401..1105420,scenario2026112200..19,one independent40-person roster perparent; samefall_end1680,pool120,T2400,window660,sensor20m,threshold2m/1mps. All20 xselectedmodel+two intact shared-fixed planners. First5rotations stress sameparents,not5extraindependentmaps. 75new full rollouts.',
        target='Original20 rescue fraction>=85% (aim90+-5 without penalizinghigher) and >=bothplanners; >=38/40scene rate >=eachplanner +15pp (atleast3extraof20). Rotation5fraction>=85% and>=bothplanners. Full40/40secondary. Pointestimates must meetall; parent-cluster uncertainty reported separately; no guarantee ofpopulation15pp.',
        unchanged='No extraepochs,rewardchange,difficultychange,futuretruth,weakeningbaseline,productionupgrade,push. Samejointavoidance everyone. Constantprediction neutral;10sservice and5m sweptseparation enforced.',
        budget='OneCPU4heavyload,3600s guard,STOP. If fails no pass claim or selection from TEST. New fields internalfinite/nonnegative/conservationchecks inherited generator, labels synthetic notrealworld.')
    write(OUT/'protocol.json',protocol)
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>3600:raise TimeoutError('3600s')
    def status(state,**kw):write(OUT/'status.json',dict(state=state,completed=len(rows),elapsed_s=time.monotonic()-start,**kw))
    try:
        for i,seed in enumerate(range(1105401,1105421)):
            guard();status('generating',seed=seed)
            with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):meta=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,'confirmation',duration=2400)
            waterhash[meta['path']]=sha(meta['path']);scenario=make_scenario(np.load(meta['path'],mmap_mode='r'),2026112200+i,cfg,fall_end=1680,pool_size=120)
            cases.append(dict(id=f'confirmation-{i}',field=meta,scenario=scenario))
        prior=read(ROOT/'uav-retrain660-20261104/run01/scenarios.json');oldparents={c['field']['parent'] for group in prior.values() for c in group}|{c['field']['parent'] for c in original}
        assert not({c['field']['parent'] for c in cases}&oldparents)
        write(OUT/'scenarios.json',cases);write(OUT/'input-lock.json',dict(scenarios_sha256=sha(OUT/'scenarios.json'),water_sha256=waterhash))
        for phase,cc,rot in [('confirmation',cases,False),('rotation',cases[:5],True)]:
            for c in cc:
                for method in ['coverage100','flow_coverage',chosen]:
                    guard();status('evaluating',case=c['id'],phase=phase,method=method)
                    w=np.load(c['field']['path'],mmap_mode='r');s=c['scenario']
                    if rot:w,s=transform(w,s)
                    e=CountCoupledEnv(w,s,pred,.25) if method==chosen else SafeFlowEnv(w,s,method)
                    minimum=float('inf')
                    while True:
                        if e.t%120==0:guard()
                        if e.t in (0,480):
                            hidden=copy.copy(e);hidden.person=np.full_like(e.person,999);hidden.status=np.full_like(e.status,3);hidden.water=None;hidden.deadline=e.deadline+100
                            np.testing.assert_array_equal(public_features(e),public_features(hidden))
                        p=e.pos.copy();done=e.step(e.choices()[1])[2]
                        minimum=min(minimum,float(swept_distances(p,e.pos).min()));assert minimum>=5-1e-8
                        if done:break
                    pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
                    for p in pp:
                        if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
                    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
                    r=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=pp,privileged=False,min_swept_separation=minimum,refloats=int(e.refloat_counts.sum()),conflicts=int(e.conflicts),avoidance_events=getattr(e,'avoidance_events',[]))
                    rows.append(r)
                    with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
                    print(json.dumps({k:v for k,v in r.items() if k not in ('persons','avoidance_events')}),flush=True)
        results={p:{m:aggregate([r for r in rows if r['phase']==p and r['method']==m]) for m in sorted({r['method'] for r in rows})} for p in ['confirmation','rotation']}
        a,b=results.values();gate=all(d[chosen]['supplied']/d[chosen]['people']>=.85 and d[chosen]['supplied']>=max(d[n]['supplied'] for n in ('coverage100','flow_coverage')) for d in (a,b)) and a[chosen]['p95_scenes']>=max(a[n]['p95_scenes'] for n in ('coverage100','flow_coverage'))+3
        for p,h in {**sources,**inputs,**waterhash}.items():assert sha(p)==h,p
        write(OUT/'summary.json',dict(results=results,chosen=chosen,target_point_estimates_passed=gate,elapsed_s=time.monotonic()-start,new_rollouts=75,source_inputs_unchanged=True))
        status('completed')
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
