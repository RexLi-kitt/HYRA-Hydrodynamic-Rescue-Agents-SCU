"""One FIT-derived scale, frozen weights; DEV gate before fresh confirmation."""
import time,copy,json,contextlib,traceback
import numpy as np
import torch
from train import ROOT,REPO,HERE,read,write,sha,MemoryGRU,predictor,FlowFogEnv,FineEnv,Config,generate,make_scenario,transform
from memory import public_features
from score_gate import GatedEnv,contrast
PRIOR=ROOT/'uav-retrain660-20261104/run01'
OUT=ROOT/'uav-score-gate-20261104/run01'

def aggregate(rr):
    return dict(scenes=len(rr),supplied=sum(r['supplied'] for r in rr),people=sum(r['people'] for r in rr),all_scenes=sum(r['all_scene'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr))

def main():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];new_count=0
    sc=read(PRIOR/'scenarios.json');priorprotocol=read(PRIOR/'protocol.json');lock=read(PRIOR/'input-lock.json')
    selected={k:v for k,v in read(PRIOR/'selection.json')['selected'].items() if k.startswith('statistical')}
    sources={**priorprotocol['source_sha256'],**{str(HERE/n):sha(HERE/n) for n in ('score_gate.py','score_gate_trial.py','test_score_gate.py')}}
    inputs={str(PRIOR/n):sha(PRIOR/n) for n in ('protocol.json','scenarios.json','summary.json','selection.json','results.jsonl','input-lock.json')}
    inputs.update({str(PRIOR/f'{n}.pt'):sha(PRIOR/f'{n}.pt') for n in selected.values()})
    inputs.update({str(PRIOR/f'data-fit-{i}.npz'):sha(PRIOR/f'data-fit-{i}.npz') for i in range(12)})
    protocol=dict(version='score-gate-v1',source_sha256=sources,input_sha256=inputs,
        config=sc['fit'][0]['scenario']['config'],models=selected,
        change='Only effective alpha=.25*min(1,S/ref). S=maxabs(p-median(p)) BEFORE unchanged flow transport. ref=model-specific median strictly-positive S over12FITx48uniform archived histories,minimum1e-6. No labels,gridsearch,weightchanges,newtraining,regionchangesorTTA.',
        hypothesis='Existing maximum normalization maps arbitrarily small contrasts to full25% correction. Continuous attenuation removes that mechanism, not a calibrated confidence guarantee.',
        development='Previous6TEST+first3rotations now DEV because inspected.2frozen ordinary GRU seeds.18new candidate fullrollouts; original2models+2completeplanners+old142adapter reuse exact hashed results,45records,clearlymarked reused.',
        gate='Both seeds: originalDEV people >= own original; rotatedDEV people >= own original AND >= best complete planner rotation; totalcandidatepeople > totaloriginalpeople. No changes to gate/ref/model after results. Otherwise STOP,no fresh generation.',
        confirmation='Only if DEVgate:6new parents1105201..1105206,scenario2026111900..05,samephysics660/2400/20m40people.Each2oldscore+2gatedmodels+2completeplanners;alsofirst3rotated.54newrollouts. No checkpoint/threshold selection. Successrequiresbothgatedseeds>=85% and>=ownold and>=strongestplanner separately onoriginal androtation;scene15ppgoal separately.',
        budget='One CPU4 workload,1800s guard,STOPfile,18newDEV+conditional54confirmation.Production unchanged,rollbackoldFineEnv;no externalpush.')
    write(OUT/'protocol.json',protocol)
    def status(state,**kw):write(OUT/'status.json',dict(state=state,new_evaluations=new_count,elapsed_s=time.monotonic()-start,**kw))
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP')
        if time.monotonic()-start>1800:raise TimeoutError('1800s budget')
    def record(r):
        rows.append(r)
        with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    def episode(c,method,pred=None,ref=None,phase='development',rotate=False):
        nonlocal new_count
        guard();status('evaluating',case=c['id'],method=method,phase=phase)
        water=np.load(c['field']['path'],mmap_mode='r');scenario=c['scenario']
        if rotate:water,scenario=transform(water,scenario)
        if method in ('coverage100','flow_coverage'):e=FlowFogEnv(water,scenario,method)
        elif ref is None:e=FineEnv(water,scenario,pred,.25)
        else:e=GatedEnv(water,scenario,pred,ref)
        while True:
            if e.t%120==0:guard()
            if e.t in (0,480):
                altered=copy.copy(e);altered.person=np.full_like(e.person,999);altered.status=np.full_like(e.status,3);altered.water=None;altered.deadline=e.deadline+100
                np.testing.assert_array_equal(public_features(e),public_features(altered))
            if e.step(e.choices()[1])[2]:break
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
        pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
        for p in pp:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        row=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),refloats=int(e.refloat_counts.sum()),persons=pp,reused=False,gate_trace=getattr(e,'gate_trace',[]))
        new_count+=1;record(row);return row
    try:
        assert sha(PRIOR/'scenarios.json')==lock['scenarios_sha256']
        assert read(PRIOR/'status.json')['state']=='completed'
        for p,h in lock['water_sha256'].items():assert sha(p)==h
        x=[]
        for i in range(12):
            with np.load(PRIOR/f'data-fit-{i}.npz') as d:
                ids=np.linspace(0,len(d['x'])-1,48).round().astype(int);x.append(d['x'][ids])
        x=np.concatenate(x);models={};refs={};distributions={}
        for name in selected.values():
            ck=torch.load(PRIOR/f'{name}.pt',map_location='cpu',weights_only=True);m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval()
            pred=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32));models[name]=pred
            scores=np.array([contrast(pred(z)) for z in x]);positive=scores[scores>0]
            assert len(positive)>0;refs[name]=max(float(np.median(positive)),1e-6);distributions[name]=scores.tolist()
        write(OUT/'fit-reference.json',dict(references=refs,raw_contrasts=distributions,histories=576,labels_used=False))
        # Previous TEST is intentionally development now; preserve full adverse rows.
        oldrows=[json.loads(line) for line in (PRIOR/'results.jsonl').read_text(encoding='utf-8').splitlines()]
        allowed=set(models)|{'coverage100','flow_coverage','old142-adapter'}
        for old in oldrows:
            if old['phase'] not in ('test','rotation') or old['method'] not in allowed:continue
            r=copy.deepcopy(old);r['source_phase']=old['phase'];r['phase']='development' if old['phase']=='test' else 'development_rotation';r['reused']=True;r['source_results_sha256']=inputs[str(PRIOR/'results.jsonl')];record(r)
        for phase,cases,rotate in (('development',sc['test'],False),('development_rotation',sc['test'][:3],True)):
            for c in cases:
                for name,pred in models.items():episode(c,name+'-gated',pred,refs[name],phase,rotate)
        def summaries(phase):return {n:aggregate([r for r in rows if r['phase']==phase and r['method']==n]) for n in sorted({r['method'] for r in rows if r['phase']==phase})}
        dev=summaries('development');rot=summaries('development_rotation');bestrot=max(rot[n]['supplied'] for n in ('coverage100','flow_coverage'))
        checks={n:dict(original_noninferior=dev[n+'-gated']['supplied']>=dev[n]['supplied'],rotation_noninferior=rot[n+'-gated']['supplied']>=rot[n]['supplied'],rotation_planner=rot[n+'-gated']['supplied']>=bestrot) for n in models}
        strict=sum(dev[n+'-gated']['supplied']+rot[n+'-gated']['supplied'] for n in models)>sum(dev[n]['supplied']+rot[n]['supplied'] for n in models)
        passed=all(all(v.values()) for v in checks.values()) and strict
        write(OUT/'development-gate.json',dict(passed=passed,per_model=checks,strict_total_improvement=strict,before_confirmation=True))
        waterhash={};confirm=[];confirm_pass=None
        if passed:
            cfg=Config(**protocol['config'])
            for i,seed in enumerate(range(1105201,1105207)):
                guard();status('generating_confirmation',seed=seed)
                with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):meta=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,'confirmation',duration=2400)
                waterhash[meta['path']]=sha(meta['path']);scenario=make_scenario(np.load(meta['path'],mmap_mode='r'),2026111900+i,cfg,fall_end=1680,pool_size=120)
                confirm.append(dict(id=f'confirmation-{i}',field=meta,scenario=scenario))
            assert not({c['field']['parent'] for c in confirm}&{c['field']['parent'] for group in sc.values() for c in group})
            write(OUT/'confirmation-scenarios.json',confirm)
            for phase,cases,rotate in (('confirmation',confirm,False),('confirmation_rotation',confirm[:3],True)):
                for c in cases:
                    for n in ('coverage100','flow_coverage'):episode(c,n,phase=phase,rotate=rotate)
                    for name,pred in models.items():
                        episode(c,name,pred,phase=phase,rotate=rotate)
                        episode(c,name+'-gated',pred,refs[name],phase,rotate)
            confirm_pass=True
            for phase in ('confirmation','confirmation_rotation'):
                scores=summaries(phase);best=max(scores[n]['supplied'] for n in ('coverage100','flow_coverage'))
                for name in models:
                    a=scores[name+'-gated'];confirm_pass &= a['supplied']/a['people']>=.85 and a['supplied']>=scores[name]['supplied'] and a['supplied']>=best
        for p,h in {**sources,**inputs,**waterhash}.items():assert sha(p)==h,p
        for p,h in lock['water_sha256'].items():assert sha(p)==h
        result=dict(development_gate_passed=passed,confirmation_gate_passed=confirm_pass,references=refs,results={phase:summaries(phase) for phase in sorted({r['phase'] for r in rows})},new_evaluations=new_count,reused_evaluations=sum(r['reused'] for r in rows),elapsed_s=time.monotonic()-start,source_inputs_weights_unchanged=True,confirmation_water_sha256=waterhash)
        write(OUT/'summary.json',result);status('completed');print(json.dumps(result),flush=True)
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
