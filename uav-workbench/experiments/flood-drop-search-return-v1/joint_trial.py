"""Shared safe joint controller: reproduce fix, all DEV, then frozen fresh test."""
import copy,time,json,contextlib,traceback
import numpy as np
import torch
from train import ROOT,REPO,HERE,read,write,sha,MemoryGRU,predictor,Config,generate,make_scenario,transform
from window_adapter import load
from memory import public_features
from joint_avoidance import SafeFineEnv,SafeFlowEnv,swept_distances
PRIOR=ROOT/'uav-retrain660-20261104/run01';OUT=ROOT/'uav-joint-avoidance-20261104/run01'
def aggregate(rr):return dict(scenes=len(rr),supplied=sum(r['supplied'] for r in rr),people=40*len(rr),all_scenes=sum(r['all_scene'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr))

def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];new_count=0;sc=read(PRIOR/'scenarios.json')
 sources={**read(PRIOR/'protocol.json')['source_sha256'],**{str(HERE/n):sha(HERE/n) for n in ('joint_avoidance.py','test_joint_avoidance.py','joint_trial.py')}}
 inputs={str(PRIOR/n):sha(PRIOR/n) for n in ('scenarios.json','input-lock.json','results.jsonl','selection.json')}
 models={};paths={}
 for name in read(PRIOR/'selection.json')['selected'].values():
  if not name.startswith('statistical'):continue
  path=PRIOR/f'{name}.pt';inputs[str(path)]=sha(path);paths[name]=str(path);ck=torch.load(path,map_location='cpu',weights_only=True)
  m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval();models[name]=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
 oldpred,oldpath,_=load(202611142,True);models['old142-adapter']=oldpred
 # load returns metadata in some versions; checkpoint source is locked by priorprotocol.
 paths['old142-adapter']=str(ROOT/'uav-fine-memory-20261104/run01/fine-s202611142-e16.pt');inputs[paths['old142-adapter']]=sha(paths['old142-adapter'])
 methods={'coverage100':None,'flow_coverage':None,**models}
 protocol=dict(version='joint-avoidance-v1',source_sha256=sources,input_sha256=inputs,config=sc['test'][0]['scenario']['config'],model_paths=paths,
  evidence='Exactreplay:rotatedDEVtest1/GRU171twoUAVstationary998seconds;1108conflictpairs;24/40rescues. Other2GRUrotationsnoconflicts. Hypothesis shared stop-both resolver admits persistentdeadlock,not allgeneralizationerrors.',
  intervention='Shared public-position/goal joint9-action resolver ONLY on predictedunsafe desiredmove. Enumerate9^4,requirecontinuous sweptseparation>=5m,prefergoalprogress andpreserveongoing10sservicewhenfeasible. Same speed,physics,futurevisibility,safety engine. Both planners+all3learningmodels use identicalresolver. NoTTA,gating,newrewardortraining.',
  feasibility='Threeunitchecks: safeactionsunchanged,headonprogressandsafe,actualstuckpaircanseparate. Firstinstrument knownfailingcase: mustsupplied>24,longestpatrolstop<60s,actualsweptsafetyvalid;elseSTOP.',
  development='All6previousTEST+3rotations nowDEV.5methods45newrollouts includingmicrocase,45oldresultsreused. Noexcludedadversecases. Frozencheckpoints; selectlearningbylargestminimum(original,rotation)peoplecompletion,thenoverallpeople,then95%scenes,thenname. Gatechosen>=85% and>=both sharedplanners people separatelyon originalandrotation. This is a new protocol,oldtwo-seedgatesremainfailed.',
  confirmation='OnlyifDEVgate:20freshparents1105301..1105320,scenario2026112100..19, same40peoplefallend1680/window660/T2400/sensor20m. Selectmodelbeforegenerating. Selectedlearningandbothsharedplanners onall20,plusfirst5rotatedstress=75newrollouts. NoTEST selection orparameterchange.',
  target='Confirmationoriginal20people>=85%,aim90+-5%withoutpenalizinghigher;learningpeople>=bothplanners;>=95%people_scene_rate >= bothplanners +15percentagepoints. Full40/40secondaryreported. Rotated5stresspeople>=85%and>=bothplanners. Pointestimatesonly;smallNdoesnotguaranteegeneralization. Failurepreservedandnotpassedoff.',
  budget='OneCPU4heavyload,4000sguard/STOPfile,noGPU/noProductionoverwrite/noPush. Up to120newfullrollouts. This is controllerrepairbenefit,not claimedneuraltrainingimprovement.')
 write(OUT/'protocol.json',protocol)
 def status(state,**kw):write(OUT/'status.json',dict(state=state,new_evaluations=new_count,elapsed_s=time.monotonic()-start,**kw))
 def guard():
  if (OUT/'STOP').exists():raise InterruptedError('STOP')
  if time.monotonic()-start>4000:raise TimeoutError('4000s cap')
 def record(r):
  rows.append(r)
  with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
 def episode(c,method,phase,rotate=False):
  nonlocal new_count
  guard();status('evaluating',case=c['id'],method=method,phase=phase)
  w=np.load(c['field']['path'],mmap_mode='r');scenario=c['scenario']
  if rotate:w,scenario=transform(w,scenario)
  e=SafeFlowEnv(w,scenario,method) if method in ('coverage100','flow_coverage') else SafeFineEnv(w,scenario,models[method],.25)
  stationary=np.zeros(4,int);longest=0;minsep=float('inf');patrolstops=0
  while True:
   if e.t%120==0:guard()
   if e.t in (0,480):
    hidden=copy.copy(e);hidden.person=np.full_like(e.person,999);hidden.status=np.full_like(e.status,3);hidden.water=None;hidden.deadline=e.deadline+100
    np.testing.assert_array_equal(public_features(e),public_features(hidden))
   _,chosen,opts,_=e.choices();patrol=np.array([opts[i][chosen[i]]['region']>=0 for i in range(4)]);before=e.pos.copy()
   done=e.step(chosen)[2];dist=swept_distances(before,e.pos);minsep=min(minsep,float(dist.min()));assert minsep>=e.cfg.separation-1e-8
   stopped=(np.linalg.norm(e.pos-before,axis=1)<1e-6)&patrol;stationary=np.where(stopped,stationary+e.cfg.step,0);longest=max(longest,int(stationary.max()));patrolstops+=int(stopped.sum())
   if done:break
  pp=[dict(id=i,fall=int(e.fall[i]),deadline=int(e.deadline[i]),first_seen=float(e.first_seen[i]),supplied_at=float(e.delivered_at[i]),status=int(e.status[i])) for i in range(40)]
  for p in pp:
   if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
  assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
  r=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),persons=pp,refloats=int(e.refloat_counts.sum()),reused=False,conflicts=int(e.conflicts),longest_patrol_stop=longest,patrol_stops=patrolstops,min_swept_separation=minsep,avoidance_events=getattr(e,'avoidance_events',[]))
  new_count+=1;record(r);print(json.dumps({k:v for k,v in r.items() if k not in ('persons','avoidance_events')}),flush=True);return r
 def summaries(phase):return {n:aggregate([r for r in rows if r['phase']==phase and r['method']==n]) for n in sorted({r['method'] for r in rows if r['phase']==phase})}
 try:
  locks=read(PRIOR/'input-lock.json');assert sha(PRIOR/'scenarios.json')==locks['scenarios_sha256']
  for p,h in locks['water_sha256'].items():assert sha(p)==h
  for old in [json.loads(l) for l in (PRIOR/'results.jsonl').read_text(encoding='utf-8').splitlines()]:
   if old['phase'] in ('test','rotation') and old['method'] in methods:
    r=copy.deepcopy(old);r['source_phase']=r['phase'];r['phase']='old_development' if old['phase']=='test' else 'old_rotation';r['reused']=True;record(r)
  micro=episode(sc['test'][1],'statistical-s202611171-e8','development_rotation',True)
  micro_pass=micro['supplied']>24 and micro['longest_patrol_stop']<60
  write(OUT/'micro-gate.json',dict(passed=micro_pass,row=micro))
  chosen=None;dev_pass=False;target_pass=None;waterhash={}
  if micro_pass:
   for phase,cases,rotate in (('development',sc['test'],False),('development_rotation',sc['test'][:3],True)):
    for c in cases:
     for n in methods:
      if phase=='development_rotation' and c['id']==sc['test'][1]['id'] and n=='statistical-s202611171-e8':continue
      episode(c,n,phase,rotate)
   dev=summaries('development');rot=summaries('development_rotation')
   chosen=max(models,key=lambda n:(min(dev[n]['supplied']/240,rot[n]['supplied']/120),dev[n]['supplied']+rot[n]['supplied'],dev[n]['p95_scenes']+rot[n]['p95_scenes'],n))
   dev_pass=all(a[chosen]['supplied']/a[chosen]['people']>=.85 and a[chosen]['supplied']>=max(a[n]['supplied'] for n in ('coverage100','flow_coverage')) for a in (dev,rot))
   write(OUT/'selection.json',dict(chosen=chosen,development_passed=dev_pass,development=dev,rotation=rot,before_confirmation=True))
   if dev_pass:
    cfg=Config(**protocol['config']);cases=[]
    for i,seed in enumerate(range(1105301,1105321)):
     guard();status('generating_confirmation',seed=seed)
     with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):meta=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,'confirmation',duration=2400)
     waterhash[meta['path']]=sha(meta['path']);scenario=make_scenario(np.load(meta['path'],mmap_mode='r'),2026112100+i,cfg,fall_end=1680,pool_size=120);cases.append(dict(id=f'confirmation-{i}',field=meta,scenario=scenario))
    assert not({c['field']['parent'] for c in cases}&{c['field']['parent'] for group in sc.values() for c in group})
    write(OUT/'confirmation-scenarios.json',cases)
    for phase,group,rotate in (('confirmation',cases,False),('confirmation_rotation',cases[:5],True)):
     for c in group:
      for n in ('coverage100','flow_coverage',chosen):episode(c,n,phase,rotate)
    a=summaries('confirmation');b=summaries('confirmation_rotation')
    target_pass=all(d[chosen]['supplied']/d[chosen]['people']>=.85 and d[chosen]['supplied']>=max(d[n]['supplied'] for n in ('coverage100','flow_coverage')) for d in (a,b)) and a[chosen]['p95_scenes']-max(a[n]['p95_scenes'] for n in ('coverage100','flow_coverage'))>=3
  for p,h in {**sources,**inputs,**locks['water_sha256'],**waterhash}.items():assert sha(p)==h,p
  write(OUT/'summary.json',dict(micro_passed=micro_pass,development_passed=dev_pass,chosen=chosen,target_point_estimates_passed=target_pass,results={phase:summaries(phase) for phase in sorted({r['phase'] for r in rows})},new_evaluations=new_count,reused_evaluations=sum(r['reused'] for r in rows),elapsed_s=time.monotonic()-start,source_inputs_unchanged=True,confirmation_water_sha256=waterhash));status('completed')
 except BaseException as ex:
  write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
