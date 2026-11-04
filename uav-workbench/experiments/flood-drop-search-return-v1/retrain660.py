"""Matched fresh training at660s: refitted statistical vs fixed metadata scales."""
import copy,time,json,contextlib,traceback
import numpy as np
import torch
from torch import nn
from train import ROOT,REPO,HERE,BASE,PREVIOUS,read,write,sha,Config,make_scenario,generate,FlowFogEnv,MemoryGRU,FineEnv,predictor,transform
from memory import FineHistory,normalizer,augment,public_features
from window_adapter import load
OUT=ROOT/'uav-retrain660-20261104/run01';SEEDS=[202611171,202611172]

def main():
 OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];count=dict(collection=0,epochs=0,evaluations=0)
 fit=read(PREVIOUS/'scenarios.json')['fit'][12:24];cfg=Config(**{**fit[0]['scenario']['config'],'duration':2400,'window':660})
 sources={str(p):sha(p) for folder in (HERE,BASE,HERE.parent/'flood-drop-fine-memory-v1',HERE.parent/'flood-drop-search-redesign-v1') for p in folder.glob('*.py')}
 sources.update({str(p):sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
 inputs={str(p):sha(p) for p in [PREVIOUS/'scenarios.json',PREVIOUS/'input-lock.json',*(PREVIOUS/f'fine-s{s}-e16.pt' for s in (202611141,202611142))]}
 protocol=dict(version='retrain660-v1',date='2026-11-04',config=cfg.__dict__,source_sha256=sources,input_sha256=inputs,
  fit='12existing FIT parents1104501..12, same40fall roster and initial positions,extend physics to2400 anddeadline660.48uniform-time training snapshots/parent,576maps eacharm. Labels are current hidden pending undiscovered counts TRAINonly.',
  splits='4new CAL1105001..04 and6new TEST1105101..06,scenario seeds2026111700..03/2026111800..05; one layout,parent-disjoint. CAL checkpoint8/16 selection, then freeze beforeTEST. first3TEST rotated stress.',
  comparison='Same64->32GRU+counthead,data,init seeds,shuffle,augmentation,Adam.0005,batch32,Poissonloss,clip5,16supervisedepochs. Statistical normalizer REFITTED on newFIT; physical arm same except mu[10:14]=0,sd[10:14]=1. Both truthful parameters. t/2400 because fixed horizon; sensor/100,N/80,window/300 fixed units. Not giving old80-valued normalization to newlytrained control.',
  checks='Both arms finite; independent FIT normalization; hiddenstate/truefuture cannot change public features; identical raw inputs/labels/seed/initial weights. This is normalization+retraining screen, not a guarantee that numerical scaling is sole cause.',
  methods='4CAL-selected freshlytrained policies,complete coverage100/flowcoverage,oldfine142raw and oldfine142windowadapter. CAL32newcheckpoint +16controls=48; TEST48; rotation24;120logicalrollouts. oldGRU trained24parents vs new12,reference only; new arms matched.',
  selection='Perarm/seed: CAL suppliedpeople,then95%scenes,thenearlierepoch. NoTESTselection or automatic extraepochs.',
  gate='For fixed-physical arm both seeds >=85% TEST and >= strongest planner TEST people,thencheck rotation andfull/95%scenes; report15pp scene objective separately. If no stable improvement,do not scale training solely becauseFITlossfalls.',
  budget='One CPU4 workload,3600s guard andSTOP;noGPU,productionchangeorpush. No new difficulty adjustment.')
 write(OUT/'protocol.json',protocol)
 def status(state,**kw):write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,**count,**kw))
 def guard():
  if (OUT/'STOP').exists():raise InterruptedError('STOP')
  if time.monotonic()-start>3600:raise TimeoutError('3600s cap')
 scenarios=dict(fit=[],calibration=[],test=[]);waterhash={};audits=[]
 def field(seed,split):
  guard();status('generating',split=split,seed=seed)
  with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):meta=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,split,duration=2400)
  waterhash[meta['path']]=sha(meta['path']);return meta,np.load(meta['path'],mmap_mode='r')
 def episode(c,method,pred=None,collect=False,phase='calibration',rotation=False):
  guard();status('collecting' if collect else 'evaluating',case=c['id'],method=method,phase=phase)
  w=np.load(c['field']['path'],mmap_mode='r');sc=c['scenario']
  if rotation:w,sc=transform(w,sc)
  e=FlowFogEnv(w,sc,method) if method in ('coverage100','flow_coverage') else FineEnv(w,sc,pred,.25)
  hist=FineHistory();xs=[];ys=[];tt=[]
  while True:
   if e.t%120==0:guard()
   if collect and e.t%30==0:xs.append(hist.get(e));ys.append(e.sector_counts(100).astype(np.float32));tt.append(e.t)
   if e.t in (0,480):
    other=copy.copy(e);other.person=np.full_like(e.person,999);other.status=np.full_like(e.status,3);other.fall=e.fall+1;other.deadline=e.deadline+1;other.water=None
    np.testing.assert_array_equal(public_features(e),public_features(other))
   if e.step(e.choices()[1])[2]:break
  pp=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(40)]
  assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
  for p in pp:
   if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
  row=dict(phase=phase,case=c['id'],parent=c['field']['parent'],method=method,supplied=int((e.status==2).sum()),people=40,all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),refloats=int(e.refloat_counts.sum()),persons=pp)
  if collect:
   count['collection']+=1;write(OUT/f'collection-{c["id"]}.json',row);return np.stack(xs),np.stack(ys),np.array(tt)
  count['evaluations']+=1;rows.append(row)
  with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
  return row
 def aggregate(rr):return dict(scenes=len(rr),supplied=sum(r['supplied'] for r in rr),people=40*len(rr),all_scenes=sum(r['all_scene'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr))
 try:
  priorlock=read(PREVIOUS/'input-lock.json');assert sha(PREVIOUS/'scenarios.json')==priorlock['scenarios_sha256']
  for i,c in enumerate(fit):
   assert sha(c['field']['path'])==priorlock['fields_sha256'][c['field']['path']]
   meta,w=field(c['field']['seed'],'fit');oldw=np.load(c['field']['path'],mmap_mode='r');assert np.array_equal(w[:len(oldw)],oldw)
   sc=copy.deepcopy(c['scenario']);sc['config']=cfg.__dict__.copy()
   for ev in sc['events']:ev['deadline']=ev['fall']+660
   scenarios['fit'].append(dict(id=f'fit-{i}',field=meta,scenario=sc));audits.append(dict(parent=meta['parent'],old_prefix_exact=True))
  for split,seeds,offset in (('calibration',range(1105001,1105005),0),('test',range(1105101,1105107),100)):
   for i,seed in enumerate(seeds):
    meta,w=field(seed,split);sc=make_scenario(w,2026111700+offset+i,cfg,fall_end=1680,pool_size=120)
    scenarios[split].append(dict(id=f'{split}-{i}',field=meta,scenario=sc))
  groups=[{c['field']['parent'] for c in group} for group in scenarios.values()];assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
  write(OUT/'scenarios.json',scenarios);write(OUT/'input-lock.json',dict(scenarios_sha256=sha(OUT/'scenarios.json'),water_sha256=waterhash));write(OUT/'audits.json',audits)
  xx=[];yy=[]
  for c in scenarios['fit']:
   x,y,t=episode(c,'flow_coverage',collect=True,phase='fit');np.savez_compressed(OUT/f'data-{c["id"]}.npz',x=x,y=y,times=t)
   ids=np.linspace(0,len(x)-1,48).round().astype(int);xx.append(x[ids]);yy.append(y[ids]);print(json.dumps(dict(collection=count['collection'])),flush=True)
  x=np.concatenate(xx);y=np.concatenate(yy);mu,sd=normalizer(x);mean=float(y.mean());xt=torch.from_numpy(x);yt=torch.from_numpy(y)
  norms={};paths={};trainrows=[]
  for arm in ('statistical','physical'):
   a=mu.copy();b=sd.copy()
   if arm=='physical':a[10:14]=0;b[10:14]=1
   norms[arm]=(a,b)
  write(OUT/'normalization.json',{k:dict(mu=a.tolist(),sd=b.tolist()) for k,(a,b) in norms.items()})
  for arm in norms:
   a,b=norms[arm];mt=torch.from_numpy(a)[None,None,:,None,None];st=torch.from_numpy(b)[None,None,:,None,None]
   for seed in SEEDS:
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=MemoryGRU(mean,False);opt=torch.optim.Adam(m.parameters(),lr=.0005)
    for ep in range(1,17):
     guard();status('training',arm=arm,seed=seed,epoch=ep);m.train();total=0.
     for ids in torch.randperm(len(x)).split(32):
      z,q=augment(xt[ids],yt[ids],int(rng.integers(4)),bool(rng.integers(2)));pred=m((z-mt)/st)
      loss=nn.functional.poisson_nll_loss(pred,q,log_input=False);assert torch.isfinite(loss)
      opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step();total+=float(loss.detach())*len(ids)
     count['epochs']+=1;trainrows.append(dict(arm=arm,seed=seed,epoch=ep,loss=total/len(x)));write(OUT/'training.json',trainrows)
     if ep in (8,16):
      name=f'{arm}-s{seed}-e{ep}';path=OUT/f'{name}.pt'
      torch.save(dict(state=m.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),aug_rng=rng.bit_generator.state,mean=mean,mu=a.tolist(),sd=b.tolist(),epoch=ep,seed=seed,arm=arm),path);paths[name]=path
  def restore(path):
   ck=torch.load(path,map_location='cpu',weights_only=True);m=MemoryGRU(ck['mean'],False);m.load_state_dict(ck['state']);m.eval()
   return predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
  stats={}
  for name,path in paths.items():
   pred=restore(path)
   for c in scenarios['calibration']:episode(c,name,pred)
   stats[name]=aggregate([r for r in rows if r['method']==name]);print(json.dumps(dict(calibration=name,result=stats[name])),flush=True)
  selected={}
  for arm in norms:
   for seed in SEEDS:
    names=[n for n in paths if n.startswith(f'{arm}-s{seed}')]
    selected[f'{arm}-s{seed}']=max(names,key=lambda n:(stats[n]['supplied'],stats[n]['p95_scenes'],-int(n.rsplit('e',1)[1])))
  write(OUT/'selection.json',dict(selected=selected,calibration=stats,before_test=True))
  oldraw,_,_=load(202611142,False);oldadapt,_,_=load(202611142,True)
  controls={'coverage100':None,'flow_coverage':None,'old142-raw':oldraw,'old142-adapter':oldadapt}
  for name,pred in controls.items():
   for c in scenarios['calibration']:episode(c,name,pred)
  methods={**controls,**{name:restore(paths[name]) for name in selected.values()}}
  for phase,cases,rotate in (('test',scenarios['test'],False),('rotation',scenarios['test'][:3],True)):
   for c in cases:
    for name,pred in methods.items():episode(c,name,pred,phase=phase,rotation=rotate)
    print(json.dumps(dict(phase=phase,case=c['id'],evaluations=count['evaluations'])),flush=True)
  for p,h in {**sources,**inputs,**waterhash}.items():assert sha(p)==h,p
  summaries={phase:{name:aggregate([r for r in rows if r['phase']==phase and r['method']==name]) for name in sorted({r['method'] for r in rows if r['phase']==phase})} for phase in ('calibration','test','rotation')}
  write(OUT/'summary.json',dict(results=summaries,selected=selected,counts=count,elapsed_s=time.monotonic()-start,source_water_unchanged=True));status('completed');print(json.dumps(summaries),flush=True)
 except BaseException as ex:
  write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':main()
