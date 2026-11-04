"""One-use, pre-registered held-out evaluation, never trains or selects on tests."""
import ctypes, hashlib, importlib.util, itertools, json, os, secrets, shutil, subprocess, sys, time, traceback
from pathlib import Path
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent;RUN=BASE/'run01'
SINGLE=ROOT/'uav-singles1024-20261101';PAIR=ROOT/'uav-all-pairs-20261101/run01'
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')/'uav-strict-eval-20261101'
DOC=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\docs\strict-eval-20261101')
ALGOS=('happo','hatrpo','hasac','maac','mat','coma')
PAIRS=['_'.join(p) for p in itertools.combinations(ALGOS,2)]
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def emit(status,**kw):
 v=dict(status=status,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw);write(RUN/'status.json',v)
 for root in (OUT,DOC):write(root/'live-run.json',dict(**v,live_path=str(RUN)))
 print(json.dumps(v,ensure_ascii=False),flush=True)
def stop():return any((r/'STOP_AFTER_CHECKPOINT').exists() for r in (RUN,OUT))
def wait_training():
 q=read(SINGLE/'run01/queue-status.json');k=ctypes.WinDLL('kernel32',use_last_error=True)
 k.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];k.OpenProcess.restype=ctypes.c_void_p
 k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.CloseHandle.argtypes=[ctypes.c_void_p]
 h=k.OpenProcess(0x00100000,False,q.get('pid',0));start=time.monotonic()
 try:
  while h:
   if stop():raise InterruptedError('Manual stop while waiting')
   result=k.WaitForSingleObject(h,30000)
   if result==0:break
   if result!=258:raise RuntimeError('Process wait failed')
   if time.monotonic()-start>12*3600:raise TimeoutError('Training wait exceeded12h')
 finally:
  if h:k.CloseHandle(h)
 for p in (SINGLE/'run01',PAIR):
  assert read(p/'queue-status.json')['status']=='completed',str(p)
  assert read(p/'integrity-audit.json')['status']=='passed',str(p)
def bootstrap():
 global e,np,torch,Mixed
 import numpy as np
 import torch
 spec=importlib.util.spec_from_file_location('strict_single_adapter',SINGLE/'singles.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s);s.bootstrap();e=s.e
 Mixed=e.Controller.__bases__[0]
 class Flexible(Mixed):
  def decide(self,models,obs,mask,roles,deterministic=False):
   if models is not None and len(models)==1:
    assert tuple(roles)==(0,0,0,0);models=[models[0],models[0]]
   return super().decide(models,obs,mask,roles,deterministic)
 e.Controller=Flexible;torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def mirror():
 for p in list(BASE.glob('*.py'))+list(BASE.glob('*.md'))+list(BASE.glob('*audit.json'))+list(RUN.glob('*.json'))+list(RUN.glob('*.jsonl'))+list(RUN.glob('*.csv')):
  for root in (OUT,DOC):root.mkdir(parents=True,exist_ok=True);shutil.copy2(p,root/p.name)
 for p in (RUN/'traces').glob('*.npz'):
  for root in (OUT,DOC):
   dest=root/'traces'/p.name
   if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
def guard(start):
 if stop():raise InterruptedError('Manual stop after current record')
 if shutil.disk_usage(RUN).free<8*1024**3:raise RuntimeError('Less than8GiB free')
 if time.monotonic()-start>8*3600:raise TimeoutError('Eight-hour evaluation budget reached')
def lock_candidates():
 candidates=[]
 for family,run,ids in [('single',SINGLE/'run01',ALGOS),('pair',PAIR,PAIRS)]:
  for cid in ids:
   b=read(run/cid/'best.json');p=run/cid/b['file'];v=read(run/cid/b['evaluation'])
   candidates.append(dict(id=cid,family=family,path=str(p),sha256=sha(p),episode=b['joint_episode'],validation_quality=e.quality(v),best_source=str(run/cid/'best.json')))
 leaders={kind:max([c for c in candidates if c['family']==kind],key=lambda c:(tuple(c['validation_quality']),c['id']))['id'] for kind in ('single','pair')}
 frozen=dict(candidates=candidates,validation_chosen_leaders=leaders,locked_before_test_generation=True,training_seed_count=1)
 write(RUN/'frozen-candidates.json',frozen);return frozen
def make_plan():
 master=secrets.randbits(63);rng=np.random.default_rng(master)
 seeds=rng.choice(np.arange(1000000000,1000100000,dtype=np.int64),140,replace=False).tolist();rows=[]
 for i in range(70):
  rows.append(dict(id=f'{"iid" if i<50 else "shift"}-{i:02d}',stratum='iid50' if i<50 else 'orientation_cluster20',water_seed=seeds[2*i],layout_seed=seeds[2*i+1],rotation_k=0 if i<50 else 1+(i-50)%3,clustered=i>=50))
 plan=dict(master_seed=master,scenarios=rows,regeneration_on_poor_results=False);write(RUN/'seed-plan.json',plan);return plan
def clustered_people(seed):
 rng=np.random.default_rng(seed);centers=rng.uniform(200,800,(2,2));yy,xx=np.indices((100,100));grid=np.stack((xx.ravel()*10+5,yy.ravel()*10+5),1)
 prob=sum(np.exp(-((grid-c)**2).sum(1)/(2*80**2)) for c in centers);prob[[0,99,9900,9999]]=0
 return grid[rng.choice(10000,20,replace=False,p=prob/prob.sum())].astype(float)
def generate(plan,start):
 cases=[]
 for r in plan['scenarios']:
  guard(start);emit('generating_test_fields',current=r['id'],completed=len(cases),total=70)
  meta=e.generate(RUN/'fields',e.REPO/'terrain-workbench/terrain-lab',r['water_seed'],'strict-test')
  assert meta['finite_nonnegative'] and abs(meta['mass_residual_m3'])<=max(1e-6,abs(meta['source_volume_m3'])*1e-9)
  original=Path(meta['path']);water=np.load(original,mmap_mode='r',allow_pickle=False);p=original
  if r['rotation_k']:
   k=r['rotation_k'];a=np.rot90(water,k,axes=(-2,-1)).copy();u=a[:,1].copy();v=a[:,2].copy()
   if k==1:a[:,1]=v;a[:,2]=-u
   elif k==2:a[:,1]=-u;a[:,2]=-v
   else:a[:,1]=-v;a[:,2]=u
   assert np.isfinite(a).all() and (a[:,0]>=0).all()
   assert np.allclose(a[:,0].sum((1,2),dtype=np.float64),water[:,0].sum((1,2),dtype=np.float64),rtol=1e-12,atol=1e-9)
   p=RUN/'fields'/f'{r["id"]}-rotated.npy';np.save(p,a);del a,u,v
  people=clustered_people(r['layout_seed']) if r['clustered'] else e.Env(water,r['layout_seed']).people
  case=dict(**r,path=str(p),sha256=sha(p),parent_meta=meta,people=people.tolist());cases.append(case);write(RUN/'scenario-manifest.partial-list.json',cases)
 write(RUN/'scenario-manifest.json',cases);return cases
def causal_preflight(frozen):
 import copy
 water=np.load(read(SINGLE/'run01/dataset.json')['cases'][0]['path'],mmap_mode='r',allow_pickle=False);checks=[]
 for c in frozen['candidates']:
  models,_=e.restore(torch.load(c['path'],map_location='cpu',weights_only=False));initial=e.model_hash(models)
  roles=[(0,0,0,0)] if c['family']=='single' else [(0,1,1,0),(1,0,0,1)]
  for role in roles:
   env=e.Env(water,770003);obs=env.observe();cf=copy.copy(env);cf.people=env.people.copy();cf.water=np.array(water,copy=True)
   rng=np.random.default_rng(1770003);hidden=np.flatnonzero(env.status==0);cells=np.argwhere(~env.known);new=cells[rng.choice(len(cells),len(hidden),replace=False)];cf.people[hidden]=new[:,::-1]*10+5
   cf.water[0,:,~env.known]=.123;cf.water[1:,0]=.74;cf.water[1:,1:]=-.12
   assert np.array_equal(obs,cf.observe());assert np.array_equal(env.available(),cf.available())
   rngstate=e.rng_state();x=e.Controller().decide(models,obs,env.available(),role);e.restore_rng(rngstate);y=e.Controller().decide(models,cf.observe(),cf.available(),role)
   assert all(np.array_equal(x[j],y[j]) for j in (0,1,2,3,4));assert x[5]==y[5]
   checks.append(dict(candidate=c['id'],roles=role,hidden_changes_leave_inputs_and_decisions_equal=True))
  assert initial==e.model_hash(models);del models,cf
 write(RUN/'frozen-information-audit.json',dict(status='passed',checks=checks))
def scenario_env(case):
 original=e.Env;people=np.asarray(case['people'],float)
 def create(water,seed=0):return original(water,seed,people=people)
 return original,create
def trial(c,models,case,role,assignment,variant,mode,seed,index):
 p=Path(case['path']);water=np.load(p,mmap_mode='r',allow_pickle=False)
 original,create=scenario_env(case);e.Env=create
 key=f'{c["id"]}-{case["id"]}-{assignment}-{variant}-{mode}-{seed}';trace=RUN/'traces'/f'{key}.npz'
 e.seed_all(seed+case['layout_seed']);before=e.model_hash(models) if models is not None else None
 try:metric,_=e.rollout(models,water,case['layout_seed'],role,deterministic=mode=='fixed',pure=variant=='planner',coverage=variant in ('full','planner'),trace_path=trace)
 finally:e.Env=original
 if models is not None:assert before==e.model_hash(models),'Evaluation modified weights'
 t=np.load(trace,allow_pickle=False);assert len(t['action'])==metric['ST'];assert int(t['info'][:,2].sum())==metric['rescued'];assert int(t['info'][:,3].sum())==metric['conflicts'];assert int((t['target_status']==2).sum())==metric['rescued'];assert abs(float(t['known'].mean())-metric['coverage'])<1e-12
 row=dict(candidate=c['id'],family=c['family'],checkpoint_episode=c.get('episode'),case=case['id'],stratum=case['stratum'],assignment=assignment,roles=role,variant=variant,mode=mode,action_seed=seed,trace=str(trace),**metric)
 with (RUN/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
 emit('evaluating',completed_records=index+1,total_records=5630,current=c['id'],case=case['id'],variant=variant)
 return row
def aggregate(rows,frozen):
 metrics=('task_complete','RCR','coverage','ACC','ST');summaries=[];rng=np.random.default_rng(20261101)
 def group(v):
  by={}
  for r in v:by.setdefault(r['case'],[]).append(r)
  return {k:np.asarray([sum(x[m] for x in rr)/len(rr) for m in metrics],float) for k,rr in by.items()}
 for cid,variant,mode,stratum in sorted({(r['candidate'],r['variant'],r['mode'],r['stratum']) for r in rows}):
  rr=[r for r in rows if (r['candidate'],r['variant'],r['mode'],r['stratum'])==(cid,variant,mode,stratum)];by=group(rr);values=np.stack(list(by.values()));n=len(values)
  boot=values[rng.integers(0,n,(10000,n))].mean(1);means=values.mean(0);ci=np.quantile(boot,[.025,.975],axis=0)
  item=dict(candidate=cid,variant=variant,mode=mode,stratum=stratum,independent_parents=n,rollouts=len(rr),means=dict(zip(metrics,means.tolist())),ci95={m:[float(ci[0,i]),float(ci[1,i])] for i,m in enumerate(metrics)},max_conflicts=max(x['conflicts'] for x in rr),failed_rollouts=sum(not x['task_complete'] for x in rr),training_seeds=1)
  complete=sum(all(x['task_complete'] for x in rr if x['case']==key) for key in by)
  rate=complete/n;z=1.959963984540054;den=1+z*z/n;center=(rate+z*z/(2*n))/den;half=z*((rate*(1-rate)/n+z*z/(4*n*n))**.5)/den
  item['all_assignments_complete']=dict(completed_parents=complete,total_parents=n,wilson_ci95=[max(0,center-half),min(1,center+half)],definition='All role/seed repetitions for this parent complete; binomial scene interval, not training-seed uncertainty.')
  if variant=='full' and mode=='fixed':
   base=group([r for r in rows if r['candidate']=='pure_planner' and r['stratum']==stratum]);keys=sorted(by);assert set(keys)==set(base)
   delta=np.stack([by[k]-base[k] for k in keys]);bootd=delta[rng.integers(0,n,(10000,n))].mean(1)
   confirm=cid in frozen['validation_chosen_leaders'].values() and stratum=='iid50';alpha=.025 if confirm else .05;ci2=np.quantile(bootd,[alpha/2,1-alpha/2],axis=0)
   item['vs_pure_planner']=dict(confirmatory=confirm,confidence=1-alpha,means=dict(zip(metrics,delta.mean(0).tolist())),intervals={m:[float(ci2[0,i]),float(ci2[1,i])] for i,m in enumerate(metrics)})
  summaries.append(item)
 result=dict(status='completed',summary=summaries,rows=len(rows),test_used_for_selection=False,training_seed_count=1,bootstrap_unit='independent parent field; roles averaged within case',caution='Percentile scene intervals with one training seed; saturated metrics can have degenerate intervals, not a guarantee or proof of100percent success.')
 write(RUN/'strict-summary.json',result)
 import csv
 with (RUN/'strict-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.writer(f);writer.writerow(['candidate','variant','mode','stratum','independent_parents','rollouts','completion','RCR','coverage','ACC','ST','completion_ci_low','completion_ci_high','failed_rollouts','max_conflicts'])
  for r in summaries:writer.writerow([r[k] for k in ('candidate','variant','mode','stratum','independent_parents','rollouts')]+[r['means'][m] for m in metrics]+r['ci95']['task_complete']+[r['failed_rollouts'],r['max_conflicts']])
def main():
 RUN.mkdir(parents=True,exist_ok=False)
 hashes=dict(read(SINGLE/'run01/protocol.json')['source_hashes']);hashes.update({str(p):sha(p) for p in (Path(__file__),BASE/'protocol.md',BASE/'audit.py')})
 proto=dict(protocol_sha256=sha(BASE/'protocol.md'),source_hashes=hashes,test_fields_generated_at_registration=False,test_seed_plan_created_at_registration=False,rollout_cap=5630)
 write(RUN/'registration.json',proto);emit('waiting_for_training',predecessor=str(SINGLE/'run01'));mirror();wait_training();start=time.monotonic();bootstrap()
 for p,h in proto['source_hashes'].items():assert sha(p)==h,p
 assert read(BASE/'information-audit.json')['status']=='passed'
 assert read(BASE/'smoke-audit.json')['status']=='passed'
 frozen=lock_candidates();causal_preflight(frozen);plan=make_plan();mirror();cases=generate(plan,start);mirror();rows=[];(RUN/'traces').mkdir()
 for c in frozen['candidates']:
  guard(start);assert sha(c['path'])==c['sha256'];models,_=e.restore(torch.load(c['path'],map_location='cpu',weights_only=False))
  roles=[(0,0,0,0)] if c['family']=='single' else [(0,1,1,0),(1,0,0,1)]
  for case in cases:
   for assignment,role in enumerate(roles):
    for variant in ('full','no-cover'):guard(start);rows.append(trial(c,models,case,role,assignment,variant,'fixed',510001,len(rows)))
  if c['id'] in frozen['validation_chosen_leaders'].values():
   for case in cases[:50]:
    for assignment,role in enumerate(roles):
     for seed in (510101,510102,510103):guard(start);rows.append(trial(c,models,case,role,assignment,'full','sample',seed,len(rows)))
  assert sha(c['path'])==c['sha256'];del models;mirror()
 for cid,variant in [('pure_planner','planner'),('uniform_service','no-cover')]:
  c=dict(id=cid,family='rule');models=None
  if cid=='uniform_service':
   m=e.Learner('happo',e.DIM)
   with torch.no_grad():
    for p in m.parameters():p.zero_()
   m.eval();models=[m]
  for case in cases:guard(start);rows.append(trial(c,models,case,(0,0,0,0),0,variant,'fixed' if models is None else 'sample',510001,len(rows)))
 assert len(rows)==5630,len(rows);aggregate(rows,frozen)
 for p,h in proto['source_hashes'].items():assert sha(p)==h
 write(RUN/'integrity-audit.json',dict(status='passed',all_trace_metrics_reconciled=True,all_weights_unchanged=True,records=len(rows),training_used_test=False))
 emit('completed',records=len(rows),elapsed_s=time.monotonic()-start);mirror()
 subprocess.run([sys.executable,str(SINGLE/'refresh.py'),'--workbook'],check=True)
if __name__=='__main__':
 try:main()
 except BaseException as ex:
  write(RUN/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));emit('manual_stopped' if isinstance(ex,InterruptedError) else 'failed',error=repr(ex));mirror();raise
