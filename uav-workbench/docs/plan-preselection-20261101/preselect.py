"""Frozen-model diagnostic screening. Policies receive observations only."""
import json, hashlib, sys, time, traceback, gc
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).parent
OUT=BASE/'run01'
ROOT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
EXP=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
sys.path[:0]=[str(EXP/'priority-20261101'),str(EXP)]
from flood1km.env import Env
from flood1km.learners import Learner
from flood1km.train import seed_all
from coverage_controller import CoverageWatchdog,service_mask
from service_commitment import ServiceCommitment
from retry_guard import RetryGuard
from priority_train import decide as original_decide

LAYOUTS=(9001,9002,9003,9004,9401,9402,9403,9404)
SEEDS=(3101,3102)
CANDIDATES={
 'P0':dict(name='纯规划',algorithm=None,checkpoint=None,planner='immediate'),
 'P1':dict(name='HATRPO1024 原策略',algorithm='hatrpo',checkpoint=ROOT/'uav6-full1024-20261101/hatrpo/checkpoint-1024.pt',planner='off'),
 'P2':dict(name='HATRPO1024 加补漏',algorithm='hatrpo',checkpoint=ROOT/'uav6-full1024-20261101/hatrpo/checkpoint-1024.pt',planner='stall5'),
 'P3':dict(name='HASAC256 加补漏',algorithm='hasac',checkpoint=ROOT/'uav6-total256-20261101/hasac/latest.pt',planner='stall5'),
 'P4':dict(name='MAAC256 加补漏',algorithm='maac',checkpoint=ROOT/'uav6-total256-20261101/maac/latest.pt',planner='stall5')}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def json_default(x):
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,Path):return str(x)
 raise TypeError(type(x).__name__)
def write(p,data):
 p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False,default=json_default),encoding='utf8')
def load(candidate):
 if not candidate['algorithm']:return None
 ck=torch.load(candidate['checkpoint'],map_location='cpu',weights_only=False)['learner']
 model=Learner(candidate['algorithm'],ck['dim']);model.restore(ck);model.eval();model.check_finite();return model
class Controller:
 def __init__(self,model,planner,deterministic):
  self.model=model;self.mode=planner;self.deterministic=deterministic
  self.service=ServiceCommitment('commit');self.guard=RetryGuard();self.planner=CoverageWatchdog()
  if planner=='immediate':self.planner.active=True;self.planner.activation_step=1
 def choose(self,obs,boundary):
  forced,service=self.service.select(obs,boundary,np.full(4,8,dtype=np.int64))
  mask=service_mask(boundary,forced,service['phases']);mask,guard=self.guard.mask(obs,mask,service['phases'])
  if self.model is None:proposed=forced.copy()
  else:proposed=self.model.act(obs,mask,self.deterministic)[0]
  if self.mode=='off':action=proposed.copy();plan=dict(active=False,planned_agents=0)
  else:action,plan=self.planner.select(obs,mask,proposed,service['phases'])
  # Guard memory must reflect executed control, including planning overrides.
  self.guard.remember(action)
  controlled=np.array([x!='learned' for x in service['phases']])
  assert np.array_equal(action[controlled],forced[controlled])
  assert mask[np.arange(4),action].all()
  return action,dict(proposed=proposed,service=service,planning=plan,guard=guard,planner_changed=int(np.count_nonzero(action!=proposed)))

def summarize(rows):
 n=len(rows);assert n
 return dict(n=n,complete_count=sum(r['task_complete'] for r in rows),completion=float(np.mean([r['task_complete'] for r in rows])),
  RCR=float(np.mean([r['RCR'] for r in rows])),coverage=float(np.mean([r['coverage'] for r in rows])),
  capped_steps=float(np.mean([r['ST'] if r['task_complete'] else 300 for r in rows])),wall_seconds=float(np.mean([r['wall_seconds'] for r in rows])),
  ACC=float(np.mean([r['ACC'] for r in rows])),max_conflicts=max(r['conflicts'] for r in rows),
  all_rescued_count=sum(r['rescued']==20 for r in rows),
  mean_first_service_step=float(np.mean([r['FRT'] for r in rows if r['FRT'] is not None])) if any(r['FRT'] is not None for r in rows) else None,
  first_service_valid=sum(r['FRT'] is not None for r in rows),
  mean_all_service_step=float(np.mean([r['CRT'] for r in rows if r['CRT'] is not None])) if any(r['CRT'] is not None for r in rows) else None,
  all_service_valid=sum(r['CRT'] is not None for r in rows),
  planner_agent_fraction=float(np.mean([r['planner_agent_steps']/(4*r['ST']) for r in rows])),
  planner_changed_fraction=float(np.mean([r['planner_changed_agent_steps']/(4*r['ST']) for r in rows])),
  neural_executed_fraction=float(np.mean([r['neural_agent_steps']/(4*r['ST']) for r in rows])),
  undiscovered=sum(r['failure_counts']['undiscovered'] for r in rows),
  discovered_with_window_unserved=sum(r['failure_counts']['discovered_with_window_unserved'] for r in rows),
  discovered_without_window_unserved=sum(r['failure_counts']['discovered_without_window_unserved'] for r in rows),
  blocked_agent_steps=sum(r['blocked_agent_steps'] for r in rows),zero_gain_steps=sum(r['zero_gain_steps'] for r in rows))

def diagnostic(env,found_at):
 counts=dict(undiscovered=0,discovered_with_window_unserved=0,discovered_without_window_unserved=0)
 targets=[]
 for j,xy in enumerate(env.people):
  col,row=(xy//10).astype(int);series=env.water[:,:,row,col]
  safe=(series[:,0]<=.08)&(np.linalg.norm(series[:,1:],axis=1)<=.3)
  starts=np.flatnonzero(safe[:-2]&safe[1:-1]&safe[2:])
  possible=starts[starts>=found_at[j]] if found_at[j]>=0 else np.array([],int)
  category='served' if env.status[j]==2 else 'undiscovered' if env.status[j]==0 else 'discovered_with_window_unserved' if len(possible) else 'discovered_without_window_unserved'
  if category in counts:counts[category]+=1
  targets.append(dict(target=j,position=xy,status=int(env.status[j]),first_discovered_step=int(found_at[j]),category=category,
    legal_windows_after_discovery=int(len(possible)),note='post-run truth diagnostic; window existence does not prove reachability'))
 assert sum(counts.values())==20-env.metrics()['rescued']
 return counts,targets

def rollout(model,candidate,water,case,layout,mode,seed,directory):
 begin=time.perf_counter()
 seed_all(seed+layout);env=Env(water,layout);obs=env.observe();control=Controller(model,candidate['planner'],mode!='sample')
 found_at=np.where(env.status>0,0,-1);counts=dict(zero_gain_steps=0,blocked_agent_steps=0,planner_agent_steps=0,planner_changed_agent_steps=0,service_agent_steps=0,neural_agent_steps=0,ordinary_hover_agent_steps=0,service_interruption_events=0)
 key=f'{mode}-{seed}-{case}-{layout}';done=False
 with (directory/(key+'.jsonl')).open('w',encoding='utf8') as trace:
  while not done:
   old=env.pos.copy();previous_status=env.status.copy();previous_progress=env.progress.copy()
   action,detail=control.choose(obs,env.available())
   if candidate['planner']!='off':assert not (control.planner.known&~env.known).any(),'Inferred unobserved coverage'
   intended=np.array([env.intent(i,int(a))[0] for i,a in enumerate(action)])
   obs,_,done,info=env.step(action)
   blocked=(np.linalg.norm(intended-old,axis=1)>1e-5)&(np.linalg.norm(env.pos-old,axis=1)<1e-5)
   counts['zero_gain_steps']+=int(info['new_cells']==0);counts['blocked_agent_steps']+=int(blocked.sum())
   planned=detail['planning']['planned_agents'];service=sum(p!='learned' for p in detail['service']['phases'])
   counts['planner_agent_steps']+=planned;counts['service_agent_steps']+=service
   counts['planner_changed_agent_steps']+=detail['planner_changed']
   counts['neural_agent_steps']+=(4-service-planned) if model is not None else 0
   found_at[(previous_status==0)&(env.status>0)]=env.t
   interrupted=np.flatnonzero((previous_progress>0)&(env.progress==0)&(env.status!=2))
   counts['service_interruption_events']+=len(interrupted)
   counts['ordinary_hover_agent_steps']+=int(sum(action[i]==8 and detail['service']['phases'][i]=='learned' for i in range(4)))
   record=dict(step=env.t,positions=env.pos,actions=action,info=info,blocked=blocked,discovered=np.flatnonzero((previous_status==0)&(env.status>0)),served=np.flatnonzero((previous_status!=2)&(env.status==2)),service_interrupted=interrupted,**detail)
   trace.write(json.dumps(record,default=json_default,allow_nan=False)+'\n')
 failure,targets=diagnostic(env,found_at)
 np.savez_compressed(directory/(key+'-final.npz'),known=env.known,missing_cells=np.argwhere(~env.known),people=env.people,status=env.status,found_at=found_at,positions=env.pos)
 write(directory/(key+'-targets.json'),targets)
 return dict(case=case,layout=layout,stratum='old' if layout<9400 else 'new',mode=mode,action_seed=seed,
  **env.metrics(),**counts,wall_seconds=time.perf_counter()-begin,planner_activation=control.planner.activation_step,failure_counts=failure,trace=key+'.jsonl',targets=key+'-targets.json')

def preflight(cases):
 model=load(CANDIDATES['P1']);water=cases[0][1]
 e=Env(water,9001);c=Controller(model,'off',False);service=ServiceCommitment('commit');guard=RetryGuard()
 for step in range(12):
  obs=e.observe();mask=e.available();seed_all(77+step);a,_=c.choose(obs,mask)
  seed_all(77+step);b=original_decide(model,service,guard,obs,mask,False)[0]
  assert np.array_equal(a,b);e.step(a)
 checks=['P1 original action stream equivalence for 12 steps']
 for cid in ('P0','P2'):
  c=Controller(model if cid=='P2' else None,CANDIDATES[cid]['planner'],True);e=Env(water,9401)
  for _ in range(12):
   a,_=c.choose(e.observe(),e.available());assert not (c.planner.known&~e.known).any();e.step(a)
  checks.append(cid+' observed coverage and legal integrated control for 12 steps')
 # Only current observations and public bounds are passed to the controller.
 w2=np.array(water,copy=True);w2[1:]=123.
 e1,e2=Env(water,9402),Env(w2,9402);assert np.array_equal(e1.observe(),e2.observe())
 c1,c2=Controller(model,'stall5',False),Controller(model,'stall5',False)
 seed_all(88);a,_=c1.choose(e1.observe(),e1.available());seed_all(88);b,_=c2.choose(e2.observe(),e2.available());assert np.array_equal(a,b)
 checks.append('first decision invariant to changed future truth with identical observation')
 write(OUT/'preflight.json',dict(status='passed',checks=checks,scope='bounded startup checks; full-run invariants remain enabled'))
 del model,w2;gc.collect()

def choose(results):
 base=results['P0'];decisions={}
 def get(cid,stratum,mode):return summarize([r for r in results[cid] if r['stratum']==stratum and r['mode']==mode])
 for cid in CANDIDATES:
  gates=[]
  for stratum in ('old','new'):
   b=get('P0',stratum,'fixed')
   for mode in (('fixed',) if cid=='P0' else ('sample','fixed')):
    c=get(cid,stratum,mode)
    gates.append(dict(stratum=stratum,mode=mode,completion_nonlower=c['completion']>=b['completion']-1e-12,rescue_nonlower=c['RCR']>=b['RCR']-1e-12,conflict_ok=c['ACC']<=.02+1e-12,
      material_gain=c['completion']>=b['completion']+1/8-1e-12 or (abs(c['completion']-b['completion'])<1e-12 and abs(c['RCR']-b['RCR'])<1e-12 and c['capped_steps']<=b['capped_steps']-5+1e-12),candidate=c,baseline=b))
  eligible=all(g[k] for g in gates for k in ('completion_nonlower','rescue_nonlower','conflict_ok'))
  benefit=cid!='P0' and all(g['material_gain'] for g in gates if g['mode']=='sample')
  decisions[cid]=dict(engineering_gates_pass=eligible,learning_investment_pass=eligible and benefit,gates=gates)
 admitted=[cid for cid,d in decisions.items() if d['learning_investment_pass']]
 def rank(cid):
  s=summarize([r for r in results[cid] if r['mode']=='sample']);return s['completion'],s['RCR'],-s['capped_steps'],-s['ACC'],-s['wall_seconds']
 admitted.sort(key=rank,reverse=True)
 return dict(decisions=decisions,selected_learning_candidate=admitted[0] if admitted else None,
   next_action='expanded-development-validation-before-new-interface-training' if admitted else 'stop-current-combination-expansion; retain full diagnostics and design one separate falsifiable interface change',
   hierarchy_training_completed=False,added_training_episodes=0)

def main():
 OUT.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 manifest=ROOT/'uav6-pilot-20261101/data/dataset.json';items=json.loads(manifest.read_text(encoding='utf8'))['cases'];cases=[]
 for item in items:
  if item['split']=='validation':
   path=Path(item['path']);assert sha(path)==item['sha256'];cases.append((path.stem,np.load(path,mmap_mode='r',allow_pickle=False)))
 assert len(cases)==2
 source_paths=[Path(__file__),EXP/'coverage_controller.py',EXP/'service_commitment.py',EXP/'priority-20261101/retry_guard.py',EXP/'flood1km/env.py',EXP/'flood1km/learners.py']
 protocol=dict(version='preselection-v1.0',frozen_before_results=True,plan_sha256=sha(BASE/'HYRA无人机算法预选与训练计划书-v1.0.docx'),source_hashes={str(p):sha(p) for p in source_paths},
  manifest_sha256=sha(manifest),candidates=CANDIDATES,checkpoint_hashes={str(c['checkpoint']):sha(c['checkpoint']) for c in CANDIDATES.values() if c['checkpoint']},
  layouts=LAYOUTS,sample_seeds=SEEDS,rollouts=208,parent_fields=2,holdout50_used=False,added_training_episodes=0,
  gate='For old and new layout strata separately, sample AND fixed: completion and rescue >= P0, ACC<=0.02. Further investment requires sample completion gain >=1/8 OR equal completion/rescue with capped steps <=P0-5 in BOTH strata.',
  no_candidate_action='No automatic expansion to 1024; preserve failure evidence and register a separate mechanism experiment.',
  planner='Existing observed-history watchdog, immediate for P0; stall5 otherwise. Same service and RetryGuard, guard remembers actual final action.',
  diagnostics='Truth only after decision or at episode end; no hidden/future input to controller. Available service windows after discovery do not prove feasible arrival.',maximum_seconds=1800)
 write(OUT/'protocol.json',protocol);write(OUT/'status.json',dict(status='preflight',completed_rollouts=0,target=208))
 preflight(cases);results={};completed=0
 previous=json.loads((ROOT/'uav6-combinations-20261101/HATRPO1024.json').read_text(encoding='utf8'))['episodes']
 for cid,candidate in CANDIDATES.items():
  directory=OUT/cid;directory.mkdir();model=load(candidate);rows=[]
  modes=[('fixed',0)] if cid=='P0' else [('sample',s) for s in SEEDS]+[('fixed',0)]
  for mode,seed in modes:
   for case,water in cases:
    for layout in LAYOUTS:
     if time.perf_counter()-start>1800:raise TimeoutError('Predeclared 1800 second cost limit reached; do not reduce denominator')
     r=rollout(model,candidate,water,case,layout,mode,seed,directory)
     if cid=='P1' and mode=='sample' and layout<9400:
      old=next(x for x in previous if Path(x['case']).stem==case and x['layout']==layout and x['action_seed']==seed)
      for k in ('coverage','RCR','ACC','ST','rescued','conflicts','task_complete'):assert abs(float(r[k])-float(old[k]))<1e-10,(case,layout,k,r[k],old[k])
     rows.append(r);completed+=1
     write(directory/'results.json',dict(episodes=rows,summary=summarize(rows),note='mixed execution summaries are inventory only; mode-separated selection follows'))
     write(OUT/'status.json',dict(status='running',candidate=cid,completed_rollouts=completed,target=208,elapsed_s=time.perf_counter()-start))
  results[cid]=rows;write(OUT/'partial-results.json',results);del model;gc.collect()
  print(cid,json.dumps({m:summarize([r for r in rows if r['mode']==m]) for m in set(r['mode'] for r in rows)},ensure_ascii=True),flush=True)
 for path,expected in protocol['source_hashes'].items():assert sha(path)==expected
 for path,expected in protocol['checkpoint_hashes'].items():assert sha(path)==expected
 assert completed==208
 write(OUT/'results.json',results);write(OUT/'selection.json',choose(results))
 write(OUT/'status.json',dict(status='completed',completed_rollouts=completed,target=208,elapsed_s=time.perf_counter()-start,checkpoints_unchanged=True,sources_unchanged=True,added_training_episodes=0))

if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():write(OUT/'failure.json',dict(traceback=traceback.format_exc()));write(OUT/'status.json',dict(status='failed',failure='failure.json'))
  raise
