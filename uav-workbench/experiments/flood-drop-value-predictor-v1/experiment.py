"""Supervised short-horizon action-value prediction; no temporal-difference RL."""
from pathlib import Path
import sys,copy,time,random,json,argparse,traceback,hashlib,shutil
import numpy as np
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2-coordination-v1'))
from train_coordination import read,write,sha,summarize
from variable_dispatch import DispatchEnv

def candidates(env):
 mask,expert,options,f=env.choices();actions=[expert.copy()]
 # Same proposal mechanism for linear and deep predictors. At most one UAV changes.
 for i in range(4):
  current=options[i][expert[i]]
  if current['person']>=0 and env.visible[current['person']] and np.linalg.norm(env.seen_xy[current['person']]-env.pos[i])<=env.cfg.drop_radius+1e-8:continue
  claimed={options[j][expert[j]]['person'] for j in range(4) if j!=i and options[j][expert[j]]['person']>=0}
  choices=[]
  for a in np.flatnonzero(mask[i]):
   o=options[i][a]
   if o['person']==current['person'] and np.allclose(o['xy'],current['xy']):continue
   if o['person'] in claimed or (o['person']<0 and o['region']<0):continue
   choices.append(int(a))
  if choices:
   a=max(choices,key=lambda a:(options[i][a]['person']>=0,options[i][a]['utility']))
   proposal=expert.copy();proposal[i]=a;actions.append(proposal)
 # All inputs are current public observations, never future water/fall/deadline.
 ids=np.flatnonzero((env.belief==1)&env.discovered)
 context=np.r_[env.pos.ravel()/1000,env.t/env.cfg.duration,env.known.mean(),len(ids)/env.cfg.people,
                np.mean(env.measured[0]),np.mean(np.linalg.norm(env.measured[1:],axis=0))]
 x=np.stack([np.r_[f[np.arange(4),a].ravel(),context] for a in actions]).astype(np.float32)
 assert np.isfinite(x).all()
 return actions,x

def clone(env):return copy.deepcopy(env,{id(env.water):env.water})

def branch(env,action,horizon):
 e=clone(env);sup=int((e.status==2).sum());end=min(e.t+horizon,e.cfg.duration)
 _,_,done,_=e.step(action)
 while not done and e.t<end:_,_,done,_=e.step(e.choices()[1])
 return int((e.status==2).sum())-sup

class Predictor(nn.Module):
 def __init__(self,dim,deep):
  super().__init__();self.net=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1)) if deep else nn.Linear(dim,1)
 def forward(self,x):return self.net(x).squeeze(-1)

def collect(out,cases,guard,status):
 records=[];started=time.monotonic()
 for index,c in enumerate(cases):
  env=DispatchEnv(np.load(c['field']['path'],mmap_mode='r',allow_pickle=False),c['scenario'])
  while True:
   guard()
   if env.t%120==0:
    actions,x=candidates(env)
    if len(actions)>1:
     values=[branch(env,a,60) for a in actions]
     record=dict(case=c['id'],parent=c['field']['parent'],time_s=env.t,x=x.tolist(),actions=[a.tolist() for a in actions],supplied_next60=values,gain_over_default=[v-values[0] for v in values])
     records.append(record)
     with (out/'counterfactual-training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
   _,_,done,_=env.step(env.choices()[1])
   if done:break
  status('collecting_training_labels',training_layouts_completed=index+1,total_training_layouts=len(cases),decision_states=len(records),collection_elapsed_s=time.monotonic()-started)
 return records

def calibrate(model,records,mu,sd):
 totals=[]
 with torch.no_grad():
  predictions=[model(torch.tensor((np.asarray(r['x'],np.float32)-mu)/sd)).numpy() for r in records]
 for margin in (0.,.05,.15,.3,1e9):
  gains=[];changed=0
  for r,v in zip(records,predictions):
   a=int(np.argmax(v));a=a if v[a]-v[0]>margin else 0
   gains.append(r['gain_over_default'][a]);changed+=int(a!=0)
  totals.append(dict(margin=margin,gain=sum(gains),changes=changed,states=len(gains)))
 # Holdout TRAIN parents only; ties prefer fewer overrides and greater margin.
 return max(totals,key=lambda r:(r['gain'],-r['changes'],r['margin'])),totals

def evaluate(model,mu,sd,margin,case):
 env=DispatchEnv(np.load(case['field']['path'],mmap_mode='r',allow_pickle=False),case['scenario']);changes=steps=0
 while True:
  actions,x=candidates(env)
  with torch.no_grad():v=model(torch.tensor((x-mu)/sd)).numpy()
  a=int(v.argmax());a=a if v[a]-v[0]>margin else 0
  changes+=int(a!=0);steps+=1
  _,_,done,_=env.step(actions[a])
  if done:break
 r=dict(**env.metrics(),case=case['id'],parent=case['field']['parent'],override_steps=changes,decision_steps=steps)
 r['persons']=[dict(id=j,fall=int(env.fall[j]),deadline=int(env.deadline[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(env.cfg.people)]
 assert r['supplied']+r['expired']+r['outside']==env.cfg.people
 for person in r['persons']:
  if person['status']==2:assert person['fall']+10<=person['supplied_at']<=person['deadline']
 return r

def run(out,previous):
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();cases=read(previous/'scenarios.json');old=read(previous/'protocol.json')
 def status(state,**kw):write(out/'status.json',dict(status=state,elapsed_s=time.monotonic()-start,**kw))
 def guard():
  if (out/'STOP').exists():raise InterruptedError('Stop requested')
  if time.monotonic()-start>7200:raise TimeoutError('Runtime limit')
 protocol=dict(version='supervised-counterfactual-value-v1',date='2026-11-03',previous=str(previous),scenario_sha256=sha(previous/'scenarios.json'),source_hashes={**old['source_hashes'],str(Path(__file__).resolve()):sha(__file__)},water_hashes=old['water_hashes'],
  label='Supplied people in next60s after one5s proposal then full nearest4 planner, relative to default; future truth only in TRAIN offline label branches.',
  features='40 observed chosen-task features plus13 current public context features; no true fall/deadline, future people, future water or status as predictor input.',
  budget='32 existing TRAIN layouts, sample every120s, at most5 branches/state, horizon60s. Linear and MLP each32 supervised epochs;0/16/32 checkpoints. These are not32 RL episodes.',
  split='First12 TRAIN parents for fitting, last4 TRAIN parents for calibration/early choice.6 DEV hydraulic parents never used for fitting/calibration. All DEV previously seen in other experiments.',
  gate='At least5 positive and5 negative nondefault training branch gains needed; otherwise stop as insufficient signal rather than train a constant predictor.',
  calibration='Select override margin from0,.05,.15,.3,infinity using TRAIN holdout branch gains only; equal gains prefer fewer changes. No DEV-selected margin.',
  baselines='All four complete archived planners on identical cases; linear predictor has same features, labels, actions and calibration as MLP.',
  metrics='People supplied primary; all and ceil95% scenes separately retained; every checkpoint/pressure panel saved. No promised15pp.',
  limitations='One-step deviations with60s oracle TRAIN continuations, deployment repeatedly overrides and may distribution-shift; no full rescue deadline lookahead. Threshold2m/1mps unchanged but prior sampled forced events0. Synthetic fields unchanged; no final test.')
 def verify():
  assert sha(previous/'scenarios.json')==protocol['scenario_sha256']
  for path,h in {**protocol['source_hashes'],**protocol['water_hashes']}.items():assert sha(path)==h,path
 write(out/'protocol.json',protocol);write(out/'scenarios.json',cases);status('preparing')
 for f in [*previous.glob('planner-*.json'),previous/'trigger-audit.json']:
  shutil.copy2(f,out/f.name)
 try:
  verify();torch.set_num_threads(2);torch.manual_seed(9501);np.random.seed(9501)
  # Public-input isolation, cloning and explicit opt-out reproduce strong planning.
  c=cases['train'][0];e=DispatchEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario']);a,x=candidates(e)
  other=clone(e);other.person[:]=999;other.fall[:]+=1;other.deadline[:]+=1;aa,xx=candidates(other)
  assert np.array_equal(x,xx) and all(np.array_equal(u,v) for u,v in zip(a,aa))
  oldtime=e.t;oldpos=e.pos.copy();branch(e,a[0],60);assert e.t==oldtime and np.array_equal(e.pos,oldpos)
  write(out/'readiness.json',dict(hidden_truth_isolated=True,branch_does_not_mutate_origin=True,feature_dim=x.shape[1]))
  records=collect(out,cases['train'],guard,status)
  parents=list(dict.fromkeys(c['field']['parent'] for c in cases['train']));fit=[r for r in records if r['parent'] in parents[:12]];hold=[r for r in records if r['parent'] in parents[12:]]
  gains=[g for r in fit for g in r['gain_over_default'][1:]]
  audit=dict(states=len(records),fit_states=len(fit),holdout_states=len(hold),positive=sum(g>0 for g in gains),negative=sum(g<0 for g in gains),zero=sum(g==0 for g in gains),fit_parents=parents[:12],holdout_parents=parents[12:]);write(out/'label-audit.json',audit)
  if audit['positive']<5 or audit['negative']<5:
   verify();write(out/'summary.json',dict(status='insufficient_training_signal',audit=audit));status('insufficient_training_signal');return
  x=np.concatenate([r['x'] for r in fit]).astype(np.float32);y=np.concatenate([r['supplied_next60'] for r in fit]).astype(np.float32)
  mu=x.mean(0);sd=np.maximum(x.std(0),.01);tx=torch.tensor((x-mu)/sd);ty=torch.tensor(y)
  results={};evals=0
  for name in ('linear','deep'):
   folder=out/name;folder.mkdir();torch.manual_seed(9501);model=Predictor(x.shape[1],name=='deep');optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);results[name]={}
   for epoch in range(33):
    guard();status('supervised_training',model=name,epoch=epoch,evaluation_rollouts=evals)
    if epoch:
     model.train();losses=[]
     for ids in torch.randperm(len(tx)).split(128):
      prediction=model(tx[ids]);loss=nn.functional.smooth_l1_loss(prediction,ty[ids]);assert torch.isfinite(loss)
      optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss))
     with (folder/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(epoch=epoch,training_huber_loss=float(np.mean(losses))))+'\n')
    if epoch in (0,16,32):
     model.eval();cal,table=calibrate(model,hold,mu,sd);write(folder/f'calibration-{epoch:04}.json',dict(selected=cal,all_margins=table))
     ck=folder/f'checkpoint-{epoch:04}.pt';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),mean=mu,std=sd,margin=cal['margin'],epoch=epoch,architecture=name,protocol_sha256=sha(out/'protocol.json')),ck)
     panel={}
     for n,group in cases['development'].items():
      rows=[]
      for c in group:
       guard();status('evaluation',model=name,epoch=epoch,profile=n,evaluation_rollouts=evals);rows.append(evaluate(model,mu,sd,cal['margin'],c));evals+=1
      panel[n]=dict(rows=rows,summary=summarize(rows))
     write(folder/f'development-{epoch:04}.json',panel);results[name][str(epoch)]={n:p['summary'] for n,p in panel.items()}
     print(json.dumps(dict(model=name,epoch=epoch,margin=cal['margin'],main=panel['main40']['summary'])),flush=True)
  verify();base=read(previous/'summary.json')['baselines']
  write(out/'summary.json',dict(status='completed',training_layouts=32,supervised_epochs_per_model=32,evaluation_rollouts=evals,results=results,baselines=base,baseline_reused=True,source_water_unchanged=True,formal_test=False));status('completed',evaluation_rollouts=evals)
 except BaseException as ex:
  write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);a=p.parse_args();run(a.output,a.previous)
