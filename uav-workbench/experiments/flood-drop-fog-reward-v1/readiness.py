from pathlib import Path
import argparse,copy
import numpy as np
import torch
from train import read,write,seed,model_for,ResidualEnv,ReferenceEnv
from fog_rollout import relative_reward

def check(out):
 p=read(out/'protocol.json');cases=read(out/'scenarios.json');c=cases['train'][0];w=np.load(c['field']['path'],mmap_mode='r',allow_pickle=False)
 # Radius reduces observable area, not actual people or flood trajectories.
 envs=[]
 for radius in (40,60,100):
  s=copy.deepcopy(c['scenario']);s['config']['sensor']=radius;envs.append(ResidualEnv(w,s))
 assert np.all(~envs[0].known|envs[1].known) and np.all(~envs[1].known|envs[2].known)
 assert all(np.array_equal(envs[0].person,e.person) and np.array_equal(envs[0].fall,e.fall) for e in envs)
 e=envs[0];obs=e.observe().copy();m=e.choices()[0].copy();e.person[:]=999;e.fall[:]+=7;e.deadline[:]+=7
 assert np.array_equal(obs,e.observe()) and np.array_equal(m,e.choices()[0])
 # Exact terminal comparison includes baseline deliveries after policy stops.
 assert sum(relative_reward(d,b,a,'relative_people') for d,b,a in [(1,0,0),(0,0,1),(2,1,5)])==10*(3-5)
 assert sum(relative_reward(d,b,a,'absolute_people') for d,b,a in [(1,0,0),(0,0,1),(2,1,5)])==30
 torch.set_num_threads(2);checks={}
 for run in p['runs']:
  seed(p['seed']);model=model_for(run['method'],run['recipe']);e=ResidualEnv(w,c['scenario']);ref=ReferenceEnv(w,c['scenario']);rows=[]
  before={k:v.detach().clone() for k,v in model.state_dict().items()}
  for _ in range(12):
   obs=e.observe();mask=e.choices()[0];a,lp,v=model.act(obs,mask,True);assert (a==8).all()
   nxt,_,done,_=e.step(a);ref.step(ref.choices()[1]);assert np.array_equal(e.pos,ref.pos) and np.array_equal(e.status,ref.status)
   rows.append(dict(obs=obs,mask=mask,action=a,reward=0.,done=float(done),next_obs=nxt,next_mask=e.choices()[0],logp=lp,value=v,next_action=a))
  if run['method']=='maac':model.update_off_policy(rows)
  else:model.update_on_policy(rows)
  model.check_finite();assert any(not torch.equal(before[k],v) for k,v in model.state_dict().items())
  restored=model_for(run['method'],run['recipe']);restored.restore(model.checkpoint());assert np.array_equal(model.act(nxt,e.choices()[0],True)[0],restored.act(nxt,e.choices()[0],True)[0])
  checks[run['id']]=dict(default_identity=True,finite_update=True,restore=True)
 write(out/'readiness.json',dict(status='passed',fog_observability_nested=True,physical_people_unchanged=True,hidden_truth_isolated=True,reward_terminal_telescoping=True,methods=checks))
 print('Readiness passed: visibility, hidden truth, terminal relative reward, six finite learners.',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);check(p.parse_args().output)
