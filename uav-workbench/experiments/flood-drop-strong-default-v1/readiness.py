import argparse,copy
from pathlib import Path
import numpy as np
import torch
from train import read,write,seed,model_for,ResidualEnv
from strong_env import DispatchEnv
from coord_rollout import ResidualEnv as OldEnv

def check(out):
 p=read(out/'protocol.json');cases=read(out/'scenarios.json');case=cases['train'][0]
 water=np.load(case['field']['path'],mmap_mode='r',allow_pickle=False);scene=case['scenario'];torch.set_num_threads(2)
 e=ResidualEnv(water,scene);assert e.observe().shape==(4,1240)
 before=e.observe().copy();mask=e.choices()[0].copy()
 e.person[:]=999.;e.fall[:]+=7;e.deadline[:]+=7
 assert np.array_equal(before,e.observe()) and np.array_equal(mask,e.choices()[0]),'future truth leakage'
 order=np.random.default_rng(123).permutation(e.cfg.people);other=copy.deepcopy(scene);other['events']=[scene['events'][i] for i in order]
 assert np.array_equal(ResidualEnv(water,scene).observe(),ResidualEnv(water,other).observe())
 checks={}
 for run in p['runs']:
  seed(p['seed']);m=model_for(run['method'],run['recipe']);e=ResidualEnv(water,scene);ref=DispatchEnv(water,scene);rows=[]
  weights={k:v.detach().clone() for k,v in m.state_dict().items()}
  for _ in range(12):
   obs=e.observe();mask=e.choices()[0];a,lp,v=m.act(obs,mask,True);assert (a==8).all()
   nxt,r,done,_=e.step(a);ref.step(ref.choices()[1]);assert np.array_equal(e.pos,ref.pos) and np.array_equal(e.status,ref.status)
   rows.append(dict(obs=obs,mask=mask,action=a,reward=r,done=float(done),next_obs=nxt,next_mask=e.choices()[0],logp=lp,value=v,next_action=a))
  if run['method']=='maac':m.update_off_policy(rows)
  else:m.update_on_policy(rows)
  m.check_finite();assert any(not torch.equal(weights[k],v) for k,v in m.state_dict().items())
  restored=model_for(run['method'],run['recipe']);restored.restore(m.checkpoint());assert np.array_equal(m.act(nxt,e.choices()[0],True)[0],restored.act(nxt,e.choices()[0],True)[0])
  checks[run['id']]=dict(initial_exact_strong_planning=True,finite_actual_update=True,restore=True)
 # Inspect old learned logits on a full deterministic DEV trajectory; no new training.
 old=Path(p['previous']);probe=cases['development']['main40'][0];w=np.load(probe['field']['path'],mmap_mode='r');diagnosis={}
 for method in ('maac','mat','happo'):
  ck0=torch.load(old/method/'checkpoint-0000.pt',map_location='cpu',weights_only=False)
  ck=torch.load(old/method/'checkpoint-0032.pt',map_location='cpu',weights_only=False)
  model=model_for(method,ck['recipe']);model.restore(ck['learner']);model.eval()
  env=OldEnv(w,probe['scenario']);margins=[];probs=[];alternative=0;free=0
  while True:
   obs=env.observe();mask=env.choices()[0];a,lp,_=model.act(obs,mask,True)
   with torch.no_grad():
    o=model.tensor(obs)[None]
    if method=='mat':logits=model.policy(o,model.tensor(a,torch.long)[None])[0][0]
    else:logits=torch.stack([net(o[:,i])[0] for i,net in enumerate(model.actors)])
    for i in range(4):
     valid=mask[i].copy();valid[8]=False
     if valid.any():
      x=logits[i].cpu().numpy();margins.append(float(x[8]-x[valid].max()));probs.append(float(torch.softmax(logits[i].masked_fill(~model.tensor(mask[i],torch.bool),-1e9),-1)[8]));free+=1;alternative+=int(a[i]!=8)
   _,_,done,_=env.step(a)
   if done:break
  diagnosis[method]=dict(case=probe['id'],checkpoint=32,free_decisions=free,nondefault_action_count=alternative,default_margin_min=min(margins),default_margin_mean=float(np.mean(margins)),default_probability_mean=float(np.mean(probs)),scope='One archived full DEV trajectory; diagnostic, not generalization.')
 write(out/'old-default-diagnostic.json',diagnosis)
 write(out/'readiness.json',dict(status='passed',hidden_truth_isolated=True,initial_ID_permutation=True,methods=checks))
 print('Readiness passed; old default-logit diagnostic saved.',flush=True)

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);check(a.parse_args().output)
