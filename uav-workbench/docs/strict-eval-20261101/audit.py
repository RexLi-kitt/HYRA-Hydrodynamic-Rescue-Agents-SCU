"""Lightweight counterfactual information audit using TRAIN data, no test access."""
import copy, importlib.util, json, sys, hashlib, time
from pathlib import Path
import numpy as np
import torch
BASE=Path(__file__).resolve().parent;ROOT=BASE.parent
SINGLE=ROOT/'uav-singles1024-20261101'
spec=importlib.util.spec_from_file_location('single_audit_adapter',SINGLE/'singles.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
s.bootstrap();e=s.e;torch.set_num_threads(1)
def run():
 row=e.read(SINGLE/'run01/dataset.json')['cases'][0];water=np.load(row['path'],mmap_mode='r',allow_pickle=False)
 tests=[]
 for alg in s.ALGOS:
  e.seed_all(54321);models=[e.adapted(alg)[0]];model_hash=e.model_hash(models)
  env=e.Env(water,770001);controller=e.Controller()
  for step in range(18):
   obs=env.observe();mask=env.available()
   if step in (0,6,12,17):
    # All historical measurements stay fixed; changes live only in hidden truth.
    cf=copy.copy(env);cf.people=env.people.copy();cf.water=np.array(env.water,copy=True)
    hidden=np.flatnonzero(env.status==0);unseen=np.argwhere(~env.known)
    rng=np.random.default_rng(980001+step)
    selected=unseen[rng.choice(len(unseen),len(hidden),replace=False)]
    cf.people[hidden]=selected[:,::-1]*10+5
    cf.water[env.t,:,~env.known]=.123
    cf.water[env.t+1:,0]=.75;cf.water[env.t+1:,1]=-.27;cf.water[env.t+1:,2]=.19
    assert np.array_equal(obs,cf.observe()) and np.array_equal(mask,cf.available())
    rng_state=e.rng_state();one=copy.deepcopy(controller).decide(models,obs,mask,s.ROLE)
    e.restore_rng(rng_state);two=copy.deepcopy(controller).decide(models,cf.observe(),cf.available(),s.ROLE)
    assert all(np.array_equal(one[i],two[i]) for i in (0,1,2,3,4)),(alg,step)
    assert one[5]==two[5];e.restore_rng(rng_state)
    tests.append(dict(algorithm=alg,step=step,hidden_targets_changed=len(hidden),future_changed=True,unknown_current_changed=True,observation_mask_and_actions_equal=True))
   action=controller.decide(models,obs,mask,s.ROLE)[0];env.step(action)
  assert e.model_hash(models)==model_hash
  del models,env,cf,controller
 result=dict(status='passed',tested_states=len(tests),tests=tests,scope='Six single-learner parent policies and shared controller; train field only; no learner updates. Pair-specific ownership tested separately by strict preflight.',
  findings=['Only observed target records are passed to policies; true target IDs/coordinates are not direct actor inputs.',
   'service_safe reads interval endpoint truth inside environment outcome calculation, after the action. It is not called by action generation.',
   'Validation has6 parentwaterfields but only4 unique targetlayout seeds; repeated checkpoint/control selection can overfit this development suite.',
   'Ideal sensing/communication and shared coverage/service controllers limit interpretation of high success.'],
  test_fields_generated=False,source_hashes={str(p):e.sha(p) for p in (Path(__file__),e.EXP/'flood1km/env.py',e.EXP/'flood1km/learners.py',ROOT/'uav-linked-training-20261101/observed_planner.py',e.EXP/'service_commitment.py')})
 e.write(BASE/'information-audit.json',result);print(json.dumps(dict(status='passed',states=len(tests),test_fields_generated=False)))
if __name__=='__main__':run()
