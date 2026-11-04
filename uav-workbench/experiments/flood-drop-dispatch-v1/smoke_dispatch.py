"""Disposable small-fixture learner update check; not part of the64 episodes."""
import json,random
import numpy as np
import torch
from dispatch_env import DispatchEnv,Config
from train import warmstart,seed,Learner,METHODS
from test_dispatch import field,scenario

def collect(model=None):
    cfg=Config(duration=60,window=30);env=DispatchEnv(field(.1),scenario(cfg,True));rows=[];done=False
    obs=env.observe();mask,preferred,_,_=env.choices()
    while not done:
        if model is None:action=preferred;logs=np.zeros(4);value=0.
        else:action,logs,value=model.act(obs,mask,False)
        nxt,reward,done,_=env.step(action);nm,npref,_,_=env.choices()
        rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;preferred=npref
    for i,r in enumerate(rows):r['next_action']=rows[i+1]['action'] if i+1<len(rows) else np.full(4,8,dtype=np.int64)
    return rows

if __name__=='__main__':
    torch.set_num_threads(2);seed(8301);teacher=collect();results=[]
    for method in METHODS:
        seed(8301);model=Learner(method,760);warm=warmstart(model,teacher,2);rows=collect(model)
        result=model.update_off_policy(rows) if method=='maac' else model.update_on_policy(rows)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        results.append(dict(method=method,warmstart=warm,update=result,finite=True))
    print(json.dumps(dict(status='passed',disposable_fixture=True,results=results),allow_nan=False))
