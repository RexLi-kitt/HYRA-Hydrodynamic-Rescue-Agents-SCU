from pathlib import Path
import json,time
import numpy as np
import torch
from environment import Config,Env,make_scenario
from pilot import read,rollout,Learner,ALGORITHMS,seed,write
import argparse

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2);seed(8701);field=read(a.dataset)['cases'][0];water=np.load(field['path'],mmap_mode='r')
    cfg=Config();case=dict(id='smoke-not-training',field=field,scenario=make_scenario(water,12345,cfg));start=time.monotonic()
    baseline,rows=rollout(None,case,collect=True);checks=[];print(json.dumps(dict(planner=baseline,seconds=time.monotonic()-start)),flush=True)
    for algorithm in ALGORITHMS:
        model=Learner(algorithm,len(rows[0]['obs'][0]));batch=[]
        # Collect short real learner rollouts, never fabricate log probabilities.
        env=Env(water,case['scenario']);obs=env.observe()
        for _ in range(16):
            mask=env.control()[0];act,logs,value=model.act(obs,mask);nxt,reward,done,_=env.step(act)
            batch.append(dict(obs=obs,mask=mask,action=act,logp=logs,value=value,reward=reward,done=float(done),next_obs=nxt,next_mask=env.control()[0]));obs=nxt
        for i,row in enumerate(batch):row['next_action']=batch[i+1]['action'] if i+1<len(batch) else np.full(4,8,int)
        # Smoke truncation is marked terminal only for this disposable update test.
        batch[-1]['done']=1.
        loss=model.update_off_policy(batch) if algorithm in ('hasac','maac') else model.update_on_policy(batch)
        model.check_finite();checks.append(dict(algorithm=algorithm,updates=model.updates,loss=loss));print(json.dumps(checks[-1]),flush=True)
    write(a.output,dict(status='passed',checks=checks,baseline=baseline,elapsed_s=time.monotonic()-start,scope='discarded smoke models; not pilot training records'))
