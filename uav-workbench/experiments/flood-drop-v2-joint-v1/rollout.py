from pathlib import Path
import sys,copy,random
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-v2'))
sys.path.insert(0,str(HERE.parent/'flood-drop-variable-v1'))
from candidate_env import DispatchEnv
from pilot import Learner,seed,read,write,sha
from screen import summarize
class ResidualEnv(DispatchEnv):
    def choices(self):
        mask,expert,options,features=super().choices()
        for i,a in enumerate(expert):
            options[i][8]=copy.deepcopy(options[i][a]);features[i,8]=features[i,a].copy();mask[i,8]=True
        return mask,np.full(self.cfg.uavs,8,dtype=np.int64),options,features

def initialize(model):
    heads=[model.policy.head] if model.algorithm=='mat' else [a[-1] for a in model.actors]
    with torch.no_grad():
        for h in heads:h.weight.zero_();h.bias.zero_();h.bias[8]=1.

def quality(result):
    s=result['summary'];return (s['people_supplied'],s['all_successes'],s['at_least95_successes'])

def episode(model,case,collect=False):
    env=ResidualEnv(np.load(case['field']['path'],mmap_mode='r',allow_pickle=False),case['scenario'])
    obs=env.observe();mask,_,options,_=env.choices();done=False;rows=[];ret=0.;free=used=0
    while not done:
        action,logs,value=model.act(obs,mask,not collect)
        choices=mask.sum(1)>1;free+=int(choices.sum())
        used+=sum(bool(choices[i] and (options[i][action[i]]['person']!=options[i][8]['person'] or
            not np.allclose(options[i][action[i]]['xy'],options[i][8]['xy'],rtol=0,atol=1e-9))) for i in range(4))
        nxt,reward,done,_=env.step(action);nm,_,noptions,_=env.choices();ret+=reward
        if collect:rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
        obs=nxt;mask=nm;options=noptions
    for i,r in enumerate(rows):r['next_action']=rows[i+1]['action'] if i+1<len(rows) else np.full(4,8,dtype=np.int64)
    result=dict(**env.metrics(),case=case['id'],parent=case['field']['parent'],return_sum=ret,alternative_task_choice_fraction=used/max(1,free))
    assert result['supplied']+result['expired']+result['outside']==env.cfg.people
    if not collect:
        result['persons']=[dict(id=j,exposure_id=case['scenario']['events'][j]['exposure_id'],fall=int(env.fall[j]),deadline=int(env.deadline[j]),
            first_seen=float(env.first_seen[j]),supplied_at=float(env.delivered_at[j]),status=int(env.status[j])) for j in range(env.cfg.people)]
        for p in result['persons']:
            if p['status']==2:assert p['fall']+env.cfg.drop_seconds<=p['supplied_at']<=p['deadline']
    return result,rows

def evaluate(model,cases):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state())
    try:
        rows=[episode(model,c)[0] for c in cases];return dict(rows=rows,summary=summarize(rows))
    finally:random.setstate(saved[0]);np.random.set_state(saved[1]);torch.set_rng_state(saved[2])

