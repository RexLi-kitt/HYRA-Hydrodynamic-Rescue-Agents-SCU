"""Unmodified referee evaluation: pure initial, checkpoint, or zero-residual.

Tuning/development maps are not blind tests. No teacher or rescue rule is run.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import core as c
import micro

s=c.search
TUNING=[41004,42012,51001]


def full(policy, parents, modes):
    ds=s.Dataset(); rows=[]
    for parent in parents:
      for mode in modes:
        s.seed_all(581000+parent); b=ds.bridge(parent); trace=[]
        while not b.done:
            o=b.observation()
            near=(np.linalg.norm(o.seen_xy[None,:,:]-o.pos[:,None,:],axis=-1)<=1.+1e-9)&o.visible[None,:]&o.active[None,:]
            owners=np.asarray(o.owner).copy()
            with torch.no_grad(): a,_,_=policy.act(c.encode(o),deterministic=mode=='greedy')
            a=a.cpu().numpy(); after,done,metrics,rec=s.step(b,a,'local_mask')
            targets=a[:,1]-1
            correct=sum(a[i,0]==8 and targets[i]>=0 and near[i,targets[i]] for i in range(4))
            continued=sum(a[i,0]==8 and targets[i]>=0 and owners[targets[i]]==i for i in range(4))
            rec.update(own_near_agents=int(near.any(axis=1).sum()),correct_near_hover=int(correct),
                       public_owner_continuations=int(continued))
            trace.append(rec)
        rows.append(dict(parent=parent,roster=0,mode=mode,evaluation_seed=581000+parent,metrics=b.env.metrics(),
            funnel={k:sum(r[k] for r in trace) for k in ('own_near_agents','correct_near_hover',
                    'public_owner_continuations','service_starts','delivered','conflicts','boundary')},trace=trace))
        print(json.dumps(dict(parent=parent,mode=mode,metrics=b.env.metrics()),allow_nan=False),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,choices=s.SEEDS,required=True)
    ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--zero-rescue',action='store_true')
    ap.add_argument('--maps',choices=('tuning','development','none'),default='tuning')
    ap.add_argument('--micro-cases',type=int,default=12)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.micro_cases<0: ap.error('negative micro count')
    out=a.output.resolve()
    if not out.is_relative_to(c.HERE.resolve()) or out.exists(): ap.error('output must be a new experiment file')
    torch.set_num_threads(1); p,_,_=c.initialize(a.seed)
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location=p.device,weights_only=False)
        if ck['schema']!='MAPPO_RESCUE_V1' or ck['seed']!=a.seed:raise ValueError('checkpoint mismatch')
        if ck['code_sha256']['core.py']!=s.sha(c.HERE/'core.py'):raise ValueError('policy code changed')
        p.actor.load_state_dict(ck['actor']);p.critic.load_state_dict(ck['critic'])
    if a.zero_rescue:
        with torch.no_grad():p.actor.rescue[-1].weight.zero_();p.actor.rescue[-1].bias.zero_()
    p.actor.eval();p.critic.eval()
    micros={mode:micro.evaluate(p,range(200000,200000+a.micro_cases),mode=mode) for mode in ('greedy','sampled')}
    parents={'tuning':TUNING,'development':s.DEV,'none':[]}[a.maps]
    data=dict(seed=a.seed,checkpoint=str(a.checkpoint) if a.checkpoint else None,
        checkpoint_sha256=s.sha(a.checkpoint) if a.checkpoint else None,zero_rescue=a.zero_rescue,
        description='Network actions; development/tuning maps are non-blind; micro fixtures idle teammates',
        micro=micros,full=full(p,parents,('sampled','greedy')))
    out.parent.mkdir(parents=True,exist_ok=True);s.json_write(out,data)


if __name__=='__main__':main()
