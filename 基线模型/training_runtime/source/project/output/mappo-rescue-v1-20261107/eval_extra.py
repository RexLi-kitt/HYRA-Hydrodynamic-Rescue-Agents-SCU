"""Pre-registered 12-case SDK dev80 cross-domain evaluation, after freezing.

This file does nothing on import. Invoke only after the candidate is fixed; it
requires an explicit panel path and checkpoint. It loads no other trial data.
The selected 12 cases are development materials, not a blind test.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
CASES={(parent,family,roster) for parent,family in (
    (72110001,'ordinary'),(95851001,'ordinary'),(88520001,'continuous'),
    (88520001,'rewet'),(95861001,'continuous'),(95861001,'rewet')) for roster in (0,1)}


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


class SourcePolicy:
    def __init__(self,c,seed,mode,device,rng_seed):
        self.c=c;self.mode=mode
        self.policy,_,_=c.initialize(seed,device)
        self.policy.actor.eval();self.policy.critic.eval()
        c.search.seed_all(rng_seed)

    def act(self,o):
        with torch.no_grad():a,_,_=self.policy.act(self.c.encode(o),deterministic=self.mode=='greedy')
        a=a.cpu().numpy();return a[:,0],a[:,1]-1


def episode(sdk,panel,case,policy):
    env=sdk.DeliveryMission(*sdk.load_trial(panel,case),(64,720,2400))
    trace=[]
    while env.t<env.duration:
        o=sdk.view(env)
        near=(np.linalg.norm(o.seen_xy[None,:,:]-o.pos[:,None,:],axis=-1)<=1.+1e-9)&o.visible[None,:]&o.active[None,:]
        actions,targets=sdk.action_pair(policy.act(o)); old_pos=env.pos.copy()
        old_progress=env.progress.copy(); old_supplied=env.supplied.copy()
        old_conflicts,old_boundary=env.conflicts,env.boundary
        correct=sum(actions[i]==8 and targets[i]>=0 and near[i,targets[i]] for i in range(4))
        continuation=sum(actions[i]==8 and targets[i]>=0 and o.owner[targets[i]]==i for i in range(4))
        env.step(actions,targets)
        trace.append(dict(t=int(o.t),moves=actions.tolist(),targets=targets.tolist(),position=old_pos.tolist(),
            displacement=np.linalg.norm(env.pos-old_pos,axis=1).tolist(),own_near_agents=int(near.any(1).sum()),
            correct_near_hover=int(correct),public_owner_continuations=int(continuation),
            service_starts=int(((env.progress>0)&(old_progress==0)).sum()),
            delivered=int(((env.supplied>=0)&(old_supplied<0)).sum()),
            conflicts=int(env.conflicts-old_conflicts),boundary=int(env.boundary-old_boundary)))
        if (env.supplied>=0).all() and (env.last>=0).all():break
    metrics=env.metrics();tape=np.array([r['moves']+r['targets'] for r in trace],dtype=np.int64)
    replay,_=sdk.evaluate(panel,case,tape=tape)
    if replay!=metrics:raise AssertionError('Official action tape replay mismatch '+case['id'])
    return dict(parent=case['parent'],family=case['family'],roster=case['roster'],id=case['id'],
        metrics=metrics,funnel={k:sum(x[k] for x in trace) for k in
        ('own_near_agents','correct_near_hover','public_owner_continuations','service_starts','delivered','conflicts','boundary')},
        trace=trace,official_replay_metrics_exact=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--panel',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--seed',type=int,choices=(77126,77127,77128),required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();output=a.output.resolve();panel=a.panel.resolve()
    if not output.is_relative_to(HERE) or output.exists():ap.error('output must be a new experiment file')
    torch.set_num_threads(1);adapter=load('extra_rescue_adapter',HERE/'policy_adapter.py');c=adapter.c;sdk=c.search.sdk
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['seed']!=a.seed:raise ValueError('Checkpoint seed mismatch')
    trial_rows=read(panel/'panel.json');manifest=read(panel/'SHA256.json')
    selected=[r for r in trial_rows if (r['parent'],r['family'],r['roster']) in CASES]
    if len(selected)!=12 or {(r['parent'],r['family'],r['roster']) for r in selected}!=CASES:
        raise ValueError('Expected precisely the 12 registered cases in panel')
    selected.sort(key=lambda r:(r['parent'],r['family'],r['roster']))
    for row in selected:
        for field in ('water','population'):
            if sha(panel/row[field])!=manifest[row[field]]:raise ValueError('Panel input hash mismatch')
    records=[];rng_seed=581000
    for case in selected:
      for mode in ('sampled','greedy'):
       for arm in ('search_source','candidate'):
        policy=(SourcePolicy(c,a.seed,mode,a.device,rng_seed) if arm=='search_source' else
                adapter.Frozen(a.checkpoint,mode=mode,device=a.device,seed=rng_seed))
        row=episode(sdk,panel,case,policy);row.update(mode=mode,arm=arm,evaluation_seed=rng_seed)
        records.append(row);print(json.dumps(dict(id=case['id'],mode=mode,arm=arm,metrics=row['metrics']),allow_nan=False),flush=True)
    result=dict(version=sdk.VERSION,status='COMPLETED',evaluation_scope='12 pre-registered development cases, not blind or full dev80',
        seed=a.seed,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint),
        panel_sha256=sha(panel/'panel.json'),panel_manifest_sha256=sha(panel/'SHA256.json'),
        rng_seed=rng_seed,source_policy_checkpoint_sha256=sha(c.source_checkpoint(a.seed)),
        code_sha256=sha(Path(__file__)),official_sdk_sha256=sha(c.search.SDK/'run.py'),records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':main()
