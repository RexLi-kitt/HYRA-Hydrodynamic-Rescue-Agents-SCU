"""Training-only public-state teacher, offline imitation of the new residual.

Not pure PPO. The search backbone is frozen during this stage. The generated
states and teacher labels are an extra supervision cost. No teacher is used
by the policy adapter or evaluator.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
import core as c
import micro
import provenance
from train import Run

s = c.search


def search_negatives(reference, seed, steps_per_map):
    """Collect teacher-abstention states on the four training maps only.

    The reference network controls all four UAVs. The public-state teacher is
    queried solely to decide whether to retain an observation for KL training;
    it never chooses the rollout action. RNG isolation makes this collection
    identical on a resumed demonstration run.
    """
    saved=s.rng_state();rows=[];details=[]
    try:
        s.seed_all(seed+930000)
        dataset=s.Dataset()
        for parent in s.TRAIN:
            b=dataset.bridge(parent); eligible=abstained=feasible=0;steps=0
            while not b.done and steps<steps_per_map:
                o=b.observation();x=c.encode(o)
                local=c.person_features(o)[:,:,0]>0
                for agent in np.flatnonzero(local.any(axis=1)):
                    eligible+=1
                    if micro.teacher(o,int(agent)) is None:
                        rows.append((x[agent].copy(),-1));abstained+=1
                    else:feasible+=1
                with torch.no_grad():a,_,_=reference.act(x)
                action=a.cpu().numpy()
                b.step(action[:,0],action[:,1]-1,local_action_order=False)
                steps+=1
            info=dict(parent=parent,roster=0,joint_steps=steps,eligible_agent_states=eligible,
                      teacher_abstentions=abstained,teacher_nonabstentions_excluded=feasible)
            details.append(info);print(json.dumps(dict(phase='SEARCH_KL_COLLECTION',**info)),flush=True)
    finally:s.set_rng(saved)
    return rows,details


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,choices=s.SEEDS,required=True)
    ap.add_argument('--cases',type=int,required=True)
    ap.add_argument('--updates',type=int,required=True)
    ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--kl-positive',type=float,default=.02)
    ap.add_argument('--kl-negative',type=float,default=1.)
    ap.add_argument('--search-steps-per-map',type=int,default=0,
                    help='collect teacher-abstention KL states from each of the four training maps, maximum 480')
    ap.add_argument('--positive-fraction',type=float,default=.5)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    a=ap.parse_args()
    if min(a.cases,a.updates,a.batch)<=0: ap.error('positive counts required')
    if not 0<=a.search_steps_per_map<=480 or not 0<a.positive_fraction<1:ap.error('invalid search steps or positive fraction')
    out=a.output.resolve()
    if not out.is_relative_to(c.HERE.resolve()) or out.exists(): ap.error('output must be a new experiment subdirectory')
    torch.set_num_threads(1); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    lock=provenance.begin(out,a.seed,a.resume)
    run=Run(s.Dataset(),a.seed)
    reference=copy.deepcopy(run.policy)
    for p in reference.actor.parameters():p.requires_grad_(False)
    if a.resume: run.restore(torch.load(a.resume,map_location=run.device,weights_only=False))
    rows=[]; cases=[]
    for i in range(a.cases):
        items,meta=micro.collect(100000+i, ('near','aligned','unaligned')[i%3])
        rows.extend(items); cases.append(meta)
    synthetic_count=len(rows)
    search_rows, search_details=search_negatives(reference,a.seed,a.search_steps_per_map)
    rows.extend(search_rows)
    if not rows: raise ValueError('No teacher-labelled examples')
    x=np.asarray([r[0] for r in rows],np.float32); y=np.asarray([r[1] for r in rows],np.int64)
    for p in run.policy.actor.parameters():p.requires_grad_(False)
    for p in run.policy.actor.rescue.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam(run.policy.actor.rescue.parameters(),lr=.001,eps=1e-5)
    if a.resume:
        state=torch.load(a.resume,map_location=run.device,weights_only=False)
        if 'demo_optimizer' not in state: raise ValueError('Resume must be a demo checkpoint')
        config=dict(cases=a.cases,batch=a.batch,kl_positive=a.kl_positive,kl_negative=a.kl_negative,
                    search_steps_per_map=a.search_steps_per_map,positive_fraction=a.positive_fraction)
        if state['demo_config'] != config: raise ValueError('Demo dataset/config changed')
        if state['demo_teacher_sha256'] != s.sha(c.HERE/'micro.py') or state['demo_code_sha256'] != s.sha(c.HERE/'demo.py'):
            raise ValueError('Demonstration code changed')
        if state['demo_planner_sha256'] != s.sha(c.HERE/'teacher_multistep.py'):raise ValueError('Teacher planner changed')
        opt.load_state_dict(state['demo_optimizer'])
        previous=state['demo_updates']
    else: previous=0
    losses=[]
    pos_ids=np.flatnonzero(y>=0);neg_ids=np.flatnonzero(y<0)
    if len(pos_ids)==0:raise ValueError('No positive demonstration labels')
    for k in range(a.updates):
        if len(neg_ids):
            num_pos=max(1,min(a.batch-1,round(a.batch*a.positive_fraction))) if a.batch>1 else 1
            idx=np.concatenate((np.random.choice(pos_ids,num_pos),np.random.choice(neg_ids,a.batch-num_pos)))
            np.random.shuffle(idx)
        else:idx=np.random.choice(pos_ids,a.batch)
        tx=run.policy.tensor(x[idx]); ty=torch.as_tensor(y[idx],dtype=torch.long,device=run.device)
        current=run.policy.joint(tx)
        with torch.no_grad():ref=reference.joint(tx)
        legal=tx[:,-585:]>.5
        kl=(ref.probs*(torch.where(legal,ref.logits,0.)-torch.where(legal,current.logits,0.))).sum(-1)
        positive=ty>=0
        safe_y=torch.where(positive,ty,torch.full_like(ty,520))
        supervised=-current.log_prob(safe_y)
        # Negative states have no heuristic rescue label; retain the exact
        # pretrained search distribution instead of forcing a no-target action.
        loss=(torch.where(positive,supervised,torch.zeros_like(supervised)) +
              torch.where(positive,a.kl_positive*kl,a.kl_negative*kl)).mean()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite teacher loss')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(run.policy.actor.rescue.parameters(),10.);opt.step()
        losses.append(float(loss.detach().item()))
        if (k+1)%20==0: print(json.dumps(dict(updates=previous+k+1,loss=losses[-1])),flush=True)
    for p in run.policy.actor.parameters():p.requires_grad_(True)
    state=run.snapshot(); state.update(demo_optimizer=opt.state_dict(),demo_updates=previous+a.updates,
        demo_config=dict(cases=a.cases,batch=a.batch,kl_positive=a.kl_positive,kl_negative=a.kl_negative,
                         search_steps_per_map=a.search_steps_per_map,positive_fraction=a.positive_fraction),
        demo_teacher_sha256=s.sha(c.HERE/'micro.py'),demo_planner_sha256=s.sha(c.HERE/'teacher_multistep.py'),
        demo_code_sha256=s.sha(c.HERE/'demo.py'))
    provenance.finish(lock)
    s.checkpoint_write(out/'checkpoint.pt',state)
    s.json_write(out/'demo_log.json',dict(method='additional public-state scripted demonstrations, residual only; not pure PPO',
        cases=a.cases, train_case_seeds=[100000,100000+a.cases-1], batch=a.batch, updates=previous+a.updates,
        labels=int((y>=0).sum()), abstentions=int((y<0).sum()), synthetic_rows=synthetic_count,
        train_map_kl_rows=len(search_rows),train_map_collection=search_details,
        positive_fraction=a.positive_fraction,kl_positive=a.kl_positive,kl_negative=a.kl_negative,
        example_draws=a.updates*a.batch, cases_detail=cases, losses=losses,
        checkpoint_sha256=s.sha(out/'checkpoint.pt'),source_integrity='PASS'))


if __name__=='__main__':main()
