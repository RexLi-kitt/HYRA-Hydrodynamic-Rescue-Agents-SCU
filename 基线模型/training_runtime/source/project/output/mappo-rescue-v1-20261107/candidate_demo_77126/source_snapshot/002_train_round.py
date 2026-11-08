"""Fixed-budget three-arm continuation; all outputs stay in this new folder."""
import argparse
import copy
import gzip
import json
import os
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import core as c
from carpet import Carpet

BUDGET=25600


class Run:
    def __init__(self,ds,arm,seed):
        self.ds=ds;self.arm=arm;self.seed=seed
        self.policy,self.trainer,ck=c.initialize(arm,seed)
        self.parent=ck['current_parent'];self.next_index=ck['next_index']
        self.bridge=c.restore_bridge(ds,self.parent,ck['env_state'],ck.get('bridge_done',False))
        assert not self.bridge.done
        self.joint=0;self.updates=0;self.episodes=[]
        c.set_rng({k:ck[k+'_rng'] for k in ('python','numpy','torch','cuda')})
    def new_episode(self):
        self.parent=c.TRAIN[self.next_index%len(c.TRAIN)];self.next_index+=1
        self.bridge=self.ds.bridge(self.parent)
    def collect(self,steps=128,reference_original=False):
        xs=[];acts=[];logps=[];vals=[];rews=[];dones=[];trace=[]
        for _ in range(steps):
            o=self.bridge.observation();x=c.original_model.encode(o) if reference_original else c.encode(o,self.arm)
            with torch.no_grad():a,lp,v=self.policy.act(x)
            a=a.cpu().numpy();lp=lp.cpu().numpy();v=v.cpu().numpy()
            after,done,metrics,rec=c.step(self.bridge,a,self.arm)
            self.joint+=1;rec.update(parent=self.parent,joint_step=self.joint,lifetime_steps=102400+self.joint)
            trace.append(rec);xs.append(x);acts.append(a);logps.append(lp);vals.append(v)
            rews.append(np.full((4,1),rec['reward'],np.float32));dones.append(float(done))
            if done:
                self.episodes.append(dict(parent=self.parent,joint_step=self.joint,metrics=metrics,
                                         includes_inherited_prefix=len(self.episodes)==0))
                self.new_episode()
        next_features=c.original_model.encode(self.bridge.observation()) if reference_original else c.encode(self.bridge.observation(),self.arm)
        with torch.no_grad():nextv=self.policy.critic(self.policy.tensor(next_features)).cpu().numpy()
        denorm=self.trainer.value_normalizer.denormalize(np.stack(vals));next_denorm=self.trainer.value_normalizer.denormalize(nextv)
        advantages=np.zeros_like(denorm);future=np.zeros((4,1),np.float32)
        for j in reversed(range(steps)):
            nxt=next_denorm if j==steps-1 else denorm[j+1];live=1-dones[j]
            delta=rews[j]+.99*live*nxt-denorm[j]
            future=delta+.99*.95*live*future;advantages[j]=future
        returns=advantages+denorm
        flat=lambda z:np.asarray(z).reshape(-1,*np.asarray(z).shape[2:])
        data=tuple(flat(z) for z in (xs,acts,logps,vals,returns,advantages))
        return data,trace
    def update(self,data):
        before=[p.detach().clone() for p in self.policy.actor.parameters()]
        stats=c.original_trainer.update(self.trainer,*data,epochs=5)
        delta=sum(float((p.detach()-q).abs().sum()) for p,q in zip(self.policy.actor.parameters(),before))
        if not np.isfinite(delta) or delta<=0:raise FloatingPointError('No finite actor update')
        for module in (self.policy.actor,self.policy.critic):
            if not all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all()) for p in module.parameters()):
                raise FloatingPointError('Nonfinite parameters or gradients')
        self.updates+=1
        local_grad=None
        if self.arm=='local_mask':local_grad=float(torch.stack([p.grad.square().sum() for p in self.policy.actor.local.parameters() if p.grad is not None]).sum().sqrt())
        return dict(joint_steps=self.joint,update=self.updates,ppo=stats[-1],epochs=stats,actor_delta_l1=delta,local_grad_norm=local_grad)
    def snapshot(self):
        p=self.policy
        return copy.deepcopy(dict(schema='MAPPO_SEARCH_V1',arm=self.arm,seed=self.seed,joint=self.joint,
            updates=self.updates,parent=self.parent,next_index=self.next_index,episodes=self.episodes,
            env=c.env_state(self.bridge),bridge_done=self.bridge.done,actor=p.actor.state_dict(),critic=p.critic.state_dict(),
            actor_optimizer=p.actor_optimizer.state_dict(),critic_optimizer=p.critic_optimizer.state_dict(),
            value_norm=c.original_trainer.norm_state(self.trainer),rng=c.rng_state()))
    def restore(self,s):
        assert s['schema']=='MAPPO_SEARCH_V1' and s['arm']==self.arm and s['seed']==self.seed
        for n in ('actor','critic','actor_optimizer','critic_optimizer'):getattr(self.policy,n).load_state_dict(s[n])
        c.original_trainer.load_norm_state(self.trainer,s['value_norm'])
        for n in ('joint','updates','parent','next_index','episodes'):setattr(self,n,copy.deepcopy(s[n]))
        self.bridge=c.restore_bridge(self.ds,self.parent,s['env'],s['bridge_done']);c.set_rng(s['rng'])


def trace_summary(trace):
    count=len(trace)*4
    return dict(joint_steps=len(trace),agent_decisions=count,
        zero_motion=sum(sum(x<=1e-8 for x in r['displacement']) for r in trace),
        boundary=sum(r['boundary'] for r in trace),conflicts=sum(r['conflicts'] for r in trace),
        near=sum(r['near'] for r in trace),correct_hover=sum(r['correct_hover'] for r in trace),
        service_starts=sum(r['service_starts'] for r in trace),delivered=sum(r['delivered'] for r in trace),
        selected=sum(r['selected'] for r in trace),selected_visible=sum(r['selected_visible'] for r in trace),
        new_cells=sum(r['new_cells'] for r in trace),discoveries=sum(r['discoveries'] for r in trace),
        reward_parts={k:sum(r['reward_parts'][k] for r in trace) for k in ('delivery','discovery','coverage','collision','time')})


def evaluate(run,parents=None):
    parents=c.DEV if parents is None else parents
    before=run.snapshot();results=[]
    try:
        for parent in parents:
            for mode in ('sampled','greedy'):
                c.seed_all(581000+parent);b=run.ds.bridge(parent);trace=[]
                while not b.done:
                    x=c.encode(b.observation(),run.arm)
                    with torch.no_grad():a,_,_=run.policy.act(x,deterministic=mode=='greedy')
                    _,_,_,rec=c.step(b,a.cpu().numpy(),run.arm);trace.append(rec)
                results.append(dict(parent=parent,roster=0,mode=mode,evaluation_seed=581000+parent,
                    metrics=b.env.metrics(),decisions=trace_summary(trace),
                    actions_and_targets=[r['moves']+r['targets'] for r in trace]))
    finally:run.restore(before)
    assert c.same(before,run.snapshot()),'Evaluation altered training state'
    return results


def audit_evaluations(ds,rows):
    steps=0
    for r in rows:
        b=ds.bridge(r['parent'])
        for a in r['actions_and_targets']:
            b.step(np.asarray(a[:4],int),np.asarray(a[4:],int),local_action_order=False);steps+=1
        assert b.done and b.env.metrics()==r['metrics']
    return dict(status='PASS',episodes=len(rows),steps=steps,all_metrics_exact=True)


def carpet_evaluate(ds):
    results=[]
    for parent in c.DEV:
        b=ds.bridge(parent);p=Carpet();trace=[];zero=0;trajectory=[]
        while not b.done:
            old=b.env.pos.copy();a,t=p.act(b.observation())
            b.step(a,t,local_action_order=False)
            zero+=int((np.linalg.norm(b.env.pos-old,axis=1)<=1e-8).sum())
            trace.append(a.tolist()+t.tolist());trajectory.append(old.tolist())
        results.append(dict(parent=parent,roster=0,mode='carpet_search_only',metrics=b.env.metrics(),
            zero_motion=zero,actions_and_targets=trace,trajectory=trajectory,
            route_index=p.p.route_index,scan_complete=p.p.scan_complete,yield_steps=p.yield_steps,yield_changed_actions=p.yield_changed))
    c.json_write(c.HERE/'carpet_evaluation.json',dict(scope='Search component only, current official 20m sensor; no delivery claims',results=results))
    c.json_write(c.HERE/'carpet_replay_audit.json',audit_evaluations(ds,results))
    return results


def freeze_protocol():
    lock=c.source_lock()
    protocol=dict(version='MAPPO_SEARCH_V1',arms=list(c.ARMS),seeds=list(c.SEEDS),
        lifetime_start=102400,new_joint_steps_each=BUDGET,lifetime_end=128000,total_training_joint_steps=BUDGET*9,
        training_parents=c.TRAIN,development_parents=c.DEV,roster=0,
        rollout=128,ppo_epochs=5,actor_lr=.0005,critic_lr=.0005,gamma=.99,gae_lambda=.95,entropy_coef=.01,
        reward='10 delivered + .25 new persons + .002 new covered cells - .1 conflict pairs - .005',
        changes=dict(original='Unchanged 998-D MAPPO actor, critic, factored sampling and PPO',
            mask='Same actor and critic, joint product distribution conditioned on exact public 4x585 boundary/target mask',
            local_mask='Mask arm plus 15x15 known-cell bitmap, 15x15 in-domain bitmap and two subcell offsets per UAV; residual 452->64->64 local branches for actor and critic, output initially zero'),
        initialization='Per-seed final baseline checkpoint; shared network weights, Adam moments, ValueNorm, current environment, map cursor and RNG continued. New local parameters begin with fresh Adam states.',
        mask_semantics='Visible-target early stopping and moving follow included; no collision/teacher/rescue override; safe no-target hover always legal; rollout masks reused exactly for PPO log probabilities and entropy',
        primary='Final development coverage, separately sampled and greedy; report paired parents and each related branch, no best-checkpoint picking',
        secondary=['discovered persons','zero displacement','boundary','conflicts','delivery and service funnel'],
        carpet='Preserved old rule-v2 quadrant routes and cleanup, adapted 20m sensor and action numbering, original local+coarse coverage ingest, no old rescue rules; original positive partial boundary motions remain legal. Added minimum-change public joint yielding because the new referee stops conflicting aircraft; archived unadapted port stalled at 7.52% coverage.',
        evaluation='Initial + final; reuse existing original initial evaluation, masked initial reused for zero-residual local arm after exact equivalence validation; seed=581000+parent; evaluation state/RNG isolation',
        stop='Stop on nonfinite values, source mismatch, illegal action or STOP_REQUESTED; checkpoint at whole rollout boundary',
        limitations=['Three related continuation branches share earlier pretraining, not three independent from-scratch seeds',
                    'Four training maps and one roster, repeatedly used local development panel; not blind generalization',
                    'Two learned heads remain; training still includes original rescue reward; this is a search-focused intervention, not a new rescue controller',
                    'Mask changes stochastic action sampling and entropy; local branch adds parameters; no claim of equal network capacity',
                    'Carpet pure search deliberately returns no target, so its zero delivery is not a rescue score'],
        source_checkpoints={str(s):dict(path=str(c.source_checkpoint(s)),sha256=c.sha(c.source_checkpoint(s))) for s in c.SEEDS},
        source_lock=lock,python=__import__('sys').version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda)
    c.json_write(c.HERE/'protocol.json',protocol)
    snap=c.HERE/'source_snapshot';snap.mkdir(exist_ok=False)
    for i,p in enumerate(lock):shutil.copyfile(p,snap/f'{i:03d}_{Path(p).name}')
    return protocol


def train_one(ds,arm,seed,protocol,resume=None):
    folder=c.HERE/f'{arm}_{seed}';folder.mkdir(exist_ok=True)
    run=Run(ds,arm,seed)
    if resume:
        ck=torch.load(resume,map_location='cuda:0',weights_only=False)
        assert ck['source_lock']==protocol['source_lock'];run.restore(ck['run'])
    initial_path=folder/'initial_evaluation.json'
    if not initial_path.exists():
        if arm=='original':
            oldpath=c.source_checkpoint(seed).parent/'evaluations'/'evaluation-total-102400.json'
            c.json_write(initial_path,dict(reused_original=True,path=str(oldpath),sha256=c.sha(oldpath),results=c.read(oldpath)['results']))
        elif arm=='local_mask':
            source=c.HERE/f'mask_{seed}'/'initial_evaluation.json'
            c.json_write(initial_path,dict(reused_zero_residual_equivalent=True,path=str(source),sha256=c.sha(source),results=c.read(source)['results']))
        else:c.json_write(initial_path,dict(results=evaluate(run)))
    began=time.perf_counter();trace_path=folder/'train_trace.jsonl.gz'
    with gzip.open(trace_path,'at' if resume else 'xt',encoding='utf8') as tracefile, (folder/'updates.jsonl').open('a' if resume else 'x',encoding='utf8') as log:
        while run.joint<BUDGET:
            data,trace=run.collect();update=run.update(data)
            update.update(decisions=trace_summary(trace),elapsed_s=time.perf_counter()-began)
            for r in trace:tracefile.write(json.dumps(r,allow_nan=False)+'\n')
            log.write(json.dumps(update,allow_nan=False)+'\n');log.flush()
            if run.joint%1024==0:
                c.json_write(folder/'status.json',dict(phase='TRAINING',arm=arm,seed=seed,joint_steps=run.joint,updates=run.updates,elapsed_s=time.perf_counter()-began))
                print(json.dumps(dict(phase='TRAINING',arm=arm,seed=seed,joint=run.joint,coverage_reward=update['decisions']['reward_parts']['coverage'])),flush=True)
            stop=(c.HERE/'STOP_REQUESTED').exists()
            if run.joint%4096==0 or run.joint==BUDGET or stop:
                c.check_lock(protocol['source_lock']);path=folder/f'checkpoint-new-{run.joint:06d}.pt'
                c.checkpoint_write(path,dict(source_lock=protocol['source_lock'],run=run.snapshot()))
                c.json_write(folder/'latest_checkpoint.json',dict(path=str(path),sha256=c.sha(path)))
            if stop:
                c.json_write(folder/'status.json',dict(phase='PAUSED',arm=arm,seed=seed,joint_steps=run.joint));return False
    c.json_write(folder/'episodes.json',run.episodes)
    results=evaluate(run);c.json_write(folder/'final_evaluation.json',dict(arm=arm,seed=seed,joint_steps=run.joint,results=results))
    c.json_write(folder/'evaluation_replay_audit.json',audit_evaluations(ds,results))
    c.json_write(folder/'status.json',dict(phase='COMPLETED',arm=arm,seed=seed,joint_steps=run.joint,updates=run.updates,
        seconds=time.perf_counter()-began,checkpoint=str(path),checkpoint_sha256=c.sha(path),peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    return True


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-round',action='store_true');a=parser.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    gate=c.read(c.HERE/'validation.json');assert gate['status']=='PASS'
    protocol=c.read(c.HERE/'protocol.json') if a.resume_round else freeze_protocol()
    assert gate['source_lock']==protocol['source_lock'];c.check_lock(protocol['source_lock'])
    ds=c.Dataset()
    if not (c.HERE/'carpet_evaluation.json').exists():carpet_evaluate(ds)
    for seed in c.SEEDS:
        for arm in c.ARMS:
            folder=c.HERE/f'{arm}_{seed}'
            if (folder/'status.json').exists() and c.read(folder/'status.json')['phase']=='COMPLETED':continue
            resume=c.read(folder/'latest_checkpoint.json')['path'] if (folder/'latest_checkpoint.json').exists() else None
            c.json_write(c.HERE/'status.json',dict(phase='RUNNING',arm=arm,seed=seed,pid=os.getpid()))
            if not train_one(ds,arm,seed,protocol,resume):
                c.json_write(c.HERE/'status.json',dict(phase='PAUSED',arm=arm,seed=seed));return
    c.check_lock(protocol['source_lock'])
    c.json_write(c.HERE/'status.json',dict(phase='COMPLETED',runs=9,total_new_training_joint_steps=BUDGET*9))


if __name__=='__main__':main()
