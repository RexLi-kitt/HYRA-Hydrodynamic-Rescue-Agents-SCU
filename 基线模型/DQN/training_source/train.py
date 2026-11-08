"""New frozen-RNN, frozen-search Double-DQN branch; no old replay is reused."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch
import demo
import policy_core as p
import replay

s=p.geometry.search


def module_hash(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        a=value.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def source_lock(bc):
    lock=demo.source_lock();lock[str(p.HERE/'train.py')]=p.sha(p.HERE/'train.py')
    lock[str(bc.resolve())]=p.sha(bc)
    return lock


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


class Run:
    def __init__(self,seed,checkpoint,device='cuda:0',epsilon_start=.2,epsilon_end=.05,anneal_steps=8192):
        self.seed=seed;self.device=device;self.parent=Path(checkpoint).resolve();self.parent_sha=p.sha(self.parent)
        seed_all(seed);self.online,bc=demo.load_model(self.parent,device)
        if bc['seed']!=seed or bc['phase'] not in ('search','rescue'):raise ValueError('BC parent mismatch')
        self.bc_phase=bc['phase']
        for q in self.online.base.parameters():q.requires_grad_(False)
        for q in self.online.search.parameters():q.requires_grad_(False)
        for q in self.online.rescue.parameters():q.requires_grad_(True)
        self.target=copy.deepcopy(self.online)
        for q in self.target.parameters():q.requires_grad_(False)
        self.optimizer=torch.optim.RMSprop(self.online.rescue.parameters(),lr=.0001,alpha=.99,eps=1e-5)
        self.initial_base=module_hash(self.online.base);self.initial_search=module_hash(self.online.search)
        self.config=dict(seed=seed,device=device,bc_phase=self.bc_phase,parent_sha256=self.parent_sha,epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,anneal_steps=anneal_steps,replay_capacity=8192,warmup=64,
            update_every=32,batch_joint_steps=32,target_every_updates=32,gamma=.99,lr=.0001,
            reward='10 delivery + .25 discoveries + .002 new cells - .1 conflict pairs - .005',
            method='fixed RNN and search; fresh sparse-context replay; rescue residual Double DQN',
            terminal='official done including 2400s deadline; only artificial interruption is truncation')
        if not (0<=epsilon_end<=epsilon_start<=1) or anneal_steps<=0:raise ValueError('Invalid epsilon schedule')
        self.ds=s.Dataset();self.replay=replay.Replay(self.config['replay_capacity'])
        self.joint=0;self.attempts=0;self.updates=0;self.skips=0;self.target_copies=0;self.next_index=0;self.episodes=[]
        self.new_episode()

    def new_episode(self):
        self.parent_map=s.TRAIN[self.next_index%len(s.TRAIN)];self.next_index+=1
        self.bridge=self.ds.bridge(self.parent_map)
        self.previous=np.full(4,-1,np.int64);self.hidden=self.online.init_hidden(4,self.device)
        self.context,self.hidden=replay.capture(self.online,self.bridge.observation(),self.previous,self.hidden)

    def epsilon(self):
        f=min(1.,self.joint/self.config['anneal_steps'])
        return self.config['epsilon_start']+(self.config['epsilon_end']-self.config['epsilon_start'])*f

    def step(self):
        epsilon=self.epsilon();actions=replay.choose(self.online,self.context,self.device,epsilon)
        pair=np.stack((actions//65,actions%65),axis=-1)
        after,done,metrics,rec=s.step(self.bridge,pair,'local_mask')
        next_context,next_hidden=replay.capture(self.online,after,actions,self.hidden)
        self.replay.insert(self.context,actions,rec['reward'],next_context,terminated=done,truncated=False)
        self.context,self.hidden,self.previous=next_context,next_hidden,actions.copy()
        self.joint+=1;rec.update(joint_step=self.joint,parent=self.parent_map,epsilon=epsilon,terminated=bool(done))
        if done:
            self.episodes.append(dict(parent=self.parent_map,joint_step=self.joint,metrics=metrics));self.new_episode()
        return rec

    def update(self):
        self.attempts+=1;items,ids=self.replay.sample(self.config['batch_joint_steps'])
        before=module_hash(self.online.rescue)
        stats=replay.double_dqn_update(self.online,self.target,self.optimizer,items,self.device,self.config['gamma'])
        if stats['updated']:
            self.updates+=1
            if self.updates%self.config['target_every_updates']==0:
                self.target.load_state_dict(self.online.state_dict());self.target_copies+=1
        else:self.skips+=1
        if module_hash(self.online.base)!=self.initial_base or module_hash(self.online.search)!=self.initial_search:
            raise AssertionError('Frozen search network changed')
        return dict(attempt=self.attempts,effective_updates=self.updates,skipped_updates=self.skips,
            target_copies=self.target_copies,joint=self.joint,replay_indices=ids,stats=stats,
            rescue_before_sha256=before,rescue_after_sha256=module_hash(self.online.rescue),
            base_sha256=self.initial_base,search_sha256=self.initial_search)

    def snapshot(self,lock):
        return copy.deepcopy(dict(schema='DQN_RESCUE_RL_V1',seed=self.seed,config=self.config,source_lock=lock,runtime=demo.runtime(),
            online=self.online.state_dict(),target=self.target.state_dict(),optimizer=self.optimizer.state_dict(),
            replay=self.replay.state(),joint=self.joint,attempts=self.attempts,updates=self.updates,skips=self.skips,
            target_copies=self.target_copies,next_index=self.next_index,parent_map=self.parent_map,episodes=self.episodes,
            env=s.env_state(self.bridge),bridge_done=self.bridge.done,previous=self.previous,hidden=self.hidden,
            context=self.context,rng=demo.rng_state(),initial_base=self.initial_base,initial_search=self.initial_search))

    def restore(self,state,lock):
        if state['schema']!='DQN_RESCUE_RL_V1' or state['seed']!=self.seed or state['config']!=self.config or state['runtime']!=demo.runtime():
            raise ValueError('Checkpoint identity/config mismatch')
        if state['source_lock']!=lock:raise ValueError('Checkpoint source lock differs')
        for name in ('online','target','optimizer'):getattr(self,name).load_state_dict(state[name])
        self.replay.restore(state['replay'])
        for name in ('joint','attempts','updates','skips','target_copies','next_index','parent_map','episodes','previous','context','initial_base','initial_search'):
            setattr(self,name,copy.deepcopy(state[name]))
        self.bridge=s.restore_bridge(self.ds,self.parent_map,state['env'],state['bridge_done'])
        self.hidden=state['hidden'].to(self.device);demo.set_rng(state['rng'])
        if module_hash(self.online.base)!=self.initial_base or module_hash(self.online.search)!=self.initial_search:
            raise ValueError('Restored frozen model mismatch')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--bc',type=Path,required=True)
    ap.add_argument('--steps',type=int,required=True);ap.add_argument('--anneal-steps',type=int,default=8192)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--resume',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=a.out.resolve();bc=a.bc.resolve()
    if a.steps<=0 or a.anneal_steps<=0 or not out.is_relative_to(p.HERE) or out.exists():ap.error('positive steps and new experiment output required')
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    lock=source_lock(bc);out.mkdir(parents=True);snap=out/'source_snapshot';snap.mkdir()
    for i,(path,digest) in enumerate(lock.items()):
        if Path(path).suffix=='.pt':continue
        target=snap/f'{i:03d}_{Path(path).name}';shutil.copyfile(path,target)
        if p.sha(target)!=digest:raise ValueError('Snapshot mismatch')
    s.json_write(out/'source_manifest.json',dict(inputs=lock,runtime=demo.runtime()))
    run=Run(a.seed,bc,a.device,anneal_steps=a.anneal_steps)
    if a.resume:run.restore(torch.load(a.resume,map_location=a.device,weights_only=False),lock)
    start=run.joint;updates=[];began=time.time();stopped=False
    with gzip.open(out/'train_trace.jsonl.gz','xt',encoding='utf8') as trace:
        while run.joint<start+a.steps:
            rec=run.step();trace.write(json.dumps(rec,allow_nan=False)+'\n')
            if run.joint>=run.config['warmup'] and run.joint%run.config['update_every']==0:
                row=run.update();updates.append(row)
                if run.attempts%10==0:print(json.dumps(row,allow_nan=False),flush=True)
            if run.joint%128==0:
                trace.flush();s.json_write(out/'status.json',dict(phase='TRAINING',joint=run.joint,updates=run.updates,
                    skipped=run.skips,replay_size=len(run.replay.items),elapsed_s=time.time()-began))
            stopped=(p.HERE/'STOP_REQUESTED').exists()
            if run.joint%2048==0 or run.joint==start+a.steps or stopped:
                s.check_lock(lock);path=out/f'checkpoint-{run.joint:06d}.pt';demo.save(path,run.snapshot(lock))
                s.json_write(out/'updates.json',updates)
            if stopped:break
    s.check_lock(lock);s.json_write(out/'status.json',dict(phase='PAUSED' if stopped else 'COMPLETED',joint=run.joint,
        updates=run.updates,attempts=run.attempts,skipped=run.skips,target_copies=run.target_copies,
        replay_size=len(run.replay.items),checkpoint=str(path),checkpoint_sha256=p.sha(path),
        base_sha256=run.initial_base,search_sha256=run.initial_search,parent_sha256=run.parent_sha,source_integrity='PASS'))


if __name__=='__main__':main()
