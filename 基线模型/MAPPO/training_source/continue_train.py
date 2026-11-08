"""Paired continuation from stable MAPPO 8192: frozen actor vs full actor PPO."""
import argparse
import copy
import gzip
import importlib.util
import json
import sys
import time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
V1=HERE.parent
V2=V1/'stable_ppo_v2'
sys.path.insert(0,str(V1))
spec=importlib.util.spec_from_file_location('continuation_stable_parent',V2/'train_v2.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
c,s=v.c,v.s
CONFIG=dict(start_joint=8192,endpoint_joint=32768,additional_steps_each=24576,
            rollout=128,ppo_epochs=5,actor_lr=.0001,critic_lr=.0005,
            arms=['frozen','joint'],seeds=[77126,77127,77128],milestones=[16384,24576,32768])
GROUPS=('net','local','move','target','rescue')


def parent_path(seed):return V2/f'seed_{seed}'/'checkpoint-008192.pt'


def groups(actor):
    return {name:v.actor_hash(getattr(actor,name)) for name in GROUPS}


class PairedRun:
    def __init__(self,ds,seed,arm,device='cuda:0'):
        if arm not in CONFIG['arms']:raise ValueError(arm)
        self.seed,self.arm,self.device=seed,arm,device
        self.parent=parent_path(seed)
        self.base=v.StableRun(ds,seed,device)
        self.base.restore(torch.load(self.parent,map_location=device,weights_only=False))
        self.run=self.base.run
        if self.run.joint!=CONFIG['start_joint']:raise ValueError('Wrong continuation parent')
        self.initial_groups=groups(self.run.policy.actor)
        self.initial_critic=v.actor_hash(self.run.policy.critic)
        self.updates_with_gradient={name:0 for name in GROUPS}
        self.set_mode()

    def set_mode(self):
        p=self.run.policy
        p.actor.zero_grad(set_to_none=True)
        for name,param in p.actor.named_parameters():
            param.requires_grad_(self.arm=='joint' or name.startswith('rescue.'))
        for param in p.critic.parameters():param.requires_grad_(True)
        for group in p.actor_optimizer.param_groups:group['lr']=CONFIG['actor_lr']
        for group in p.critic_optimizer.param_groups:group['lr']=CONFIG['critic_lr']

    def collect(self,steps):return self.run.collect(steps)

    def update(self,data,epochs=5):
        before=groups(self.run.policy.actor)
        stats=self.run.update(data,epochs=epochs)
        after=groups(self.run.policy.actor)
        norms={}
        for name in GROUPS:
            module=getattr(self.run.policy.actor,name)
            grads=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
            norms[name]=float(torch.stack(grads).sum().sqrt()) if grads else 0.
            self.updates_with_gradient[name]+=int(norms[name]>0)
            if self.arm=='frozen' and name!='rescue':
                if after[name]!=self.initial_groups[name] or before[name]!=after[name] or grads:
                    raise AssertionError('Frozen actor parameter changed: '+name)
        return dict(ppo=stats,actor_before=before,actor_after=after,gradient_norms=norms)

    def snapshot(self,lock):
        state=self.run.snapshot()
        state['paired_continuation']=copy.deepcopy(dict(config=CONFIG,arm=self.arm,
            parent_checkpoint=str(self.parent),parent_sha256=s.sha(self.parent),
            initial_groups=self.initial_groups,current_groups=groups(self.run.policy.actor),
            initial_critic=self.initial_critic,current_critic=v.actor_hash(self.run.policy.critic),
            updates_with_gradient=self.updates_with_gradient,source_lock=lock))
        return state

    def restore(self,state):
        meta=state.get('paired_continuation')
        if not meta or meta['config']!=CONFIG or meta['arm']!=self.arm:
            raise ValueError('Incompatible continuation checkpoint')
        if meta['parent_sha256']!=s.sha(self.parent) or meta['initial_groups']!=self.initial_groups:
            raise ValueError('Continuation parent changed')
        s.check_lock(meta['source_lock'])
        self.run.restore(state);self.set_mode()
        if groups(self.run.policy.actor)!=meta['current_groups']:
            raise ValueError('Actor fingerprint mismatch')
        self.updates_with_gradient=copy.deepcopy(meta['updates_with_gradient'])


def prepare():
    path=HERE/'source_lock.json'
    if path.exists():s.check_lock(s.read(path));return
    if s.read(HERE/'validation.json')['status']!='PASS':raise ValueError('Validation missing')
    spec=importlib.util.spec_from_file_location('verify_stable_training_parent',V2/'orchestrate_v2.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    identities={};lock={}
    for seed in CONFIG['seeds']:
        identities[str(seed)]=audit.train_check(dict(seed=seed,output=str(parent_path(seed).parent)))
        state=torch.load(parent_path(seed),map_location='cpu',weights_only=False)
        lock.update(state['stable_ppo_v2']['source_lock'])
        for file in (parent_path(seed),parent_path(seed).parent/'source_manifest.json',
                     V2/f'eval_{seed}_dev.json',V2/f'audit_{seed}.json'):
            lock[str(file)]=s.sha(file)
    for file in (V1/'eval_extra.py',V1/'audit_results.py',V1/'orchestrate.py',
                 HERE/'continue_train.py',HERE/'validate.py',HERE/'run_experiment.py',
                 HERE/'protocol.json',HERE/'validation.json'):
        lock[str(file)]=s.sha(file)
    extra=Path(r'C:\Users\李\Desktop\hyra\HYRA-dev80-20261106.1')
    for name in ('panel.json','SHA256.json'):lock[str(extra/name)]=s.sha(extra/name)
    s.check_lock(lock)
    s.json_write(HERE/'parent_audit.json',identities)
    s.json_write(path,lock)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--arm',choices=CONFIG['arms']);ap.add_argument('--seed',type=int,choices=CONFIG['seeds'])
    ap.add_argument('--until',type=int,choices=CONFIG['milestones']);ap.add_argument('--resume',type=Path)
    ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if a.prepare:return prepare()
    if a.arm is None or a.seed is None or a.until is None:ap.error('arm, seed and until required')
    lock=s.read(HERE/'source_lock.json');s.check_lock(lock)
    run=PairedRun(s.Dataset(),a.seed,a.arm,a.device)
    if a.resume:run.restore(torch.load(a.resume,map_location=a.device,weights_only=False))
    start=run.run.joint
    if a.until-start!=8192:raise ValueError('Each stage adds precisely 8192 joint steps')
    folder=HERE/a.arm/f'seed_{a.seed}';folder.mkdir(parents=True,exist_ok=True)
    trace_path=folder/f'trace_{start:06d}_{a.until:06d}.jsonl.gz'
    updates_path=folder/f'updates_{start:06d}_{a.until:06d}.json'
    if trace_path.exists() or updates_path.exists():raise FileExistsError('Stage already exists')
    s.json_write(folder/'source_manifest.json',dict(inputs=lock,parent=str(run.parent),arm=a.arm,config=CONFIG))
    began=time.monotonic();records=[]
    with gzip.open(trace_path,'xt',encoding='utf-8') as tracefile:
        while run.run.joint<a.until:
            data,trace=run.collect(CONFIG['rollout']);stats=run.update(data,CONFIG['ppo_epochs'])
            for row in trace:tracefile.write(json.dumps(row,allow_nan=False)+'\n')
            tracefile.flush()
            stats.update(joint=run.run.joint,updates=run.run.updates,
                         delivered=sum(x['delivered'] for x in trace),service_starts=sum(x['service_starts'] for x in trace))
            records.append(stats)
            status=dict(phase='TRAINING',arm=a.arm,seed=a.seed,joint=run.run.joint,until=a.until,
                        added_this_stage=run.run.joint-start,elapsed_seconds=time.monotonic()-began)
            if run.run.joint%1024==0:
                s.check_lock(lock)
                cp=folder/f'checkpoint-{run.run.joint:06d}.pt';s.checkpoint_write(cp,run.snapshot(lock))
                s.json_write(updates_path,records)
                status.update(checkpoint=str(cp),checkpoint_sha256=s.sha(cp))
                print(json.dumps(status),flush=True)
            s.json_write(folder/'status.json',status)
            if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('Stopped at a rollout boundary')
    s.check_lock(lock)
    if run.run.joint!=a.until:raise AssertionError('Incomplete stage')
    s.json_write(folder/f'stage_{a.until:06d}.json',dict(status,phase='COMPLETED',
        updates=len(records),updates_with_gradient=run.updates_with_gradient,
        initial_groups=run.initial_groups,final_groups=groups(run.run.policy.actor),
        critic_changed=v.actor_hash(run.run.policy.critic)!=run.initial_critic,
        trace=str(trace_path),trace_sha256=s.sha(trace_path),updates_sha256=s.sha(updates_path),
        source_integrity='PASS'))
    s.json_write(folder/'status.json',dict(status,phase='COMPLETED'))


if __name__=='__main__':main()
