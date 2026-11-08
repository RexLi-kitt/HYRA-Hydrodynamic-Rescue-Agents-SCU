"""Freeze the v1 actor backbone and train its rescue head with 1e-4 PPO.

Outputs remain compatible with the v1 SDK adapter. Fresh and resumed runs
preserve all v1 Run state, Adam optimizers, ValueNorm, environment and RNG.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
V1=HERE.parent
sys.path.insert(0,str(V1))
import core as c
import train as original
s=c.search

CONFIG=dict(variant='STABLE_PPO_V2',actor_trainable_prefix='rescue.',actor_lr=0.0001,
            critic_lr=0.0005,rollout=128,ppo_epochs=5,endpoint_joint_steps=8192)
VARIANT_FILES=('train_v2.py','protocol_v2.json','validate_v2.py')
V1_FILES=('core.py','train.py','demo.py','micro.py','teacher_multistep.py','evaluate.py','policy_adapter.py','provenance.py')


def actor_hash(actor,rescue=False):
    h=hashlib.sha256()
    for name,tensor in sorted(actor.state_dict().items()):
        if name.startswith('rescue.') != rescue:continue
        array=tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode('utf8'));h.update(str(array.dtype).encode('ascii'))
        h.update(str(array.shape).encode('ascii'));h.update(array.tobytes())
    return h.hexdigest()


def bc_checkpoint(seed):return V1/f'candidate_demo_{seed}'/'checkpoint.pt'


def load_bc(seed,device):
    path=bc_checkpoint(seed)
    # The locked v1 runner validates the original source manifest, checkpoint,
    # demo budget and teacher/code identities without running training.
    spec=importlib.util.spec_from_file_location('v1_stable_parent_audit',V1/'orchestrate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checked=module.train_ok(dict(kind='demo',seed=seed,output=str(path.parent)))
    state=torch.load(path,map_location=device,weights_only=False)
    if state['seed']!=seed or state['joint']!=0 or state['demo_updates']!=1000:
        raise ValueError('Expected completed demonstration checkpoint')
    if checked['checkpoint_sha256']!=s.sha(path):raise ValueError('Demonstration parent changed')
    return state,checked


def source_lock(seed,resume=None):
    lock=s.source_lock()
    # Revalidate and include every source used to create the BC parent, rather
    # than relying on the manifest file's own digest alone. Retain its
    # archived snapshots as well so an in-flight alteration is detected.
    parent_manifest=s.read(bc_checkpoint(seed).parent/'source_manifest.json')
    for path,digest in parent_manifest['inputs'].items():
        resolved=Path(path).resolve()
        if s.sha(resolved)!=digest:raise ValueError('Changed BC parent input '+str(resolved))
        lock[str(resolved)]=digest
    for relative,digest in parent_manifest['snapshots'].items():
        resolved=(bc_checkpoint(seed).parent/relative).resolve()
        if s.sha(resolved)!=digest:raise ValueError('Changed BC parent snapshot '+str(resolved))
        lock[str(resolved)]=digest
    for name in V1_FILES:lock[str(V1/name)]=s.sha(V1/name)
    for name in VARIANT_FILES:lock[str(HERE/name)]=s.sha(HERE/name)
    for path in (V1/'orchestrate.py',c.source_checkpoint(seed),bc_checkpoint(seed),bc_checkpoint(seed).parent/'source_manifest.json',
                 bc_checkpoint(seed).parent/'demo_log.json'):
        lock[str(path)]=s.sha(path)
    if resume:lock[str(Path(resume).resolve())]=s.sha(resume)
    return lock


def provenance_begin(folder,seed,resume):
    folder.mkdir(parents=True,exist_ok=False);lock=source_lock(seed,resume)
    snapshot=folder/'source_snapshot';snapshot.mkdir();saved={}
    for i,(name,digest) in enumerate(lock.items()):
        dest=snapshot/f'{i:03d}_{Path(name).name}';shutil.copyfile(name,dest)
        if s.sha(dest)!=digest:raise ValueError('Snapshot differs from source')
        saved[str(dest.relative_to(folder))]=digest
    s.json_write(folder/'source_manifest.json',dict(inputs=lock,snapshots=saved,config=CONFIG))
    return lock


class StableRun:
    def __init__(self,ds,seed,device='cuda:0'):
        self.seed=seed;self.device=device
        self.parent_path=bc_checkpoint(seed);self.parent_sha=s.sha(self.parent_path)
        parent,self.parent_provenance=load_bc(seed,device)
        self.run=original.Run(ds,seed,device)
        self.run.restore(parent)
        self.initial_backbone_hash=actor_hash(self.run.policy.actor)
        self.freeze()

    def freeze(self):
        actor=self.run.policy.actor
        actor.zero_grad(set_to_none=True)
        for name,p in actor.named_parameters():p.requires_grad_(name.startswith('rescue.'))
        if not any(p.requires_grad for p in actor.parameters()):raise ValueError('No trainable rescue parameters')
        for group in self.run.policy.actor_optimizer.param_groups:group['lr']=CONFIG['actor_lr']

    def collect(self,steps):return self.run.collect(steps)

    def update(self,data,epochs=5):
        if actor_hash(self.run.policy.actor)!=self.initial_backbone_hash:raise ValueError('Backbone changed before PPO update')
        old_head=actor_hash(self.run.policy.actor,rescue=True)
        stats=self.run.update(data,epochs=epochs)
        actual=actor_hash(self.run.policy.actor)
        if actual!=self.initial_backbone_hash:raise AssertionError('Frozen actor backbone changed')
        if any(p.grad is not None for n,p in self.run.policy.actor.named_parameters() if not n.startswith('rescue.')):
            raise AssertionError('Frozen actor gradient present')
        return dict(ppo=stats,backbone_before_sha256=self.initial_backbone_hash,backbone_after_sha256=actual,
                    rescue_head_before_sha256=old_head,rescue_head_after_sha256=actor_hash(self.run.policy.actor,rescue=True))

    def snapshot(self,lock):
        if actor_hash(self.run.policy.actor)!=self.initial_backbone_hash:raise AssertionError('Frozen actor changed')
        state=self.run.snapshot()
        state['stable_ppo_v2']=copy.deepcopy(dict(config=CONFIG,parent_bc_path=str(self.parent_path.resolve()),
            parent_bc_sha256=self.parent_sha,parent_source_manifest_sha256=self.parent_provenance['source_manifest_sha256'],
            frozen_actor_initial_sha256=self.initial_backbone_hash,frozen_actor_current_sha256=actor_hash(self.run.policy.actor),
            rescue_actor_current_sha256=actor_hash(self.run.policy.actor,rescue=True),
            variant_code_sha256={n:s.sha(HERE/n) for n in VARIANT_FILES},source_lock=lock))
        return state

    def restore(self,state):
        meta=state.get('stable_ppo_v2')
        if not meta or meta['config']!=CONFIG or meta['parent_bc_sha256']!=self.parent_sha:
            raise ValueError('Not a compatible stable continuation')
        if meta['parent_bc_path']!=str(self.parent_path.resolve()):raise ValueError('Parent path mismatch')
        if meta['variant_code_sha256']!={n:s.sha(HERE/n) for n in VARIANT_FILES}:raise ValueError('Variant code changed')
        s.check_lock(meta['source_lock'])
        if meta['frozen_actor_initial_sha256']!=self.initial_backbone_hash:raise ValueError('Frozen parent differs')
        self.run.restore(state);self.freeze()
        if actor_hash(self.run.policy.actor)!=self.initial_backbone_hash or actor_hash(self.run.policy.actor,rescue=True)!=meta['rescue_actor_current_sha256']:
            raise ValueError('Actor fingerprint mismatch')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,choices=s.SEEDS,required=True)
    ap.add_argument('--steps',type=int,default=8192,help='additional joint steps; final total must not exceed 8192')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--resume',type=Path)
    ap.add_argument('--device',default='cuda:0')
    a=ap.parse_args();out=a.output.resolve()
    if a.steps<=0 or a.steps%128 or not out.is_relative_to(HERE) or out.exists():
        ap.error('steps must be positive multiples of 128; output must be a new v2 subdirectory')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    lock=provenance_begin(out,a.seed,a.resume)
    run=StableRun(s.Dataset(),a.seed,a.device)
    if a.resume:run.restore(torch.load(a.resume,map_location=a.device,weights_only=False))
    if run.run.joint+a.steps>8192:raise ValueError('Requested steps exceed registered final endpoint')
    records=[];rollouts=a.steps//128;began=time.time();stop=False
    with gzip.open(out/'trace.jsonl.gz','xt',encoding='utf8') as tracefile:
        for i in range(rollouts):
            data,trace=run.collect(128);stats=run.update(data,epochs=5)
            for row in trace:tracefile.write(json.dumps(row,allow_nan=False)+'\n')
            tracefile.flush();stats.update(joint=run.run.joint,updates=run.run.updates,
                delivered=sum(r['delivered'] for r in trace),service_starts=sum(r['service_starts'] for r in trace))
            records.append(stats);print(json.dumps(stats,allow_nan=False),flush=True)
            s.json_write(out/'status.json',dict(phase='TRAINING',seed=a.seed,joint=run.run.joint,
                updates=run.run.updates,elapsed_s=time.time()-began,frozen_actor_sha256=run.initial_backbone_hash))
            stop=(HERE/'STOP_REQUESTED').exists()
            if (i+1)%8==0 or i+1==rollouts or stop:
                s.check_lock(lock)
                path=out/f'checkpoint-{run.run.joint:06d}.pt';s.checkpoint_write(path,run.snapshot(lock))
                s.json_write(out/'updates.json',records)
            if stop:break
    s.check_lock(lock)
    s.json_write(out/'status.json',dict(phase='PAUSED' if stop else 'COMPLETED',seed=a.seed,
        joint=run.run.joint,updates=run.run.updates,endpoint_reached=run.run.joint==8192,
        frozen_actor_initial_sha256=run.initial_backbone_hash,
        frozen_actor_final_sha256=actor_hash(run.run.policy.actor),parent_bc_sha256=run.parent_sha,
        checkpoint=str(path),checkpoint_sha256=s.sha(path),source_integrity='PASS'))


if __name__=='__main__':main()
