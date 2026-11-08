"""Frozen-backbone VDN with optional retained demonstrations."""
import argparse
from collections import deque
import copy
import gzip
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V4 = HERE.parent/'retained_demo_v4'
sys.path.insert(0, str(HERE.parent))
import demo
import policy_core as p
import replay
import train
sys.path.insert(0, str(V4))
import data
from team_td import team_loss
s = p.geometry.search


def source_lock(bc):
    lock = train.source_lock(bc)
    for path in [HERE/'train_vdn.py', HERE/'team_td.py', HERE/'protocol.json'] + [V4/name for name in ('train_v4.py','data.py','protocol.json','demonstrations.pt','data_manifest.json')]:
        lock[str(path)] = p.sha(path)
    return lock


def retained_update(online, target, reference, optimizer, items, examples, device, weight=5.):
    td, team_stats = team_loss(online, target, items, device)
    examples_states = [r[0] for r in examples]
    dq, dm = replay.qvalues(online, examples_states, device)
    indices = torch.as_tensor([4*i+r[1] for i, r in enumerate(examples)], device=device)
    dq, dm = dq[indices], dm[indices]
    truth = torch.as_tensor([r[2] for r in examples], device=device)
    positive = truth >= 0
    if not bool(dm[positive].gather(1, truth[positive, None]).all()): raise AssertionError('Illegal demonstration')
    logp = torch.log_softmax(dq.masked_fill(~dm, -torch.inf), -1)
    ce = -logp[positive].gather(1, truth[positive, None]).squeeze(1)
    with torch.no_grad():
        reference_q, reference_mask = replay.qvalues(reference, examples_states, device)
        if not torch.equal(dm, reference_mask[indices]): raise AssertionError('Reference mask differs')
        reference_logp = torch.log_softmax(reference_q[indices].masked_fill(~dm, -torch.inf), -1)
    kl = (reference_logp.exp() * (torch.where(dm, reference_logp, 0.) - torch.where(dm, logp, 0.))).sum(-1)
    supervised = (ce.sum() + kl[~positive].sum()) / len(examples)
    loss = td + weight * supervised
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite retained-demo loss')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    parameters = [x for x in online.parameters() if x.requires_grad]
    gradients = [x.grad for x in parameters if x.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients): raise FloatingPointError('Invalid gradient')
    if not any(bool((g != 0).any()) for g in gradients): raise AssertionError('No rescue gradient')
    norm = torch.nn.utils.clip_grad_norm_(parameters, 10.); optimizer.step()
    return dict(loss=float(loss.detach()), td_loss=float(td.detach()),
                demo_loss=float(supervised.detach()), positive_ce=float(ce.detach().mean()),
                abstention_kl=float(kl[~positive].detach().mean()),
                grad_norm=float(norm), td_abs=team_stats['td_abs'], q_tot=team_stats['q_tot'], target=team_stats['target'], bootstrap=team_stats['bootstrap'],
                updated=True, demo_positive=int(positive.sum()), demo_abstention=int((~positive).sum()))


class Run(train.Run):
    def __init__(self, seed, checkpoint, arm, device='cuda:0'):
        super().__init__(seed, checkpoint, device, anneal_steps=8192)
        if arm not in ('plain', 'hard') or self.bc_phase != 'rescue': raise ValueError('Wrong arm/parent')
        self.arm = arm
        dataset = torch.load(V4/'demonstrations.pt', map_location='cpu', weights_only=False)
        s.check_lock(dataset['source_lock'])
        if dataset['base_sha256'] != self.initial_base: raise ValueError('Cached recurrent features incompatible')
        if demo.data_hash(dataset['old']) != dataset['old_sha256'] or demo.data_hash(dataset['hard']) != dataset['hard_sha256']:
            raise ValueError('Demonstration content changed')
        self.examples = {name: rows for name, rows in (
            ('positive', [x for x in dataset['old'] if x[2] >= 0]),
            ('negative', [x for x in dataset['old'] if x[2] < 0]),
            ('hard', [x for x in dataset['hard'] if x[2] >= 0]))}
        self.reference = copy.deepcopy(self.online).eval()
        for x in self.reference.parameters(): x.requires_grad_(False)
        self.reference_hash = train.module_hash(self.reference)
        self.aux_rng = np.random.default_rng(seed + 940000)
        self.success = deque(maxlen=1024); self.recent = deque(maxlen=8)
        self.last_success_added = -1; self.success_draws = self.demo_draws = 0
        self.config.update(algorithm='VDN', method='示范初始化、冻结搜索骨干的 VDN 微调',
                           variant='VDN_FROZEN_BACKBONE_V1', arm=arm, demo_weight=5., demo_batch=32,
                           success_capacity=1024, success_history=8, success_quota=8,
                           data_sha256=p.sha(V4/'demonstrations.pt'), reference_sha256=self.reference_hash)
        if arm == 'plain':
            self.config.update(demo_weight=0., demo_batch=0, success_quota=0)

    def step(self):
        rec = super().step()
        if self.arm == 'plain': return rec
        item = self.replay.items[(self.replay.index - 1) % self.replay.capacity]
        self.recent.append((self.joint, item))
        if rec['delivered'] > 0:
            for step, transition in self.recent:
                if step > self.last_success_added:
                    self.success.append((step, transition)); self.last_success_added = step
        if rec['terminated']: self.recent.clear()
        return rec

    def update(self):
        self.attempts += 1
        if self.arm == 'plain':
            return self.update_plain()
        success_count = 8 if self.success else 0
        items, ids = self.replay.sample(32-success_count)
        success_ids = []
        if success_count:
            chosen = self.aux_rng.choice(len(self.success), success_count, replace=len(self.success)<success_count)
            items += [self.success[int(i)][1] for i in chosen]
            success_ids = [self.success[int(i)][0] for i in chosen]
        quotas = [('positive', 8), ('hard', 8), ('negative', 16)]
        examples, example_ids = [], []
        for kind, count in quotas:
            chosen = self.aux_rng.integers(len(self.examples[kind]), size=count)
            examples.extend(self.examples[kind][int(i)] for i in chosen)
            example_ids.extend((kind, int(i)) for i in chosen)
        before = train.module_hash(self.online.rescue)
        stats = retained_update(self.online, self.target, self.reference, self.optimizer, items, examples, self.device)
        self.updates += 1; self.demo_draws += len(examples); self.success_draws += success_count
        if self.updates % self.config['target_every_updates'] == 0:
            self.target.load_state_dict(self.online.state_dict()); self.target_copies += 1
        if train.module_hash(self.online.base) != self.initial_base or train.module_hash(self.online.search) != self.initial_search:
            raise AssertionError('Frozen backbone changed')
        return dict(joint=self.joint, attempt=self.attempts, effective_updates=self.updates,
                    stats=stats, uniform_indices=ids, success_steps=success_ids, demo_indices=example_ids,
                    success_buffer_size=len(self.success), demo_draws=self.demo_draws, success_draws=self.success_draws,
                    rescue_before_sha256=before, rescue_after_sha256=train.module_hash(self.online.rescue))

    def update_plain(self):
        items, ids = self.replay.sample(32)
        loss, stats = team_loss(self.online, self.target, items, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite team loss')
        if loss.requires_grad: loss.backward()
        params = [x for x in self.online.parameters() if x.requires_grad]
        grads = [x.grad for x in params if x.grad is not None]
        if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Nonfinite gradient')
        updated = bool(grads) and any(bool((g != 0).any()) for g in grads)
        norm = float(torch.nn.utils.clip_grad_norm_(params, 10.)) if updated else 0.
        if updated:
            self.optimizer.step(); self.updates += 1
            if self.updates % self.config['target_every_updates'] == 0:
                self.target.load_state_dict(self.online.state_dict()); self.target_copies += 1
        else: self.skips += 1
        if train.module_hash(self.online.base) != self.initial_base or train.module_hash(self.online.search) != self.initial_search:
            raise AssertionError('Frozen backbone changed')
        stats.update(loss=float(loss.detach()), grad_norm=norm, updated=updated)
        return dict(joint=self.joint, attempt=self.attempts, effective_updates=self.updates,
                    skipped_updates=self.skips, target_copies=self.target_copies, uniform_indices=ids, stats=stats)

    def snapshot(self, lock):
        state = super().snapshot(lock)
        state['retained_demo'] = copy.deepcopy(dict(arm=self.arm, success=self.success, recent=self.recent,
                last_success_added=self.last_success_added, success_draws=self.success_draws,
                demo_draws=self.demo_draws, aux_rng=self.aux_rng.bit_generator.state))
        return state

    def restore(self, state, lock):
        if state['retained_demo']['arm'] != self.arm: raise ValueError('Wrong replay arm')
        super().restore(state, lock)
        for name, value in state['retained_demo'].items():
            if name == 'aux_rng': self.aux_rng.bit_generator.state = copy.deepcopy(value)
            else: setattr(self, name, copy.deepcopy(value))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--arm', choices=('plain','hard'), required=True)
    ap.add_argument('--steps', type=int, default=8192); ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--resume', type=Path); ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(); out = a.out.resolve(); bc = data.parent(a.seed)
    if out.exists() or not out.is_relative_to(HERE) or a.steps <= 0: ap.error('New output under VDN directory required')
    torch.set_num_threads(1); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    lock = source_lock(bc); out.mkdir(parents=True); snap = out/'source_snapshot'; snap.mkdir()
    for i, (name, digest) in enumerate(lock.items()):
        if Path(name).suffix == '.pt': continue
        dest = snap/f'{i:03d}_{Path(name).name}'; shutil.copyfile(name, dest)
        if p.sha(dest) != digest: raise ValueError('Snapshot mismatch')
    run = Run(a.seed, bc, a.arm, a.device)
    if a.resume: run.restore(torch.load(a.resume, map_location=a.device, weights_only=False), lock)
    s.json_write(out/'source_manifest.json', dict(inputs=lock, config=run.config, runtime=demo.runtime()))
    start = run.joint; updates = []; began = time.time(); stopped = False
    with gzip.open(out/'train_trace.jsonl.gz', 'xt', encoding='utf8') as trace:
        while run.joint < start+a.steps:
            trace.write(json.dumps(run.step(), allow_nan=False)+'\n')
            if run.joint >= 64 and run.joint % 32 == 0:
                updates.append(run.update())
            if run.joint % 128 == 0:
                trace.flush()
                status = dict(phase='TRAINING', joint=run.joint, updates=run.updates,
                              success_size=len(run.success), elapsed_s=time.time()-began)
                s.json_write(out/'status.json', status); print(json.dumps(status), flush=True)
            stopped = (HERE/'STOP_REQUESTED').exists()
            if run.joint % 2048 == 0 or run.joint == start+a.steps or stopped:
                s.check_lock(lock); path = out/f'checkpoint-{run.joint:06d}.pt'
                demo.save(path, run.snapshot(lock)); s.json_write(out/'updates.json', updates)
            if stopped: break
    s.check_lock(lock)
    if train.module_hash(run.reference) != run.reference_hash: raise AssertionError('Reference changed')
    s.json_write(out/'status.json', dict(phase='PAUSED' if stopped else 'COMPLETED', arm=a.arm, seed=a.seed,
        joint=run.joint, updates=run.updates, target_copies=run.target_copies, checkpoint=str(path),
        checkpoint_sha256=p.sha(path), parent_sha256=run.parent_sha, base_sha256=run.initial_base,
        search_sha256=run.initial_search, demo_draws=run.demo_draws, success_draws=run.success_draws,
        source_integrity='PASS', elapsed_s=time.time()-began))


if __name__ == '__main__': main()


