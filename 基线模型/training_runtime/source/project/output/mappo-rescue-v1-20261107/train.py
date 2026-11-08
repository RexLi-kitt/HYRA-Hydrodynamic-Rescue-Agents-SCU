"""Explicitly budgeted continuation using the original upstream MAPPO trainer.

No experiment runs on import. Checkpoints include optimizer, ValueNorm,
environment and all RNG states. A completed rollout is the checkpoint unit.
"""
import argparse
import copy
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import core as c
import provenance

s = c.search


class Run:
    def __init__(self, dataset, seed, device='cuda:0'):
        self.dataset, self.seed, self.device = dataset, seed, device
        self.policy, self.trainer, source = c.initialize(seed, device)
        self.parent, self.next_index = source['parent'], source['next_index']
        self.bridge = s.restore_bridge(dataset, self.parent, source['env'], source['bridge_done'])
        self.joint, self.updates, self.episodes = 0, 0, []
        s.set_rng(source['rng'])
        if self.bridge.done: self.new_episode()

    def new_episode(self):
        self.parent = s.TRAIN[self.next_index % len(s.TRAIN)]
        self.next_index += 1
        self.bridge = self.dataset.bridge(self.parent)

    def collect(self, steps):
        xs, actions, logps, values, rewards, dones, trace = [], [], [], [], [], [], []
        for _ in range(steps):
            x = c.encode(self.bridge.observation())
            with torch.no_grad(): a, logp, value = self.policy.act(x)
            a, logp, value = a.cpu().numpy(), logp.cpu().numpy(), value.cpu().numpy()
            _, done, metrics, rec = s.step(self.bridge, a, 'local_mask')
            rec.update(parent=self.parent, joint=self.joint+1, done=bool(done), terminal_metrics=metrics)
            xs.append(x); actions.append(a); logps.append(logp); values.append(value)
            rewards.append(np.full((4, 1), rec['reward'], np.float32)); dones.append(float(done))
            trace.append(rec); self.joint += 1
            if done:
                self.episodes.append(dict(parent=self.parent, joint=self.joint, metrics=metrics))
                self.new_episode()
        with torch.no_grad():
            nextv = self.policy.critic(self.policy.tensor(c.encode(self.bridge.observation()))).cpu().numpy()
        vals = self.trainer.value_normalizer.denormalize(np.stack(values))
        nextv = self.trainer.value_normalizer.denormalize(nextv)
        adv = np.zeros_like(vals); future = np.zeros((4, 1), np.float32)
        for j in reversed(range(steps)):
            nxt = nextv if j == steps-1 else vals[j+1]
            live = 1-dones[j]
            delta = rewards[j]+.99*live*nxt-vals[j]
            future = delta+.99*.95*live*future; adv[j] = future
        flat = lambda z: np.asarray(z).reshape(-1, *np.asarray(z).shape[2:])
        return tuple(flat(z) for z in (xs, actions, logps, values, adv+vals, adv)), trace

    def update(self, data, epochs=5):
        stats = s.original_trainer.update(self.trainer, *data, epochs=epochs)
        for module in (self.policy.actor, self.policy.critic):
            if not all(torch.isfinite(p).all() and (p.grad is None or torch.isfinite(p.grad).all())
                       for p in module.parameters()):
                raise FloatingPointError('Nonfinite parameters or gradients')
        self.updates += 1
        return stats

    def snapshot(self):
        p = self.policy
        return copy.deepcopy(dict(schema='MAPPO_RESCUE_V1', seed=self.seed, joint=self.joint,
            updates=self.updates, parent=self.parent, next_index=self.next_index, episodes=self.episodes,
            env=s.env_state(self.bridge), bridge_done=self.bridge.done,
            actor=p.actor.state_dict(), critic=p.critic.state_dict(),
            actor_optimizer=p.actor_optimizer.state_dict(), critic_optimizer=p.critic_optimizer.state_dict(),
            value_norm=s.original_trainer.norm_state(self.trainer), rng=s.rng_state(),
            source_sha256=s.sha(c.source_checkpoint(self.seed)),
            code_sha256={name:s.sha(c.HERE/name) for name in ('core.py','train.py')}))

    def restore(self, state):
        if state['schema'] != 'MAPPO_RESCUE_V1' or state['seed'] != self.seed:
            raise ValueError('Checkpoint identity mismatch')
        if state['source_sha256'] != s.sha(c.source_checkpoint(self.seed)):
            raise ValueError('Source checkpoint changed')
        if state['code_sha256'] != {name:s.sha(c.HERE/name) for name in ('core.py','train.py')}:
            raise ValueError('Training code changed')
        for key in ('actor', 'critic', 'actor_optimizer', 'critic_optimizer'):
            getattr(self.policy, key).load_state_dict(state[key])
        s.original_trainer.load_norm_state(self.trainer, state['value_norm'])
        for key in ('joint', 'updates', 'parent', 'next_index', 'episodes'):
            setattr(self, key, copy.deepcopy(state[key]))
        self.bridge = s.restore_bridge(self.dataset, self.parent, state['env'], state['bridge_done'])
        s.set_rng(state['rng'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, choices=s.SEEDS, required=True)
    ap.add_argument('--steps', type=int, required=True, help='additional joint steps, divisible by rollout')
    ap.add_argument('--rollout', type=int, default=128)
    ap.add_argument('--checkpoint-every', type=int, default=8, help='completed rollouts between checkpoints')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--resume', type=Path)
    args = ap.parse_args()
    if args.steps <= 0 or args.rollout <= 0 or args.checkpoint_every <= 0 or args.steps % args.rollout:
        ap.error('steps must be positive and divisible by rollout')
    target = args.output.resolve()
    if not target.is_relative_to(c.HERE.resolve()) or target.exists():
        ap.error('output must be a new directory inside this experiment')
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    lock=provenance.begin(target,args.seed,args.resume)
    run = Run(s.Dataset(), args.seed)
    if args.resume: run.restore(torch.load(args.resume, map_location=run.device, weights_only=False))
    records = []
    num_rollouts = args.steps // args.rollout
    with gzip.open(target/'trace.jsonl.gz','xt',encoding='utf-8') as logfile:
        for iteration in range(num_rollouts):
            data, trace = run.collect(args.rollout)
            stats = run.update(data)
            for row in trace:logfile.write(json.dumps(row,allow_nan=False)+'\n')
            logfile.flush()
            records.append(dict(joint=run.joint, ppo=stats, delivered=sum(r['delivered'] for r in trace),
                                service_starts=sum(r['service_starts'] for r in trace)))
            print(json.dumps(records[-1], allow_nan=False), flush=True)
            s.json_write(target/'status.json',dict(phase='TRAINING',joint=run.joint,updates=run.updates,
                                                   requested_new_steps=args.steps,rollout=args.rollout))
            stop = (c.HERE / 'STOP_REQUESTED').exists()
            if (iteration+1) % args.checkpoint_every == 0 or iteration+1 == num_rollouts or stop:
                provenance.finish(lock)
                path = target / f'checkpoint-{run.joint:06d}.pt'
                s.checkpoint_write(path, run.snapshot())
                s.json_write(target/'updates.json', records)
            if stop: break
    provenance.finish(lock)
    s.json_write(target/'status.json',dict(phase='PAUSED' if stop else 'COMPLETED',joint=run.joint,
        updates=run.updates,checkpoint=str(path),checkpoint_sha256=s.sha(path),source_integrity='PASS'))


if __name__ == '__main__': main()
