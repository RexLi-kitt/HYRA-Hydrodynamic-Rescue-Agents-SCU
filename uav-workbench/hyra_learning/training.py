"""Full recurrent QMIX preflight with bounded, hash-checked resume.

The actor never receives centralized state. Targets and execution use clean
features; current training Qs optionally use the registered Sur mechanism.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid
import zipfile

from .runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F

from hyra_uav.core import Config, FloodProcess
from .networks import SearchAgent, make_mixer
from .observations import encode, pool_mean
from .team_env import TeamConfig, TeamEnv

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    partial.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    partial.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def source_files():
    paths = [ROOT/'learning.py', ROOT/'learning_configs/sur_preflight_20261025.json']
    paths += list((ROOT/'hyra_learning').rglob('*.py'))
    paths += [ROOT/'hyra_learning/vendor/PYMARL_LICENSE', ROOT/'hyra_learning/vendor/SOURCE.json']
    paths += list((ROOT/'hyra_uav').glob('*.py'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in sorted(paths)}


def central_state(env):
    view = env.view()
    depth = pool_mean(env.training_state()[None], 8).ravel()
    bed = pool_mean(view.bed[None], 8).ravel()
    bed = (bed-bed.mean())/max(float(bed.std()), .05)
    poses = []
    for agent in view.agents:
        angle = np.radians(agent.pose.heading_deg)
        poses.extend([(agent.pose.x_m+agent.dx/2)/(agent.shape[1]*agent.dx),
                      (agent.pose.z_m+agent.dx/2)/(agent.shape[0]*agent.dx), np.sin(angle), np.cos(angle)])
    return np.concatenate((depth, bed, poses, [env.time_s/env.config.duration_s])).astype(np.float32)


class EpisodeReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.episodes = []

    def add(self, episode):
        self.episodes.append(episode)
        self.episodes = self.episodes[-self.capacity:]

    def sample(self, count, rng, device):
        indices = rng.choice(len(self.episodes), count, replace=False)
        selected = [self.episodes[i] for i in indices]
        length = max(len(e['actions']) for e in selected)
        result = {}
        for key in selected[0]:
            size = length if key in ('actions', 'rewards', 'terminal') else length+1
            sample = selected[0][key]
            padded = torch.zeros((count, size, *sample.shape[1:]), dtype=sample.dtype)
            for i, episode in enumerate(selected):
                padded[i, :len(episode[key])] = episode[key]
            result[key] = padded.to(device)
        result['lengths'] = torch.tensor([len(e['actions']) for e in selected], device=device)
        result['valid'] = torch.arange(length, device=device)[None] < result['lengths'][:, None]
        return result


def rollout(agent, process, plan, epsilon, rng, device):
    env = TeamEnv(process, Config(**plan['config']), TeamConfig(**plan['team']))
    view, _ = env.reset()
    current = encode(view)
    items = {key: [] for key in ('images', 'poses', 'states', 'actions', 'rewards', 'terminal')}
    hidden = agent.initial_hidden(env.n_agents, device)
    previous = torch.zeros(env.n_agents, 4, device=device)
    agent.eval()
    total = 0.
    done = False
    while not done:
        items['images'].append(current['images'])
        items['poses'].append(current['poses'])
        items['states'].append(central_state(env))
        with torch.no_grad():
            q, hidden, _, _ = agent(torch.as_tensor(current['images'], device=device),
                                    torch.as_tensor(current['poses'], device=device), previous,
                                    len(items['actions']), hidden, enhance=False)
        actions = q.argmax(dim=-1).cpu().numpy()
        if epsilon:
            explore = rng.random(env.n_agents) < epsilon
            actions[explore] = rng.integers(0, 4, int(explore.sum()))
        view, reward, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        items['actions'].append(actions.copy())
        items['rewards'].append(reward)
        # The registered finite task ends at 600s; no bootstrap beyond cutoff.
        items['terminal'].append(float(done))
        total += reward
        previous = F.one_hot(torch.as_tensor(actions, device=device), 4).float()
        current = encode(view)
    items['images'].append(current['images'])
    items['poses'].append(current['poses'])
    items['states'].append(central_state(env))
    tensors = {key: torch.as_tensor(np.asarray(value), dtype=torch.long if key == 'actions' else torch.float32)
               for key, value in items.items()}
    metrics = dict(time_average_correct_coverage=env.integrated_quality/env.config.duration_s,
                   distance_m=env.distance_m, turn_s=env.turn_s, collision_events=env.collision_events,
                   final=env.snapshots[-1]['metrics'], return_=total, steps=len(items['actions']))
    return tensors, metrics, env


def recurrent(agent, batch, enhance):
    b, time_count, agents = batch['images'].shape[:3]
    hidden = agent.initial_hidden(b*agents, batch['images'].device)
    output, embeddings = [], []
    for t in range(time_count):
        previous = (torch.zeros(b*agents, 4, device=hidden.device) if t == 0 else
                    F.one_hot(batch['actions'][:, t-1].reshape(-1), 4).float())
        valid = (batch['lengths'] >= t)[:, None].expand(-1, agents).reshape(-1)
        q, hidden, latent, _ = agent(batch['images'][:, t].reshape(b*agents, *agent.image_shape),
                                     batch['poses'][:, t].reshape(b*agents, -1), previous, t,
                                     hidden, enhance=enhance, valid=valid)
        output.append(q.reshape(b, agents, 4))
        embeddings.append(latent.reshape(b, agents, -1))
    return torch.stack(output, dim=1), torch.stack(embeddings, dim=1)


class Learner:
    def __init__(self, n_agents, state_dim, plan, device, surprise):
        self.device, self.surprise, self.plan = device, surprise, plan
        self.actor = SearchAgent().to(device)
        self.mixer = make_mixer(n_agents, state_dim).to(device)
        self.target_actor = deepcopy(self.actor).eval().requires_grad_(False)
        self.target_mixer = deepcopy(self.mixer).eval().requires_grad_(False)
        cfg = plan['training']
        self.parameters = list(self.actor.parameters())+list(self.mixer.parameters())
        self.optimizer = torch.optim.RMSprop(self.parameters, lr=cfg['lr'], alpha=cfg['alpha'], eps=cfg['eps'])
        self.updates = 0

    def additional_loss(self, batch):
        """Independent study subclasses may add registered auxiliary losses."""
        return None, {}

    def train_batch(self, batch):
        self.actor.train()
        q, _ = recurrent(self.actor, batch, self.surprise)
        chosen = q[:, :-1].gather(-1, batch['actions'][..., None]).squeeze(-1)
        current = self.mixer(chosen, batch['states'][:, :-1]).squeeze(-1)
        with torch.no_grad():
            clean, _ = recurrent(self.actor, batch, False)
            target, _ = recurrent(self.target_actor, batch, False)
            actions = clean[:, 1:].argmax(dim=-1, keepdim=True)
            values = target[:, 1:].gather(-1, actions).squeeze(-1)
            target_total = self.target_mixer(values, batch['states'][:, 1:]).squeeze(-1)
            expected = batch['rewards']+self.plan['training']['gamma']*(1-batch['terminal'])*target_total
        mask = batch['valid'].float()
        td_loss = ((current-expected).square()*mask).sum()/mask.sum().clamp_min(1)
        extra, diagnostics = self.additional_loss(batch)
        loss = td_loss if extra is None else td_loss+extra
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite TD loss')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.plan['training']['gradient_norm_clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient norm')
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.plan['training']['target_update_every_optimizer_steps'] == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        return dict(td_loss=float(td_loss.detach().cpu()), gradient_norm=float(norm.detach().cpu()), updates=self.updates, **diagnostics)

    def checkpoint(self, replay, rng, episode, history):
        return dict(actor=self.actor.state_dict(), mixer=self.mixer.state_dict(),
                    target_actor=self.target_actor.state_dict(), target_mixer=self.target_mixer.state_dict(),
                    optimizer=self.optimizer.state_dict(), updates=self.updates, episode=episode,
                    replay=replay.episodes, numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], history=history)

    def restore(self, checkpoint, replay, rng):
        for name in ('actor', 'mixer', 'target_actor', 'target_mixer'):
            getattr(self, name).load_state_dict(checkpoint[name])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.updates = checkpoint['updates']
        replay.episodes = checkpoint['replay']
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'])
        if torch.cuda.is_available() and checkpoint['cuda_rng']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])


def memory_usage():
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(name, ctypes.c_size_t) for name in (
                'peak_working_set', 'working_set', 'peak_paged', 'paged', 'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        api = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        api.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if api(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return dict(working_set_bytes=counters.working_set, process_peak_working_set_bytes=counters.peak_working_set,
                        scope='process-wide native working set, peak includes prior methods in this process')
        raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(process_peak_working_set_bytes=peak*(1 if sys.platform == 'darwin' else 1024))


def save_checkpoint(folder, learner, replay, rng, episode, history):
    path = folder/f'checkpoint-{episode:03d}.pt'
    partial = path.with_suffix('.partial')
    torch.save(learner.checkpoint(replay, rng, episode, history), partial)
    partial.replace(path)
    write(folder/'latest.json', dict(path=path.name, sha256=digest(path), episode=episode))


def run_method(root, method, plan, processes, stop_after):
    folder = root/method
    folder.mkdir(exist_ok=True)
    seed = plan['seeds'][0]
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    device = plan['runtime']['device']
    env = TeamEnv(processes[0], Config(**plan['config']), TeamConfig(**plan['team']))
    env.reset()
    learner = Learner(env.n_agents, len(central_state(env)), plan, device, method != 'QMIX')
    replay = EpisodeReplay(plan['training']['replay_capacity_episodes'])
    rng = np.random.default_rng(seed)
    history, first = [], 0
    if (folder/'latest.json').exists():
        pointer = read(folder/'latest.json')
        path = folder/pointer['path']
        if digest(path) != pointer['sha256']:
            raise ValueError('Checkpoint hash changed')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        learner.restore(checkpoint, replay, rng)
        first, history = checkpoint['episode'], checkpoint['history']
    write(folder/'capacity.json', dict(actor=sum(p.numel() for p in learner.actor.parameters()),
                                      mixer=sum(p.numel() for p in learner.mixer.parameters()), state_dim=len(central_state(env))))
    start = time.perf_counter()
    final_episode = min(stop_after or plan['training_episodes_per_method'], plan['training_episodes_per_method'])
    if final_episode < first:
        raise ValueError('Registered resume cannot rewind completed episodes')
    for episode in range(first, final_episode):
        epsilon = 1-.9*episode/(plan['training_episodes_per_method']-1)
        data, metrics, _ = rollout(learner.actor, processes[episode % len(processes)], plan, epsilon, rng, device)
        replay.add(data)
        update = None
        if len(replay.episodes) >= plan['training']['batch_episodes']:
            update = learner.train_batch(replay.sample(plan['training']['batch_episodes'], rng, device))
        history.append(dict(episode=episode+1, epsilon=epsilon, event=plan['data'][episode % len(processes)]['id'],
                            metrics=metrics, update=update))
        save_checkpoint(folder, learner, replay, rng, episode+1, history)
        write(folder/'progress.json', dict(episode=episode+1, updates=learner.updates, env_steps=(episode+1)*60,
                                          segment_elapsed_s=time.perf_counter()-start))
        print(method, 'episode', episode+1, 'updates', learner.updates, flush=True)
    elapsed = time.perf_counter()-start
    resource = memory_usage()
    if device == 'cuda':
        torch.cuda.synchronize()
        resource.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    segments = read(folder/'segments.json') if (folder/'segments.json').exists() else []
    segments.append(dict(first_episode=first, last_episode=final_episode, elapsed_s=elapsed, memory=resource))
    write(folder/'segments.json', segments)
    write(folder/'training-history.json', history)
    if final_episode < plan['training_episodes_per_method']:
        return dict(method=method, state='checkpointed_for_registered_resume', episode=final_episode)
    evaluations = []
    for entry, process in zip(plan['data'], processes):
        data, metrics, env = rollout(learner.actor, process, plan, 0., np.random.default_rng(0), device)
        write(folder/('evaluation-'+entry['id']+'.json'), env.result())
        evaluations.append(dict(event=entry['id'], **metrics))
    result = dict(method=method, state='completed', episodes=len(history), training_joint_steps=sum(r['metrics']['steps'] for r in history),
                  updates=learner.updates, evaluation_joint_steps=sum(r['steps'] for r in evaluations),
                  capacity=read(folder/'capacity.json'), segments=segments, evaluations=evaluations,
                  limitation='Engineering preflight only: one seed, seen fixtures, twelve episodes. No effectiveness conclusion.')
    write(folder/'summary.json', result)
    return result


def preflight(root, resume, stop_after, device):
    root = root.resolve()
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        plan = read(ROOT/'learning_configs/sur_preflight_20261025.json')
        plan['status'] = 'executable_preregistered_before_first_training'
        plan['source_sha256'] = source_files()
        plan['runtime'] = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__, torch=torch.__version__,
                               device=device, cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None)
        write(root/'plan.json', plan)
        with zipfile.ZipFile(root/'frozen-sources.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in plan['source_sha256']:
                archive.write(ROOT/name, name)
    else:
        plan = read(root/'plan.json')
        if plan['source_sha256'] != source_files() or plan['runtime']['device'] != device:
            raise ValueError('Sources or device changed; preserve existing run')
    for entry in plan['data']:
        if digest(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Frozen development input changed')
    token = uuid.uuid4().hex
    lock = root/'run.lock'
    with lock.open('x', encoding='utf-8') as file:
        json.dump(dict(pid=os.getpid(), token=token), file)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        processes = [FloodProcess(ROOT/e['path']) for e in plan['data']]
        results = []
        for method in plan['methods']:
            write(root/'state.json', dict(state='running', method=method, pid=os.getpid()))
            if (root/method/'summary.json').exists():
                results.append(read(root/method/'summary.json'))
                continue
            results.append(run_method(root, method, plan, processes, stop_after))
        complete = all(r['state'] == 'completed' for r in results)
        write(root/'summary.json', dict(status='completed' if complete else 'checkpointed_for_registered_resume', methods=results))
        write(root/'state.json', dict(state='completed' if complete else 'checkpointed_for_registered_resume', pid=os.getpid()))
    except BaseException as error:
        write(root/'state.json', dict(state='stopped', pid=os.getpid(), error=type(error).__name__+': '+str(error)))
        raise
    finally:
        if lock.exists() and read(lock).get('token') == token:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight'])
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA is not available')
    if args.stop_after is not None and not 1 <= args.stop_after <= 12:
        raise ValueError('Stop must be within the registered twelve episodes')
    preflight(args.output, args.resume, args.stop_after, args.device)
