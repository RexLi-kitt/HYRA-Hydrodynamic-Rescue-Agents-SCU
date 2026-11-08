"""Learned target-conditioned joint-action residual over the search policy.

Only referee public observation fields are consumed. The safety mask is the
unchanged search mask, stored verbatim in every PPO sample. There is no rescue
controller at inference. See README for the finite local-context design.
"""
import copy
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
SEARCH = HERE.parent / 'mappo-search-v1-20261107'
sys.path.insert(0, str(SEARCH))
_spec = importlib.util.spec_from_file_location('hyra_search_core', SEARCH/'core.py')
search = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(search)

BASE = 998 + search.LOCAL_DIM
PERSON_DIM = 19
ACTION_DIM = 12
FEATURE_DIM = 64 * PERSON_DIM + 64 * 9 * ACTION_DIM
MASK_DIM = 585
INPUT_DIM = BASE + FEATURE_DIM + MASK_DIM


def person_features(o):
    """Per-agent, per-person context built exclusively from public data.

    Fresh known targets remain eligible for 30 seconds after last detection,
    supporting multi-step approach after they leave the sensor footprint.
    Eligibility only gates a learned residual; it never changes legal actions.
    """
    pos = np.asarray(o.pos, dtype=np.float64)
    seen = np.asarray(o.seen_at)
    active = np.asarray(o.active, dtype=bool)
    visible = np.asarray(o.visible, dtype=bool)
    owner = np.asarray(o.owner)
    xy = np.asarray(o.seen_xy, dtype=np.float64)
    v = np.asarray(o.seen_v, dtype=np.float64)
    observed = (seen >= 0) & active
    age = np.maximum(0., float(o.t) - seen)
    result = np.zeros((4, 64, PERSON_DIM), dtype=np.float32)
    for i in range(4):
        rel = np.where(observed[:, None], xy-pos[i], 0.)
        distance = np.linalg.norm(rel, axis=1)
        eligible = observed & (age <= 30.) & (distance <= 120.)
        team = np.delete(pos, i, axis=0)
        team_rel = (team-pos[i])/100.
        target_team = np.linalg.norm(xy[:, None, :]-team[None, :, :], axis=-1)
        f = np.column_stack((eligible, visible & observed, np.minimum(age, 120.)/120.,
              np.clip(rel/100., -2., 2.), np.minimum(distance, 200.)/100.,
              np.where(observed[:, None], np.clip(v/3., -2., 2.), 0.),
              owner == i, (owner >= 0) & (owner != i),
              np.broadcast_to(np.clip(team_rel.reshape(6), -10., 10.), (64, 6)),
              np.clip(target_team/100., 0., 10.)))
        assert f.shape == (64, PERSON_DIM)
        f[~observed] = 0.
        result[i] = f
    if not np.isfinite(result).all(): raise ValueError('Nonfinite public features')
    return result


def encode(o):
    old = search.encode(o, 'local_mask')
    person = person_features(o).reshape(4, 64 * PERSON_DIM)
    speed, duration, bound, _ = search.geometry.intent_geometry(o)
    speed = speed.reshape(4, 9, 65, 2)[:, :, 1:, :].transpose(0, 2, 1, 3)
    duration = duration.reshape(4, 9, 65)[:, :, 1:].transpose(0, 2, 1)
    bound = bound.reshape(4, 9, 65)[:, :, 1:].transpose(0, 2, 1)
    end = np.asarray(o.pos)[:, None, None, :] + speed * duration[..., None]
    xy = np.asarray(o.seen_xy)[None, :, None, :]
    vel = np.asarray(o.seen_v)[None, :, None, :]
    observed = ((np.asarray(o.seen_at) >= 0) & np.asarray(o.active, bool))[None, :, None]
    current_gap = np.where(observed[..., None], (xy-end)/50., 0.)
    future_gap = np.where(observed[..., None], (xy+5.*vel-end)/50., 0.)
    current_distance = np.linalg.norm(current_gap, axis=-1)*50.
    future_distance = np.linalg.norm(future_gap, axis=-1)*50.
    initial_distance = np.broadcast_to(np.linalg.norm(xy - np.asarray(o.pos)[:, None, None, :], axis=-1), current_distance.shape)
    fine = np.stack((np.minimum(current_distance, 5.)/5., np.minimum(future_distance, 5.)/5.,
        current_distance <= 1., future_distance <= 1., initial_distance <= 1.), axis=-1)
    fine = np.where(observed[..., None], fine, 0.)
    action = np.concatenate((np.clip(current_gap, -20., 20.), np.clip(future_gap, -20., 20.),
        (duration/5.)[..., None], (bound >= duration-1e-9)[..., None],
        (duration < 5.-1e-9)[..., None], fine), axis=-1).astype(np.float32)
    out = np.concatenate((old[:, :BASE], person, action.reshape(4, -1), old[:, -MASK_DIM:]), axis=1)
    assert out.shape == (4, INPUT_DIM)
    return out


class Actor(search.Actor):
    def __init__(self):
        super().__init__(True)
        self.rescue = nn.Sequential(nn.Linear(PERSON_DIM + ACTION_DIM + 9, 128), nn.ReLU(),
                                    nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.rescue[-1].weight)
        nn.init.zeros_(self.rescue[-1].bias)

    def joint_logits(self, x):
        m, t = self.distributions(x)
        logits = m.logits[:, :, None] + t.logits[:, None, :]
        split = BASE + 64 * PERSON_DIM
        person = x[:, BASE:split].reshape(-1, 64, PERSON_DIM)
        action = x[:, split:BASE+FEATURE_DIM].reshape(-1, 64, 9, ACTION_DIM)
        shape = (x.shape[0], 64, 9)
        move = torch.eye(9, device=x.device).view(1, 1, 9, 9).expand(*shape, 9)
        features = torch.cat((person[:, :, None, :].expand(*shape, PERSON_DIM), action, move), dim=-1)
        extra = self.rescue(features).squeeze(-1) * person[:, :, :1]
        # no-target score remains exactly the pretrained search score
        return logits + torch.cat((torch.zeros_like(extra[:, :1, :]), extra), dim=1).transpose(1, 2)


class Policy(search.Policy):
    def __init__(self, device='cuda:0'):
        self.arm = 'local_mask'
        self.device = device
        self.actor = Actor().to(device)
        self.critic = search.Critic(True).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=.0005, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=.0005, eps=1e-5)

    def joint(self, x):
        logits = self.actor.joint_logits(x).reshape(-1, MASK_DIM)
        mask = x[:, -MASK_DIM:] > .5
        if not bool(mask[:, 520].all()): raise ValueError('Safe action absent')
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf))


def make(device='cuda:0'):
    args = search.original_trainer.get_config().parse_args([])
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False
    args.ppo_epoch = 5
    args.num_mini_batch = 1
    args.lr = args.critic_lr = .0005
    args.gamma, args.gae_lambda, args.clip_param, args.entropy_coef = .99, .95, .2, .01
    p = Policy(device)
    return p, search.original_trainer.R_MAPPO(args, p, device=torch.device(device))


def source_checkpoint(seed):
    if seed not in search.SEEDS: raise ValueError('Unknown search continuation branch')
    return SEARCH / f'local_mask_{seed}' / 'checkpoint-new-025600.pt'


def initialize(seed, device='cuda:0'):
    search.seed_all(seed)
    source = torch.load(source_checkpoint(seed), map_location=device, weights_only=False)
    search.check_lock(source['source_lock'])
    s = source['run']
    if s['arm'] != 'local_mask' or s['joint'] != 25600 or s['seed'] != seed:
        raise ValueError('Unexpected source checkpoint')
    p, tr = make(device)
    missing, unexpected = p.actor.load_state_dict(s['actor'], strict=False)
    if unexpected or not missing or any(not k.startswith('rescue.') for k in missing):
        raise ValueError('Incompatible actor')
    p.critic.load_state_dict(s['critic'])
    search.extend_optimizer(p.actor_optimizer, s['actor_optimizer'])
    p.critic_optimizer.load_state_dict(s['critic_optimizer'])
    search.original_trainer.load_norm_state(tr, s['value_norm'])
    return p, tr, copy.deepcopy(s)
