"""QPLEX joint masked Double-Q loss, matching the official DMAQ learner."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay


def mix(mixer, values, states, actions, max_values):
    return (mixer(values, states, is_v=True) + mixer(
        values, states, actions=actions, max_q_i=max_values, is_v=False)).reshape(-1)


def team_loss(online, target, mixer, target_mixer, items, device, gamma=.99):
    q, mask = replay.qvalues(online, [x['state'] for x in items], device)
    actions = torch.as_tensor(np.concatenate([x['action'] for x in items]), dtype=torch.long, device=device)
    if not bool(mask.gather(1, actions[:, None]).all()):
        raise ValueError('Illegal stored action')
    states = torch.as_tensor(np.stack([x['state']['public'].reshape(-1) for x in items]), dtype=torch.float32, device=device)
    selected = q.gather(1, actions[:, None]).reshape(-1, 4)
    max_values = q.detach().masked_fill(~mask, -torch.inf).max(-1).values.reshape(-1, 4)
    onehot = F.one_hot(actions, 585).float().reshape(-1, 4 * 585)
    chosen = mix(mixer, selected, states, onehot, max_values)
    with torch.no_grad():
        following = [x['next_state'] for x in items]
        nq, nm = replay.qvalues(online, following, device)
        best = nq.masked_fill(~nm, -torch.inf).argmax(-1)
        tq, tm = replay.qvalues(target, following, device)
        if not torch.equal(nm, tm):
            raise ValueError('Target mask differs')
        target_q = tq.gather(1, best[:, None]).reshape(-1, 4)
        target_max = tq.masked_fill(~tm, -torch.inf).max(-1).values.reshape(-1, 4)
        next_states = torch.as_tensor(np.stack([x['next_state']['public'].reshape(-1) for x in items]), dtype=torch.float32, device=device)
        target_actions = F.one_hot(best, 585).float().reshape(-1, 4 * 585)
        bootstrap = mix(target_mixer, target_q, next_states, target_actions, target_max)
        reward = torch.tensor([x['reward'] for x in items], dtype=torch.float32, device=device)
        terminal = torch.tensor([x['terminated'] for x in items], dtype=torch.bool, device=device)
        expected = reward + gamma * (~terminal).float() * bootstrap
    td = chosen - expected
    return td.square().mean(), dict(q_tot=float(chosen.detach().mean()), target=float(expected.mean()),
                                   td_abs=float(td.detach().abs().mean()), bootstrap=float(bootstrap.mean()))
