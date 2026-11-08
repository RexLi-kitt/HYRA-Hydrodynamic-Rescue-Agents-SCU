"""VDN team Double Q loss over joint transitions."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import replay


def team_loss(online, target, mixer, target_mixer, items, device, gamma=.99):
    q, mask = replay.qvalues(online, [x['state'] for x in items], device)
    actions = torch.as_tensor(np.concatenate([x['action'] for x in items]), dtype=torch.long, device=device)
    if not bool(mask.gather(1, actions[:, None]).all()):
        raise ValueError('Illegal stored action')
    states = torch.as_tensor(np.stack([x['state']['public'].reshape(-1) for x in items]), dtype=torch.float32, device=device)
    chosen = mixer(q.gather(1, actions[:, None]).reshape(-1, 4), states).reshape(-1)
    with torch.no_grad():
        nq, nm = replay.qvalues(online, [x['next_state'] for x in items], device)
        best = nq.masked_fill(~nm, -torch.inf).argmax(-1)
        tq, tm = replay.qvalues(target, [x['next_state'] for x in items], device)
        if not torch.equal(nm, tm):
            raise ValueError('Target mask differs')
        next_states = torch.as_tensor(np.stack([x['next_state']['public'].reshape(-1) for x in items]), dtype=torch.float32, device=device)
        bootstrap = target_mixer(tq.gather(1, best[:, None]).reshape(-1, 4), next_states).reshape(-1)
        reward = torch.tensor([x['reward'] for x in items], dtype=torch.float32, device=device)
        terminal = torch.tensor([x['terminated'] for x in items], dtype=torch.bool, device=device)
        expected = reward + gamma * (~terminal).float() * bootstrap
    td = chosen - expected
    return td.square().mean(), dict(q_tot=float(chosen.detach().mean()), target=float(expected.mean()),
                                   td_abs=float(td.detach().abs().mean()), bootstrap=float(bootstrap.mean()))
