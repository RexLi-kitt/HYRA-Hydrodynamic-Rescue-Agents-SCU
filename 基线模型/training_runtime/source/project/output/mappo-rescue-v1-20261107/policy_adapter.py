"""SDK policy factory: network chooses both action and target, with no override."""
import os
import importlib.util
from pathlib import Path
import torch

_spec = importlib.util.spec_from_file_location('hyra_rescue_core', Path(__file__).resolve().parent/'core.py')
c = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c)


class Frozen:
    def __init__(self, checkpoint, mode='greedy', device=None, seed=581000):
        if mode not in ('greedy','sampled'): raise ValueError('mode must be greedy or sampled')
        self.mode = mode
        device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        state = torch.load(Path(checkpoint), map_location=device, weights_only=False)
        if state['schema'] != 'MAPPO_RESCUE_V1': raise ValueError('Wrong checkpoint schema')
        if state['code_sha256']['core.py'] != c.search.sha(c.HERE/'core.py'):
            raise ValueError('Policy code differs from training checkpoint')
        self.policy = c.Policy(device)
        self.policy.actor.load_state_dict(state['actor'])
        self.policy.critic.load_state_dict(state['critic'])
        self.policy.actor.eval(); self.policy.critic.eval()
        c.search.seed_all(seed)

    def act(self, obs):
        with torch.no_grad(): actions, _, _ = self.policy.act(c.encode(obs), deterministic=self.mode == 'greedy')
        a = actions.cpu().numpy()
        return a[:,0], a[:,1]-1


def make_policy():
    checkpoint = os.environ.get('HYRA_RESCUE_CHECKPOINT')
    if not checkpoint: raise ValueError('Set HYRA_RESCUE_CHECKPOINT to a rescue checkpoint')
    return Frozen(checkpoint, os.environ.get('HYRA_RESCUE_MODE','greedy'),
                  os.environ.get('HYRA_RESCUE_DEVICE'), int(os.environ.get('HYRA_RESCUE_SEED','581000')))
