"""Temporal MAT with explicit memory-bearing inference and training entrypoints."""
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from .temporal_model import TemporalMAT


class SafeTemporalMAT(TemporalMAT):
    """Do not expose inherited APIs which silently omit temporal state."""
    def _unsafe(self, *args, **kwargs):
        raise RuntimeError('This API omits temporal memory or geometry; use act or parallel_step with previous and hidden.')

    forward = _unsafe
    get_actions = _unsafe
    get_values = _unsafe
    sequential = _unsafe
    parallel = _unsafe

    def action_logits(self, shifted, obs, person, geometry, representation, mask, previous):
        return self.logits(shifted, obs, person, geometry, representation, mask)

    def parallel_step(self, obs, person, geometry, mask, previous, hidden, actions, reset=None, *, return_logits=False):
        values, rep, hidden = self.representation(obs, previous, hidden, reset)
        shift = obs.new_zeros((*obs.shape[:2], 586))
        shift[:, 0, 0] = 1
        shift[:, 1:, 1:] = F.one_hot(actions[:, :-1].long(), 585).to(obs.dtype)
        logits = self.action_logits(shift, obs, person, geometry, rep, mask, previous).masked_fill(~mask.bool(), -1e10)
        dist = Categorical(logits=logits)
        result = (dist.log_prob(actions.long()), dist.entropy(), values, hidden)
        return result + (logits,) if return_logits else result

    @torch.no_grad()
    def act(self, obs, person, geometry, mask, previous, hidden, deterministic=False, reset=None):
        values, rep, hidden = self.representation(obs, previous, hidden, reset)
        shift = obs.new_zeros((*obs.shape[:2], 586))
        shift[:, 0, 0] = 1
        actions, logs = [], []
        for i in range(4):
            logits = self.action_logits(shift, obs, person, geometry, rep, mask, previous)[:, i].masked_fill(~mask[:, i].bool(), -1e10)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            actions.append(action)
            logs.append(dist.log_prob(action))
            if i < 3:
                shift[:, i + 1, 1:] = F.one_hot(action, 585).to(obs.dtype)
        return torch.stack(actions, 1), torch.stack(logs, 1), values, hidden
