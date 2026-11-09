"""Learned causal coordination and approach residual on the public temporal MAT.

Only ``coordination_head.*`` are new trainable parameters.  Freezing every
other named parameter trains this residual alone; unfreezing the parent allows
end-to-end learning.  The final layer starts at zero, preserving base logits.
"""
import torch
from torch import nn
from .safe_temporal import SafeTemporalMAT


class CoordinatedMAT(SafeTemporalMAT):
    def __init__(self, device='cpu', memory=True):
        super().__init__(device=device, memory=memory)
        # Public context + action identity + (prefix, own prior, team prior).
        self.coordination_head = nn.Sequential(
            nn.Linear(128 + 19 + 12 + 9 + 3, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1),
        )
        nn.init.zeros_(self.coordination_head[-1].weight)
        nn.init.zeros_(self.coordination_head[-1].bias)
        self.coordination_head.to(device)

    def action_logits(self, shifted, obs, person, geometry, representation, mask, previous):
        base = super().action_logits(shifted, obs, person, geometry, representation, mask, previous)
        b = obs.shape[0]
        # At slot i the shifted input contains only slot i-1's action.  Summing
        # through i therefore includes exactly agents < i, never a future slot.
        prefix = shifted[..., 1:].reshape(b, 4, 9, 65).sum(dim=2).cumsum(dim=1)[..., 1:]
        # 0 is no target in each action; 585 is the no-previous-action sentinel.
        valid_previous = (previous >= 0) & (previous < 585)
        previous_target = previous.long().remainder(65)
        targets = torch.arange(1, 65, device=obs.device)
        prior_match = ((previous_target[..., None] == targets) & valid_previous[..., None]).to(obs.dtype)
        teammate_match = prior_match.sum(dim=1, keepdim=True) - prior_match
        cues = torch.stack((prefix, prior_match, teammate_match), dim=-1).reshape(b * 4, 64, 3)
        valid = mask.reshape(b * 4, 9, 65).bool().any(dim=1)[:, 1:]
        ids = torch.nonzero(valid, as_tuple=False)
        if not ids.numel():
            return base
        sample, target = ids.unbind(dim=1)
        k = ids.shape[0]
        moves = torch.eye(9, device=obs.device, dtype=obs.dtype).unsqueeze(0).expand(k, -1, -1)
        context = representation.reshape(b * 4, -1)[sample, None, :].expand(-1, 9, -1)
        people = person.reshape(b * 4, 64, 19)[sample, target, None, :].expand(-1, 9, -1)
        geo = geometry.reshape(b * 4, 64, 9, 12)[sample, target]
        history = cues[sample, target, None, :].expand(-1, 9, -1)
        score = self.coordination_head(torch.cat((context, people, geo, moves, history), dim=-1)).squeeze(-1)
        delta = obs.new_zeros((b * 4, 64, 9)).index_put((sample, target), score)
        delta = torch.cat((torch.zeros_like(delta[:, :1]), delta), dim=1).transpose(1, 2)
        return base + delta.reshape(b, 4, 585)
