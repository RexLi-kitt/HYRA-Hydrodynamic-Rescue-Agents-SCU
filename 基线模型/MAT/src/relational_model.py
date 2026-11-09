"""Generalizable target relationships on top of CoordinatedMAT.

Only ``relational_head.*`` is new.  Fine-tune this prefix alone or unfreeze
parent modules as desired.  Its zero-initialized final layer preserves the
parent's output exactly; no-target and uncommitted targets receive zero delta.
"""
import torch
from torch import nn
from .coordinated_model import CoordinatedMAT


class RelationalMAT(CoordinatedMAT):
    def __init__(self, device='cpu', memory=True):
        super().__init__(device=device, memory=memory)
        # log distances, nearness, their differences, sight/age/ownership,
        # and whether this aircraft committed to the target on the prior step.
        self.relational_head = nn.Sequential(nn.Linear(11, 64), nn.ReLU(),
                                             nn.Linear(64, 64), nn.ReLU(),
                                             nn.Linear(64, 1))
        nn.init.zeros_(self.relational_head[-1].weight)
        nn.init.zeros_(self.relational_head[-1].bias)
        self.relational_head.to(device)

    def action_logits(self, shifted, obs, person, geometry, representation, mask, previous):
        base = super().action_logits(shifted, obs, person, geometry, representation, mask, previous)
        b = obs.shape[0]
        # Person team-distance columns follow ascending aircraft ID excluding self.
        team_ids = torch.tensor([[j for j in range(4) if j != i] for i in range(4)],
                                device=obs.device)
        team_previous = previous[:, team_ids]  # [B, 4, 3]
        target_ids = torch.arange(1, 65, device=obs.device)
        claims = ((team_previous[..., None] >= 0) & (team_previous[..., None] < 585) &
                  (team_previous[..., None].long().remainder(65) == target_ids))
        # Columns 16:19 correspond to the same ordered three teammate IDs.
        team_distance = person[..., 16:19].transpose(-1, -2) * 100.0
        distance = team_distance.masked_fill(~claims, float('inf')).min(dim=2).values
        has_claim = claims.any(dim=2)
        legal_person = mask.reshape(b, 4, 9, 65).bool().any(dim=2)[..., 1:]
        ids = torch.nonzero(has_claim & legal_person, as_tuple=False)
        if not ids.numel():
            return base
        batch, agent, target = ids.unbind(dim=1)
        public = person[batch, agent, target]
        own_dist = (public[:, 5] * 100.0).clamp_min(0)
        team_dist = distance[batch, agent, target].clamp_min(0)
        own_log = torch.log1p(own_dist) / 7.0
        team_log = torch.log1p(team_dist) / 7.0
        own_near = torch.exp(-own_dist / 50.0)
        team_near = torch.exp(-team_dist / 50.0)
        own_previous = previous[batch, agent]
        same_target = ((own_previous >= 0) & (own_previous < 585) &
                       (own_previous.long().remainder(65) == target + 1)).to(obs.dtype)
        z = torch.stack((own_log, team_log, own_log - team_log,
                         own_near, team_near, own_near - team_near,
                         public[:, 1], public[:, 2], public[:, 8], public[:, 9], same_target), dim=-1)
        score = self.relational_head(z).squeeze(-1)
        delta = obs.new_zeros((b, 4, 65)).index_put((batch, agent, target + 1), score)
        return base + delta.unsqueeze(2).expand(-1, -1, 9, -1).reshape(b, 4, 585)
