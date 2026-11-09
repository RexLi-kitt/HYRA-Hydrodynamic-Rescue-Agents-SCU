"""Learned short-loss reacquisition on public, temporally encoded inputs.

Only ``reacquisition_head.*`` is new. The final layer is zero initialized;
its raw logit is shared by the policy residual and supervised outcome training.
Public position, velocity and age inputs retain their source encoding's clips.
"""
import math
import torch
from torch import nn
from .relational_model import RelationalMAT


class ReacquisitionMAT(RelationalMAT):
    def __init__(self, device='cpu', memory=True):
        super().__init__(device=device, memory=memory)
        # Relative endpoint, log distance, heading-relative bearing, velocity,
        # age, duration and move one-hot; no private state or target identity.
        self.reacquisition_head = nn.Sequential(nn.Linear(18, 64), nn.ReLU(),
                                                nn.Linear(64, 64), nn.ReLU(),
                                                nn.Linear(64, 1))
        nn.init.zeros_(self.reacquisition_head[-1].weight)
        nn.init.zeros_(self.reacquisition_head[-1].bias)
        self.reacquisition_head.to(device)

    @staticmethod
    def reacquisition_features(obs, person, geometry, agent, move):
        """Selected rows: obs[N,1450], person[N,19], geometry[N,12], indices[N].

        Returns [N,18] for prediction from the identical features used by the
        policy. Each row must describe one actual or candidate agent/target/move.
        """
        agent, move = agent.long(), move.long()
        velocity = person[:, 6:8] * 3.0
        age = person[:, 2] * 120.0
        relative = geometry[:, :2] * 50.0 + (age[:, None] + 5.0) * velocity
        distance = torch.linalg.vector_norm(relative, dim=-1)
        bearing = relative / distance.clamp_min(1e-6)[:, None]
        root2 = math.sqrt(0.5)
        directions = obs.new_tensor(((0, 1), (0, -1), (-1, 0), (1, 0),
                                     (-root2, root2), (-root2, -root2),
                                     (root2, root2), (root2, -root2), (0, 0)))
        heading = obs[:, 408:416].reshape(-1, 4, 2)[torch.arange(len(obs), device=obs.device), agent]
        heading = torch.nn.functional.normalize(heading, dim=-1, eps=1e-6)
        direction = directions[move]
        stationary = (move == 8) | (geometry[:, 4] <= 1e-7)
        heading = torch.where(stationary[:, None], heading, direction)
        dot = (heading * bearing).sum(dim=-1)
        cross = heading[:, 0] * bearing[:, 1] - heading[:, 1] * bearing[:, 0]
        onehot = torch.nn.functional.one_hot(move, 9).to(obs.dtype)
        return torch.cat((relative / 100., (torch.log1p(distance) / 7.)[:, None],
                          dot[:, None], cross[:, None], velocity / 3.,
                          (age / 120.)[:, None], geometry[:, 4:5], onehot), dim=-1)

    def score_reacquisition(self, obs, person, geometry, agent, move):
        return self.reacquisition_head(self.reacquisition_features(obs, person, geometry, agent, move)).squeeze(-1)

    def action_logits(self, shifted, obs, person, geometry, representation, mask, previous):
        base = super().action_logits(shifted, obs, person, geometry, representation, mask, previous)
        b = obs.shape[0]
        # Person age has float32 rounding from seconds/120; small tolerance at
        # endpoints avoids accidentally excluding a 5-second observation.
        short_loss = ((person[..., 1] < 0.5) & (person[..., 2] >= 5.0 / 120.0 - 1e-7) &
                      (person[..., 2] <= 30.0 / 120.0 + 1e-7))
        legal = mask.reshape(b, 4, 9, 65).bool()[..., 1:].permute(0, 1, 3, 2)
        ids = torch.nonzero(short_loss[..., None] & legal, as_tuple=False)
        if not ids.numel(): return base
        batch, agent, target, move = ids.unbind(dim=1)
        score = self.score_reacquisition(obs[batch, agent], person[batch, agent, target],
                                         geometry[batch, agent, target, move], agent, move)
        delta = obs.new_zeros((b, 4, 9, 65)).index_put((batch, agent, move, target + 1), score)
        return base + delta.reshape(b, 4, 585)
