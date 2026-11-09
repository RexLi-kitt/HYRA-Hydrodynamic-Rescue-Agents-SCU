"""Public-observation MAT with per-aircraft trainable temporal memory."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .geometry_mat import GeometryMAT


class TemporalMAT(GeometryMAT):
    def __init__(self, device='cpu', memory=True):
        super().__init__(device)
        self.memory = memory
        self.action_history = nn.Embedding(586, 32)
        self.recurrent = nn.GRUCell(160, 128)
        self.memory_projection = nn.Linear(128, 128)
        nn.init.zeros_(self.memory_projection.weight)
        nn.init.zeros_(self.memory_projection.bias)
        self.to(device)

    def initial(self, batch, device=None):
        return torch.zeros(batch, 4, 128, device=device or next(self.parameters()).device)

    def representation(self, obs, previous, hidden, reset=None):
        if reset is not None:
            hidden = hidden * (1 - reset.to(obs.dtype).reshape(-1, 1, 1))
        x = self.encoder.obs_encoder(obs)
        if self.memory:
            a = self.action_history(previous.long())
            hidden = self.recurrent(torch.cat((x, a), -1).reshape(-1, 160), hidden.reshape(-1, 128)).reshape(-1, 4, 128)
            x = x + self.memory_projection(hidden)
        rep = self.encoder.blocks(self.encoder.ln(x))
        return self.encoder.head(rep).squeeze(-1), rep, hidden

    def parallel_step(self, obs, person, geometry, mask, previous, hidden, actions, reset=None):
        values, rep, hidden = self.representation(obs, previous, hidden, reset)
        shift = obs.new_zeros((*obs.shape[:2], 586)); shift[:, 0, 0] = 1
        shift[:, 1:, 1:] = F.one_hot(actions[:, :-1].long(), 585).to(obs.dtype)
        logits = self.logits(shift, obs, person, geometry, rep, mask).masked_fill(~mask.bool(), -1e10)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions.long()), dist.entropy(), values, hidden

    @torch.no_grad()
    def act(self, obs, person, geometry, mask, previous, hidden, deterministic=False, reset=None):
        values, rep, hidden = self.representation(obs, previous, hidden, reset)
        shift = obs.new_zeros((*obs.shape[:2], 586)); shift[:, 0, 0] = 1
        actions, logs = [], []
        for i in range(4):
            logits = self.logits(shift, obs, person, geometry, rep, mask)[:, i].masked_fill(~mask[:, i].bool(), -1e10)
            dist = Categorical(logits=logits)
            action = logits.argmax(-1) if deterministic else dist.sample()
            actions.append(action); logs.append(dist.log_prob(action))
            if i < 3: shift[:, i + 1, 1:] = F.one_hot(action, 585).to(obs.dtype)
        return torch.stack(actions, 1), torch.stack(logs, 1), values, hidden
