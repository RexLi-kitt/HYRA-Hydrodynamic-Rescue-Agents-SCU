"""Paper-informed CNN/action/time/GRU and training-only surprise enhancement.

Known adaptation choices: pose projection, min/max normalization over VALID
batch-agent rows, fixed all-ones unit direction, stopped surprise gradients,
clean targets and clean execution. No claim of author-code reproduction.
"""
import math
from types import SimpleNamespace

from .runtime import enable_existing_packages
enable_existing_packages()
import torch
from torch import nn
from torch.nn import functional as F

from .vendor.pymarl_qmix import QMixer


class SearchAgent(nn.Module):
    def __init__(self, image_shape=(16, 7, 7), pose_dim=6, n_actions=4, embed=32, hidden=64):
        super().__init__()
        if embed < 2 or embed % 2 or n_actions < 2 or hidden < 1:
            raise ValueError('Positive dimensions and an even embedding width are required')
        self.n_actions, self.embed, self.hidden_dim = n_actions, embed, hidden
        self.image_shape = tuple(image_shape)
        self.conv = nn.Sequential(nn.Conv2d(image_shape[0], 16, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Flatten())
        self.image_embed = nn.Linear(16*image_shape[1]*image_shape[2], embed)
        self.pose_embed = nn.Linear(pose_dim, embed)
        self.action_embed = nn.Linear(n_actions, embed)
        self.gru = nn.GRUCell(3*embed, hidden)
        self.output = nn.Linear(hidden, n_actions)
        self.register_buffer('direction', torch.ones(embed)/math.sqrt(embed))

    def initial_hidden(self, count, device=None):
        return torch.zeros(count, self.hidden_dim, device=device or self.direction.device)

    def encode(self, images, poses):
        return F.relu(self.image_embed(self.conv(images))+self.pose_embed(poses))

    def time_embedding(self, step, count, device):
        frequency = torch.exp(torch.arange(0, self.embed, 2, device=device)*(-math.log(10000.)/self.embed))
        value = torch.as_tensor(step, dtype=frequency.dtype, device=device)*frequency
        return torch.stack((value.sin(), value.cos()), dim=-1).flatten()[None].expand(count, -1)

    def decode(self, latent, previous, step, hidden):
        joined = torch.cat((latent, self.action_embed(previous),
                            self.time_embedding(step, len(latent), latent.device)), dim=-1)
        state = self.gru(joined, hidden)
        return self.output(state), state

    def forward(self, images, poses, previous, step, hidden, enhance=False, valid=None):
        latent = self.encode(images, poses)
        base_latent = latent
        surprise = latent.new_zeros(len(latent))
        if enhance:
            if not self.training or not torch.is_grad_enabled():
                raise RuntimeError('Surprise enhancement is training-only')
            virtual, _ = self.decode(latent, previous, step, hidden)
            gradient = torch.autograd.grad(virtual.max(dim=-1).values.sum(), latent,
                                           retain_graph=True, create_graph=False)[0]
            norm = gradient.detach().norm(dim=-1)
            mask = torch.ones(len(latent), dtype=torch.bool, device=latent.device) if valid is None else valid.bool()
            if mask.shape != norm.shape:
                raise ValueError('Valid mask must match flattened batch-agent rows')
            if mask.any():
                low, high = norm[mask].min(), norm[mask].max()
                surprise = torch.where(mask, (norm-low)/(high-low).clamp_min(1e-12), 0.)
                latent = latent+surprise[:, None]*self.direction
        q, state = self.decode(latent, previous, step, hidden)
        # The auxiliary head must use THIS policy encoder, before surprise.
        return q, state, base_latent, surprise


def make_mixer(n_agents, state_dim, embed=32):
    return QMixer(SimpleNamespace(n_agents=n_agents, state_shape=state_dim,
                                 mixing_embed_dim=embed, hypernet_layers=2, hypernet_embed=64))
