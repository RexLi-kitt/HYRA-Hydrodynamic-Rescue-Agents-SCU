"""One parameter-matched, variable-grid backbone for all five information rules.

All raw shared history reaches a cell GRU and the common region/global bank.
H has unrestricted learned attention over that bank. E/P only reorganize the
message path with frozen sparse links, lags and (for P) explicit expiry/decay.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
from .model_features import HISTORY_CHANNELS

METHODS = ("H", "E", "P", "P_wrong_links", "P_uniform_time")


def regions_for_positions(positions, side):
    positions = np.asarray(positions)
    col = np.clip((positions[:, 0] / (side * 10.) * 4).astype(int), 0, 3)
    row = np.clip((positions[:, 1] / (side * 10.) * 4).astype(int), 0, 3)
    return row * 4 + col


def wrong_link_permutation(seed=2026101803):
    # Sattolo cycle: no source remains itself; source permutation preserves
    # every target's degree and the multiset of source degrees.
    rng = np.random.default_rng(seed)
    p = np.arange(16)
    for index in range(15, 0, -1):
        other = int(rng.integers(0, index))
        p[index], p[other] = p[other], p[index]
    return p


def graph_arrays(method, graph):
    if method not in METHODS:
        raise ValueError(method)
    if method == "H":
        return np.ones((16, 16), bool), np.zeros((16, 16), int), np.zeros((16, 16), np.float32)
    prefix = "E" if method == "E" else "P"
    mask = np.asarray(graph[prefix + "_mask"], bool)
    lag = np.asarray(graph[prefix + "_lag_steps"], int)
    score = np.asarray(graph[prefix + "_score"], np.float32)
    if mask.shape != (16, 16) or lag.shape != mask.shape or score.shape != mask.shape or np.any((lag < 0) | (lag > 12)):
        raise ValueError("Invalid frozen graph")
    if method == "P_wrong_links":
        perm = wrong_link_permutation()
        mask, lag, score = mask[:, perm], lag[:, perm], score[:, perm]
    return mask, lag, score


class SharedForecastBackbone(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.hidden = hidden
        self.history = nn.GRU(HISTORY_CHANNELS, hidden, batch_first=True)
        self.geometry = nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.pair = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.global_projection = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU())
        self.decoder = nn.Sequential(nn.Linear(3 * hidden + 30, 96), nn.SiLU(), nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, 13))

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def encode(self, history, static, region_ids):
        if history.ndim != 3 or history.shape[1:] != (13, HISTORY_CHANNELS) or static.shape != (len(history), 4):
            raise ValueError("Expected shared full raw history and 4 public static features")
        hidden, _ = self.history(history)
        node = hidden[:, -1] + self.geometry(static)
        observed = history[:, :, 3] > .5
        available = observed.cumsum(dim=1) > 0
        region_hidden = torch.zeros(16, 13, self.hidden, device=history.device, dtype=history.dtype)
        region_count = torch.zeros(16, 13, device=history.device, dtype=history.dtype)
        region_hidden.index_add_(0, region_ids, hidden * available.unsqueeze(-1))
        region_count.index_add_(0, region_ids, available.to(history.dtype))
        region_hidden = region_hidden / region_count.clamp_min(1).unsqueeze(-1)
        indices = torch.arange(13, device=history.device).unsqueeze(0).expand(len(history), -1)
        latest = torch.where(observed, indices, -1000).cummax(dim=1).values
        selected = latest.clamp_min(0)
        ages = history[:, :, 2].gather(1, selected) * 60
        region_age = torch.zeros(16, 13, device=history.device, dtype=history.dtype)
        region_age.index_add_(0, region_ids, ages * available)
        region_age /= region_count.clamp_min(1)
        region_static = torch.zeros(16, 4, device=history.device, dtype=history.dtype)
        region_sizes = torch.bincount(region_ids, minlength=16).to(history.dtype)
        region_static.index_add_(0, region_ids, static)
        region_static /= region_sizes.clamp_min(1).unsqueeze(-1)
        global_value = self.global_projection(torch.cat((node.mean(dim=0), node.amax(dim=0)), dim=0))
        residual = history[:, :, 0] - history[:, :, 1]
        last_residual = residual.gather(1, selected[:, -1:]).squeeze(1)
        last_age = ages[:, -1]
        has_data = available[:, -1]
        valid = observed.to(history.dtype)
        relative_t = -history[:, :, 2] * 60
        n = valid.sum(1)
        mean_t = (relative_t * valid).sum(1) / n.clamp_min(1)
        mean_r = (residual * valid).sum(1) / n.clamp_min(1)
        covariance = ((relative_t - mean_t[:, None]) * (residual - mean_r[:, None]) * valid).sum(1)
        variance = ((relative_t - mean_t[:, None]).square() * valid).sum(1)
        slope = torch.where(n >= 3, covariance / variance.clamp_min(1), 0.).clamp(-.005, .005)
        global_slope = slope.sum() / (n >= 3).sum().clamp_min(1)
        global_residual = ((last_residual + global_slope * last_age) * has_data).sum() / has_data.sum().clamp_min(1)
        local_weight = (1 - last_age / 60).clamp(0, 1) * has_data
        prior = local_weight * (last_residual + slope * last_age) + (1 - local_weight) * global_residual
        trend = local_weight * slope + (1 - local_weight) * global_slope
        return dict(node=node, regions=region_hidden, region_count=region_count, region_age=region_age,
                    region_static=region_static, region_ids=region_ids, global_value=global_value,
                    prior=prior, trend=trend, last_age=last_age, has_data=has_data)

    def decode(self, encoded, query_ids, nominal_future, future_mask, graph, method):
        device, dtype = nominal_future.device, nominal_future.dtype
        mask_np, lag_np, score_np = graph_arrays(method, graph)
        mask = torch.as_tensor(mask_np, dtype=torch.bool, device=device)
        lag = torch.as_tensor(lag_np, dtype=torch.long, device=device)
        score = torch.as_tensor(score_np, dtype=dtype, device=device)
        target_regions = encoded["region_ids"][query_ids]
        sources = torch.arange(16, device=device).expand(len(query_ids), -1)
        lags = lag[target_regions]
        cutoff = 12 - lags
        source_value = encoded["regions"][sources, cutoff]
        source_count = encoded["region_count"][sources, cutoff]
        source_age = encoded["region_age"][sources, cutoff]
        active = mask[target_regions] & (source_count > 0) & (source_age <= 60 + 1e-6)
        source_static = encoded["region_static"][sources]
        target_static = encoded["region_static"][target_regions].unsqueeze(1)
        delta = source_static - target_static
        pair = torch.stack((delta[:, :, 0], delta[:, :, 1], delta[:, :, 2], lags.to(dtype) / 12,
                            score[target_regions], source_age / 60), dim=-1)
        node = encoded["node"][query_ids]
        logits = (self.query(node).unsqueeze(1) * self.key(source_value)).sum(-1) / math.sqrt(self.hidden)
        logits += self.pair(pair).squeeze(-1)
        if method in ("P", "P_wrong_links"):
            logits -= (source_age - lags * 5).clamp_min(0) / 20.
        weights = torch.softmax(logits.masked_fill(~active, -1e4), dim=1) * active
        weights /= weights.sum(1, keepdim=True).clamp_min(1e-12)
        message = (weights.unsqueeze(-1) * self.value(source_value)).sum(1)
        prior = encoded["prior"][query_ids]
        trend = encoded["trend"][query_ids]
        extra = torch.stack((prior, trend * 60, encoded["last_age"][query_ids] / 60,
                             encoded["has_data"][query_ids].to(dtype)), dim=-1)
        features = torch.cat((node, message, encoded["global_value"].expand(len(query_ids), -1),
                              nominal_future, future_mask.to(dtype), extra), dim=-1)
        learned = .25 * torch.tanh(self.decoder(features))
        horizon = torch.arange(13, device=device, dtype=dtype) * 5
        prediction = nominal_future + prior[:, None] + trend[:, None] * horizon + learned
        return prediction.clamp_min(0.)

    def forward(self, history, static, region_ids, query_ids, nominal_future, future_mask, graph, method):
        return self.decode(self.encode(history, static, region_ids), query_ids, nominal_future, future_mask, graph, method)
