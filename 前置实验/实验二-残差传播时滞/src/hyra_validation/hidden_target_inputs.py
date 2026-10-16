"""Causal six-source inputs for a future hidden-target comparison.

This module neither fits links nor reads reference labels. Physics and recency
use identical sources, bin selection, source normalization and support fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCES = (10, 20, 30, 60, 70, 80)
FIELDS = ('observation_m', 'prior_m', 'residual_m', 'age_s', 'actual_lead_s', 'present')


@dataclass(frozen=True)
class SourceRecord:
    record_id: int
    source: int
    valid_s: float
    arrival_s: float
    prior_release_s: float
    observation_m: float
    prior_m: float


def causal_history(records, now, history_s=60., bin_s=10., minimum_lead_s=20.):
    """Newest arrived record per source and fixed age bin, with actual references."""
    if history_s <= 0 or bin_s <= 0 or history_s % bin_s or not np.isfinite(now):
        raise ValueError('Invalid history grid')
    bins = int(history_s/bin_s)
    values = np.zeros((len(SOURCES), bins, len(FIELDS)))
    references = np.full((len(SOURCES), bins), -1, dtype=int)
    selected_valid = np.full((len(SOURCES), bins), -np.inf)
    ids = set()
    for row in records:
        if row.source not in SOURCES:
            raise ValueError('Record outside the common upstream network')
        if row.record_id < 0 or row.record_id in ids:
            raise ValueError('Record identities must be unique nonnegative integers')
        ids.add(row.record_id)
        if not np.all(np.isfinite([row.valid_s, row.arrival_s, row.prior_release_s, row.observation_m, row.prior_m])):
            raise ValueError('Nonfinite record')
        if row.arrival_s < row.valid_s or row.prior_release_s > row.valid_s-minimum_lead_s:
            raise ValueError('Invalid arrival or historical forecast version')
        age = now-row.valid_s
        if not (row.arrival_s <= now and 0 <= age < history_s):
            continue
        source, slot = SOURCES.index(row.source), int(age//bin_s)
        if row.valid_s <= selected_valid[source, slot]:
            continue
        selected_valid[source, slot] = row.valid_s
        references[source, slot] = row.record_id
        values[source, slot] = [row.observation_m, row.prior_m, row.observation_m-row.prior_m,
                                age, row.valid_s-row.prior_release_s, 1.]
    return dict(values=values, record_ids=references, source_ids=np.array(SOURCES), fields=FIELDS)


def aligned_summary(history, links, horizons, mode='physical', bandwidth_s=10.):
    """Target-specific summaries; links must come from legal training/nominal data.

Each link is (source, nonnegative_weight, nonnegative_lag_s). A source's kernel
is normalized within its available records before normalizing across sources.
Support and mean age are independent of kernel mode, preventing an accidental
P4/P5 input-budget difference. Empty support is represented explicitly.
"""
    if mode not in ('physical', 'recency', 'wrong_lag') or bandwidth_s <= 0:
        raise ValueError('Invalid kernel')
    values = np.asarray(history['values'])
    if values.shape != (6, 6, 6) or not np.array_equal(history['source_ids'], SOURCES):
        raise ValueError('Unexpected common history contract')
    if len({link[0] for link in links}) != len(links):
        raise ValueError('Duplicate link source')
    summaries, weight_rows = [], []
    for horizon in np.asarray(horizons, dtype=float):
        if not np.isfinite(horizon) or horizon < 0:
            raise ValueError('Invalid prediction horizon')
        weights = np.zeros((6, 6)); support = 0; ages = []
        for source, strength, lag in links:
            if source not in SOURCES or not np.isfinite(strength+lag) or strength < 0 or lag < 0:
                raise ValueError('Invalid physical link')
            if not strength: continue
            i = SOURCES.index(source); active = values[i, :, 5] > 0
            if not active.any(): continue
            age = values[i, active, 3]
            center = max(lag-horizon, 0.)
            if mode == 'recency': center = 0.
            if mode == 'wrong_lag': center = (center+20.) % 60.
            log_kernel = -.5*((age-center)/bandwidth_s)**2
            kernel = np.exp(log_kernel-log_kernel.max())
            weights[i, active] = strength*kernel/kernel.sum()
            support += int(active.sum()); ages.extend(age)
        total = weights.sum()
        if total: weights /= total
        summaries.append([float(np.sum(weights*values[:, :, 2])), support/36.,
                          float(np.mean(ages))/60 if ages else 0., float(total > 0)])
        weight_rows.append(weights)
    return dict(features=np.array(summaries), normalized_weights=np.array(weight_rows),
                record_ids=np.array(history['record_ids'], copy=True))
