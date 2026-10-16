"""Diagnostics for finite numerical ensembles, without probabilistic claims."""
from __future__ import annotations

import numpy as np


def comparison_summary(old, new, threshold=.8):
    old, new = np.asarray(old, float), np.asarray(new, float)
    if old.shape != new.shape or old.ndim != 1 or not len(old):
        raise ValueError('Expected nonempty paired vectors')
    if not np.all(np.isfinite(old)) or not np.all(np.isfinite(new)):
        raise ValueError('Nonfinite depth')
    delta = abs(old-new)
    return dict(windows=len(old), label_changes=int(np.count_nonzero((old <= threshold) != (new <= threshold))),
                max_window_change_m=float(delta.max()), mean_window_change_m=float(delta.mean()))


def stable_controls(rows, threshold=.8):
    """Freeze both near-threshold and distant controls before new simulations."""
    groups = {}
    for index, row in enumerate(rows):
        if row['label_disagrees']:
            continue
        group = tuple(row[k] for k in ('family', 'anchor_s', 'physical_center_y_m', 'condition', 'target'))
        distance = min(abs(row['lower_m']-threshold), abs(row['upper_m']-threshold))
        groups.setdefault(group, []).append((distance, index))
    selected = set()
    for values in groups.values():
        selected.add(min(values)[1])
        selected.add(min(values, key=lambda v: (-v[0], v[1]))[1])
    return sorted(selected)


def bed_readout(centroids, bed, weights, query, params):
    analytic = -params['slope']*query[:, 0]+params['ridge']*np.exp(-((query[:, 1]-50)/12)**2)
    linear = np.sum(bed[weights['vertices']]*weights['weights'], axis=1)
    nearest = bed[weights['nearest']]
    return dict(linear_max_analytic_difference_m=float(np.max(abs(linear-analytic))),
                nearest_max_analytic_difference_m=float(np.max(abs(nearest-analytic))),
                nearest_max_distance_m=float(np.max(np.linalg.norm(centroids[weights['nearest']]-query, axis=1))),
                analytic_at_points_m=analytic.tolist(), linear_at_points_m=linear.tolist())
