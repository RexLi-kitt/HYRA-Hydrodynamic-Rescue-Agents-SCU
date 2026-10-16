"""Exploratory residual-to-decision audit, isolated from the QMIX pipeline."""
from bisect import bisect_right
from collections import deque
import numpy as np

from .data import observation
from .diagnostics import score_fixed_order
from .envs import MOVES
from .lead import MONITORS, available_history

METHODS = ('raw_self', 'raw_full', 'recent', 'physical', 'uniform', 'wrong')
OFFSETS = np.arange(0, 51, 2, dtype=float)
SUMMARY_HORIZONS = (6, 12, 20)


def issued_curve(event, now, offsets=OFFSETS):
    """Interpolate within ONE already published version; never extrapolate."""
    ids = np.flatnonzero(event['release_s'] <= now)
    if not len(ids):
        raise ValueError('No published version')
    f = int(ids[-1]); valid = event['forecast_valid_s'][f]
    query = now + np.asarray(offsets)
    if query.min() < valid[0] or query.max() > valid[-1]:
        raise ValueError('Outside published horizon')
    depth = event['forecast_depth_m'][f]
    values = np.column_stack([np.interp(query, valid, depth[:, r])
                              for r in range(depth.shape[1])])
    return f, values


def scheduled_records(event, revisit, delay=0, phase=0):
    """Synthetic fixed-monitor sampling; phase is a fraction of a revisit.

    New contract: latest actual issue <= valid-20; actual lead is logged.
    There is no interpolated issue and no new hydrodynamic simulation.
    """
    if revisit <= 0 or delay < 0 or not 0 <= phase < 1:
        raise ValueError('Invalid schedule')
    records = {r: [] for r in MONITORS}
    for t in np.arange(20 + phase*revisit, 300 + 1e-9, revisit):
        ids = np.flatnonzero(event['release_s'] <= t-20)
        for region in MONITORS:
            value = float(np.interp(t, event['time_s'], event['truth_depth_m'][:, region]))
            row = dict(region_id=region, valid_time_s=float(t),
                       arrival_time_s=float(t+delay), observed_depth_m=value,
                       residual_m=None, matched_forecast_id=None,
                       prior_release_s=None, prior_depth_m=None, actual_lead_s=None)
            if len(ids):
                f = int(ids[-1]); valid = event['forecast_valid_s'][f]
                if valid[0] <= t <= valid[-1]:
                    prior = float(np.interp(t, valid, event['forecast_depth_m'][f, :, region]))
                    row.update(matched_forecast_id=f, prior_release_s=float(event['release_s'][f]),
                               prior_depth_m=prior, residual_m=value-prior,
                               actual_lead_s=float(t-event['release_s'][f]))
            records[region].append(row)
    return records


def binned_history(records, region, now):
    """Six fixed age bins spanning 60 s, newest causal record in each bin."""
    bins = [None]*6
    for row in records.get(region, []):
        age = now-row['valid_time_s']
        if not (0 <= age < 60 and row['arrival_time_s'] <= now
                and row.get('residual_m') is not None):
            continue
        slot = min(int(age//10), 5)
        if bins[slot] is None or row['valid_time_s'] > bins[slot]['valid_time_s']:
            bins[slot] = row
    return bins


def normalized_summary(histories, links, now, tau, kernel):
    values, weights, all_ages = [], [], []
    for link, history in zip(links, histories):
        h = [r for r in history if r is not None]
        if not h:
            continue
        age = np.array([now-r['valid_time_s'] for r in h])
        residual = np.array([r['residual_m'] for r in h])
        center = max(link['travel_s']-tau, 0)
        if kernel == 'recent': center = 0
        if kernel == 'wrong': center = (center+20) % 40
        k = np.ones(len(h)) if kernel == 'uniform' else np.exp(-.5*((age-center)/10)**2)
        values.append(float(k@residual/k.sum())); weights.append(link['weight'])
        all_ages.extend(age)
    if not values:
        return [0., 0., 0.]
    return [float(np.average(values, weights=weights)), len(all_ages)/18,
            float(np.mean(all_ages))/60]


def feature_groups(event, records, graph, region, now, forecast=None):
    links = graph[str(region)]
    own = binned_history(records, region, now)
    upstream = [binned_history(records, l['source'], now) for l in links]
    raw = np.zeros(96)
    for j, history in enumerate([own]+upstream):
        for b, row in enumerate(history):
            if row is not None:
                raw[j*24+b*4:j*24+b*4+4] = [row['observed_depth_m'], row['prior_depth_m'],
                                                           (now-row['valid_time_s'])/60, 1]
    if forecast is None: _, forecast = issued_curve(event, now)
    xy = event['xy_m'][region]/100
    base = np.r_[raw, forecast[:, region], now/300, xy, np.zeros(9)]
    out = {m: base.copy() for m in METHODS}
    out['raw_self'][24:96] = 0
    for method in METHODS[2:]:
        out[method][-9:] = np.concatenate([normalized_summary(upstream, links, now, tau, method)
                                           for tau in SUMMARY_HORIZONS])
    return out


def shortest_paths(blocked, origin, n=10):
    paths = {int(origin): [int(origin)]}; queue = deque([int(origin)])
    while queue:
        r = queue.popleft(); x, y = r % n, r // n
        for dx, dy in MOVES:
            xx, yy = x+int(dx), y+int(dy)
            if not (0 <= xx < n and 0 <= yy < n): continue
            rr = yy*n+xx
            if blocked[rr] or rr in paths: continue
            if dx and dy and (blocked[y*n+xx] or blocked[yy*n+x]): continue
            paths[rr] = paths[r]+[rr]; queue.append(rr)
    return paths


def candidate_orders(paths, origin, targets):
    a, b = targets
    return [([a, b], [len(paths[origin][a])-1, len(paths[a][b])-1]),
            ([b, a], [len(paths[origin][b])-1, len(paths[b][a])-1])]


def max_duration(orders):
    return max(sum(max(d, 1) for d in ds)*6+24 for _, ds in orders)


def order_scores(times, depths, orders, now):
    return [score_fixed_order(times, depths, order, ds, now, 6, 12, .8)
            for order, ds in orders]


def choose(scores):
    """Tie-breaking is independent of labels and reference truth."""
    return int(np.argmax([s['service_yield'] for s in scores]))


def prefix_checkpoints(paths, origin, targets, chosen, now):
    """States on the baseline prefix, before first service commitment.

    This is a local alternative-order comparison, not a new closed-loop policy.
    """
    first = targets[chosen]
    return [(now+6*k, node) for k, node in enumerate(paths[origin][first][:-1])]


def record_evidence(records, graph, region, now):
    rows = []
    for link in graph[str(region)]:
        h = [r for r in binned_history(records, link['source'], now) if r is not None]
        latest = max(h, key=lambda r: r['valid_time_s']) if h else None
        rows.append(dict(source=link['source'], travel_s=link['travel_s'],
            valid_s=latest['valid_time_s'] if latest else None,
            arrival_s=latest['arrival_time_s'] if latest else None,
            age_s=now-latest['valid_time_s'] if latest else None,
            actual_lead_s=latest['actual_lead_s'] if latest else None,
            residual_m=latest['residual_m'] if latest else None,
            prior_release_s=latest['prior_release_s'] if latest else None,
            nominal_remaining_s=link['travel_s']-(now-latest['valid_time_s']) if latest else None))
    return rows


def fit_reader(x, y, alpha):
    mean = x.mean(axis=0); std = x.std(axis=0); std[std < 1e-8] = 1
    z = np.c_[np.ones(len(x)), (x-mean)/std]
    penalty = np.eye(z.shape[1])*alpha; penalty[0, 0] = 0
    weights = np.linalg.solve(z.T@z+penalty, z.T@y)
    return dict(mean=mean, std=std, weights=weights, alpha=alpha)


def predict_reader(model, x):
    return np.c_[np.ones(len(x)), (x-model['mean'])/model['std']]@model['weights']
