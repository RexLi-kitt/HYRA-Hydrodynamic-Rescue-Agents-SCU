"""Local-service experiment: causal inputs, conventional updates and vector scoring.

No truth labels are read by candidate prediction or action selection functions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .decision_value import OFFSETS, issued_curve, shortest_paths, candidate_orders
from .lead import MONITORS, TARGETS


def load_design(root):
    root = Path(root)
    config = json.loads((root / 'configs/local_service_regret.design.v1.json').read_text(encoding='utf-8'))
    preview = json.loads((root / 'configs/local_service_regret.geometry.preview.v1.json').read_text(encoding='utf-8'))
    return config, preview


def task_table(preview):
    """Canonical case order; geometry and sampled times only."""
    rows = []
    for layout in preview['layouts']:
        blocked = np.zeros(100, bool)
        blocked[layout['blocked_regions']] = True
        origin = layout['origin']; targets = layout['targets']
        paths = {r: shortest_paths(blocked, r) for r in [origin, *targets]}
        orders = candidate_orders(paths, origin, targets)
        for now in layout['sampled_times_s']:
            rows.append(dict(layout=layout['layout_index'], now_s=now, origin=origin,
                             targets=targets, orders=orders))
    assert len(rows) == 128
    return rows


def schedules(config):
    for noise in config['observations']['noise_std_m']:
        repetitions = config['observations']['noisy_replicates'] if noise else 1
        for delay in config['observations']['delay_s']:
            for phase_index, phase in enumerate(config['observations']['phase_fractions']):
                for replicate in range(repetitions):
                    yield dict(noise_std_m=noise, delay_s=delay, phase=phase,
                               phase_index=phase_index, replicate=replicate,
                               weight=1 / repetitions)


def sensor_arrays(event, setting, seed_identity, master_seed):
    """One shared observation network, version matched with actual lead logged."""
    valid = np.arange(20 + 5 * setting['phase'], 300 + 1e-9, 5)
    regions = np.asarray(MONITORS)
    truth = np.column_stack([np.interp(valid, event['time_s'], event['truth_depth_m'][:, r]) for r in regions])
    rng = np.random.default_rng(np.random.SeedSequence(
        [master_seed, 5, *seed_identity, setting['phase_index'], setting['replicate']]))
    values = truth + setting['noise_std_m'] * rng.standard_normal(truth.shape)
    issue_index = np.searchsorted(event['release_s'], valid - 20, side='right') - 1
    prior = np.zeros_like(values); lead = np.zeros(len(valid)); has_prior = issue_index >= 0
    for i in np.flatnonzero(has_prior):
        f = issue_index[i]; times = event['forecast_valid_s'][f]
        if not times[0] <= valid[i] <= times[-1]:
            has_prior[i] = False
            continue
        prior[i] = [np.interp(valid[i], times, event['forecast_depth_m'][f, :, r]) for r in regions]
        lead[i] = valid[i] - event['release_s'][f]
    return dict(valid_s=valid, arrival_s=valid + setting['delay_s'], values_m=values,
                prior_m=prior, residual_m=values-prior, has_prior=has_prior,
                actual_lead_s=lead, issue_index=issue_index, regions=regions)


def event_foundation(event):
    """Immutable per-event curves/labels, shared across sensor conditions."""
    times = np.arange(24, 241, 6)
    row_times = np.repeat(times, len(TARGETS))
    row_regions = np.tile(TARGETS, len(times))
    n = len(row_times)
    base = np.empty((n, len(OFFSETS)))
    release_age = np.empty(n)
    for i, now in enumerate(times):
        f, curve = issued_curve(event, now)
        sl = slice(i*len(TARGETS), (i+1)*len(TARGETS))
        base[sl] = curve[:, TARGETS].T
        release_age[sl] = now-event['release_s'][f]
    truth_future = np.empty_like(base)
    for j, region in enumerate(TARGETS):
        truth_future[j::len(TARGETS)] = np.interp(
            (times[:, None]+OFFSETS).ravel(), event['time_s'], event['truth_depth_m'][:, region]).reshape(len(times), -1)
    common = np.c_[base, row_times/300, event['xy_m'][row_regions]/100, release_age/10]
    return dict(base=base, truth_future=truth_future, labels=truth_future-base,
                row_times=row_times, row_regions=row_regions, common=common)


def build_batch(event, setting, seed_identity, master_seed, foundation=None):
    """Rows are time x target. Labels are separated from all input arrays."""
    foundation = event_foundation(event) if foundation is None else foundation
    row_times = foundation['row_times']; row_regions = foundation['row_regions']; n = len(row_times)
    sensors = sensor_arrays(event, setting, seed_identity, master_seed)
    history = np.zeros((n, 6, 4))  # residual, age/60, actual lead/30, mask
    record_ids = np.full((n, 6), -1, int)
    source = np.array([list(MONITORS).index(int(r)) for r in row_regions])
    age = row_times[:, None]-sensors['valid_s'][None]
    available = ((age >= 0) & (age < 60) & (sensors['arrival_s'][None] <= row_times[:, None])
                 & sensors['has_prior'][None])
    all_indices = np.arange(len(sensors['valid_s']))
    for b in range(6):
        eligible = available & (age >= b*10) & (age < (b+1)*10)
        indices = np.where(eligible, all_indices, -1).max(axis=1)
        active = indices >= 0; rows = np.flatnonzero(active); jj = indices[active]
        record_ids[:, b] = indices
        history[rows, b, 0] = sensors['residual_m'][jj, source[active]]
        history[rows, b, 1] = (row_times[active]-sensors['valid_s'][jj])/60
        history[rows, b, 2] = sensors['actual_lead_s'][jj]/30
        history[rows, b, 3] = 1
    latest = np.zeros((n, 4))
    present = history[:, :, 3] > 0
    indices = present.argmax(axis=1)
    latest[:] = history[np.arange(n), indices]
    common = foundation['common']
    inputs = dict(forecast=common, latest=np.c_[common, latest],
                  history=np.c_[common, history.reshape(n, -1)])
    # Audit actual input references, not just candidate availability.
    used_rows, used_bins = np.nonzero(record_ids >= 0)
    ids = record_ids[used_rows, used_bins]
    assert np.all(sensors['arrival_s'][ids] <= row_times[used_rows])
    assert np.all(sensors['valid_s'][ids] <= row_times[used_rows])
    assert np.all(sensors['actual_lead_s'][ids] >= 20)
    checksum = hashlib.sha256()
    for key in ('valid_s', 'arrival_s', 'values_m', 'prior_m', 'issue_index'):
        checksum.update(np.ascontiguousarray(sensors[key]).tobytes())
    return dict(inputs=inputs, history=history, base=foundation['base'], labels=foundation['labels'], truth_future=foundation['truth_future'],
                row_times=row_times, row_regions=row_regions, sensors=sensors,
                record_ids=record_ids, observation_sha256=checksum.hexdigest())


class RidgeMoments:
    """Weighted streaming sufficient statistics; no test/validation fitting."""
    def __init__(self):
        self.weight = 0.
        self.sx = self.sy = self.sxx = self.sxy = None

    def add(self, x, y, weight=1.):
        if self.sx is None:
            self.sx = np.zeros(x.shape[1]); self.sy = np.zeros(y.shape[1])
            self.sxx = np.zeros((x.shape[1], x.shape[1])); self.sxy = np.zeros((x.shape[1], y.shape[1]))
        self.weight += len(x)*weight
        self.sx += weight*x.sum(axis=0); self.sy += weight*y.sum(axis=0)
        self.sxx += weight*(x.T@x); self.sxy += weight*(x.T@y)

    def fit(self, alpha):
        mean = self.sx/self.weight; ym = self.sy/self.weight
        covariance = self.sxx-self.weight*np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(covariance)/self.weight, 0))
        std[std < 1e-8] = 1
        ztz = covariance/std[:, None]/std[None, :]
        zty = (self.sxy-self.weight*np.outer(mean, ym))/std[:, None]
        weights = np.linalg.solve(ztz+alpha*np.eye(len(mean)), zty)
        return dict(mean=mean, std=std, intercept=ym, weights=weights, alpha=float(alpha),
                    training_weight=self.weight,
                    effective_rank=int(np.linalg.matrix_rank(ztz)))


def learned_predict(model, x):
    return model['intercept'] + ((x-model['mean'])/model['std'])@model['weights']


def conventional_predictions(history, config, wanted=None):
    """Replay exactly the same causal bins, oldest first, for all 26 updates."""
    n = len(history); h = history[:, ::-1]
    residual = h[:, :, 0]; valid_time = -60*h[:, :, 1]; masks = h[:, :, 3] > 0
    result = {'nominal': np.zeros((n, len(OFFSETS)))}
    latest = np.zeros(n)
    for k in range(6):
        latest = np.where(masks[:, k], residual[:, k], latest)
    result['persistence'] = np.repeat(latest[:, None], len(OFFSETS), axis=1)
    mc = config['models']
    for half in mc['ewma_half_life_s']:
        if wanted is not None and f'ewma_{half}' not in wanted: continue
        value = np.zeros(n); last = np.zeros(n); initialized = np.zeros(n, bool)
        for k in range(6):
            active = masks[:, k]; first = active & ~initialized
            alpha = 1-2**(-np.maximum(valid_time[:, k]-last, 0)/half)
            value = np.where(first, residual[:, k], np.where(active, value+alpha*(residual[:, k]-value), value))
            last = np.where(active, valid_time[:, k], last); initialized |= active
        result[f'ewma_{half}'] = np.repeat(value[:, None], len(OFFSETS), axis=1)
    for kind, q_values in [('rw', mc['kf_rw_process_variance_rate_m2_per_s']),
                            ('cv', mc['kf_cv_process_spectral_density_m2_per_s3'])]:
        for q in q_values:
            for r in mc['kf_measurement_std_m']:
                name = f'kf_{kind}_q{q:g}_r{r:g}'
                if wanted is not None and name not in wanted: continue
                x = np.zeros((n, 2)); p = np.zeros((n, 2, 2))
                p[:, 0, 0] = mc['kf_initial_bias_std_m']**2
                p[:, 1, 1] = mc['kf_initial_bias_rate_std_m_per_s']**2 if kind == 'cv' else 0
                last = np.zeros(n); initialized = np.zeros(n, bool)
                for k in range(6):
                    active = masks[:, k]
                    delta = np.where(active & initialized, valid_time[:, k]-last, 0)
                    assert np.all(delta >= 0)
                    if kind == 'cv':
                        f = np.broadcast_to(np.eye(2), (n, 2, 2)).copy(); f[:, 0, 1] = delta
                        x = np.einsum('nij,nj->ni', f, x)
                        p = f@p@np.transpose(f, (0, 2, 1))
                        p[:, 0, 0] += q*delta**3/3; p[:, 0, 1] += q*delta**2/2
                        p[:, 1, 0] += q*delta**2/2; p[:, 1, 1] += q*delta
                    else:
                        p[:, 0, 0] += q*delta
                    gain = p[:, :, 0]/(p[:, 0, 0]+r*r)[:, None]
                    innovation = residual[:, k]-x[:, 0]
                    new_x = x+gain*innovation[:, None]
                    # Joseph form preserves covariance symmetry/PSD.
                    ikh = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
                    ikh[:, :, 0] -= gain
                    new_p = ikh@p@np.transpose(ikh, (0, 2, 1)) + r*r*gain[:, :, None]*gain[:, None, :]
                    x = np.where(active[:, None], new_x, x)
                    p = np.where(active[:, None, None], new_p, p)
                    last = np.where(active, valid_time[:, k], last); initialized |= active
                value = x[:, 0, None] + (OFFSETS[None, :]-last[:, None])*x[:, 1, None]
                result[name] = value
    if wanted is None: assert len(result) == mc['ordinary_update_candidate_count']
    return result if wanted is None else {k: v for k, v in result.items() if k in wanted}


def trend_predictions(history, windows=(20, 40, 60)):
    x = -60*history[:, :, 1]; y = history[:, :, 0]
    out = {}
    for window in windows:
        mask = (history[:, :, 3] > 0) & (-x <= window)
        count = mask.sum(axis=1); safe = np.maximum(count, 1)
        xm = (x*mask).sum(axis=1)/safe; ym = (y*mask).sum(axis=1)/safe
        centered = x-xm[:, None]
        denom = (centered**2*mask).sum(axis=1)
        slope = np.divide((centered*(y-ym[:, None])*mask).sum(axis=1), denom,
                          out=np.zeros(len(x)), where=denom > 1e-12)
        out[f'trend_{window}'] = (ym-slope*xm)[:, None] + slope[:, None]*OFFSETS
    return out


def predict_candidates(batch, models, config, wanted=None):
    if wanted is not None:
        predictions = conventional_predictions(batch['history'], config, wanted)
        for name in wanted:
            if name in models:
                kind = name.split('_ridge_')[0]
                predictions[name] = learned_predict(models[name], batch['inputs'][kind])
        trends = [int(name.split('_')[1]) for name in wanted if name.startswith('trend_')]
        if trends: predictions.update(trend_predictions(batch['history'], trends))
        assert all(name in predictions for name in wanted)
        return list(wanted), np.stack([predictions[name] for name in wanted]), None
    ordinary = conventional_predictions(batch['history'], config)
    predictions = dict(ordinary)
    groups = {'A0': ['nominal'], 'A2': list(ordinary), 'A1': [], 'A3': [], 'A5': []}
    for kind in ('forecast', 'latest', 'history'):
        for alpha in config['models']['ridge_alpha']:
            name = f'{kind}_ridge_{alpha:g}'
            predictions[name] = learned_predict(models[name], batch['inputs'][kind])
            if kind == 'forecast': groups['A1'].append(name)
            else: groups['A3'].append(name)
            if kind == 'history': groups['A5'].append(name)
    trends = trend_predictions(batch['history'], config['models']['ols_window_s'])
    predictions.update(trends); groups['A3'] += list(trends)+['persistence']
    groups['A4'] = list(groups['A3'])
    assert groups['A3'] == groups['A4'] and len(groups['A3']) == 10
    names = list(predictions)
    errors = np.stack([predictions[name] for name in names])
    assert np.isfinite(errors).all()
    return names, errors, groups


def interval_max(curves, begin_s, duration_s):
    """Exact maximum for the shared piecewise-linear 2 s grid; no extrapolation."""
    starts = np.asarray(begin_s, int)//2
    assert np.all(np.asarray(begin_s) % 2 == 0)
    indices = starts[..., None] + np.arange(duration_s//2+1)
    if np.min(indices) < 0 or np.max(indices) >= curves.shape[-1]:
        raise ValueError('Service interval outside released prediction')
    return np.take_along_axis(curves, indices, axis=-1).max(axis=-1)


def score_orders(curves, cases, row_times, row_regions):
    """Vectorized reference-equivalent scoring. Shape curves=(models,rows,26)."""
    lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(row_times, row_regions))}
    counts = []; margins = []; starts = []; maxima = []
    model_count = curves.shape[0]
    for order_index in range(2):
        indices = np.array([[lookup[c['now_s'], target] for target in c['orders'][order_index][0]] for c in cases])
        travel = np.array([[max(d, 1)*6 for d in c['orders'][order_index][1]] for c in cases])
        first = curves[:, indices[:, 0]]; second = curves[:, indices[:, 1]]
        start1 = np.broadcast_to(travel[:, 0], (model_count, len(cases)))
        max1_six = interval_max(first, start1, 6)
        max1_full = interval_max(first, start1, 12)
        start2 = start1 + np.where(max1_six <= .8, 12, 6)+travel[:, 1]
        max2_full = interval_max(second, start2, 12)
        counts.append((max1_full <= .8).astype(np.int8)+(max2_full <= .8).astype(np.int8))
        margins.append(np.minimum(.8-max1_full, .8-max2_full))
        starts.append(np.stack([start1, start2], axis=-1))
        maxima.append(np.stack([max1_full, max2_full], axis=-1))
    return dict(count=np.stack(counts, axis=-1), margin=np.stack(margins, axis=-1),
                starts=np.stack(starts, axis=2), maxima=np.stack(maxima, axis=2))


def choose_order(scores, cases):
    counts = scores['count']; margin = scores['margin']
    prefer1 = np.array([c['orders'][1][0] < c['orders'][0][0] for c in cases])
    same_count = counts[..., 1] == counts[..., 0]
    same_margin = np.abs(margin[..., 1]-margin[..., 0]) <= 1e-12
    return ((counts[..., 1] > counts[..., 0]) |
            (same_count & ((margin[..., 1] > margin[..., 0]+1e-12) |
                           (same_margin & prefer1)))).astype(np.int8)


def evaluate_batch(batch, errors, cases):
    curves = np.maximum(0, batch['base'][None] + errors)
    scores = score_orders(curves, cases, batch['row_times'], batch['row_regions'])
    chosen = choose_order(scores, cases)
    truth_curves = batch.get('truth_future', batch['base']+batch['labels'])[None]
    truth = score_orders(truth_curves, cases, batch['row_times'], batch['row_regions'])
    true_yields = truth['count'][0]
    actual = np.take_along_axis(np.broadcast_to(true_yields, (*chosen.shape, 2)), chosen[..., None], axis=-1)[..., 0]
    oracle = true_yields.max(axis=-1)
    # Pointwise MAE over the SAME case/target weighting used for service selection.
    lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(batch['row_times'], batch['row_regions']))}
    rows = [lookup[c['now_s'], r] for c in cases for r in c['targets']]
    mae = np.mean(abs(errors[:, rows]-batch['labels'][rows]), axis=(1, 2))
    nominal_maxima = scores['maxima'][0]
    critical = np.min(abs(.8-nominal_maxima), axis=(1, 2)) <= .05
    return dict(chosen=chosen, actual=actual, regret=oracle[None]-actual, mae=mae,
                true_yields=true_yields, oracle=oracle, order_sensitive=true_yields[:, 0] != true_yields[:, 1],
                nominal_critical=critical, predicted_counts=scores['count'],
                predicted_maxima=scores['maxima'], predicted_starts=scores['starts'],
                truth_maxima=truth['maxima'][0], truth_starts=truth['starts'][0])


def select_models(candidate_names, groups, validation):
    """One candidate per group across all conditions; no test parameter exists."""
    score = np.asarray(validation['yield']); mae = np.asarray(validation['mae'])
    selected = {}
    for group in ('A0', 'A1', 'A2', 'A3', 'A4', 'A5'):
        allowed = [candidate_names.index(name) for name in groups[group]]
        # Python's stable min retains the prespecified candidate order on ties.
        index = min(allowed, key=lambda i: mae[i] if group == 'A3' else -score[i])
        selected[group] = dict(candidate=candidate_names[index], candidate_index=index,
                               validation_yield=float(score[index]), validation_mae_m=float(mae[index]))
    return selected
