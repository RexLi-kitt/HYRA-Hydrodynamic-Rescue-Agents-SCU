"""Causal task selection and offline opportunity bounds for alignment calibration.

No fitted predictor, detector, or online use of reference labels is implemented.
"""
from __future__ import annotations

import numpy as np

from .decision_value import issued_curve
from .local_service import score_orders, choose_order
from .lead import TARGETS, SOURCES, XY


def interpolate_checked(times, depths, query):
    times = np.asarray(times); query = np.asarray(query)
    if query.min() < times[0] or query.max() > times[-1]:
        raise ValueError('Requested time is outside saved trajectory/version')
    return np.column_stack([np.interp(query.ravel(), times, depths[:, r])
                            for r in range(depths.shape[1])]).reshape(*query.shape, depths.shape[1])


def make_cases(layouts, anchor, offsets=(0, 6, 12)):
    return [dict(c, now_s=int(anchor+offset)) for c in layouts for offset in offsets]


def curve_rows(cases, curve_at):
    times = np.array(sorted({c['now_s'] for c in cases}))
    return (np.concatenate([curve_at(t)[:, TARGETS].T for t in times]),
            np.repeat(times, len(TARGETS)), np.tile(TARGETS, len(times)))


def anchor_selection(event, layouts, anchor, band=.05):
    """Only the named issue is read; reference and later issues are irrelevant."""
    cases = make_cases(layouts, anchor)
    ids = np.flatnonzero(event['release_s'] == anchor)
    if len(ids) != 1: raise ValueError('Anchor is not an actual released version')
    index = int(ids[0]); valid = event['forecast_valid_s'][index]
    curves, times, regions = curve_rows(cases, lambda t: interpolate_checked(
        valid, event['forecast_depth_m'][index], t+np.arange(0, 49, 2)))
    scores = score_orders(curves[None], cases, times, regions)
    critical = np.min(abs(.8-scores['maxima'][0]), axis=(1, 2)) <= band
    assert max(c['now_s']-anchor+sum(max(d, 1)*6 for d in order[1])+24
               for c in cases for order in c['orders']) <= 60
    return cases, critical, scores


def evaluate_trajectory(event, depth, cases):
    curves, times, regions = curve_rows(cases, lambda t: interpolate_checked(
        event['time_s'], depth, t+np.arange(0, 51, 2)))
    scores = score_orders(curves[None], cases, times, regions)
    return {key: value[0] for key, value in scores.items()}


def nominal_actions(event, cases):
    curves, times, regions = curve_rows(cases, lambda t: issued_curve(event, t)[1])
    scores = score_orders(curves[None], cases, times, regions)
    return choose_order(scores, cases)[0], {k: v[0] for k, v in scores.items()}


def sampled_records(event, base, perturbed, phase, noise, replicate, seed_identity):
    """Exactly six upstream channels. Paired events share noise, no local sensor."""
    valid = np.arange(20+5*phase, 300+1e-9, 5)
    ids = np.searchsorted(event['release_s'], valid-20, side='right')-1
    if np.min(ids) < 0: raise ValueError('No matching historical version')
    prior = np.array([interpolate_checked(event['forecast_valid_s'][i],
        event['forecast_depth_m'][i], np.array([s]))[0, SOURCES] for i, s in zip(ids, valid)])
    e0 = interpolate_checked(event['time_s'], base, valid)[:, SOURCES]
    e1 = interpolate_checked(event['time_s'], perturbed, valid)[:, SOURCES]
    rng = np.random.default_rng(np.random.SeedSequence([2026101517, 61, *seed_identity, replicate]))
    epsilon = noise*rng.standard_normal(e0.shape)
    issue = event['release_s'][ids]
    assert np.all(issue <= valid-20) and np.all((valid-issue >= 20) & (valid-issue < 30))
    return dict(valid_s=valid, prior_release_s=issue, prior_m=prior,
                E0_observed_m=e0+epsilon, E1_observed_m=e1+epsilon,
                paired_reference_response_m=e1-e0, actual_lead_s=valid-issue)


def causal_bin_ids(valid, now, delay):
    age = now-valid
    available = (valid+delay <= now) & (age >= 0) & (age < 60)
    return np.array([np.flatnonzero(available & (age >= b*10) & (age < (b+1)*10))[-1]
                     if np.any(available & (age >= b*10) & (age < (b+1)*10)) else -1
                     for b in range(6)], dtype=int)


def marker(times, response, start):
    mask = times >= start
    peak = float(np.max(abs(response[mask])))
    if peak < .001: return None
    found = np.flatnonzero(mask & (abs(response) >= .1*peak))
    return float(times[found[0]]) if len(found) else None


def propagation_edges(times, nominal_delta, truth_delta, family, anchor, center):
    rows = []
    for target in TARGETS:
        if (XY[target, 1] < 50) != (center == 25): continue
        for source in SOURCES:
            if (XY[source, 1] < 50) != (center == 25): continue
            sm = marker(times, nominal_delta[:, source], anchor)
            tm = marker(times, nominal_delta[:, target], anchor)
            ts = marker(times, truth_delta[:, source], anchor)
            tt = marker(times, truth_delta[:, target], anchor)
            rows.append(dict(family=family, anchor_s=anchor, center_y_m=center,
                source=source, target=target, nominal_source_marker_s=sm,
                nominal_target_marker_s=tm, signed_nominal_lag_s=None if sm is None or tm is None else tm-sm,
                kernel_lag_s=None if sm is None or tm is None else max(tm-sm, 0),
                reference_source_marker_s=ts, reference_target_marker_s=tt,
                nominal_target_peak_m=float(np.max(abs(nominal_delta[:, target]))),
                reference_target_peak_m=float(np.max(abs(truth_delta[:, target])))))
    return rows


def kernel_distance(ages, lag, horizon):
    if len(ages) == 0: return None
    p = np.exp(-.5*((ages-max(lag-horizon, 0))/10)**2)
    r = np.exp(-.5*(ages/10)**2)
    return float(np.sum(abs(p/p.sum()-r/r.sum())))


def gained_targets(case, truth_scores, row, baseline):
    counts = truth_scores['count'][row]
    best = int(np.argmax(counts))
    if counts[best] <= counts[baseline]: return []
    successes = []
    for order in (baseline, best):
        successes.append({region for k, region in enumerate(case['orders'][order][0])
                          if truth_scores['maxima'][row, order, k] <= .8})
    return sorted(successes[1]-successes[0])


def paired_response_available(records, now, delay, targets):
    """Offline necessary opportunity bound, NOT an operational detector.

    Requires a stored (binned) record with a >=1mm paired noiseless response
    in at least one same-bank upstream channel for a gained target.
    """
    ids = causal_bin_ids(records['valid_s'], now, delay)
    ids = ids[ids >= 0]
    if not len(ids): return False
    channels = [i for i, source in enumerate(SOURCES)
                if any((XY[source, 1] < 50) == (XY[t, 1] < 50) for t in targets)]
    return bool(channels and np.any(abs(records['paired_reference_response_m'][np.ix_(ids, channels)]) >= .001))
