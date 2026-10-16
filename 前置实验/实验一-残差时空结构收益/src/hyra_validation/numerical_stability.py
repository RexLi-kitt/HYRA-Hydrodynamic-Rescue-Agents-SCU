"""Fixed-coordinate sampling and arbitrary-output-grid service diagnostics."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .lead import hydraulic_domain, XY, SOURCES


def boundary_footprint(domain):
    intervals = []
    for (cell, edge), tag in domain.boundary.items():
        if tag != 'pulse': continue
        x, y = domain.get_edge_midpoint_coordinate(cell, edge)
        assert abs(x) < 1e-10
        length = float(domain.edgelengths[cell, edge])
        intervals.append((float(y-length/2), float(y+length/2)))
    intervals.sort()
    if not intervals: return None
    assert all(abs(a[1]-b[0]) < 1e-8 for a, b in zip(intervals, intervals[1:]))
    return [intervals[0][0], intervals[-1][1]]


def fixed_domain(params, mesh, factor, center=None, anchor=210):
    domain = hydraulic_domain(params, mesh, factor=factor, pulse_amplitude=.1 if center is not None else 0,
                              pulse_start=anchor, pulse_center=center if center is not None else 20, plateau=20)
    if center is not None:
        np.testing.assert_allclose(boundary_footprint(domain), [center-10, center+10], atol=1e-8, rtol=0)
    return domain


def point_weights(centroids, xy=XY):
    triangulation = Delaunay(centroids)
    simplex = triangulation.find_simplex(xy)
    if np.any(simplex < 0): raise ValueError('Fixed point outside centroid interpolation hull')
    transform = triangulation.transform[simplex]
    first = np.einsum('ijk,ik->ij', transform[:, :2, :], xy-transform[:, 2, :])
    weights = np.c_[first, 1-first.sum(axis=1)]
    if weights.min() < -1e-9: raise ValueError('Nonconvex interpolation')
    weights = np.maximum(weights, 0); weights /= weights.sum(axis=1, keepdims=True)
    distance, nearest = cKDTree(centroids).query(xy)
    return dict(vertices=triangulation.simplices[simplex], weights=weights,
                nearest=nearest, nearest_distance_m=distance)


def read_points(stage, bed, weights):
    depth = np.maximum(stage-bed, 0)
    return dict(nearest=depth[:, weights['nearest']],
                linear=np.einsum('tpk,pk->tp', depth[:, weights['vertices']], weights['weights']))


def interval_maximum(times, values, begin, end):
    if not times[0] <= begin <= end <= times[-1]: raise ValueError('Interval outside trajectory')
    inside = values[(times >= begin) & (times <= end)]
    return max(float(np.interp(begin, times, values)), float(np.interp(end, times, values)),
               float(np.max(inside)) if len(inside) else -np.inf)


def scalar_order(times, depth, order, distances, now, deadline):
    clock = now; maxima = []; starts = []; succeeded = []
    for region, distance in zip(order, distances):
        begin = clock+max(distance, 1)*6
        if begin+12 > deadline: raise ValueError('Plan exceeds absolute task deadline')
        maximum = interval_maximum(times, depth[:, region], begin, begin+12)
        first = interval_maximum(times, depth[:, region], begin, begin+6)
        starts.append(begin); maxima.append(maximum)
        succeeded.append(int(region) if maximum <= .8 else None)
        clock = begin+(6 if first > .8 else 12)
    return dict(count=sum(s is not None for s in succeeded), maxima=maxima, starts=starts,
                succeeded=succeeded, margin=min(.8-m for m in maxima))


def score_cases(times, depth, cases, anchor):
    rows = [[scalar_order(times, depth, *order, c['now_s'], anchor+60) for order in c['orders']] for c in cases]
    return dict(count=np.array([[o['count'] for o in row] for row in rows]),
                maxima=np.array([[o['maxima'] for o in row] for row in rows]),
                starts=np.array([[o['starts'] for o in row] for row in rows]),
                succeeded=[[o['succeeded'] for o in row] for row in rows])


def forecast_plans(forecasts, cases, anchor):
    chosen = []; critical = []
    for c in cases:
        release = max(r for r in forecasts if r <= c['now_s'])
        t, depth = forecasts[release]
        scores = [scalar_order(t, depth, *order, c['now_s'], anchor+60) for order in c['orders']]
        fixed_t, fixed_depth = forecasts[anchor]
        fixed_scores = [scalar_order(fixed_t, fixed_depth, *order, c['now_s'], anchor+60) for order in c['orders']]
        critical.append(min(abs(.8-m) for s in fixed_scores for m in s['maxima']) <= .05)
        winner = 0
        if scores[1]['count'] > scores[0]['count']: winner = 1
        elif scores[1]['count'] == scores[0]['count']:
            if scores[1]['margin'] > scores[0]['margin']+1e-12: winner = 1
            elif abs(scores[1]['margin']-scores[0]['margin']) <= 1e-12 and c['orders'][1][0] < c['orders'][0][0]: winner = 1
        chosen.append(winner)
    return np.array(chosen, dtype=int), np.array(critical, dtype=bool)


def visible_response(times, base, pulse, now, phase, delay, gained):
    if not gained: return False
    valid = np.arange(20+5*phase, 300+1e-9, 5)
    age = now-valid; ids = []
    for b in range(6):
        eligible = np.flatnonzero((valid+delay <= now) & (age >= b*10) & (age < b*10+10))
        if len(eligible): ids.append(int(eligible[-1]))
    for source in SOURCES:
        if not any((source//10 < 5) == (target//10 < 5) for target in gained): continue
        delta = pulse[:, source]-base[:, source]
        if ids and np.max(abs(np.interp(valid[ids], times, delta))) >= .001: return True
    return False


def opportunity_flags(times, base, pulse, cases, anchor, chosen, scores):
    # Only two orders: positive difference fixes the alternative uniquely.
    gap = scores['count'][np.arange(len(cases)), 1-chosen]-scores['count'][np.arange(len(cases)), chosen]
    flags = np.zeros((len(cases), 4, 2), bool)
    for j, c in enumerate(cases):
        if gap[j] <= 0: continue
        gained = sorted(set(t for t in scores['succeeded'][j][1-chosen[j]] if t is not None)
                        - set(t for t in scores['succeeded'][j][chosen[j]] if t is not None))
        for p, phase in enumerate((0, .25, .5, .75)):
            for d, delay in enumerate((0, 10)):
                flags[j, p, d] = visible_response(times, base, pulse, c['now_s'], phase, delay, gained)
    return gap, flags


def common_windows(cases, anchor):
    rows = []
    for j, case in enumerate(cases):
        for o, (order, distances) in enumerate(case['orders']):
            start = case['now_s']+max(distances[0], 1)*6
            second_travel = max(distances[1], 1)*6
            for branch, target, begin in [('first', order[0], start),
                    ('second_after_abort', order[1], start+6+second_travel),
                    ('second_after_full', order[1], start+12+second_travel)]:
                assert begin+12 <= anchor+60
                rows.append(dict(case_index=j, order=o, branch=branch, target=int(target),
                                 begin_s=begin, end_s=begin+12))
    return rows


def window_maxima(times, depth, windows):
    return np.array([interval_maximum(times, depth[:, r['target']], r['begin_s'], r['end_s']) for r in windows])
