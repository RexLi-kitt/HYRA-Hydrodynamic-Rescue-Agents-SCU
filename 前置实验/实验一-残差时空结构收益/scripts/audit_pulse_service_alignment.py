"""Independent saved-artifact audit. Does not import the campaign scorer/helpers."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def distances(origin, blocked):
    result = {origin: 0}; queue = deque([origin])
    while queue:
        r = queue.popleft(); x, y = r % 10, r//10
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0: continue
                xx, yy = x+dx, y+dy
                v = yy*10+xx
                if not (0 <= xx < 10 and 0 <= yy < 10) or v in blocked or v in result: continue
                if dx and dy and (y*10+xx in blocked or yy*10+x in blocked): continue
                result[v] = result[r]+1; queue.append(v)
    return result


def maximum(times, depths, region, begin, end):
    assert times[0] <= begin <= end <= times[-1]
    inside = depths[(times >= begin) & (times <= end), region]
    return max(float(np.interp(begin, times, depths[:, region])),
               float(np.interp(end, times, depths[:, region])), float(np.max(inside)) if len(inside) else -np.inf)


def score(times, depths, order, ds, now, deadline):
    clock = now; maxima = []; successes = []
    for region, distance in zip(order, ds):
        start = clock+max(distance, 1)*6
        assert start+12 <= deadline
        maxima.append(maximum(times, depths, region, start, start+12))
        success = maxima[-1] <= .8; successes.append(region if success else None)
        clock = start+(6 if maximum(times, depths, region, start, start+6) > .8 else 12)
    return sum(r is not None for r in successes), min(.8-m for m in maxima), maxima, successes


def response_marker(times, delta, start):
    after = times >= start
    peak = np.max(np.abs(delta[after]))
    if peak < .001: return None
    indices = np.flatnonzero(after & (np.abs(delta) >= peak/10))
    return float(times[indices[0]]) if len(indices) else None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args(); run = args.run.resolve(); root = run.parents[2]
    started = time.perf_counter(); config = read(run/'config.json'); parent = Path(config['parent'])
    assert read(run/'completion.json')['completed']
    provenance = read(run/'provenance.json')
    for name, value in provenance['files'].items(): assert digest(root/name) == value, name
    geometry = read(parent/'geometry.json')['layouts']
    all_orders = []
    for layout in geometry:
        a, b = layout['targets']; blocked = set(layout['blocked_regions']); origin = layout['origin']
        origin_dist = distances(origin, blocked)
        ab = distances(a, blocked)[b]; ba = distances(b, blocked)[a]
        all_orders.append([([a, b], [origin_dist[a], ab]), ([b, a], [origin_dist[b], ba])])
    trajectories = list((run/'trajectories').glob('*.complete.json'))
    for path in trajectories:
        row = read(path); assert digest(run/row['file']) == row['sha256']
    counters = dict(families=0, raw_trajectory_hashes=len(trajectories), independent_scalar_orders=0,
        causal_channel_references=0, observed_scalar_values_checked=0,
        opportunity_cells=0, action_sets=0, propagation_edges=0, kernel_rows=0)
    family_files = sorted((run/'families').glob('local_v1_train_???.json'))
    assert len(family_files) == read(run/'completion.json')['completed_families']
    for file in family_files:
        family = read(file); name = family['family']
        for source, value in family['source_hashes'].items():
            assert 'train' in Path(source).name and digest(source) == value
        metadata = read(parent/'families'/(name+'.json')); assert metadata['split'] == 'train'
        old = load(parent/'families'/(name+'.npz'))
        saved = load(run/'families'/(name+'_task_scores.npz'))
        records = load(run/'families'/(name+'_causal_records.npz'))
        assert records['source_region_ids'].tolist() == [10, 20, 30, 60, 70, 80]
        base = load(run/'trajectories'/(name+'_mesh10_truth_base_a0_y25.npz'))
        time_s = base['time_s']; base = base['depth_m']
        evaluated = {}; true_maxima = {}; reference_delta = {}
        for anchor in (90, 120, 150, 180, 210):
            for center in (25, 75):
                prefix = f'a{anchor}_y{center}'
                pulse = load(run/'trajectories'/(name+f'_mesh10_truth_pulse_a{anchor}_y{center}.npz'))['depth_m']
                reference_delta[anchor, center] = pulse-base
                np.testing.assert_allclose(pulse[time_s <= anchor], base[time_s <= anchor], atol=1e-8, rtol=0)
                critical = []; chosen = []; e0 = []; e1 = []; gained = []
                for layout_index, orders in enumerate(all_orders):
                    for offset in (0, 6, 12):
                        now = anchor+offset; deadline = anchor+60
                        f = int(np.flatnonzero(old['release_s'] <= now)[-1])
                        af = int(np.flatnonzero(old['release_s'] == anchor)[0])
                        nominal = [score(old['forecast_valid_s'][f], old['M0_forecast_depth_m'][f], order, ds, now, deadline)
                                   for order, ds in orders]
                        anchor_scores = [score(old['forecast_valid_s'][af], old['M0_forecast_depth_m'][af], order, ds, now, deadline)
                                         for order, ds in orders]
                        # Same count, margin and target-ID tie breaking without campaign choose_order.
                        best = 0
                        if nominal[1][0] > nominal[0][0]: best = 1
                        elif nominal[1][0] == nominal[0][0]:
                            if nominal[1][1] > nominal[0][1]+1e-12: best = 1
                            elif abs(nominal[1][1]-nominal[0][1]) <= 1e-12 and orders[1][0] < orders[0][0]: best = 1
                        chosen.append(best)
                        critical.append(min(abs(.8-x) for s in anchor_scores for x in s[2]) <= .05)
                        scored0 = [score(time_s, base, order, ds, now, deadline) for order, ds in orders]
                        scored1 = [score(time_s, pulse, order, ds, now, deadline) for order, ds in orders]
                        e0.append([s[0] for s in scored0]); e1.append([s[0] for s in scored1])
                        truth_best = int(np.argmax(e1[-1]))
                        gained.append(sorted(set(r for r in scored1[truth_best][3] if r is not None)
                                             - set(r for r in scored1[best][3] if r is not None))
                                      if e1[-1][truth_best] > e1[-1][best] else [])
                        counters['independent_scalar_orders'] += 8
                for key, actual in [('critical_mask', critical), ('baseline_order', chosen), ('E0_counts', e0), ('E1_counts', e1)]:
                    np.testing.assert_array_equal(saved[prefix+'_'+key], actual)
                evaluated[anchor, center] = dict(critical=np.array(critical), chosen=np.array(chosen),
                    E0=np.array(e0), E1=np.array(e1), gained=gained)
                for phase_index, phase in enumerate((0, .25, .5, .75)):
                    key = prefix+f'_p{phase_index}'; valid = records[key+'_valid_s']
                    issue = records[key+'_prior_release_s']
                    assert np.all(issue <= valid-20) and np.all((valid-issue >= 20) & (valid-issue < 30))
                    for i, (v, r) in enumerate(zip(valid, issue)):
                        fi = int(np.flatnonzero(old['release_s'] == r)[0])
                        for channel, source in enumerate(records['source_region_ids']):
                            expected = np.interp(v, old['forecast_valid_s'][fi], old['M0_forecast_depth_m'][fi, :, source])
                            assert abs(expected-records[key+'_prior_m'][i, channel]) < 1e-12
                    truth0 = np.column_stack([np.interp(valid, time_s, base[:, s]) for s in records['source_region_ids']])
                    truth1 = np.column_stack([np.interp(valid, time_s, pulse[:, s]) for s in records['source_region_ids']])
                    for noise in (0, .02):
                        for rep in range(3 if noise else 1):
                            rng = np.random.default_rng(np.random.SeedSequence([2026101517, 61,
                                metadata['index'], anchor, center, phase_index, rep]))
                            eps = noise*rng.standard_normal(truth0.shape)
                            for e, truth in [('E0', truth0), ('E1', truth1)]:
                                np.testing.assert_allclose(records[key+f'_n{noise}_rep{rep}_{e}_observed_m'], truth+eps, atol=1e-14, rtol=0)
                                counters['observed_scalar_values_checked'] += truth.size
                    for delay in (0, 10):
                        ids_saved = records[key+f'_d{delay}_bin_ids']
                        for row, offset in enumerate((0, 6, 12)):
                            age = anchor+offset-valid; expected = []
                            for b in range(6):
                                eligible = np.flatnonzero((valid+delay <= anchor+offset) & (age >= 10*b) & (age < 10*b+10))
                                expected.append(int(eligible[-1]) if len(eligible) else -1)
                            np.testing.assert_array_equal(ids_saved[row], expected)
                            counters['causal_channel_references'] += 6*8*sum(i >= 0 for i in expected)
        for row in family['opportunities']:
            data = evaluated[row['anchor_s'], row['center_y_m']]
            counts = data[row['condition']]; actual = counts[np.arange(96), data['chosen']]
            mask = np.ones(96, bool) if row['population'] == 'general' else data['critical']
            mask = mask & (np.tile([0, 6, 12], 32) == row['decision_offset_s'])
            assert mask.sum() == row['retained_cases']
            assert counts[mask].max(axis=1).sum()-actual[mask].sum() == row['gap_sum']
            counters['opportunity_cells'] += 1
        for row in family['actions']:
            anchor = row['anchor_s']; center = row['center_y_m']; delay = row['delay_s']
            data = evaluated[anchor, center]; key = f"a{anchor}_y{center}_p{row['phase_index']}"
            valid = records[key+'_valid_s']; paired = records[key+'_paired_reference_response_m']
            gap_times = []; visible = []; cg = []; cv = []; eligible = []
            for off_index, offset in enumerate((0, 6, 12)):
                j = row['layout']*3+off_index; now = anchor+offset
                if data['critical'][j]: eligible.append(now)
                if data['gained'][j]:
                    gap_times.append(now)
                    if data['critical'][j]: cg.append(now)
                    ids = records[key+f'_d{delay}_bin_ids'][off_index]; ids = ids[ids >= 0]
                    channels = [q for q, s in enumerate(records['source_region_ids']) if any(
                        (s//10 < 5) == (t//10 < 5) for t in data['gained'][j])]
                    yes = bool(len(ids) and channels and np.any(abs(paired[np.ix_(ids, channels)]) >= .001))
                    if yes:
                        visible.append(now)
                        if data['critical'][j]: cv.append(now)
            assert row['critical_eligible_times_s'] == eligible
            assert row['oracle_improvement_times_s'] == gap_times
            assert row['paired_response_and_oracle_times_s'] == visible
            assert row['critical_oracle_improvement_times_s'] == cg
            assert row['critical_paired_response_and_oracle_times_s'] == cv
            counters['action_sets'] += 1
        nominal_base = load(run/'trajectories'/(name+'_mesh10_nominal_base_a0_y25.npz'))['depth_m']
        nominal_deltas = {}
        for anchor in (90, 120, 150, 180, 210):
            for center in (25, 75):
                nominal_deltas[anchor, center] = load(run/'trajectories'/(
                    name+f'_mesh10_nominal_pulse_a{anchor}_y{center}.npz'))['depth_m']-nominal_base
        lag_lookup = {}
        for row in family['edges']:
            anchor = row['anchor_s']; center = row['center_y_m']; source = row['source']; target = row['target']
            nominal_delta = nominal_deltas[anchor, center]
            reference = reference_delta[anchor, center]
            sm = response_marker(time_s, nominal_delta[:, source], anchor)
            tm = response_marker(time_s, nominal_delta[:, target], anchor)
            assert row['nominal_source_marker_s'] == sm and row['nominal_target_marker_s'] == tm
            assert row['reference_source_marker_s'] == response_marker(time_s, reference[:, source], anchor)
            assert row['reference_target_marker_s'] == response_marker(time_s, reference[:, target], anchor)
            expected_lag = None if sm is None or tm is None else max(tm-sm, 0)
            assert row['kernel_lag_s'] == expected_lag
            lag_lookup[anchor, center, source, target] = expected_lag
            counters['propagation_edges'] += 1
        for row in family['kernels']:
            anchor = row['anchor_s']; center = row['center_y_m']
            key = f"a{anchor}_y{center}_p{row['phase_index']}"
            bin_index = (0, 6, 12).index(row['decision_offset_s'])
            ids = records[key+f"_d{row['delay_s']}_bin_ids"][bin_index]
            ages = anchor+row['decision_offset_s']-records[key+'_valid_s'][ids[ids >= 0]]
            lag = lag_lookup[anchor, center, row['source'], row['target']]
            if len(ages):
                c = max(lag-row['horizon_s'], 0)
                weights1 = np.exp(-((ages-c)/10)**2/2); weights1 /= weights1.sum()
                weights0 = np.exp(-(ages/10)**2/2); weights0 /= weights0.sum()
                assert abs(np.sum(abs(weights1-weights0))-row['weight_L1']) < 1e-12
            else: assert row['weight_L1'] is None
            counters['kernel_rows'] += 1
        counters['families'] += 1
        print('AUDIT '+name, flush=True)
    grid_review_path = run/'positive_case_grid_review.json'
    if grid_review_path.exists():
        grid_review = read(grid_review_path)
        counters['posthoc_grid_positive_pair_checks'] = 0
        for row in grid_review['comparisons']:
            name = row['family']; anchor = row['anchor_s']; center = row['center_y_m']
            base = load(run/'trajectories'/(name+'_mesh20_truth_base_a0_y25.npz'))
            times = base['time_s']; base = base['depth_m']
            pulse = load(run/'trajectories'/(name+f'_mesh20_truth_pulse_a{anchor}_y{center}.npz'))['depth_m']
            saved = load(run/'families'/(name+'_task_scores.npz'))
            records = load(run/'families'/(name+'_causal_records.npz'))
            prefix = f'a{anchor}_y{center}'; keep = saved[prefix+'_critical_mask']
            chosen = saved[prefix+'_baseline_order']; gains = []; counts = []
            for layout_index, orders in enumerate(all_orders):
                for offset_index, offset in enumerate((0, 6, 12)):
                    j = layout_index*3+offset_index
                    fine = [score(times, pulse, order, ds, anchor+offset, anchor+60) for order, ds in orders]
                    counts.append([s[0] for s in fine]); best = int(np.argmax(counts[-1]))
                    gains.append(sorted(set(r for r in fine[best][3] if r is not None)
                                        - set(r for r in fine[int(chosen[j])][3] if r is not None))
                                 if fine[best][0] > fine[int(chosen[j])][0] else [])
            counts = np.array(counts)
            assert int((counts.max(axis=1)-counts[np.arange(96), chosen])[keep].sum()) == row['fine_gap_sum']
            for delay in (0, 10):
                hits = []
                for phase in range(4):
                    key = prefix+f'_p{phase}'; valid = records[key+'_valid_s']
                    paired = np.column_stack([np.interp(valid, times, pulse[:, s]-base[:, s])
                                              for s in records['source_region_ids']])
                    for j in range(96):
                        ids = records[key+f'_d{delay}_bin_ids'][j % 3]; ids = ids[ids >= 0]
                        channels = [q for q, s in enumerate(records['source_region_ids']) if any(
                            (s//10 < 5) == (t//10 < 5) for t in gains[j])]
                        if keep[j] and len(ids) and channels and np.any(abs(paired[np.ix_(ids, channels)]) >= .001):
                            hits.append([j, phase])
                assert hits == row['fine_visible_oracle_indices'][str(delay)]
                assert len(hits) == row['fine_visible_oracle_decisions'][str(delay)]
            counters['posthoc_grid_positive_pair_checks'] += 1
    output = dict(passed=True, counters=counters, seconds=time.perf_counter()-started,
        methods='Independent BFS, scalar interval execution, source record reconstruction, action-set recomputation and hashes.',
        limitations='Audit passes arithmetic/references only. Positive-case grid review shows numerical instability; numerical validity gate fails.',
        auditor_source_sha256=digest(Path(__file__)))
    (run/'independent_audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
