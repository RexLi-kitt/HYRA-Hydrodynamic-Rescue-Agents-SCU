"""Independent field, restart, interpolation and task audit of numerical diagnostics."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('independent_scalar_reference', HERE/'audit_pulse_service_alignment.py')
reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)


def view(registry, key, method, decimate=False):
    data = reference.load(registry[key]['projected_path'])
    step = 2 if decimate else 1
    return data['time_s'][::step], data[method][::step]


def key(family, n, dt, kind, anchor=0, center=0):
    return f'{family}_n{n}_dt{dt}_{kind}_a{anchor}_y{center}'


def parse_variant(name):
    parts = name.split('_'); return int(parts[0][1:]), int(parts[1][2:]), parts[-1], 'decimated2' in parts


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    run = p.parse_args().run.resolve(); started = time.perf_counter(); root = run.parents[2]
    read, load, sha = reference.read, reference.load, reference.digest
    config = read(run/'config.json'); parent = Path(config['source_family_parent']); oldrun = Path(config['alignment_run'])
    for name, value in read(run/'provenance.json')['files'].items(): assert sha(root/name) == value, name
    for path, value in read(run/'source_family_hashes.json').items(): assert sha(path) == value
    registry = read(run/'field_registry.json'); counters = dict(raw_fields=0, projected_scalar_values=0,
        nominal_restarts=0, independent_order_evaluations=0, common_windows=0,
        causal_visibility_flags=0, robust_cases=0)
    fixed_xy = np.array([(5+10*x, 5+10*y) for y in range(10) for x in range(10)])
    families = config['design']['selected_training_families']
    sampling = []
    for k, info in registry.items():
        assert sha(info['raw_path']) == info['raw_sha256']
        assert sha(info['projected_path']) == info['projected_sha256']
        raw = load(info['raw_path']); projection = load(info['projected_path'])
        vertices = projection['vertices']; weights = projection['weights']; indices = projection['nearest_index']
        assert weights.min() >= 0
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
        xy = np.sum(raw['centroid_xy_m'][vertices]*weights[:, :, None], axis=1)
        np.testing.assert_allclose(xy, fixed_xy, atol=1e-10)
        depth = np.maximum(raw['stage']-raw['bed_m'], 0)
        manual = np.zeros((len(raw['time_s']), len(fixed_xy)))
        for i in range(3): manual += depth[:, vertices[:, i]]*weights[:, i]
        np.testing.assert_allclose(projection['linear'], manual, atol=1e-12, rtol=0)
        np.testing.assert_allclose(projection['nearest'], depth[:, indices], atol=1e-12, rtol=0)
        counters['projected_scalar_values'] += manual.size*2; counters['raw_fields'] += 1
        if info.get('restart'):
            restart = info['restart']; assert restart['source_kind'] == 'independent_nominal_base'
            assert 'nominal_base' in Path(restart['source_path']).name
            assert sha(restart['source_path']) == restart['source_sha256']
            base = load(restart['source_path']); ix = int(np.flatnonzero(base['time_s'] == restart['state_time_s'])[0])
            for quantity in ('stage', 'xmomentum', 'ymomentum'):
                np.testing.assert_allclose(raw[quantity][0], base[quantity][ix], atol=1e-12, rtol=0)
            counters['nominal_restarts'] += 1
        if '_truth_base_' in k and '_dt2_' in k:
            family = k.split('_n')[0]; meta = read(parent/'families'/(family+'.json'))
            n = int(k.split('_n')[1].split('_')[0]); par = meta['params']
            analytic = -par['slope']*fixed_xy[:, 0]+par['ridge']*np.exp(-((fixed_xy[:, 1]-50)/12)**2)
            linear_bed = (raw['bed_m'][vertices]*weights).sum(axis=1)
            nearest_bed = raw['bed_m'][indices]
            sampling.append(dict(family=family, mesh=n,
                nearest_max_distance_m=float(np.max(np.linalg.norm(raw['centroid_xy_m'][indices]-fixed_xy, axis=1))),
                nearest_max_analytic_bed_difference_m=float(np.max(abs(nearest_bed-analytic))),
                linear_max_analytic_bed_difference_m=float(np.max(abs(linear_bed-analytic)))))
    geometry = read(parent/'geometry.json')['layouts']; orders = []
    for layout in geometry:
        a, b = layout['targets']; blocked = set(layout['blocked_regions']); origin = layout['origin']
        d = reference.distances(origin, blocked)
        orders.append([([a, b], [d[a], reference.distances(a, blocked)[b]]),
                       ([b, a], [d[b], reference.distances(b, blocked)[a]])])
    for family in families:
        result = read(run/(family+'.json')); scores = load(run/(family+'_scores.npz'))
        fields_cache = {}; counts_cache = {}; flag_cache = {}; matched_cache = {}; native_cache = {}
        def field(n, dt, kind, anchor=0, center=0, method='nearest', decimate=False):
            cache_key = (n, dt, kind, anchor, center, method, decimate)
            if cache_key not in fields_cache:
                fields_cache[cache_key] = view(registry, key(family, n, dt, kind, anchor, center), method, decimate)
            return fields_cache[cache_key]
        names = sorted({r['variant'] for r in result['opportunities']})
        for anchor in (180, 210):
            for center in (20, 70):
                for name in names:
                    n, dt, method, decimate = parse_variant(name)
                    prefix = f'a{anchor}_y{center}_{name}'
                    frozen = scores[prefix+'_frozen_action']; mask = scores[prefix+'_frozen_mask']
                    tt, base = field(n, dt, 'truth_base', method=method, decimate=decimate)
                    _, pulse = field(n, dt, 'truth_pulse', anchor, center, method, decimate)
                    all_count = {}; succeeded = []; matched = []; native = []
                    for condition, depth in (('E0', base), ('E1', pulse)):
                        counts = []; successes = []
                        for layout_index, alternatives in enumerate(orders):
                            for offset in (0, 6, 12):
                                now = anchor+offset
                                executed = [reference.score(tt, depth, order, ds, now, anchor+60) for order, ds in alternatives]
                                counts.append([s[0] for s in executed]); successes.append([s[3] for s in executed])
                                counters['independent_order_evaluations'] += 2
                                if condition == 'E1':
                                    release = anchor if offset < 10 else anchor+10
                                    ft, fd = field(n, dt, 'forecast', release, method=method, decimate=decimate)
                                    af, ad = field(n, dt, 'forecast', anchor, method=method, decimate=decimate)
                                    pred = [reference.score(ft, fd, order, ds, now, anchor+60) for order, ds in alternatives]
                                    prior = [reference.score(af, ad, order, ds, now, anchor+60) for order, ds in alternatives]
                                    winner = 0
                                    if pred[1][0] > pred[0][0]: winner = 1
                                    elif pred[1][0] == pred[0][0]:
                                        if pred[1][1] > pred[0][1]+1e-12: winner = 1
                                        elif abs(pred[1][1]-pred[0][1]) <= 1e-12 and alternatives[1][0] < alternatives[0][0]: winner = 1
                                    matched.append(winner)
                                    native.append(min(abs(.8-m) for s in prior for m in s[2]) <= .05)
                        all_count[condition] = np.array(counts)
                        counts_cache[anchor, center, name, condition] = np.array(counts)
                        if condition == 'E1': succeeded = successes
                    np.testing.assert_array_equal(scores[prefix+'_counts'], all_count['E1'])
                    np.testing.assert_array_equal(scores[prefix+'_matched_action'], matched)
                    np.testing.assert_array_equal(scores[prefix+'_native_mask'], native)
                    matched_cache[anchor, center, name] = np.array(matched)
                    native_cache[anchor, center, name] = np.array(native)
                    count = all_count['E1']; gap = count[np.arange(96), 1-frozen]-count[np.arange(96), frozen]
                    np.testing.assert_array_equal(scores[prefix+'_frozen_gap'], gap)
                    flags = np.zeros((96, 4, 2), bool)
                    for j in range(96):
                        if gap[j] <= 0: continue
                        gained = set(r for r in succeeded[j][1-frozen[j]] if r is not None)-set(r for r in succeeded[j][frozen[j]] if r is not None)
                        now = anchor+(j % 3)*6
                        for pi, phase in enumerate((0, .25, .5, .75)):
                            samples = np.arange(20+5*phase, 300+1e-9, 5)
                            for di, delay in enumerate((0, 10)):
                                ids = []
                                for b in range(6):
                                    candidates = np.flatnonzero((samples+delay <= now) & (now-samples >= b*10) & (now-samples < b*10+10))
                                    if len(candidates): ids.append(int(candidates[-1]))
                                for source in (10, 20, 30, 60, 70, 80):
                                    if not any((source//10 < 5) == (target//10 < 5) for target in gained): continue
                                    if ids and max(abs(np.interp(samples[ids], tt, pulse[:, source]-base[:, source]))) >= .001:
                                        flags[j, pi, di] = True
                    np.testing.assert_array_equal(scores[prefix+'_visible'], flags)
                    flag_cache[anchor, center, name] = flags
                    counters['causal_visibility_flags'] += flags.size
        for row in result['opportunities']:
            a, c, v = row['anchor_s'], row['physical_center_y_m'], row['variant']
            counts = counts_cache[a, c, v, row['condition']]; prefix = f'a{a}_y{c}_{v}'
            actions = scores[prefix+'_frozen_action'] if row['scope'].startswith('frozen') else matched_cache[a, c, v]
            mask = native_cache[a, c, v] if row['scope'].endswith('native_mask') else scores[prefix+'_frozen_mask']
            actual = counts[np.arange(96), actions]
            assert (counts.max(axis=1)-actual)[mask].sum() == row['gap_sum']
            assert mask.sum() == row['retained_cases']
        for row in result['windows']:
            for name in names:
                n, dt, method, decimate = parse_variant(name)
                kind = 'truth_base' if row['condition'] == 'E0' else 'truth_pulse'
                tt, depth = field(n, dt, kind, row['anchor_s'] if kind.endswith('pulse') else 0,
                    row['physical_center_y_m'] if kind.endswith('pulse') else 0, method, decimate)
                value = reference.maximum(tt, depth, row['target'], row['begin_s'], row['end_s'])
                assert abs(value-row[name]) < 1e-12
            values = [row[name] for name in names]
            assert abs(row['lower_m']-min(values)) < 1e-12 and abs(row['upper_m']-max(values)) < 1e-12
            assert row['label_disagrees'] == (min(values) <= .8 < max(values))
            counters['common_windows'] += 1
        for row in result['robust_cases']:
            a = row['anchor_s']; c = row['physical_center_y_m']; j = row['layout']*3+(row['now_s']-a)//6
            deltas = []; flags = []
            for name in names:
                prefix = f'a{a}_y{c}_{name}'; deltas.append(scores[prefix+'_frozen_gap'][j])
                flags.append(flag_cache[a, c, name][j])
            assert min(deltas) == row['minimum_alternative_minus_baseline']
            assert row['all_variants_improve'] == all(d > 0 for d in deltas)
            np.testing.assert_array_equal(np.all(np.array(flags)[:, :, 0], axis=0), row['all_variants_visible_D0_by_phase'])
            np.testing.assert_array_equal(np.all(np.array(flags)[:, :, 1], axis=0), row['all_variants_visible_D10_by_phase'])
            counters['robust_cases'] += 1
        print('AUDIT '+family, flush=True)
    out = dict(passed=True, counters=counters, sampling_geometry=sampling,
        audit_source_sha256=sha(Path(__file__)), seconds=time.perf_counter()-started,
        limitations='Numerical consistency is assessed separately; this pass only confirms saved fields, interpolation, restarts and arithmetic.')
    (run/'independent_audit.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
