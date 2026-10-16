"""Independent scalar audit of the training-only physical/task scale screen."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

from audit_pulse_service_alignment import read, digest, load, distances, maximum, score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def visible(times, response, now, phase, delay):
    sample = np.arange(120+5*phase, times[-1]+1e-9, 5.)
    kept = []
    for b in range(6):
        valid = [t for t in sample if t+delay <= now and b*10 <= now-t < (b+1)*10]
        if valid: kept.append(valid[-1])
    return any(abs(np.interp(t, times, response[:, source])) >= .001
               for t in kept for source in (10, 20, 30, 60, 70, 80))


def choose(scores, orders):
    if scores[1][0] > scores[0][0]: return 1
    if scores[1][0] < scores[0][0]: return 0
    if scores[1][1] > scores[0][1]+1e-12: return 1
    if abs(scores[1][1]-scores[0][1]) <= 1e-12 and orders[1][0] < orders[0][0]: return 1
    return 0


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve(); config = read(run/'config.json'); design = config['design']
    assert read(run/'completion.json')['completed']
    for rel, checksum in read(run/'provenance.json')['files'].items(): assert digest(ROOT/rel) == checksum, rel
    for path, checksum in read(run/'source_hashes.json').items(): assert digest(path) == checksum
    parent = Path(design['source_family_parent']); geometry = read(parent/'geometry.json')['layouts']
    registry = read(run/'field_registry.json'); cache = {}; counters = defaultdict(int)
    def field(family, length, kind, anchor=0, center=0, amp=0.):
        key = f'{family}_L{length}_n40_{kind}_a{anchor}_y{center}_mm{round(amp*1000)}'
        if key not in cache:
            info = registry[key]; assert digest(info['projected_path']) == info['projected_sha256']
            cache[key] = load(info['projected_path'])
        return cache[key]
    for key, info in registry.items():
        assert info['split'] == 'train' and info['family'] in design['family_ids']
        assert digest(info['raw_path']) == info['raw_sha256']
        assert digest(info['projected_path']) == info['projected_sha256']
        raw = load(info['raw_path']); p = load(info['projected_path'])
        expected_xy = np.array([(5+10*x, 5+10*y) for y in range(10) for x in range(10)], float)
        expected_xy[:, 0] *= info['length_x_m']/100.
        vertices, weights = p['vertices'], p['weights']
        assert weights.min() >= 0
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)
        np.testing.assert_allclose((raw['centroid_xy_m'][vertices]*weights[:, :, None]).sum(axis=1), expected_xy, atol=1e-10)
        depth = np.maximum(raw['stage']-raw['bed_m'], 0.)
        manual = sum(depth[:, vertices[:, j]]*weights[:, j] for j in range(3))
        np.testing.assert_allclose(manual, p['linear'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(depth[:, p['nearest_index']], p['nearest'], atol=1e-12, rtol=0)
        counters['raw_fields'] += 1; counters['projection_scalar_values'] += manual.size*2
        if info['pulse_amplitude_m']:
            assert info['physical_interval_y_m'] == [info['center_y_m']-10., info['center_y_m']+10.]
        if info['restart']:
            restart = info['restart']; assert restart['source_kind'] == 'same_mesh_independent_nominal_base'
            assert 'nominal_base' in Path(restart['source_path']).name
            assert digest(restart['source_path']) == restart['source_sha256']
            base = load(restart['source_path']); index = np.flatnonzero(base['time_s'] == restart['state_time_s'])[0]
            for name in ('stage','xmomentum','ymomentum'):
                np.testing.assert_allclose(raw[name][0], base[name][index], atol=1e-12, rtol=0)
            start = info['anchor_s']; finish = start+60; counters['nominal_restarts'] += 1
        else: start, finish = 0, 390
        np.testing.assert_array_equal(raw['time_s'], np.arange(start, finish+1, 2))
    orders = {}
    for layout in geometry:
        a, b = layout['targets']; blocked = set(layout['blocked_regions']); initial = distances(layout['origin'], blocked)
        orders[layout['layout_index']] = [([a,b], [initial[a],distances(a,blocked)[b]]),
                                          ([b,a], [initial[b],distances(b,blocked)[a]])]
    all_table_rows = []
    for family in design['family_ids']:
        meta = read(parent/'families'/(family+'.json')); assert meta['split'] == 'train'
        for length in design['domain_lengths_x_m']:
            data = read(run/(family+f'_L{length}.json')); all_table_rows.extend(data['rows'])
            base = field(family, length, 'truth_base')
            # Cache nominal decisions independently of pulse amplitude/location.
            nominal = {}
            for anchor in design['anchors_s']:
                for layout_index, alternatives in orders.items():
                    af = field(family, length, 'nominal_forecast', anchor)
                    for offset in design['decision_offsets_s']:
                        now = anchor+offset; release = int(now//10*10)
                        current = field(family, length, 'nominal_forecast', release)
                        forecasts = [score(current['time_s'], current['linear'], order, ds, now, anchor+60)
                                     for order, ds in alternatives]
                        anchor_scores = [score(af['time_s'], af['linear'], order, ds, now, anchor+60)
                                         for order, ds in alternatives]
                        nominal[anchor,layout_index,now] = (choose(forecasts, alternatives),
                            min(abs(.8-m) for s in anchor_scores for m in s[2]) <= .05)
            groups = defaultdict(list)
            for row in data['actions']:
                anchor, center, amp, now = row['anchor_s'], row['center_y_m'], row['amplitude_m'], row['now_s']
                action, critical = nominal[anchor, row['layout'], now]
                assert action == row['nominal_action'] and critical == row['critical']
                pulse = field(family, length, 'truth_pulse', anchor, center, amp)
                alternatives = orders[row['layout']]
                e0 = [score(base['time_s'], base['linear'], order, ds, now, anchor+60) for order,ds in alternatives]
                e1 = [score(pulse['time_s'], pulse['linear'], order, ds, now, anchor+60) for order,ds in alternatives]
                counters['scalar_order_evaluations'] += 4
                assert [s[0] for s in e0] == row['E0_counts'] and [s[0] for s in e1] == row['E1_counts']
                g0 = e0[1-action][0]-e0[action][0]; g1 = e1[1-action][0]-e1[action][0]
                assert g0 == row['E0_alternative_gap'] and g1 == row['E1_alternative_gap']
                specific = g1 > 0 and g0 <= 0; assert specific == row['pulse_specific']
                for delay in (0,10):
                    flags = [specific and visible(pulse['time_s'], pulse['linear']-base['linear'], now, phase, delay)
                             for phase in (0,.25,.5,.75)]
                    assert flags == row[f'visible_D{delay}_by_phase']
                    counters['visibility_flags'] += 4
                groups[anchor, center, amp].append(row)
            for table in data['rows']:
                selected = groups[table['anchor_s'],table['center_y_m'],table['amplitude_m']]
                if table['population'] == 'critical': selected = [r for r in selected if r['critical']]
                name = table['condition']+'_counts'
                assert len(selected) == table['cases']
                actual = sum(r[name][r['nominal_action']] for r in selected)
                oracle = sum(max(r[name]) for r in selected)
                assert actual == table['nominal_sum'] and oracle == table['oracle_sum']
                assert oracle-actual == table['headroom_sum']
                for delay in (0,10):
                    expected = sum(sum(r[f'visible_D{delay}_by_phase'])*max(r['E1_alternative_gap'],0)
                                   for r in selected) if table['condition'] == 'E1' else 0
                    assert expected == table[f'visible_D{delay}_sum']
            for row in data['windows']:
                af = field(family, length, 'truth_base') if row['condition']=='E0' else field(
                    family,length,'truth_pulse',row['anchor_s'],row['center_y_m'],row['amplitude_m'])
                layout_index = geometry[row['case_index']//3]['layout_index']; offset = design['decision_offsets_s'][row['case_index']%3]
                order, ds = orders[layout_index][row['order']]; first = row['anchor_s']+offset+max(ds[0],1)*6
                if row['branch']=='first': target, begin = order[0],first
                else: target,begin = order[1],first+(6 if row['branch']=='second_after_abort' else 12)+max(ds[1],1)*6
                assert row['target']==target and row['begin_s']==begin and row['end_s']==begin+12
                value = maximum(af['time_s'],af['linear'],target,begin,begin+12)
                assert abs(value-row['maximum_m']) < 1e-12
                counters['scalar_window_evaluations'] += 1
            counters['contexts'] += 1
    summary = read(run/'summary.json'); expected_selected = None
    for item in summary['conditions']:
        for fam in item['families']:
            selected = [r for r in all_table_rows if r['family']==fam['family'] and r['length_x_m']==item['length_x_m']
                        and r['amplitude_m']==item['amplitude_m'] and r['population']==item['population']]
            count = sum(r['cases'] for r in selected)
            assert fam['cases']==count and fam['unique_design_cases']==count//2
            assert fam['nonempty_anchor_center_cells']==sum(r['cases']>0 for r in selected if r['condition']=='E0')
            if count:
                expected = sum(r['headroom_sum'] for r in selected)/count*100
                assert abs(fam['headroom_per100']-expected)<1e-12
                for delay in (0,10):
                    expected = sum(r[f'visible_D{delay}_sum'] for r in selected)/(count*4)*100
                    assert abs(fam[f'visible_D{delay}_per100']-expected)<1e-12
        nonempty=[r for r in item['families'] if r['cases']]
        assert item['nonempty_families']==len(nonempty)
        if nonempty:
            expected=float(np.mean([r['headroom_per100'] for r in nonempty]))
            assert abs(item['headroom_family_macro_per100']-expected)<1e-12
            for delay in (0,10):
                expected=float(np.mean([r[f'visible_D{delay}_per100'] for r in nonempty]))
                assert abs(item[f'visible_D{delay}_family_macro_per100']-expected)<1e-12
        if item['population']=='critical':
            passed = all(r['unique_design_cases']>=10 and r['nonempty_anchor_center_cells']>=2 for r in item['families'])
            passed = passed and (item['headroom_family_macro_per100'] or 0)>=4 and (item['visible_D0_family_macro_per100'] or 0)>=2
            assert passed == item['training_screen_passed']
            if passed and expected_selected is None: expected_selected=dict(length_x_m=item['length_x_m'],amplitude_m=item['amplitude_m'])
    assert expected_selected == summary['selected_for_next_stage']
    assert counters['raw_fields']==504 and counters['nominal_restarts']==264 and counters['contexts']==12
    write_json(run/'independent_audit.json',dict(passed=True,**counters,audit_source_sha256=digest(__file__),
        interpretation='Raw states, actual issue restarts, fixed-point projections, all windows, scalar orders, pulse-specific visibility, selection rule. No model efficacy or new-geometry convergence claim.'))
    pointer=read(ROOT/'results/latest_physical_scale_calibration.json'); assert Path(pointer['run'])==run
    pointer.update(audit_pending=False,independent_audit_passed=True)
    write_json(ROOT/'results/latest_physical_scale_calibration.json',pointer)
    print('AUDIT_PASSED '+str(dict(counters)))


if __name__=='__main__': main()
