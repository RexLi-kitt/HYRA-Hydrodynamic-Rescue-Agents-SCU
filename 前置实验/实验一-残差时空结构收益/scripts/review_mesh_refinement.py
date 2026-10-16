"""One bounded 40->80 review. Old runs and confirmation labels are read-only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from diagnose_numerical_stability import Fields, read, load, sha, csv_write, nominal_lags
from hyra_validation.artifacts import new_run, write_json
from hyra_validation.lead import XY, TARGETS
from hyra_validation.local_service import task_table
from hyra_validation.alignment import make_cases
from hyra_validation.numerical_stability import (point_weights, window_maxima, common_windows,
    score_cases, opportunity_flags)
from hyra_validation.refinement import comparison_summary, stable_controls, bed_readout


def field_key(family, mesh, kind, anchor=0, center=0):
    return f'{family}_n{mesh}_dt2_{kind}_a{anchor}_y{center}'


class BoundedFields(Fields):
    def get(self, *args, **kwargs):
        result = super().get(*args, **kwargs)
        write_json(self.run/'field_registry.json', self.registry)
        write_json(self.run/'progress.json', dict(completed=False, registered_fields=len(self.registry),
                   last_field=result['info']['key']))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    design = read(ROOT/'configs/numerical_refinement.design.v1.json')
    parent_run = Path(design['parent_run'])
    parent_config = read(parent_run/'config.json')
    families_path = Path(parent_config['source_family_parent'])
    run = args.run.resolve() if args.run else new_run('bounded_mesh80_review', dict(
        design=design, parent_run=str(parent_run), source_family_parent=str(families_path)))
    if args.run:
        for rel, checksum in read(run/'provenance.json')['files'].items():
            assert sha(ROOT/rel) == checksum, rel
    print('RUN '+str(run), flush=True)
    parent_registry = read(parent_run/'field_registry.json')
    parent_rows = [r for family in design['families'] for r in read(parent_run/(family+'.json'))['windows']]
    controls = stable_controls(parent_rows)
    selection = dict(parent_windows=len(parent_rows),
        parent_disagreement_indices=[i for i, r in enumerate(parent_rows) if r['label_disagrees']],
        stable_control_indices=controls, all_windows_recomputed=True,
        selection_precedes_new_integration=True, parent_windows_sha256=sha(parent_run/'windows.csv'))
    if (run/'coverage_selection.json').exists():
        assert read(run/'coverage_selection.json') == selection
    else:
        write_json(run/'coverage_selection.json', selection)
    fields = BoundedFields(run, Path(parent_config['alignment_run']))
    layout_map = {}
    for case in task_table(read(families_path/'geometry.json')):
        layout_map.setdefault(case['layout'], case)
    layouts = list(layout_map.values())
    started = time.perf_counter()
    all_rows, differences, bed_rows, opportunity_rows = [], [], [], []
    source_hashes = {}
    for family in design['families']:
        meta_path = families_path/'families'/(family+'.json')
        meta = read(meta_path)
        assert meta['split'] == 'train'
        source_hashes[str(meta_path)] = sha(meta_path)
        frozen = load(parent_run/(family+'_scores.npz'))
        parent_result = read(parent_run/(family+'.json'))
        old_cache = {}

        def old(kind, anchor=0, center=0):
            key = field_key(family, 40, kind, anchor, center)
            if key not in old_cache:
                info = parent_registry[key]
                assert sha(info['projected_path']) == info['projected_sha256']
                old_cache[key] = load(info['projected_path'])
            return old_cache[key]

        base = fields.get(meta, 80, 2, 'truth_base')
        nominal = fields.get(meta, 80, 2, 'nominal_base')
        forecasts = fields.forecasts(meta, 80, 2, nominal)
        # Diagnose the bed separately from the evolved water surface; no readout is silently changed.
        for mesh, info in [(40, parent_registry[field_key(family, 40, 'truth_base')]), (80, base['info'])]:
            assert sha(info['raw_path']) == info['raw_sha256']
            raw = load(info['raw_path'])
            weights = point_weights(raw['centroid_xy_m'])
            bed_rows.append(dict(family=family, mesh=mesh, **bed_readout(
                raw['centroid_xy_m'], raw['bed_m'], weights, XY, meta['params'])))
        family_rows, family_diff, family_opportunities = [], [], []
        for anchor in design['anchors_s']:
            cases = make_cases(layouts, anchor)
            windows = common_windows(cases, anchor)
            for center in design['physical_centers_y_m']:
                pulse = fields.get(meta, 80, 2, 'truth_pulse', anchor, center)
                probe = fields.get(meta, 80, 2, 'nominal_pulse', anchor, center)
                before = pulse['time_s'] <= anchor
                for method in design['readouts']:
                    np.testing.assert_allclose(pulse[method][before], base[method][before], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(probe[method][before], nominal[method][before], atol=1e-8, rtol=0)
                    a = old('truth_pulse', anchor, center)[method]
                    b = pulse[method]
                    oldlag = nominal_lags(old('nominal_base')['time_s'],
                        old('nominal_pulse', anchor, center)[method]-old('nominal_base')[method], center)
                    newlag = nominal_lags(nominal['time_s'], probe[method]-nominal[method], center)
                    lag_changes = [abs(oldlag[e]-newlag[e]) for e in oldlag
                                   if oldlag[e] is not None and newlag[e] is not None]
                    family_diff.append(dict(family=family, anchor_s=anchor, center_y_m=center, readout=method,
                        target_max_depth_change_m=float(np.max(abs(a[:, TARGETS]-b[:, TARGETS]))),
                        max_nominal_lag_change_s=max(lag_changes) if lag_changes else None,
                        missing_lag_edges=sum(oldlag[e] is None or newlag[e] is None for e in oldlag)))
                for condition, oldfield, newfield in [('E0', old('truth_base'), base),
                        ('E1', old('truth_pulse', anchor, center), pulse)]:
                    maxima = {f'n{n}_{method}': window_maxima(field['time_s'], field[method], windows)
                              for n, field in [(40, oldfield), (80, newfield)] for method in design['readouts']}
                    previous = [r for r in parent_result['windows'] if r['anchor_s'] == anchor
                                and r['physical_center_y_m'] == center and r['condition'] == condition]
                    assert len(previous) == len(windows)
                    for i, (window, prev) in enumerate(zip(windows, previous)):
                        for k, value in window.items(): assert prev[k] == value
                        low = min(prev['lower_m'], maxima['n80_nearest'][i], maxima['n80_linear'][i])
                        high = max(prev['upper_m'], maxima['n80_nearest'][i], maxima['n80_linear'][i])
                        family_rows.append(dict(prev, parent_lower_m=prev['lower_m'], parent_upper_m=prev['upper_m'],
                            parent_label_disagrees=prev['label_disagrees'], lower_m=float(low), upper_m=float(high),
                            label_disagrees=bool(low <= .8 < high), spread_m=float(high-low),
                            **{k+'_maximum_m': float(v[i]) for k, v in maxima.items()}))
                # Extend the old finite-set opportunity check, never use it as an online detector.
                prefix = f'a{anchor}_y{center}_n40_dt2_linear'
                actions, mask = frozen[prefix+'_frozen_action'], frozen[prefix+'_frozen_mask']
                scores = score_cases(pulse['time_s'], pulse['linear'], cases, anchor)
                gap, flags = opportunity_flags(pulse['time_s'], base['linear'], pulse['linear'],
                                               cases, anchor, actions, scores)
                nearest_scores = score_cases(pulse['time_s'], pulse['nearest'], cases, anchor)
                nearest_gap, nearest_flags = opportunity_flags(pulse['time_s'], base['nearest'],
                    pulse['nearest'], cases, anchor, actions, nearest_scores)
                previous_robust = [r for r in parent_result['robust_cases']
                    if r['anchor_s'] == anchor and r['physical_center_y_m'] == center]
                cursor = 0
                for j, case in enumerate(cases):
                    if not mask[j]: continue
                    prev = previous_robust[cursor]; cursor += 1
                    assert prev['layout'] == case['layout'] and prev['now_s'] == case['now_s']
                    survives = prev['all_variants_improve'] and gap[j] > 0 and nearest_gap[j] > 0
                    family_opportunities.append(dict(family=family, anchor_s=anchor, center_y_m=center,
                        layout=case['layout'], now_s=case['now_s'], parent_improves=prev['all_variants_improve'],
                        n80_linear_alternative_minus_baseline=int(gap[j]), expanded_improves=bool(survives),
                        n80_nearest_alternative_minus_baseline=int(nearest_gap[j]),
                        expanded_visible_D0_by_phase=(survives & np.array(prev['all_variants_visible_D0_by_phase']) & flags[j, :, 0] & nearest_flags[j, :, 0]).tolist(),
                        expanded_visible_D10_by_phase=(survives & np.array(prev['all_variants_visible_D10_by_phase']) & flags[j, :, 1] & nearest_flags[j, :, 1]).tolist()))
        write_json(run/(family+'.json'), dict(windows=family_rows, differences=family_diff, opportunities=family_opportunities))
        all_rows.extend(family_rows); differences.extend(family_diff); opportunity_rows.extend(family_opportunities)
        print('FAMILY_DONE '+family, flush=True)
    assert len(all_rows) == len(parent_rows) == 18432
    for previous, current in zip(parent_rows, all_rows):
        assert all(previous[k] == current[k] for k in ('family', 'anchor_s', 'physical_center_y_m', 'condition', 'case_index', 'order', 'branch', 'target', 'begin_s', 'end_s'))
    comparisons = {}
    for method in design['readouts']:
        comparisons[method] = comparison_summary([r[f'n40_{method}_maximum_m'] for r in all_rows],
                                                [r[f'n80_{method}_maximum_m'] for r in all_rows])
    linear_diff = [r for r in differences if r['readout'] == 'linear']
    max_depth = max(r['target_max_depth_change_m'] for r in linear_diff)
    max_lag = max(r['max_nominal_lag_change_s'] for r in linear_diff if r['max_nominal_lag_change_s'] is not None)
    limits = design['engineering_thresholds']
    checks = dict(max_target_change_le_5mm=max_depth <= limits['max_target_trajectory_depth_change_m'],
        max_nominal_lag_change_le_2s=max_lag <= limits['max_nominal_lag_change_s'],
        no_common_window_label_changes=comparisons['linear']['label_changes'] == 0)
    discord = selection['parent_disagreement_indices']
    groups = {}
    for name, indices in [('old_disagreements', discord), ('stable_controls', controls), ('all', range(len(all_rows)))]:
        rows = [all_rows[i] for i in indices]
        groups[name] = dict(comparison_summary([r['n40_linear_maximum_m'] for r in rows],
                                              [r['n80_linear_maximum_m'] for r in rows]),
            expanded_disagreements=sum(r['label_disagrees'] for r in rows))
    summary = dict(training_families=len(design['families']), comparisons=comparisons, coverage=groups,
        max_target_depth_change_m=max_depth, max_nominal_lag_change_s=max_lag,
        engineering_checks=checks, numerical_gate_passed=all(checks.values()),
        parent_disagreements=len(discord), expanded_disagreements=sum(r['label_disagrees'] for r in all_rows),
        parent_robust_opportunities=sum(r['parent_improves'] for r in opportunity_rows),
        expanded_robust_opportunities=sum(r['expanded_improves'] for r in opportunity_rows),
        expanded_visible_D0=sum(sum(r['expanded_visible_D0_by_phase']) for r in opportunity_rows),
        expanded_visible_D10=sum(sum(r['expanded_visible_D10_by_phase']) for r in opportunity_rows),
        new_field_integrations=len(fields.registry), nominal_restarts=sum(bool(r.get('restart')) for r in fields.registry.values()),
        old_confirmation_labels_read=0, learned_models=0, seconds=time.perf_counter()-started,
        next_action='freeze_numerical_contract_then_training_comparison' if all(checks.values()) else
                    'finite_budget_exhausted_report_uncertain_labels_and_review_physical_task_scale_before_new_confirmation')
    assert summary['new_field_integrations'] <= design['integration_budget']['max_new_fields']
    for name, data in [('windows', all_rows), ('field_differences', differences), ('bed_readout', bed_rows), ('opportunities', opportunity_rows)]:
        csv_write(run/(name+'.csv'), data)
    write_json(run/'summary.json', summary)
    write_json(run/'source_family_hashes.json', source_hashes)
    write_json(run/'progress.json', dict(completed=True, registered_fields=len(fields.registry)))
    write_json(run/'completion.json', dict(completed=True, numerical_gate_passed=summary['numerical_gate_passed']))
    write_json(ROOT/'results/latest_numerical_refinement.json', dict(run=str(run), completed=True,
        numerical_gate_passed=summary['numerical_gate_passed'], audit_pending=True))
    print('SUMMARY '+str(summary), flush=True)


if __name__ == '__main__': main()
