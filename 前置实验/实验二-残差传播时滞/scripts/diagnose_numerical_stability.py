"""Training-only fixed-forcing, spatial-readout and output-grid sensitivity study."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import new_run, write_json
from hyra_validation.lead import XY, SOURCES, TARGETS, hydraulic_domain
from hyra_validation.local_service import task_table
from hyra_validation.local_service_data import nominal_domain
from hyra_validation.alignment import make_cases, marker
from hyra_validation.adapters.anuga import simulate
from hyra_validation.numerical_stability import (fixed_domain, boundary_footprint, point_weights, read_points,
    score_cases, forecast_plans, opportunity_flags, common_windows, window_maxima)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p, allow_pickle=False) as z: return {k: z[k] for k in z.files}
def csv_write(p, rows):
    if not rows: return
    with Path(p).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows: w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


class Fields:
    def __init__(self, run, oldrun):
        self.run = run; self.oldrun = oldrun; self.registry = {}; self.projections = {}; self.weights = {}
        (run/'raw').mkdir(exist_ok=True); (run/'projected').mkdir(exist_ok=True)

    def key(self, meta, mesh, step, kind, anchor=0, center=0):
        return f"{meta['family_id']}_n{mesh}_dt{step}_{kind}_a{anchor}_y{center}"

    def get(self, meta, mesh, step, kind, anchor=0, center=0, restart=None):
        key = self.key(meta, mesh, step, kind, anchor, center)
        if key in self.projections: return self.projections[key]
        marker_path = self.run/'raw'/(key+'.json')
        if marker_path.exists():
            info = read(marker_path); rawpath = Path(info['raw_path']); assert sha(rawpath) == info['raw_sha256']
        else:
            rawpath = None
            if step == 2 and kind in ('truth_base', 'nominal_base') and mesh in (10, 20):
                rawpath = self.oldrun/'trajectories'/f"{meta['family_id']}_mesh{mesh}_{kind}_a0_y25.npz"
            elif step == 2 and mesh == 10 and kind in ('truth_pulse', 'nominal_pulse'):
                legacy_center = center+5
                rawpath = self.oldrun/'trajectories'/f"{meta['family_id']}_mesh10_{kind}_a{anchor}_y{legacy_center}.npz"
            if rawpath is not None:
                old_marker = read(rawpath.with_suffix('.complete.json'))
                assert sha(rawpath) == old_marker['sha256']
                info = dict(key=key, raw_path=str(rawpath), raw_sha256=sha(rawpath), reused=True,
                            physical_interval_y_m=[center-10, center+10] if center else None,
                            old_requested_center_y_m=center+5 if center else None)
            else:
                factor = 1 if kind.startswith('truth') else meta['nominal_specifications'][0]['factor']
                if kind == 'forecast':
                    initial = restart['state']
                    domain = nominal_domain(meta['params'], mesh, factor, initial=initial, start=anchor)
                    end = anchor+60
                else:
                    domain = fixed_domain(meta['params'], mesh, factor, center if kind.endswith('pulse') else None, anchor)
                    end = 300
                started = time.perf_counter(); tt, depth, states, mapping = simulate(domain, end, XY, yieldstep=step)
                rawpath = self.run/'raw'/(key+'.npz')
                raw = dict(time_s=tt, depth_m=depth, centroid_xy_m=domain.get_centroid_coordinates(),
                           bed_m=domain.quantities['elevation'].centroid_values.copy())
                for name in states[0]: raw[name] = np.asarray([s[name] for s in states])
                np.savez_compressed(rawpath, **raw)
                info = dict(key=key, raw_path=str(rawpath), raw_sha256=sha(rawpath), reused=False,
                    physical_interval_y_m=boundary_footprint(domain), factor=factor,
                    output_interval_s=step, nmesh=mesh, seconds=time.perf_counter()-started,
                    restart=None if restart is None else {k: v for k, v in restart.items() if k != 'state'})
                print('FIELD '+key, flush=True)
            write_json(marker_path, info)
        data = load(rawpath)
        weights = self.weights.setdefault(mesh, point_weights(data['centroid_xy_m']))
        points = read_points(data['stage'], data['bed_m'], weights)
        np.testing.assert_allclose(points['nearest'], data['depth_m'], atol=1e-12, rtol=0)
        projected = self.run/'projected'/(key+'.npz')
        np.savez_compressed(projected, time_s=data['time_s'], **points,
            vertices=weights['vertices'], weights=weights['weights'], nearest_index=weights['nearest'],
            nearest_distance_m=weights['nearest_distance_m'])
        info.update(projected_path=str(projected), projected_sha256=sha(projected))
        write_json(marker_path, info); self.registry[key] = info
        out = dict(time_s=data['time_s'], **points, info=info)
        self.projections[key] = out
        return out

    def forecasts(self, meta, mesh, step, base):
        raw = load(base['info']['raw_path']); out = {}
        for release in (180, 190, 210, 220):
            i = int(np.flatnonzero(raw['time_s'] == release)[0])
            restart = dict(state={k: raw[k][i] for k in ('stage', 'xmomentum', 'ymomentum')},
                source_kind='independent_nominal_base', source_path=base['info']['raw_path'],
                source_sha256=base['info']['raw_sha256'], state_time_s=release)
            forecast = self.get(meta, mesh, step, 'forecast', anchor=release, restart=restart)
            for method in ('nearest', 'linear'):
                np.testing.assert_allclose(forecast[method][0], base[method][i], atol=1e-8, rtol=0)
            out[release] = forecast
        return out


def variants():
    return [(f'n{n}_dt2_{m}', n, 2, m, False) for n in (10, 20, 40) for m in ('nearest', 'linear')]+[
        (f'n40_dt1_{m}', 40, 1, m, False) for m in ('nearest', 'linear')]+[
        (f'n40_dt1_decimated2_{m}', 40, 1, m, True) for m in ('nearest', 'linear')]


def view(field, method, decimate):
    ids = np.arange(0, len(field['time_s']), 2) if decimate else np.arange(len(field['time_s']))
    return field['time_s'][ids], field[method][ids]


def nominal_lags(times, delta, center):
    out = {}
    for target in TARGETS:
        if (target//10 < 5) != (center < 50): continue
        for source in SOURCES:
            if (source//10 < 5) != (center < 50): continue
            # Start is inferred from the experiment condition, offline prior only.
            sm = marker(times, delta[:, source], 0); tm = marker(times, delta[:, target], 0)
            out[source, target] = None if sm is None or tm is None else tm-sm
    return out


def analyze_family(run, oldrun, parent, fields, meta, layouts):
    family = meta['family_id']; result = run/(family+'.json')
    dataset = {}
    for mesh, step in ((10, 2), (20, 2), (40, 2), (40, 1)):
        base = fields.get(meta, mesh, step, 'truth_base')
        nominal = fields.get(meta, mesh, step, 'nominal_base')
        forecasts = fields.forecasts(meta, mesh, step, nominal)
        dataset[mesh, step] = dict(base=base, nominal=nominal, forecasts=forecasts, pulses={}, probes={})
        for anchor in (180, 210):
            for center in (20, 70):
                pulse = fields.get(meta, mesh, step, 'truth_pulse', anchor, center)
                probe = fields.get(meta, mesh, step, 'nominal_pulse', anchor, center)
                before = pulse['time_s'] <= anchor
                for method in ('nearest', 'linear'):
                    np.testing.assert_allclose(pulse[method][before], base[method][before], atol=1e-8, rtol=0)
                    np.testing.assert_allclose(probe[method][before], nominal[method][before], atol=1e-8, rtol=0)
                dataset[mesh, step]['pulses'][anchor, center] = pulse
                dataset[mesh, step]['probes'][anchor, center] = probe
    old_scores = load(oldrun/'families'/(family+'_task_scores.npz'))
    all_windows = []; rows = []; robust = []; differences = []; geometry = []; comparisons = []; dumps = {}
    ids = [v[0] for v in variants()]
    for anchor in (180, 210):
        cases = make_cases(layouts, anchor); windows = common_windows(cases, anchor)
        for center in (20, 70):
            legacy = center+5; oldkey = f'a{anchor}_y{legacy}'
            frozen = old_scores[oldkey+'_baseline_order']; mask = old_scores[oldkey+'_critical_mask']
            maxima_by_condition = {'E0': {}, 'E1': {}}; data_by_variant = {}
            for name, mesh, step, method, decimated in variants():
                ds = dataset[mesh, step]
                times, base = view(ds['base'], method, decimated)
                _, pulse = view(ds['pulses'][anchor, center], method, decimated)
                _, nominal = view(ds['nominal'], method, decimated)
                _, probe = view(ds['probes'][anchor, center], method, decimated)
                forecasts = {r: view(f, method, decimated) for r, f in ds['forecasts'].items()}
                chosen, native_mask = forecast_plans(forecasts, cases, anchor)
                score = score_cases(times, pulse, cases, anchor)
                gap, visible = opportunity_flags(times, base, pulse, cases, anchor, frozen, score)
                lag = nominal_lags(times, probe-nominal, center)
                data_by_variant[name] = dict(times=times, base=base, pulse=pulse, score=score, gap=gap,
                    visible=visible, lags=lag, matched_chosen=chosen, native_mask=native_mask)
                dumpkey = f'a{anchor}_y{center}_{name}'
                for k, v in [('frozen_gap', gap), ('visible', visible), ('counts', score['count']),
                             ('frozen_mask', mask), ('frozen_action', frozen), ('matched_action', chosen), ('native_mask', native_mask)]:
                    dumps[dumpkey+'_'+k] = v
                for condition, depth in (('E0', base), ('E1', pulse)):
                    evaluated = score if condition == 'E1' else score_cases(times, depth, cases, anchor)
                    for scope, actions, keep in [('frozen_actions_fixed_mask', frozen, mask),
                            ('matched_actions_fixed_mask', chosen, mask), ('matched_actions_native_mask', chosen, native_mask)]:
                        actual = evaluated['count'][np.arange(len(cases)), actions]
                        loss = evaluated['count'].max(axis=1)-actual
                        rows.append(dict(family=family, anchor_s=anchor, physical_center_y_m=center,
                            variant=name, condition=condition, scope=scope, retained_cases=int(keep.sum()),
                            available_cases=len(cases), gap_sum=int(loss[keep].sum()),
                            nominal_actual_sum=int(actual[keep].sum()), finite_oracle_sum=int(evaluated['count'][keep].max(axis=1).sum()),
                            visible_oracle_D0=int(visible[keep, :, 0].sum()) if condition == 'E1' and scope == 'frozen_actions_fixed_mask' else None,
                            visible_oracle_D10=int(visible[keep, :, 1].sum()) if condition == 'E1' and scope == 'frozen_actions_fixed_mask' else None))
                    maxima_by_condition[condition][name] = window_maxima(times, depth, windows)
            for condition, maxima in maxima_by_condition.items():
                matrix = np.stack([maxima[name] for name in ids])
                low, high = matrix.min(axis=0), matrix.max(axis=0)
                for j, window in enumerate(windows):
                    all_windows.append(dict(family=family, anchor_s=anchor, physical_center_y_m=center,
                        condition=condition, critical=bool(mask[window['case_index']]), **window,
                        lower_m=float(low[j]), upper_m=float(high[j]), spread_m=float(high[j]-low[j]),
                        label_disagrees=bool(low[j] <= .8 < high[j]),
                        **{n: float(maxima[n][j]) for n in ids}))
                for a, b, label in [('n10_dt2_nearest', 'n20_dt2_nearest', 'mesh10_20_nearest'),
                        ('n20_dt2_nearest', 'n40_dt2_nearest', 'mesh20_40_nearest'),
                        ('n10_dt2_linear', 'n20_dt2_linear', 'mesh10_20_linear'),
                        ('n20_dt2_linear', 'n40_dt2_linear', 'mesh20_40_linear'),
                        ('n40_dt2_linear', 'n40_dt1_linear', 'independent_output2_1_linear'),
                        ('n40_dt1_decimated2_linear', 'n40_dt1_linear', 'same_run_decimation2_1_linear'),
                        ('n40_dt2_linear', 'n40_dt1_decimated2_linear', 'integration_with_output_times_linear'),
                        ('n40_dt2_nearest', 'n40_dt2_linear', 'spatial_readout_same_mesh40')]:
                    aa = maxima[a]; bb = maxima[b]
                    comparisons.append(dict(family=family, anchor_s=anchor, physical_center_y_m=center,
                        condition=condition, comparison=label, windows=len(windows),
                        max_window_change_m=float(np.max(abs(aa-bb))),
                        mean_window_change_m=float(np.mean(abs(aa-bb))),
                        label_changes=int(np.count_nonzero((aa <= .8) != (bb <= .8))),
                        critical_label_changes=int(sum(mask[w['case_index']] and ((aa[j] <= .8) != (bb[j] <= .8)) for j, w in enumerate(windows)))))
            # Common physical inputs only; no old shifted mesh20 in this ensemble.
            gaps = np.stack([data_by_variant[n]['gap'] for n in ids])
            visibility = np.stack([data_by_variant[n]['visible'] for n in ids])
            for j, c in enumerate(cases):
                if not mask[j]: continue
                robust.append(dict(family=family, anchor_s=anchor, physical_center_y_m=center,
                    layout=c['layout'], now_s=c['now_s'], frozen_order=int(frozen[j]),
                    minimum_alternative_minus_baseline=int(gaps[:, j].min()),
                    maximum_alternative_minus_baseline=int(gaps[:, j].max()),
                    all_variants_improve=bool(np.all(gaps[:, j] > 0)),
                    all_variants_visible_D0_by_phase=np.all(visibility[:, j, :, 0], axis=0).tolist(),
                    all_variants_visible_D10_by_phase=np.all(visibility[:, j, :, 1], axis=0).tolist()))
            for a, b, label in [('n10_dt2_linear', 'n20_dt2_linear', 'mesh10_20_linear'),
                    ('n20_dt2_linear', 'n40_dt2_linear', 'mesh20_40_linear'),
                    ('n40_dt2_linear', 'n40_dt1_linear', 'independent_output2_1_linear')]:
                aa = data_by_variant[a]; bb = data_by_variant[b]
                query = np.arange(0, 301, 2)
                xa = np.column_stack([np.interp(query, aa['times'], aa['pulse'][:, r]) for r in TARGETS])
                xb = np.column_stack([np.interp(query, bb['times'], bb['pulse'][:, r]) for r in TARGETS])
                changes = [abs(aa['lags'][edge]-bb['lags'][edge]) for edge in aa['lags']
                           if aa['lags'][edge] is not None and bb['lags'][edge] is not None]
                differences.append(dict(family=family, anchor_s=anchor, physical_center_y_m=center,
                    comparison=label, target_depth_max_change_m=float(np.max(abs(xa-xb))),
                    target_depth_mean_change_m=float(np.mean(abs(xa-xb))),
                    max_nominal_lag_change_s=max(changes) if changes else None,
                    missing_lag_edges=sum(aa['lags'][e] is None or bb['lags'][e] is None for e in aa['lags'])))
            if anchor == 210:
                # Same mesh/readout; isolate the old 5m forcing displacement.
                path = oldrun/'trajectories'/f'{family}_mesh20_truth_pulse_a210_y{legacy}.npz'
                expected = read(path.with_suffix('.complete.json'))['sha256']; assert sha(path) == expected
                old = load(path)
                old_scored = score_cases(old['time_s'], old['depth_m'], cases, anchor)
                old_base_path = oldrun/'trajectories'/f'{family}_mesh20_truth_base_a0_y25.npz'
                assert sha(old_base_path) == read(old_base_path.with_suffix('.complete.json'))['sha256']
                old_base = load(old_base_path)['depth_m']
                _, old_visible = opportunity_flags(old['time_s'], old_base, old['depth_m'], cases, anchor, frozen, old_scored)
                fixed = data_by_variant['n20_dt2_nearest']
                geometry.append(dict(family=family, anchor_s=anchor, old_requested_center_y_m=legacy,
                    old_physical_interval_y_m=[center-5, center+15], fixed_interval_y_m=[center-10, center+10],
                    readout='nearest', mesh=20, old_visible_D0=int(old_visible[mask, :, 0].sum()),
                    fixed_visible_D0=int(fixed['visible'][mask, :, 0].sum()),
                    max_target_depth_change_m=float(np.max(abs(old['depth_m'][:, TARGETS]-fixed['pulse'][:, TARGETS]))),
                    order_counts_changed=int(np.count_nonzero(old_scored['count'] != fixed['score']['count']))))
    np.savez_compressed(run/(family+'_scores.npz'), **dumps)
    out = dict(family=family, opportunities=rows, robust_cases=robust, windows=all_windows,
               field_differences=differences, window_comparisons=comparisons, boundary_geometry_control=geometry)
    write_json(result, out)
    print('FAMILY_DONE '+family, flush=True)
    return out


def summarize(run, results, registry):
    flat = {key: [r for f in results for r in f[key]] for key in
        ('opportunities', 'robust_cases', 'windows', 'field_differences', 'window_comparisons', 'boundary_geometry_control')}
    for k, rows in flat.items(): csv_write(run/(k+'.csv'), rows)
    per_variant = []
    for name, *_ in variants():
        for scope in ('frozen_actions_fixed_mask', 'matched_actions_fixed_mask', 'matched_actions_native_mask'):
            rows = [r for r in flat['opportunities'] if r['variant'] == name and r['scope'] == scope]
            supported = defaultdict(lambda: [0, 0])
            for r in rows:
                supported[r['family']][0] += r['gap_sum']; supported[r['family']][1] += r['retained_cases']
            values = [g/n*100 for g, n in supported.values() if n]
            per_variant.append(dict(variant=name, scope=scope, retained=sum(r['retained_cases'] for r in rows),
                headroom_family_macro_per100=float(np.mean(values)) if values else None,
                visible_D0=sum(r['visible_oracle_D0'] or 0 for r in rows),
                visible_D10=sum(r['visible_oracle_D10'] or 0 for r in rows)))
    compare = []
    for label in sorted({r['comparison'] for r in flat['window_comparisons']}):
        rows = [r for r in flat['window_comparisons'] if r['comparison'] == label]
        fields = [r for r in flat['field_differences'] if r['comparison'] == label]
        compare.append(dict(comparison=label, windows=sum(r['windows'] for r in rows),
            max_window_change_m=max(r['max_window_change_m'] for r in rows),
            label_changes=sum(r['label_changes'] for r in rows),
            critical_label_changes=sum(r['critical_label_changes'] for r in rows),
            max_target_depth_change_m=max((r['target_depth_max_change_m'] for r in fields), default=None),
            max_nominal_lag_change_s=max((r['max_nominal_lag_change_s'] for r in fields if r['max_nominal_lag_change_s'] is not None), default=None)))
    spatial = next(r for r in compare if r['comparison'] == 'mesh20_40_linear')
    temporal = next(r for r in compare if r['comparison'] == 'independent_output2_1_linear')
    checks = dict(target_depth_change_le_5mm=spatial['max_target_depth_change_m'] <= .005,
        nominal_lag_change_le_2s=spatial['max_nominal_lag_change_s'] <= 2,
        common_window_labels_unchanged=spatial['label_changes'] == 0,
        independent_output_window_change_le_1mm=temporal['max_window_change_m'] <= .001)
    out = dict(independent_training_families=len(results), variants=per_variant, comparisons=compare,
        robust_critical_cases=len(flat['robust_cases']),
        robust_alternative_improvement_cases=sum(r['all_variants_improve'] for r in flat['robust_cases']),
        all_variants_visible_D0=sum(sum(r['all_variants_visible_D0_by_phase']) for r in flat['robust_cases']),
        all_variants_visible_D10=sum(sum(r['all_variants_visible_D10_by_phase']) for r in flat['robust_cases']),
        common_windows=len(flat['windows']),
        common_window_labels_disagree=sum(r['label_disagrees'] for r in flat['windows']),
        critical_common_window_labels_disagree=sum(r['label_disagrees'] and r['critical'] for r in flat['windows']),
        geometry_old_visible_D0=sum(r['old_visible_D0'] for r in flat['boundary_geometry_control']),
        geometry_fixed_visible_D0=sum(r['fixed_visible_D0'] for r in flat['boundary_geometry_control']),
        reused_fields=sum(r['reused'] for r in registry.values()), new_field_integrations=sum(not r['reused'] for r in registry.values()),
        engineering_checks=checks, engineering_consistency_passed=all(checks.values()),
        old_confirmation_labels_read=0, learned_models=0,
        interpretation='Selected training diagnostics; finite numerical ensemble, no certified error bound or task efficacy claim.')
    write_json(run/'summary.json', out)
    write_json(run/'field_registry.json', registry)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--alignment-run', type=Path, required=True)
    p.add_argument('--run', type=Path); args = p.parse_args()
    oldrun = args.alignment_run.resolve(); parent = Path(read(oldrun/'config.json')['parent'])
    design = read(ROOT/'configs/numerical_stability.design.v1.json')
    run = args.run.resolve() if args.run else new_run('numerical_stability_fixed_footprint',
        dict(design=design, alignment_run=str(oldrun), source_family_parent=str(parent)))
    if args.run:
        for rel, checksum in read(run/'provenance.json')['files'].items(): assert sha(ROOT/rel) == checksum, rel
    layout_map = {}
    for c in task_table(read(parent/'geometry.json')): layout_map.setdefault(c['layout'], c)
    layouts = list(layout_map.values()); results = []; fields = Fields(run, oldrun)
    footprints = []
    for n in (10, 20, 40):
        meta0 = read(parent/'families'/(design['selected_training_families'][0]+'.json'))
        for legacy in (25, 75):
            olddomain = hydraulic_domain(meta0['params'], n, factor=1, pulse_amplitude=.1, pulse_center=legacy)
            domain = fixed_domain(meta0['params'], n, 1, legacy-5)
            footprints.append(dict(mesh=n, old_requested_center_y_m=legacy,
                old_interval_y_m=boundary_footprint(olddomain), fixed_interval_y_m=boundary_footprint(domain)))
    write_json(run/'boundary_footprints.json', footprints)
    started = time.perf_counter(); sources = {}
    for family in design['selected_training_families']:
        meta_path = parent/'families'/(family+'.json'); meta = read(meta_path); assert meta['split'] == 'train'
        complete = read(parent/'families'/(family+'.complete.json'))
        for filename, checksum in complete['sha256'].items():
            source_path = parent/'families'/filename; assert sha(source_path) == checksum
            sources[str(source_path)] = checksum
        results.append(analyze_family(run, oldrun, parent, fields, meta, layouts))
    summary = summarize(run, results, fields.registry)
    write_json(run/'source_family_hashes.json', sources)
    write_json(run/'completion.json', dict(completed=True, training_families=4,
        numerical_engineering_gate_passed=summary['engineering_consistency_passed'], seconds=time.perf_counter()-started))
    write_json(ROOT/'results/latest_numerical_stability.json', dict(run=str(run), completed=True, report_pending=True))
    print('RUN '+str(run), flush=True)


if __name__ == '__main__': main()
