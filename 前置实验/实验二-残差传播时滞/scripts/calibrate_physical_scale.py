"""Preregistered training-only screen of physical propagation/task scales."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from diagnose_numerical_stability import read, sha, load, csv_write
from hyra_validation.artifacts import new_run, write_json
from hyra_validation.adapters.anuga import simulate
from hyra_validation.scaled_hydraulics import scaled_domain, sampling_xy
from hyra_validation.numerical_stability import (point_weights, read_points, boundary_footprint,
    score_cases, forecast_plans, common_windows, window_maxima)
from hyra_validation.alignment import make_cases, marker
from hyra_validation.lead import SOURCES, TARGETS
from hyra_validation.local_service import task_table


class ScaleFields:
    def __init__(self, run, design):
        self.run, self.design = run, design
        self.registry, self.cache, self.weights = {}, {}, {}
        (run/'raw').mkdir(exist_ok=True); (run/'projected').mkdir(exist_ok=True)

    def get(self, meta, length, kind, anchor=0, center=0, amplitude=0., restart=None):
        mesh = self.design['mesh']; step = self.design['output_interval_s']
        key = f"{meta['family_id']}_L{length}_n{mesh}_{kind}_a{anchor}_y{center}_mm{round(amplitude*1000)}"
        if key in self.cache: return self.cache[key]
        marker_path = self.run/'raw'/(key+'.json')
        if marker_path.exists():
            info = read(marker_path)
            assert sha(info['raw_path']) == info['raw_sha256']
            assert sha(info['projected_path']) == info['projected_sha256']
        else:
            factor = 1. if kind.startswith('truth') else meta['nominal_specifications'][0]['factor']
            domain = scaled_domain(meta['params'], mesh, length, factor=factor,
                initial=restart['state'] if restart else None, start=anchor if restart else 0.,
                amplitude_m=amplitude, anchor_s=anchor, center_y_m=center)
            end = anchor+self.design['forecast_horizon_s'] if restart else self.design['final_time_s']
            started = time.perf_counter()
            times, nearest, states, _ = simulate(domain, end, sampling_xy(length), yieldstep=step)
            coordinates = domain.get_centroid_coordinates()
            bed = domain.quantities['elevation'].centroid_values.copy()
            weights = self.weights.get((length, mesh))
            if weights is None:
                weights = self.weights[length, mesh] = point_weights(coordinates, sampling_xy(length))
            stage = np.asarray([s['stage'] for s in states])
            projection = read_points(stage, bed, weights)
            np.testing.assert_allclose(projection['nearest'], nearest, atol=1e-12, rtol=0)
            rawpath = self.run/'raw'/(key+'.npz')
            np.savez_compressed(rawpath, time_s=times, centroid_xy_m=coordinates, bed_m=bed,
                **{name: np.asarray([s[name] for s in states]) for name in states[0]})
            projected = self.run/'projected'/(key+'.npz')
            np.savez_compressed(projected, time_s=times, **projection,
                vertices=weights['vertices'], weights=weights['weights'],
                nearest_index=weights['nearest'], nearest_distance_m=weights['nearest_distance_m'])
            info = dict(key=key, family=meta['family_id'], split=meta['split'], length_x_m=length,
                mesh=mesh, kind=kind, anchor_s=anchor, center_y_m=center, pulse_amplitude_m=amplitude,
                output_interval_s=step, physical_interval_y_m=boundary_footprint(domain), factor=factor,
                raw_path=str(rawpath), raw_sha256=sha(rawpath), projected_path=str(projected),
                projected_sha256=sha(projected), seconds=time.perf_counter()-started,
                restart={k:v for k,v in restart.items() if k != 'state'} if restart else None)
            write_json(marker_path, info)
            print('FIELD '+key, flush=True)
        data = load(info['projected_path'])
        out = dict(time_s=data['time_s'], linear=data['linear'], nearest=data['nearest'], info=info)
        self.registry[key] = info; self.cache[key] = out
        assert len(self.registry) <= self.design['maximum_new_integrations']
        write_json(self.run/'field_registry.json', self.registry)
        write_json(self.run/'progress.json', dict(completed=False, registered_fields=len(self.registry), last_field=key))
        return out

    def forecasts(self, meta, length, base):
        raw = load(base['info']['raw_path']); result = {}
        grid = self.design['forecast_releases_s']
        for release in range(grid['start'], grid['stop_inclusive']+1, grid['step']):
            i = int(np.flatnonzero(raw['time_s'] == release)[0])
            restart = dict(state={k:raw[k][i] for k in ('stage','xmomentum','ymomentum')},
                source_kind='same_mesh_independent_nominal_base', source_path=base['info']['raw_path'],
                source_sha256=base['info']['raw_sha256'], state_time_s=release)
            f = self.get(meta, length, 'nominal_forecast', anchor=release, restart=restart)
            np.testing.assert_allclose(f['linear'][0], base['linear'][i], atol=1e-8, rtol=0)
            result[release] = (f['time_s'], f['linear'])
        return result


def source_visibility(times, delta, now, phase, delay):
    valid = np.arange(120+5*phase, times[-1]+1e-9, 5.)
    age = now-valid; ids = []
    for b in range(6):
        eligible = np.flatnonzero((valid+delay <= now) & (age >= b*10) & (age < b*10+10))
        if len(eligible): ids.append(int(eligible[-1]))
    if not ids: return False
    return any(np.max(abs(np.interp(valid[ids], times, delta[:, source]))) >= .001 for source in SOURCES)


def analyze_context(run, design, fields, meta, length, layouts):
    base = fields.get(meta, length, 'truth_base')
    nominal = fields.get(meta, length, 'nominal_base')
    forecasts = fields.forecasts(meta, length, nominal)
    rows, windows_out, edges, action_rows = [], [], [], []
    for anchor in design['anchors_s']:
        cases = make_cases(layouts, anchor, design['decision_offsets_s'])
        chosen, critical = forecast_plans(forecasts, cases, anchor)
        windows = common_windows(cases, anchor)
        base_scores = score_cases(base['time_s'], base['linear'], cases, anchor)
        ix = np.arange(len(cases))
        base_gap = base_scores['count'][ix, 1-chosen]-base_scores['count'][ix, chosen]
        for center in design['physical_centers_y_m']:
            probe = fields.get(meta, length, 'nominal_probe', anchor, center, design['nominal_probe_amplitude_m'])
            before = probe['time_s'] <= anchor
            np.testing.assert_allclose(probe['linear'][before], nominal['linear'][before], atol=1e-8, rtol=0)
            delta = probe['linear']-nominal['linear']
            for target in TARGETS:
                tm = marker(probe['time_s'], delta[:, target], anchor)
                for source in SOURCES:
                    sm = marker(probe['time_s'], delta[:, source], anchor)
                    edges.append(dict(family=meta['family_id'], length_x_m=length, anchor_s=anchor,
                        center_y_m=center, source=source, target=target,
                        source_marker_s=sm, target_marker_s=tm,
                        signed_nominal_lag_s=None if sm is None or tm is None else tm-sm,
                        nominal_source_peak_m=float(np.max(abs(delta[:, source]))),
                        nominal_target_peak_m=float(np.max(abs(delta[:, target])))))
            for amplitude in design['truth_pulse_amplitudes_m']:
                pulse = fields.get(meta, length, 'truth_pulse', anchor, center, amplitude)
                np.testing.assert_allclose(pulse['linear'][before], base['linear'][before], atol=1e-8, rtol=0)
                pulse_scores = score_cases(pulse['time_s'], pulse['linear'], cases, anchor)
                pulse_gap = pulse_scores['count'][ix, 1-chosen]-pulse_scores['count'][ix, chosen]
                pulse_specific = (pulse_gap > 0) & (base_gap <= 0)
                flags = np.zeros((len(cases), 4, 2), dtype=bool)
                for j, case in enumerate(cases):
                    if not pulse_specific[j]: continue
                    for p, phase in enumerate(design['sample_phase_fractions']):
                        for d, delay in enumerate(design['observation_delays_s']):
                            flags[j, p, d] = source_visibility(pulse['time_s'], pulse['linear']-base['linear'],
                                case['now_s'], phase, delay)
                for population, keep in [('general', np.ones(len(cases), bool)), ('critical', critical)]:
                    for condition, scored in [('E0', base_scores), ('E1', pulse_scores)]:
                        actual = scored['count'][ix, chosen]; oracle = scored['count'].max(axis=1)
                        rows.append(dict(family=meta['family_id'], length_x_m=length,
                            amplitude_m=amplitude, anchor_s=anchor, center_y_m=center, population=population,
                            condition=condition, cases=int(keep.sum()), available_cases=len(cases),
                            nominal_sum=int(actual[keep].sum()), oracle_sum=int(oracle[keep].sum()),
                            headroom_sum=int((oracle-actual)[keep].sum()),
                            pulse_specific_gain_sum=int(np.maximum(pulse_gap, 0)[keep & pulse_specific].sum()) if condition == 'E1' else 0,
                            visible_D0_sum=int((flags[:, :, 0]*np.maximum(pulse_gap, 0)[:, None])[keep].sum()) if condition == 'E1' else 0,
                            visible_D10_sum=int((flags[:, :, 1]*np.maximum(pulse_gap, 0)[:, None])[keep].sum()) if condition == 'E1' else 0))
                for condition, field in [('E0', base), ('E1', pulse)]:
                    maxima = window_maxima(field['time_s'], field['linear'], windows)
                    for window, maximum in zip(windows, maxima):
                        windows_out.append(dict(family=meta['family_id'], length_x_m=length, amplitude_m=amplitude,
                            anchor_s=anchor, center_y_m=center, condition=condition,
                            critical=bool(critical[window['case_index']]), **window, maximum_m=float(maximum)))
                for j, case in enumerate(cases):
                    action_rows.append(dict(family=meta['family_id'], length_x_m=length, amplitude_m=amplitude,
                        anchor_s=anchor, center_y_m=center, layout=case['layout'], now_s=case['now_s'],
                        critical=bool(critical[j]), nominal_action=int(chosen[j]),
                        E0_counts=base_scores['count'][j].tolist(), E1_counts=pulse_scores['count'][j].tolist(),
                        E0_alternative_gap=int(base_gap[j]), E1_alternative_gap=int(pulse_gap[j]),
                        pulse_specific=bool(pulse_specific[j]), visible_D0_by_phase=flags[j, :, 0].tolist(),
                        visible_D10_by_phase=flags[j, :, 1].tolist()))
    result = dict(rows=rows, windows=windows_out, propagation=edges, actions=action_rows)
    write_json(run/(meta['family_id']+f'_L{length}.json'), result)
    print('CONTEXT_DONE '+meta['family_id']+f' L{length}', flush=True)
    return result


def summarize(design, results):
    rows = [r for result in results for r in result['rows']]
    summary = []
    for length in design['domain_lengths_x_m']:
        for amplitude in design['truth_pulse_amplitudes_m']:
            for population in ('general', 'critical'):
                values = []
                for family in design['family_ids']:
                    subset = [r for r in rows if r['length_x_m'] == length and r['amplitude_m'] == amplitude
                              and r['population'] == population and r['family'] == family]
                    count = sum(r['cases'] for r in subset)
                    # E0/E1 both represented; visibility has four phase repeats, never independent families.
                    values.append(dict(family=family, cases=count, unique_design_cases=count//2,
                        nonempty_anchor_center_cells=sum(r['cases'] > 0 for r in subset if r['condition'] == 'E0'),
                        headroom_per100=sum(r['headroom_sum'] for r in subset)/count*100 if count else None,
                        visible_D0_per100=sum(r['visible_D0_sum'] for r in subset)/(count*4)*100 if count else None,
                        visible_D10_per100=sum(r['visible_D10_sum'] for r in subset)/(count*4)*100 if count else None))
                nonempty = [r for r in values if r['cases']]
                summary.append(dict(length_x_m=length, amplitude_m=amplitude, population=population,
                    families=values, nonempty_families=len(nonempty),
                    headroom_family_macro_per100=float(np.mean([r['headroom_per100'] for r in nonempty])) if nonempty else None,
                    visible_D0_family_macro_per100=float(np.mean([r['visible_D0_per100'] for r in nonempty])) if nonempty else None,
                    visible_D10_family_macro_per100=float(np.mean([r['visible_D10_per100'] for r in nonempty])) if nonempty else None))
    rule = design['training_screen_selection_rule']; selected = None
    for row in summary:
        if row['population'] != 'critical': continue
        passed = all(r['unique_design_cases'] >= 10 and r['nonempty_anchor_center_cells'] >= 2 for r in row['families'])
        passed &= (row['headroom_family_macro_per100'] or 0) >= rule['minimum_family_macro_nominal_headroom_per100']
        passed &= (row['visible_D0_family_macro_per100'] or 0) >= rule['minimum_family_macro_visible_pulse_gain_D0_per100']
        row['training_screen_passed'] = bool(passed)
        if passed and selected is None: selected = dict(length_x_m=row['length_x_m'], amplitude_m=row['amplitude_m'])
    return dict(training_family_count=len(design['family_ids']), conditions=summary, selected_for_next_stage=selected,
        interpretation='Training-only necessary optimistic screen vs nominal. Not B-star headroom, online detectability, numerical convergence of new geometry or policy efficacy.',
        old_confirmation_labels_read=0, learned_models=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path)
    args = parser.parse_args(); design = read(ROOT/'configs/physical_scale_calibration.design.v1.json')
    parent = Path(design['source_family_parent'])
    run = args.run.resolve() if args.run else new_run('physical_scale_training_screen', dict(design=design))
    if args.run:
        for rel, checksum in read(run/'provenance.json')['files'].items(): assert sha(ROOT/rel) == checksum, rel
    print('RUN '+str(run), flush=True)
    layout_map = {}
    for case in task_table(read(parent/'geometry.json')): layout_map.setdefault(case['layout'], case)
    fields = ScaleFields(run, design); results = []; source_hashes = {str(parent/'geometry.json'): sha(parent/'geometry.json')}
    started = time.perf_counter()
    for family in design['family_ids']:
        path = parent/'families'/(family+'.json'); meta = read(path); assert meta['split'] == 'train'
        source_hashes[str(path)] = sha(path)
        for length in design['domain_lengths_x_m']:
            results.append(analyze_context(run, design, fields, meta, length, list(layout_map.values())))
    for field, filename in [('rows','opportunity_table'),('propagation','propagation'),('actions','actions'),('windows','windows')]:
        csv_write(run/(filename+'.csv'), [r for result in results for r in result[field]])
    summary = summarize(design, results)
    summary.update(new_integrations=len(fields.registry), seconds=time.perf_counter()-started)
    write_json(run/'summary.json', summary); write_json(run/'source_hashes.json', source_hashes)
    write_json(run/'completion.json', dict(completed=True, contexts=len(results), new_integrations=len(fields.registry)))
    write_json(run/'progress.json', dict(completed=True, registered_fields=len(fields.registry)))
    write_json(ROOT/'results/latest_physical_scale_calibration.json', dict(run=str(run), completed=True,
               audit_pending=True, selected_for_next_stage=summary['selected_for_next_stage']))
    print('SUMMARY '+str(summary), flush=True)


if __name__ == '__main__': main()
