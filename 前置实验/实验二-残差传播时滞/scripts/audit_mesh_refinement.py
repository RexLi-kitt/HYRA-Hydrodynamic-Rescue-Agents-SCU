"""Recompute fixed-point readouts and every window with an independent scalar implementation."""
import argparse
from pathlib import Path
import sys

import numpy as np

from audit_pulse_service_alignment import read, digest, load, maximum

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    config = read(run/'config.json'); parent = Path(config['parent_run'])
    assert read(run/'completion.json')['completed']
    for rel, checksum in read(run/'provenance.json')['files'].items():
        assert digest(ROOT/rel) == checksum, rel
    for path, checksum in read(run/'source_family_hashes.json').items(): assert digest(path) == checksum
    registry = read(run/'field_registry.json')
    previous_registry = read(parent/'field_registry.json')
    counters = dict(raw_fields=0, point_projection_scalar_values=0, nominal_restarts=0,
                    scalar_window_evaluations=0, covered_parent_disagreements=0)
    fixed_xy = np.array([(5+10*x, 5+10*y) for y in range(10) for x in range(10)])
    cache = {}
    def field(family, mesh, kind, anchor=0, center=0):
        key = f'{family}_n{mesh}_dt2_{kind}_a{anchor}_y{center}'
        if key not in cache:
            info = (registry if mesh == 80 else previous_registry)[key]
            assert digest(info['projected_path']) == info['projected_sha256']
            cache[key] = load(info['projected_path'])
        return cache[key]
    for key, info in registry.items():
        assert digest(info['raw_path']) == info['raw_sha256']
        assert digest(info['projected_path']) == info['projected_sha256']
        raw, projection = load(info['raw_path']), load(info['projected_path'])
        v, w = projection['vertices'], projection['weights']
        assert np.min(w) >= 0
        np.testing.assert_allclose(w.sum(axis=1), 1, atol=1e-12)
        np.testing.assert_allclose((raw['centroid_xy_m'][v]*w[:, :, None]).sum(axis=1), fixed_xy, atol=1e-10)
        depth = np.maximum(raw['stage']-raw['bed_m'], 0)
        manual = sum(depth[:, v[:, j]]*w[:, j] for j in range(3))
        np.testing.assert_allclose(manual, projection['linear'], atol=1e-12, rtol=0)
        np.testing.assert_allclose(depth[:, projection['nearest_index']], projection['nearest'], atol=1e-12, rtol=0)
        counters['raw_fields'] += 1; counters['point_projection_scalar_values'] += manual.size*2
        pulse_interval = info['physical_interval_y_m']
        if pulse_interval is not None: assert pulse_interval in [[10., 30.], [60., 80.]]
        if info.get('restart'):
            restart = info['restart']
            assert restart['source_kind'] == 'independent_nominal_base'
            assert 'nominal_base' in Path(restart['source_path']).name
            assert digest(restart['source_path']) == restart['source_sha256']
            base = load(restart['source_path']); i = np.flatnonzero(base['time_s'] == restart['state_time_s'])[0]
            for quantity in ('stage', 'xmomentum', 'ymomentum'):
                np.testing.assert_allclose(raw[quantity][0], base[quantity][i], atol=1e-12, rtol=0)
            counters['nominal_restarts'] += 1
    windows = []
    for family in config['design']['families']:
        assert read(Path(config['source_family_parent'])/'families'/(family+'.json'))['split'] == 'train'
        windows.extend(read(run/(family+'.json'))['windows'])
    changes = dict(linear=0, nearest=0); max_change = dict(linear=0., nearest=0.)
    for row in windows:
        values = {}
        for mesh in (40, 80):
            f = field(row['family'], mesh, 'truth_base' if row['condition'] == 'E0' else 'truth_pulse',
                      0 if row['condition'] == 'E0' else row['anchor_s'],
                      0 if row['condition'] == 'E0' else row['physical_center_y_m'])
            for method in ('nearest', 'linear'):
                value = maximum(f['time_s'], f[method], row['target'], row['begin_s'], row['end_s'])
                assert abs(value-row[f'n{mesh}_{method}_maximum_m']) < 1e-12
                values[mesh, method] = value; counters['scalar_window_evaluations'] += 1
        for method in changes:
            changes[method] += (values[40, method] <= .8) != (values[80, method] <= .8)
            max_change[method] = max(max_change[method], abs(values[40, method]-values[80, method]))
        lower = min(row['parent_lower_m'], values[80, 'linear'], values[80, 'nearest'])
        upper = max(row['parent_upper_m'], values[80, 'linear'], values[80, 'nearest'])
        assert abs(lower-row['lower_m']) < 1e-12 and abs(upper-row['upper_m']) < 1e-12
        assert row['label_disagrees'] == (lower <= .8 < upper)
        counters['covered_parent_disagreements'] += row['parent_label_disagrees']
    summary = read(run/'summary.json')
    assert len(windows) == 18432 and counters['covered_parent_disagreements'] == 563
    for method in changes:
        assert summary['comparisons'][method]['label_changes'] == changes[method]
        assert abs(summary['comparisons'][method]['max_window_change_m']-max_change[method]) < 1e-12
    assert summary['new_field_integrations'] == counters['raw_fields'] == 56
    assert counters['nominal_restarts'] == 16
    write_json(run/'independent_audit.json', dict(passed=True, **counters,
        scope='raw hashes, convex fixed-point projection, independent nominal restart, all window maxima and envelopes; not a model efficacy audit'))
    pointer = read(ROOT/'results/latest_numerical_refinement.json')
    assert Path(pointer['run']) == run
    pointer.update(audit_pending=False, independent_audit_passed=True)
    write_json(ROOT/'results/latest_numerical_refinement.json', pointer)
    print('AUDIT_PASSED '+str(counters))


if __name__ == '__main__': main()
