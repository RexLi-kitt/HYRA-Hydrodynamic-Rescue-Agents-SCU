"""Independent hydraulic families for the local-service protocol.

Reference trajectories are never used to initialize nominal forecasts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .adapters.anuga import simulate
from .lead import XY, hydraulic_domain
from .artifacts import write_json


SPLIT_IDS = {'train': 1, 'validation': 2, 'pilot': 3, 'confirmation': 4}


def boundary_level(t, params, factor, switch_s=None, next_factor=None):
    if switch_s is None:
        elapsed = factor*max(t-params['onset'], 0)
    else:
        elapsed = factor*max(min(t, switch_s)-params['onset'], 0)
        elapsed += next_factor*max(t-max(switch_s, params['onset']), 0)
    return .15+params['rate']*elapsed


def nominal_domain(params, mesh, factor, switch_s=None, next_factor=None, initial=None, start=0):
    import anuga
    domain = hydraulic_domain(params, mesh, factor=factor, initial=initial, start=start)
    if switch_s is not None:
        wall = anuga.Reflective_boundary(domain)
        left = anuga.Time_boundary(domain, lambda t: [boundary_level(t, params, factor, switch_s, next_factor), 0, 0])
        domain.set_boundary(dict(left=left, right=wall, top=wall, bottom=wall))
    return domain


def issue_domain(params, mesh, specification, release, initial):
    # A forecast before the switch persists the then-known input law.
    switch = specification.get('switch_s')
    if switch is not None and release < switch:
        switch = None
    return nominal_domain(params, mesh, specification['factor'], switch,
                          specification.get('next_factor') if switch is not None else None,
                          initial=initial, start=release)


def draw_family(config, split, index):
    rng = np.random.default_rng(np.random.SeedSequence([config['master_seed'], SPLIT_IDS[split], index]))
    limits = config['hydraulic_parameters_uniform']
    params = {name: float(rng.uniform(*limits[key])) for name, key in
              [('slope', 'slope'), ('ridge', 'ridge_m'), ('rate', 'rate_m_per_s'),
               ('onset', 'onset_s'), ('friction', 'friction')]}
    f = config['forecast']
    low = float(rng.uniform(*f['underestimate_factor_uniform']))
    high = float(rng.uniform(*f['overestimate_factor_uniform']))
    forward = index % 2 == 0
    specifications = [dict(mode='M0', factor=float(rng.uniform(*f['near_accurate_factor_uniform']))),
                      dict(mode='M1', factor=low), dict(mode='M2', factor=high),
                      dict(mode='M3', factor=low if forward else high,
                           next_factor=high if forward else low,
                           switch_s=f['switch_time_s_candidates'][index % 3])]
    return dict(family_id=f'local_v1_{split}_{index:03}', split=split, index=index,
                seed_identity=[SPLIT_IDS[split], index], params=params,
                pulse_start_s=int(rng.choice(config['pulse']['start_s_candidates'])),
                pulse_center_y_m=int(rng.choice(config['pulse']['center_y_m_candidates'])),
                nominal_specifications=specifications)


def generate_family(directory, config, split, index, mesh=10):
    """Persist one atomic family; resume only from a completed checked snapshot."""
    import anuga
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    description = draw_family(config, split, index)
    basename = description['family_id']
    marker = directory/f'{basename}.complete.json'
    if marker.exists():
        complete = json.loads(marker.read_text(encoding='utf-8'))
        for name, checksum in complete['sha256'].items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Existing family checksum mismatch: '+name)
        return complete
    data_file = directory/f'{basename}.npz'; raw_file = directory/f'{basename}.solver.npz'
    if data_file.exists() or raw_file.exists():
        raise FileExistsError('Incomplete family outputs exist; preserve and inspect before resuming')
    start = time.perf_counter(); arrays = {'xy_m': XY}; raw = {}
    params = description['params']
    for condition, amplitude in [('E0', 0), ('E1', config['pulse']['amplitude_m'])]:
        domain = hydraulic_domain(params, mesh, factor=1, pulse_amplitude=amplitude,
                                  pulse_start=description['pulse_start_s'],
                                  pulse_center=description['pulse_center_y_m'], plateau=config['pulse']['plateau_s'])
        times, depth, states, mapping = simulate(domain, 360, XY)
        arrays['time_s'] = times; arrays[f'{condition}_truth_depth_m'] = depth
        for key in states[0]: raw[f'{condition}_{key}'] = np.asarray([s[key] for s in states])
        if condition == 'E0':
            raw['centroid_xy_m'] = domain.get_centroid_coordinates()
            raw['bed_m'] = domain.quantities['elevation'].centroid_values.copy()
    np.testing.assert_allclose(arrays['E0_truth_depth_m'][times < description['pulse_start_s']],
                               arrays['E1_truth_depth_m'][times < description['pulse_start_s']], atol=1e-8)
    raw['time_s'] = times
    releases = np.arange(0, 301, 10, dtype=float)
    arrays['release_s'] = releases
    for spec in description['nominal_specifications']:
        domain = nominal_domain(params, mesh, spec['factor'], spec.get('switch_s'), spec.get('next_factor'))
        nominal_times, nominal_depth, states, _ = simulate(domain, 300, XY)
        mode = spec['mode']; forecasts = []; valid_times = []
        for release in releases:
            state_index = int(np.flatnonzero(nominal_times == release)[0])
            forecast = issue_domain(params, mesh, spec, release, states[state_index])
            ft, fd, _, _ = simulate(forecast, release+60, XY)
            np.testing.assert_allclose(fd[0], nominal_depth[state_index], atol=1e-8)
            forecasts.append(fd); valid_times.append(ft)
        arrays[f'{mode}_forecast_depth_m'] = np.asarray(forecasts)
        arrays['forecast_valid_s'] = np.asarray(valid_times)
        arrays[f'{mode}_nominal_depth_m'] = nominal_depth
        for key in states[0]: raw[f'{mode}_nominal_{key}'] = np.asarray([s[key] for s in states])
    raw['nominal_time_s'] = nominal_times
    # M3's pre-switch issues must reproduce the matching fixed-law predictor.
    spec = description['nominal_specifications'][3]
    match = 'M1' if spec['factor'] < 1 else 'M2'
    earlier = releases < spec['switch_s']
    np.testing.assert_allclose(arrays['M3_forecast_depth_m'][earlier], arrays[f'{match}_forecast_depth_m'][earlier], atol=1e-7)
    for key, value in arrays.items():
        assert np.isfinite(value).all(), key
        if 'depth' in key: assert np.min(value) >= 0, key
    np.savez_compressed(data_file, **arrays); np.savez_compressed(raw_file, **raw)
    description.update(solver_version=anuga.__version__, nmesh=mesh, mapping=mapping,
                       units={'time': 's', 'xy': 'm', 'depth': 'm'},
                       independent_nominal_no_truth_reset=True, switch_lookahead_audit_passed=True,
                       seconds=time.perf_counter()-start, solver_advances=130)
    metadata_file = directory/f'{basename}.json'
    write_json(metadata_file, description)
    complete = dict(family_id=basename, split=split, index=index, data_file=data_file.name,
                    metadata_file=metadata_file.name, seconds=description['seconds'],
                    sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [data_file, raw_file, metadata_file]})
    write_json(marker, complete)
    return complete


def load_family(directory, complete):
    directory = Path(directory)
    meta = json.loads((directory/complete['metadata_file']).read_text(encoding='utf-8'))
    with np.load(directory/complete['data_file'], allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return arrays, meta


def family_events(arrays, metadata):
    for condition in ('E0', 'E1'):
        for mode in ('M0', 'M1', 'M2', 'M3'):
            yield condition, mode, dict(time_s=arrays['time_s'], truth_depth_m=arrays[f'{condition}_truth_depth_m'],
                xy_m=arrays['xy_m'], release_s=arrays['release_s'], forecast_valid_s=arrays['forecast_valid_s'],
                forecast_depth_m=arrays[f'{mode}_forecast_depth_m'],
                metadata=dict(metadata, event_id=f"{metadata['family_id']}_{condition}_{mode}",
                              event_family_id=metadata['family_id'], condition=condition, mode=mode))
