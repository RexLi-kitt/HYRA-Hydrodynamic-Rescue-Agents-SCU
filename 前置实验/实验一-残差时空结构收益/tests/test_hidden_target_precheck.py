import copy
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hidden_target_precheck_test', ROOT/'scripts/precheck_hidden_target_physics.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_critical_task_selection_does_not_read_truth(simple_event):
    event = simple_event
    arrays = {k: event[k] for k in ('time_s', 'xy_m', 'release_s', 'forecast_valid_s')}
    arrays['E0_truth_depth_m'] = event['truth_depth_m'].copy()
    arrays['E1_truth_depth_m'] = event['truth_depth_m'].copy()
    for mode in ('M0', 'M1', 'M2', 'M3'):
        arrays[mode+'_forecast_depth_m'] = event['forecast_depth_m'].copy()
    meta = dict(family_id='fixture', split='train')
    parent = ROOT/'configs/local_service_regret.geometry.preview.v1.json'
    cases = module.all_geometry_cases(module.read(parent))[:4]
    original = module.opportunity(arrays, meta, cases)
    changed = copy.deepcopy(arrays)
    changed['E0_truth_depth_m'][:] = .9; changed['E1_truth_depth_m'][:] = 1
    altered = module.opportunity(changed, meta, cases)
    assert [r['retained_cases'] for r in original] == [r['retained_cases'] for r in altered]
    assert original[0]['nominal_mean_yield'] != altered[0]['nominal_mean_yield']


def test_long_horizon_kernel_collapse_and_time_varying_lag():
    edges = [dict(family='f', source=10, target=13, probe_start_s=90,
                  source_marker_s=95, target_marker_s=110, signed_response_lag_s=15, nonnegative_kernel_lag_s=15),
             dict(family='f', source=10, target=13, probe_start_s=180,
                  source_marker_s=185, target_marker_s=202, signed_response_lag_s=17, nonnegative_kernel_lag_s=17)]
    out = module.kernel_and_sampling(edges)
    assert out['probe_time_absolute_lag_change_max_s'] == 2
    for row in out['kernel']:
        if row['future_horizon_s'] == 30:
            assert row['exact_center_collapse_fraction'] == 1 and row['mean_weight_L1_distance'] == 0
        if row['future_horizon_s'] == 6: assert row['mean_weight_L1_distance'] > 0
    assert out['sampling_marker_bound'][0]['median_marker_lead_s'] >= out['sampling_marker_bound'][1]['median_marker_lead_s']


def test_extended_case_grid_is_geometry_only_and_has_full_time_support():
    preview = module.read(ROOT/'configs/local_service_regret.geometry.preview.v1.json')
    cases = module.all_geometry_cases(preview)
    assert len(cases) == 32*37
    assert sorted({r['now_s'] for r in cases}) == list(range(24, 241, 6))
