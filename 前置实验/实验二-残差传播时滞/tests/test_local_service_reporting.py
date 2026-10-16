import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('local_service_runner_test', ROOT/'scripts/run_local_service_experiment.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture_rows(identical=False):
    names = ['nominal', 'forecast', 'filter', 'mae', 'task']
    selected = {g: {'candidate': c} for g, c in zip(['A0', 'A1', 'A2', 'A3', 'A4', 'A5'],
                                                  ['nominal', 'forecast', 'filter', 'task' if identical else 'mae', 'task', 'task'])}
    rows = []
    for i in range(48):
        values = np.array([1, 1.001, 1.002, 1.003, 1.025])+i*.0001
        entry = dict(yield_=None, **{'yield': values, 'order_sensitive_fraction': .1})
        rows.append(dict(family_id=f'f{i}', candidate_names=names, aggregate={'yield': values},
                         pulse_observation_effect=np.zeros(5), strata={
                             'E0/M0/noise0/delay0': entry,
                             'E0/M1/noise0.02/delay10': entry}))
    return rows, selected


def test_confirmation_bootstrap_uses_paired_families_and_primary_levels(tmp_path, monkeypatch):
    (tmp_path/'results').mkdir(); monkeypatch.setattr(runner, 'ROOT', tmp_path)
    rows, selected = fixture_rows()
    config = {'master_seed': 5, 'statistics': {'bootstrap_family_resamples': 1000}}
    runner.produce_confirmation_report(tmp_path, config, rows, selected)
    output = runner.read(tmp_path/'confirmation_summary.json')
    primary = [c for c in output['comparisons'] if c['primary']]
    assert len(primary) == 2 and all(c['interval_level'] == .975 for c in primary)
    np.testing.assert_allclose(primary[0]['interval'], [.022, .022], atol=1e-12)
    assert output['full_core_claim_supported'] and output['engineering_expansion_gate_passed']


def test_identical_selected_models_cannot_support_selector_superiority(tmp_path, monkeypatch):
    (tmp_path/'results').mkdir(); monkeypatch.setattr(runner, 'ROOT', tmp_path)
    rows, selected = fixture_rows(identical=True)
    runner.produce_confirmation_report(tmp_path, {'master_seed': 5, 'statistics': {'bootstrap_family_resamples': 1000}}, rows, selected)
    output = runner.read(tmp_path/'confirmation_summary.json')
    comparison = next(c for c in output['comparisons'] if c['baseline'] == 'A3')
    assert comparison['mean_gain_services_per_case'] == 0 and comparison['interval'] == [0, 0]
    assert not comparison['hypothesis_supported'] and not output['full_core_claim_supported']
