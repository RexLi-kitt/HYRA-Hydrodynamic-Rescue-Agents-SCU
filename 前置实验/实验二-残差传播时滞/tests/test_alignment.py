import copy

import numpy as np
import pytest

from hyra_validation.alignment import (anchor_selection, causal_bin_ids, sampled_records,
    evaluate_trajectory, gained_targets, paired_response_available, interpolate_checked)
from hyra_validation.diagnostics import score_fixed_order


def layouts():
    return [dict(layout=0, origin=14, targets=[13, 15],
                 orders=[([13, 15], [2, 2]), ([15, 13], [2, 2])])]


def test_anchor_uses_full_sixty_seconds_and_no_later_issue_or_truth(simple_event):
    event = simple_event
    event['forecast_depth_m'][9, -1, 13] = .8
    cases, critical, _ = anchor_selection(event, layouts(), 90)
    assert critical.tolist() == [False, False, True]
    changed = copy.deepcopy(event)
    changed['truth_depth_m'][:] = 4
    changed['forecast_depth_m'][10:] = 4
    _, other, _ = anchor_selection(changed, layouts(), 90)
    np.testing.assert_array_equal(critical, other)
    assert cases[-1]['now_s']+48 == 150


def test_delay_excludes_newest_record_and_expired_history():
    valid = np.array([20., 40., 60., 80., 90., 95.])
    ids = causal_bin_ids(valid, 96, 10)
    assert set(ids[ids >= 0]) == {1, 2, 3}
    assert np.all(valid[ids[ids >= 0]]+10 <= 96)


def test_shared_noise_and_source_only_records(simple_event):
    event = simple_event; base = event['truth_depth_m']
    pulse = base.copy(); pulse[event['time_s'] >= 90, 10] += .01
    records = sampled_records(event, base, pulse, .25, .02, 2, [0, 90, 25, 1])
    assert records['E0_observed_m'].shape[1] == 6
    assert np.all((records['actual_lead_s'] >= 20) & (records['actual_lead_s'] < 30))
    assert not paired_response_available(records, 96, 10, [13])
    assert paired_response_available(records, 96, 0, [13])
    assert not paired_response_available(records, 96, 0, [63])
    assert np.max(abs((records['E1_observed_m']-records['E0_observed_m'])[:, 1:])) < 1e-14


def test_full_interval_failure_and_scalar_execution_agree(simple_event):
    event = simple_event; depth = event['truth_depth_m'].copy()
    # Failure after the first 6s check still consumes the full first 12s.
    depth[event['time_s'] == 112, 13] = .9
    cases, _, _ = anchor_selection(event, layouts(), 90)
    scores = evaluate_trajectory(event, depth, cases)
    for j, case in enumerate(cases):
        for o, (order, ds) in enumerate(case['orders']):
            actual = score_fixed_order(event['time_s'], depth, order, ds, case['now_s'], 6, 12, .8, 150)
            assert actual['service_yield'] == scores['count'][j, o]
            np.testing.assert_array_equal([r['start_s'] for r in actual['trace']],
                case['now_s']+scores['starts'][j, o])
    assert gained_targets(cases[0], scores, 0, 0) == [13]


def test_no_forecast_extrapolation():
    with pytest.raises(ValueError):
        interpolate_checked(np.array([0, 60]), np.zeros((2, 3)), np.array([62]))
