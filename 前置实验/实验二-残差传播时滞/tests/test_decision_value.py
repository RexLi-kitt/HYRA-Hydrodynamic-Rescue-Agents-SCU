import copy
import numpy as np
import pytest
from hyra_validation.data import observation
from hyra_validation.lead import MONITORS
from hyra_validation.decision_value import (scheduled_records, binned_history,
    feature_groups, issued_curve, shortest_paths, prefix_checkpoints,
    candidate_orders, max_duration, choose, order_scores, normalized_summary)


def graph():
    return {'23': [dict(source=s, travel_s=t, weight=1/3)
                   for s, t in [(10, 7), (20, 13), (30, 23)]]}


def test_dense_records_use_only_existing_past_versions(simple_event):
    records = scheduled_records(simple_event, 5, 10, .25)
    for row in records[20]:
        assert 20 <= row['actual_lead_s'] < 30
        assert row['prior_release_s'] in simple_event['release_s']
        assert row['prior_release_s'] <= row['valid_time_s']-20
        assert row['arrival_time_s'] == row['valid_time_s']+10
    old = scheduled_records(simple_event, 10)
    for row in old[20]:
        expected = observation(simple_event, 20, row['valid_time_s'], row['valid_time_s'],
                               row['observed_depth_m'], 20)
        assert row['residual_m'] == expected['residual_m']
        assert row['matched_forecast_id'] == expected['matched_forecast_id']


def test_bins_keep_fixed_span_and_respect_arrivals():
    records = {20: [dict(valid_time_s=t, arrival_time_s=t+5, residual_m=t)
                    for t in range(0, 81, 2)]}
    rows = binned_history(records, 20, 60)
    assert [r['valid_time_s'] for r in rows] == [54, 50, 40, 30, 20, 10]
    assert all(r['arrival_time_s'] <= 60 for r in rows)


def test_feature_groups_share_all_raw_inputs_and_kernel_support(simple_event):
    records = scheduled_records(simple_event, 5)
    for source in (10, 20, 30):
        for row in records[source]: row['residual_m'] = source/100
    ff = feature_groups(simple_event, records, graph(), 23, 60)
    for m in ('recent', 'physical', 'uniform', 'wrong'):
        np.testing.assert_array_equal(ff[m][:-9], ff['raw_full'][:-9])
        np.testing.assert_allclose(ff[m][-9::3], .2)
        np.testing.assert_array_equal(ff[m][-8::3], ff['recent'][-8::3])
        np.testing.assert_array_equal(ff[m][-7::3], ff['recent'][-7::3])
    assert not ff['raw_self'][24:96].any()


def test_future_truth_forecast_and_late_packets_do_not_change_features(simple_event):
    records = scheduled_records(simple_event, 5)
    before = feature_groups(simple_event, records, graph(), 23, 60)
    changed = copy.deepcopy(simple_event)
    changed['truth_depth_m'][changed['time_s'] > 60] = 999
    changed['forecast_depth_m'][changed['release_s'] > 60] = 999
    rr = copy.deepcopy(records)
    rr[20].append(dict(region_id=20, valid_time_s=55, arrival_time_s=70,
                      residual_m=999, observed_depth_m=999, prior_depth_m=0))
    after = feature_groups(changed, rr, graph(), 23, 60)
    for m in before: np.testing.assert_array_equal(before[m], after[m])


def test_observation_versions_do_not_use_later_publications(simple_event):
    old = scheduled_records(simple_event, 5)[20]
    changed = copy.deepcopy(simple_event)
    changed['forecast_depth_m'][changed['release_s'] > 0] = 999
    new = scheduled_records(changed, 5)[20]
    for a, b in zip(old, new):
        if a['valid_time_s'] < 30: assert a == b


def test_missing_versions_remain_missing(simple_event):
    simple_event['release_s'] += 1000
    rr = scheduled_records(simple_event, 5)
    assert all(r['residual_m'] is None for r in rr[20])
    assert binned_history(rr, 20, 60) == [None]*6


def test_forecast_never_extrapolates_or_uses_future_issue(simple_event):
    f, curve = issued_curve(simple_event, 66)
    assert simple_event['release_s'][f] == 60
    with pytest.raises(ValueError): issued_curve(simple_event, 66, [0, 55])


def test_no_diagonal_corner_cutting_and_prefix_is_reachable():
    blocked = np.zeros(100, dtype=bool); blocked[1] = True
    paths = {r: shortest_paths(blocked, r) for r in (0, 2, 22)}
    assert paths[0][11] == [0, 10, 11]
    checkpoints = prefix_checkpoints(paths, 0, [2, 22], 0, 24)
    assert checkpoints[0] == (24, 0)
    assert all(node != 2 for _, node in checkpoints)
    assert all(t == 24+6*k for k, (t, _) in enumerate(checkpoints))
    orders = candidate_orders(paths, 0, [2, 22])
    assert max_duration(orders) == max(sum(max(d, 1) for d in ds)*6+24 for _, ds in orders)


def test_same_origin_still_costs_one_move_and_truth_changes_no_tie_break(simple_event):
    orders = [([1, 2], [0, 1]), ([2, 1], [1, 1])]
    ss = order_scores(simple_event['time_s'], simple_event['truth_depth_m'], orders, 0)
    assert ss[0]['trace'][0]['start_s'] == 6
    assert choose(ss) == 0


def test_kernel_collapse_and_zero_missing_support():
    links = graph()['23']
    histories = [[dict(valid_time_s=55, residual_m=.01), dict(valid_time_s=40, residual_m=.03)]]*3
    assert normalized_summary(histories, links, 60, 30, 'physical') == normalized_summary(histories, links, 60, 30, 'recent')
    assert normalized_summary([[None]*6]*3, links, 60, 20, 'physical') == [0, 0, 0]
