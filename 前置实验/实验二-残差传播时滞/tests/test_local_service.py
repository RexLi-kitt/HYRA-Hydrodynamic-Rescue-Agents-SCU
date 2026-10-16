import copy
import json
from pathlib import Path

import numpy as np

from hyra_validation.local_service import (
    load_design, task_table, schedules, build_batch, RidgeMoments, learned_predict,
    conventional_predictions, trend_predictions, score_orders, choose_order,
    select_models, evaluate_batch)
from hyra_validation.local_service_data import boundary_level, draw_family
from hyra_validation.decision_value import OFFSETS, order_scores, fit_reader, predict_reader
from hyra_validation.lead import TARGETS

ROOT = Path(__file__).resolve().parents[1]


def setup():
    config, preview = load_design(ROOT)
    return config, task_table(preview)


def test_delayed_future_and_new_issues_cannot_change_inputs(simple_event):
    config, _ = setup(); setting = next(s for s in schedules(config) if s['delay_s'] == 10 and s['phase'] == .25)
    first = build_batch(simple_event, setting, [1, 0], config['master_seed'])
    changed = copy.deepcopy(simple_event)
    changed['truth_depth_m'][changed['time_s'] > 50] += 10
    changed['forecast_depth_m'][changed['release_s'] > 50] += 10
    second = build_batch(changed, setting, [1, 0], config['master_seed'])
    rows = first['row_times'] <= 50
    for key in first['inputs']:
        np.testing.assert_array_equal(first['inputs'][key][rows], second['inputs'][key][rows])
    assert np.max(first['history'][first['row_times'] == 24]) == 0
    used = first['record_ids'][rows]; used = used[used >= 0]
    assert np.all(first['sensors']['actual_lead_s'][used] >= 20)


def test_streaming_weighted_ridge_matches_repeated_batch_fit():
    rng = np.random.default_rng(23)
    x = rng.normal(size=(60, 8)); x[:, -1] = 0
    y = rng.normal(size=(60, 26))
    moments = RidgeMoments(); moments.add(x[:30], y[:30], 2); moments.add(x[30:], y[30:])
    xx = np.r_[x[:30], x[:30], x[30:]]; yy = np.r_[y[:30], y[:30], y[30:]]
    model = moments.fit(.1); ordinary = fit_reader(xx, yy, .1)
    np.testing.assert_allclose(learned_predict(model, x), predict_reader(ordinary, x), atol=1e-10)


def test_conventional_no_observations_means_zero_and_count_fixed():
    config, _ = setup()
    predictions = conventional_predictions(np.zeros((4, 6, 4)), config)
    assert len(predictions) == 26
    assert all(not value.any() for value in predictions.values())


def test_valid_time_kalman_prediction_and_nonrepeated_consumption():
    config, _ = setup(); h = np.zeros((2, 6, 4))
    # Same observed value with different actual ages. CV propagation must use valid time.
    h[:, 1] = [.1, 10/60, 20/30, 1]
    h[:, 2] = [.05, 20/60, 20/30, 1]
    h[1, 1, 1] += 5/60; h[1, 2, 1] += 5/60
    pred = conventional_predictions(h, config)
    name = next(n for n in pred if n.startswith('kf_cv'))
    # Same two measurements and gap -> same terminal slope. Older evidence adds 5 s propagation.
    slope = (pred[name][0, -1]-pred[name][0, 0])/50
    np.testing.assert_allclose(pred[name][1]-pred[name][0], slope*5, atol=1e-12)
    duplicate_call = conventional_predictions(h, config)
    for n in pred: np.testing.assert_array_equal(pred[n], duplicate_call[n])
    np.testing.assert_array_equal(pred['persistence'], .1*np.ones((2, 26)))


def test_trend_known_linear_residual_compensates_age():
    h = np.zeros((1, 6, 4))
    h[0, 1] = [.07, 10/60, 20/30, 1]
    h[0, 2] = [.04, 20/60, 20/30, 1]
    actual = trend_predictions(h)["trend_40"]
    np.testing.assert_allclose(actual[0], .1+.003*OFFSETS, atol=1e-12)


def test_vector_service_matches_independent_scalar_on_nonmonotone_curves():
    _, cases = setup(); rng = np.random.default_rng(14)
    times = np.repeat(np.arange(24, 241, 6), len(TARGETS)); regions = np.tile(TARGETS, 37)
    curves = rng.uniform(.65, .9, size=(3, len(times), 26))
    scores = score_orders(curves, cases, times, regions)
    lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(times, regions))}
    for m in range(3):
        for j, case in enumerate(cases):
            depths = np.zeros((26, 100))
            for target in case['targets']: depths[:, target] = curves[m, lookup[case['now_s'], target]]
            independent = order_scores(case['now_s']+OFFSETS, depths, case['orders'], case['now_s'])
            for o, score in enumerate(independent):
                assert score['service_yield'] == scores['count'][m, j, o]
                for node, trace in enumerate(score['trace']):
                    assert trace['start_s']-case['now_s'] == scores['starts'][m, j, o, node]


def test_mid_service_spike_invalidates_completion_and_lexicographic_tie():
    _, cases = setup(); cases = cases[:1]
    times = np.repeat(np.arange(24, 241, 6), len(TARGETS)); regions = np.tile(TARGETS, 37)
    curve = np.full((1, len(times), 26), .2)
    score = score_orders(curve, cases, times, regions)
    expected = int(cases[0]['orders'][1][0] < cases[0]['orders'][0][0])
    assert choose_order(score, cases)[0, 0] == expected
    start = int(score['starts'][0, 0, 0, 0]); target = cases[0]['orders'][0][0][0]
    row = np.flatnonzero((times == cases[0]['now_s']) & (regions == target))[0]
    curve[0, row, (start+4)//2] = .81
    changed = score_orders(curve, cases, times, regions)
    assert changed['count'][0, 0, 0] == 1
    assert changed['starts'][0, 0, 0, 1] == start+6+max(cases[0]['orders'][0][1][1], 1)*6


def test_selection_only_objective_changes_shared_pool():
    names = ['zero', 'good_mae', 'good_task']
    groups = {g: names for g in ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']}
    selected = select_models(names, groups, {'yield': [1, 1.1, 1.2], 'mae': [.1, .01, .02]})
    assert selected['A3']['candidate'] == 'good_mae'
    assert selected['A4']['candidate'] == 'good_task'
    assert selected['A2']['candidate'] == 'good_task'


def test_pairing_schedule_weights_and_split_seed_isolation():
    config, cases = setup()
    ss = list(schedules(config)); assert len(ss) == 32
    assert sum(s['weight'] for s in ss) == 16
    a = draw_family(config, 'train', 0); b = draw_family(config, 'validation', 0)
    assert a['params'] != b['params'] and a['seed_identity'] != b['seed_identity']
    assert all(max(sum(max(d, 1)*6 for d in ds)+24 for _, ds in c['orders']) <= 48 for c in cases)


def test_switch_continuous_and_before_switch_forecast_can_persist_old_rule():
    p = {'onset': 20, 'rate': .003}
    assert boundary_level(90, p, .9, 90, 1.1) == boundary_level(90, p, .9)
    assert abs(boundary_level(90+1e-7, p, .9, 90, 1.1)-boundary_level(90, p, .9, 90, 1.1)) < 1e-9
    assert boundary_level(100, p, .9) != boundary_level(100, p, .9, 90, 1.1)
