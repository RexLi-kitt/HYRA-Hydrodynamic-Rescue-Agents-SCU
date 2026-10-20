import numpy as np
import pytest

from hyra_exp3.planning import RollingPlanner
from hyra_exp3.simulation import EpisodeConfig, run_episode
from hyra_exp3.windows import NominalPredictor


def test_batched_windows_match_scalar_with_multiple_crossings_and_fractional_publication():
    rng = np.random.default_rng(42)
    times = np.arange(0., 121., 2.)
    depth = rng.uniform(.5, 1.1, (len(times), 3, 4))
    depth[:20, 0, 0] = .2
    depth[30:, 1, 1] = .8
    predictor = NominalPredictor(times, depth)
    for now in (0., 1.25, 60., 119.):
        predictor.issue(now, [])
        cells = [(r, c) for r in range(3) for c in range(4)]
        for cell, batch in zip(cells, predictor.estimate_many(cells, now)):
            scalar = predictor.estimate(cell, now)
            assert batch.future_depths == pytest.approx(scalar.future_depths)
            assert np.asarray(batch.service_intervals).ravel() == pytest.approx(np.asarray(scalar.service_intervals).ravel())
            for start in np.arange(now, min(now + 60, 120), .5):
                assert batch.can_start(start) == scalar.can_start(start)


def test_routes_schedule_multiple_visits_without_conflicts_or_payload_excess():
    times = np.arange(121.)
    depth = np.full((121, 8, 8), .2)
    c = EpisodeConfig(map_shape=(8, 8), n_drones=2, n_targets=4, duration_s=120, payload=2)
    predictor = NominalPredictor(times, depth)
    predictor.issue(0, [])
    planner = RollingPlanner(forecast_exploration=False)
    cells = {0: (0, 1), 1: (0, 2), 2: (7, 6), 3: (7, 5)}
    robots = [dict(id=0, position=np.array([5., 5.]), payload=2), dict(id=1, position=np.array([75., 75.]), payload=2)]
    planner(now=0., robots=robots, known_targets=cells, covered=np.zeros((8, 8), bool),
            allowed=np.ones((8, 8), bool), predictor=predictor, config=c)
    routes = planner.last_diagnostics["selected_routes"]
    all_tasks = [tid for route in routes for tid in route["targets"]]
    assert sorted(all_tasks) == list(cells)
    for route in routes:
        assert len(route["targets"]) <= 2
        position, finish = robots[route["drone"]]["position"], 0.
        for tid, start in zip(route["targets"], route["starts_s"]):
            r, col = cells[tid]
            target_xy = np.array([(col + .5) * 10, (r + .5) * 10])
            assert start >= finish + np.linalg.norm(target_xy - position) / 5 - 1e-8
            assert predictor.estimate(cells[tid], 0).can_start(start) is True
            finish, position = start + 12, target_xy
        assert finish <= 60


def test_forecast_changes_exploration_without_hidden_target_input():
    c = EpisodeConfig(map_shape=(20, 20), n_drones=1, n_targets=0, duration_s=120)
    times = np.arange(121.)
    depth = np.full((121, 20, 20), .2)
    covered = np.zeros((20, 20), bool)
    covered[6:11, 6:11] = True
    robots = [dict(id=0, position=np.array([85., 85.]), payload=4)]
    arguments = dict(now=0., robots=robots, known_targets={}, covered=covered, allowed=np.ones((20, 20), bool), config=c)
    predictor = NominalPredictor(times, depth.copy())
    predictor.issue(0, [])
    baseline = RollingPlanner(forecast_exploration=False, multi_visit=False)(predictor=predictor, **arguments)
    # Make the baseline destination and its neighborhood too deep throughout.
    row, col = baseline[0][0]
    depth[:, max(0, row-3):row+4, max(0, col-3):col+4] = 1.
    wet = NominalPredictor(times, depth)
    wet.issue(0, [])
    changed = RollingPlanner(multi_visit=False)(predictor=wet, **arguments)
    assert changed[0][0] != baseline[0][0]


def test_disabled_features_reproduce_legacy_actions_and_discovery():
    times = np.arange(121.)
    depth = np.full((121, 12, 12), .2)
    c = EpisodeConfig(map_shape=(12, 12), n_drones=3, n_targets=8, duration_s=120, planner_budget_s=1.)
    a = run_episode(depth, c, times=times, nominal_depths=depth.copy())
    b = run_episode(depth, c, times=times, nominal_depths=depth.copy(),
                    planner=RollingPlanner(forecast_exploration=False, multi_visit=False))
    assert a["trajectories"] == b["trajectories"]
    assert a["targets"] == b["targets"]


def test_custom_planner_only_receives_discovered_targets_and_arrived_coverage():
    class Spy(RollingPlanner):
        def __call__(self, **kwargs):
            assert set(kwargs) == {"now", "robots", "known_targets", "covered", "allowed", "predictor", "config"}
            if kwargs["now"] < 10:
                assert not kwargs["known_targets"]
                assert not kwargs["covered"].any()
            return super().__call__(**kwargs)
    times = np.arange(41.)
    depth = np.full((41, 6, 6), .2)
    c = EpisodeConfig(map_shape=(6, 6), n_drones=1, n_targets=1, duration_s=40, observation_delay_s=10, planner_budget_s=1.)
    result = run_episode(depth, c, times=times, nominal_depths=depth.copy(), targets=[(0, 0)],
                         starts=[(0, 0)], planner=Spy())
    assert result["targets"][0]["discovered_at"] == 10


def test_two_near_visits_win_over_one_far_visit_inside_published_support():
    times = np.arange(121.)
    depth = np.full((121, 2, 16), .2)
    c = EpisodeConfig(map_shape=(2, 16), n_drones=1, n_targets=3, duration_s=120, forecast_horizon_s=50)
    predictor = NominalPredictor(times, depth, horizon_s=50)
    predictor.issue(0, [])
    planner = RollingPlanner(forecast_exploration=False)
    planner(now=0., robots=[dict(id=0, position=np.array([5., 5.]), payload=4)],
            known_targets={0: (0, 13), 1: (0, 1), 2: (0, 2)}, covered=np.zeros((2, 16), bool),
            allowed=np.ones((2, 16), bool), predictor=predictor, config=c)
    route = planner.last_diagnostics["selected_routes"][0]
    assert set(route["targets"]) == {1, 2}
    assert route["finish_s"] <= 50


def test_no_route_when_entire_service_exceeds_forecast_support():
    times = np.arange(121.)
    depth = np.full((121, 3, 3), .2)
    predictor = NominalPredictor(times, depth, horizon_s=11)
    predictor.issue(0, [])
    planner = RollingPlanner()
    c = EpisodeConfig(map_shape=(3, 3), n_drones=1, n_targets=1, duration_s=120)
    proposals = planner(now=0., robots=[dict(id=0, position=np.array([5., 5.]), payload=4)],
                        known_targets={0: (0, 0)}, covered=np.zeros((3, 3), bool),
                        allowed=np.ones((3, 3), bool), predictor=predictor, config=c)
    assert not planner.last_diagnostics["selected_routes"]
    assert all(proposal[1] is None for proposal in proposals.values())
