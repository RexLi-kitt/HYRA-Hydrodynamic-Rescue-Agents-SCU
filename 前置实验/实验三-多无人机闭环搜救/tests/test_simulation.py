import math
import json
import numpy as np
import pytest

from hyra_exp3.simulation import EpisodeConfig, DepthField, run_episode
from hyra_exp3.windows import NominalPredictor, WindowEstimate, service_start_intervals


def fixture(shape=(12, 12), duration=40):
    times = np.arange(duration + 1, dtype=float)
    return times, np.full((len(times), *shape), .2)


def test_windows_check_interior_peak_and_do_not_extrapolate():
    times = np.arange(31.)
    depth = np.full(31, .2)
    depth[6] = 1.
    intervals = service_start_intervals(times, depth)
    estimate = WindowEstimate((0, 0), 0., 1, tuple(times), tuple(depth), intervals, True, 30.)
    assert estimate.can_start(0) is False
    assert estimate.can_start(7) is True
    assert estimate.can_start(20) is None


def test_fractional_truth_interpolation_matches_float64_independent_auditor():
    times=np.array([0.,1.])
    values=np.array([.79,.81],np.float32)
    field=DepthField(times,values.reshape(2,1,1))
    for now in np.linspace(0.,1.,101):
        assert field.at(now,[(0,0)])[0] == pytest.approx(np.interp(now,times,values.astype(float)),abs=1e-15,rel=0.)


def test_forecast_versions_exclude_future_observations():
    times, depth = fixture()
    predictor = NominalPredictor(times, depth, horizon_s=20)
    with pytest.raises(ValueError):
        predictor.issue(5, [dict(sample_time=4, arrival_time=14)])
    predictor.issue(5, [dict(sample_time=1, arrival_time=4)])
    estimate = predictor.estimate((0, 0), 5)
    assert estimate.valid_until == 25
    assert estimate.can_start(14) is None
    assert estimate.can_start(13) is True
    predictor.issue(10, [])
    assert predictor.estimate((0, 0), 10).version > estimate.version


def test_diagonal_flight_and_entire_service_use_fractional_clock():
    times, depth = fixture((5, 5), 30)
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=30,
                           observation_noise_m=0, sensor_radius_cells=2)
    result = run_episode(depth, config, times=times, nominal_depths=depth.copy(),
                         targets=[(1, 1)], starts=[(0, 0)])
    arrival = next(e for e in result["events"] if e["event"] == "target_arrival")
    assert arrival["time_s"] == pytest.approx(math.sqrt(200) / 5)
    assert result["targets"][0]["served_at"] == pytest.approx(math.sqrt(200) / 5 + 12)
    assert result["drones"][0]["payload"] == 3
    json.dumps(result, allow_nan=False)


def test_in_transit_sensor_samples_actual_position_not_destination():
    times, depth = fixture((20, 20), 12)
    config = EpisodeConfig(map_shape=(20, 20), n_drones=1, n_targets=0, duration_s=12,
                           observation_noise_m=0)
    result = run_episode(depth, config, times=times, nominal_depths=depth.copy(), targets=[], starts=[(0, 0)])
    positions = [r for r in result["trajectories"] if r["time_s"] == 5]
    assert positions
    assert np.linalg.norm(np.array(positions[0]["position_m"]) - [5, 5]) <= 25 + 1e-8
    center = positions[0]["sensor_center_cell"]
    assert center == list(np.floor(np.array(positions[0]["position_m"])[::-1] / 10).astype(int))


def test_target_discovery_waits_for_observation_arrival():
    times, depth = fixture((5, 5), 40)
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=40,
                           observation_delay_s=10, observation_noise_m=0)
    result = run_episode(depth, config, times=times, nominal_depths=depth.copy(),
                         targets=[(0, 0)], starts=[(0, 0)])
    assert result["targets"][0]["first_sensed_at"] == 0
    assert result["targets"][0]["discovered_at"] == 10
    assert all(o["arrival_time"] >= o["sample_time"] + 10 for o in result["raw_observations"])
    assert all(e["time_s"] >= 10 for e in result["events"] if e["event"] == "service_start")


def test_failed_service_consumes_full_time_but_keeps_payload():
    times, nominal = fixture((5, 5), 20)
    truth = nominal.copy()
    truth[6, 0, 0] = 1.
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=20,
                           observation_noise_m=0)
    result = run_episode(truth, config, times=times, nominal_depths=nominal,
                         targets=[(0, 0)], starts=[(0, 0)])
    finished = [e for e in result["events"] if e["event"] == "service_complete"]
    assert len(finished) == 1
    assert not finished[0]["success"] and finished[0]["time_s"] == 12
    assert result["drones"][0]["payload"] == 4
    assert result["summary"]["successful_services"] == 0


def test_hidden_undiscovered_targets_remain_in_denominator():
    times, depth = fixture((20, 20), 1)
    config = EpisodeConfig(map_shape=(20, 20), n_drones=1, n_targets=2, duration_s=1,
                           sensor_radius_cells=0, observation_noise_m=0)
    result = run_episode(depth, config, times=times, nominal_depths=depth.copy(),
                         targets=[(0, 0), (19, 19)], starts=[(0, 0)])
    assert result["summary"]["hidden_targets"] == 2
    assert result["summary"]["success_rate"] == 0
    assert result["summary"]["failure_categories"]["undiscovered"] == 1


def test_target_layout_is_independent_of_truth_and_noise_stream():
    times, depth = fixture((8, 8), 1)
    c = EpisodeConfig(map_shape=(8, 8), n_drones=1, n_targets=3, duration_s=1)
    a = run_episode(depth, c, times=times, nominal_depths=depth.copy())
    b = run_episode(depth + .1, c, times=times, nominal_depths=depth.copy())
    assert [r["cell"] for r in a["targets"]] == [r["cell"] for r in b["targets"]]


def test_no_fallback_to_true_field_as_prediction():
    times, depth = fixture((5, 5))
    with pytest.raises(ValueError, match="Independent nominal"):
        run_episode(depth, EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=40), times=times)


def test_timeout_leaves_drone_waiting_when_no_feasible_previous_plan_exists():
    times, depth = fixture((5, 5), 20)
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=1, duration_s=20,
                           planner_budget_s=1e-12, observation_noise_m=0)
    result = run_episode(depth, config, times=times, nominal_depths=depth.copy(),
                         targets=[(0, 1)], starts=[(0, 0)])
    assert result["summary"]["planning_timeouts"] > 0
    assert result["summary"]["distance_m"] == 0
    assert result["summary"]["successful_services"] == 0
