import copy
import numpy as np
import pytest

from hyra_exp3.planner_replay import QueryPredictor, audit_episode
from hyra_exp3.planning import RollingPlanner
from hyra_exp3.simulation import EpisodeConfig, run_episode


def episode():
    times = np.arange(41.)
    depth = np.full((41, 5, 5), .2)
    config = EpisodeConfig(map_shape=(5, 5), n_drones=1, n_targets=2, duration_s=40, planner_budget_s=1.)
    predictor = QueryPredictor(times, depth)
    result = run_episode(depth, config, times=times, predictor=predictor, targets=[(0, 0), (0, 1)],
                         starts=[(0, 0)], planner=RollingPlanner())
    truth = dict(times_s=times, depths_m=depth.reshape(41, -1))
    return result, predictor, truth, config


def test_replay_audit_checks_services_and_rejects_forged_success():
    result, predictor, truth, config = episode()
    assert audit_episode(result, predictor, truth, config)["complete_services_rescored"] == 2
    corrupted = copy.deepcopy(result)
    event = next(e for e in corrupted["events"] if e["event"] == "service_complete")
    event["success"] = False
    with pytest.raises(AssertionError, match="score or duration"):
        audit_episode(corrupted, predictor, truth, config)


def test_replay_audit_distinguishes_public_frontier_queries_from_hidden_task_queries():
    result, predictor, truth, config = episode()
    predictor.queries.append((0., 4, 4, 1, 1))
    audit_episode(result, predictor, truth, config)
    predictor.queries.append((0., 4, 4, 1, 0))
    with pytest.raises(AssertionError, match="preceded target discovery"):
        audit_episode(result, predictor, truth, config)
