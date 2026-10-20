import numpy as np

from hyra_exp3.frontier_control import NeutralFrontierPredictor
from hyra_exp3.planner_replay import QueryPredictor
from hyra_exp3.planning import RollingPlanner
from hyra_exp3.simulation import EpisodeConfig


def test_neutral_frontier_changes_no_known_task_forecasts_and_preserves_unknown_support():
    depth = np.full((121, 4, 4), 1.)
    predictor = NeutralFrontierPredictor(np.arange(121.), depth)
    predictor.issue(10, [])
    assert predictor.estimate((0, 0), 10).can_start(10) is False
    frontier = predictor.estimate_many([(0, 0)], 10)[0]
    assert frontier.can_start(10) is True
    assert frontier.can_start(59) is None
    assert frontier.valid_until == 70
    assert predictor.queries[-1][-1] == 1


def test_neutral_control_matches_weighted_controller_on_uniform_safe_field():
    times = np.arange(121.)
    depth = np.full((121, 20, 20), .2)
    c = EpisodeConfig(map_shape=(20, 20), n_drones=2, n_targets=0, duration_s=120)
    arguments = dict(now=0., robots=[dict(id=0, position=np.array([85., 85.]), payload=4),
                                    dict(id=1, position=np.array([115., 115.]), payload=4)],
                     known_targets={}, covered=np.zeros((20, 20), bool), allowed=np.ones((20, 20), bool), config=c)
    predictor, neutral = QueryPredictor(times, depth), NeutralFrontierPredictor(times, depth)
    predictor.issue(0, [])
    neutral.issue(0, [])
    assert RollingPlanner()(predictor=predictor, **arguments) == RollingPlanner()(predictor=neutral, **arguments)
