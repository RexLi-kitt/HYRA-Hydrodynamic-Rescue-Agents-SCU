from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hyra_uav.core import Config, Pose, footprint
from hyra_learning.team_env import TeamConfig, TeamEnv
from hyra_learning.observations import encode
from test_core import TEST_TEMP, bundle


class TeamLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        self.root = Path(self.temp.name)
        labels = np.zeros((5, 5, 9), dtype=int)
        labels[1:, 2, 3] = 1
        self.process = bundle(self.root/'team.zip', labels)
        self.config = Config(speed_m_s=1, angular_speed_deg_s=90, range_m=1.1,
                             fov_deg=60, action_s=2, sample_s=1, duration_s=4)
        self.poses = [Pose(0, 0), Pose(8, 4, 180)]

    def tearDown(self):
        self.temp.cleanup()

    def test_joint_clock_observations_shared_union_and_reward(self):
        env = TeamEnv(self.process, self.config, poses=self.poses)
        views, _ = env.reset(seed=11)
        expected = footprint(self.poses[0], self.config, self.process.shape, 1) | footprint(self.poses[1], self.config, self.process.shape, 1)
        np.testing.assert_array_equal(views.agents[0].labels >= 0, expected)
        np.testing.assert_array_equal(views.agents[0].labels, views.agents[1].labels)
        total = 0.
        for actions in ((0, 0), (2, 1)):
            view, reward, terminated, truncated, _ = env.step(actions)
            total += reward
            self.assertFalse(terminated)
            self.assertEqual([a.time_s for a in view.agents], [env.time_s]*2)
        self.assertTrue(truncated)
        self.assertEqual(len(env.snapshots), 5)
        self.assertEqual(len(env.observations), 10)
        self.assertEqual([o['observed_s'] for o in env.observations], [0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        for obs in env.observations:
            self.assertEqual(obs['observed_s'], obs['arrived_s'])
            np.testing.assert_array_equal(obs['labels'], self.process.labels[obs['frame_index']].ravel()[obs['cells']])
        result = env.result()
        self.assertEqual(result['n_agents'], 2)
        self.assertEqual(len(result['truth']), 5)
        expected_reward = (result['summary']['time_average_correct_coverage']
                           -env.team.distance_cost_per_m*env.distance_m
                           -env.team.turn_cost_per_s*env.turn_s
                           -env.team.collision_cost*env.collision_events)
        self.assertAlmostEqual(total, expected_reward, places=12)
        with self.assertRaises(ValueError):
            env.step((3, 3))

    def test_crossing_paths_cancel_translation_and_charge_time(self):
        config = replace(self.config, speed_m_s=4, action_s=1)
        env = TeamEnv(self.process, config, TeamConfig(separation_m=1),
                      [Pose(2, 2), Pose(6, 2, 180)])
        env.reset()
        env.step((0, 0))
        self.assertEqual(env.poses, [Pose(2, 2), Pose(6, 2, 180)])
        self.assertEqual(env.distance_m, 0)
        self.assertEqual(env.collision_events, 1)
        self.assertEqual(env.time_s, 1)
        env.step((1, 2))
        self.assertEqual(env.turn_s, 2)

    def test_invalid_joint_action_is_atomic_and_reset_is_deterministic(self):
        env = TeamEnv(self.process, self.config, poses=self.poses)
        original, _ = env.reset(seed=3)
        for actions in [(0,), (0, 4), ('forward', 'invalid'), (True, 1)]:
            with self.assertRaises(ValueError):
                env.step(actions)
            self.assertEqual(env.time_s, 0)
            self.assertEqual(env.actions, [])
        env.step((0, 3))
        again, _ = env.reset(seed=3)
        np.testing.assert_array_equal(encode(original)['images'], encode(again)['images'])

    def test_actor_encoding_cannot_read_hidden_current_or_future_water(self):
        # Different hidden present and future fields, identical available data.
        labels = np.zeros((5, 5, 9), dtype=int)
        labels[:, 2, 4] = 1
        other = bundle(self.root/'different-hidden.zip', labels)
        dry = bundle(self.root/'dry.zip', np.zeros_like(labels))
        a, b = (TeamEnv(p, self.config, poses=self.poses) for p in (dry, other))
        va, _ = a.reset()
        vb, _ = b.reset()
        for key in ('images', 'poses'):
            np.testing.assert_array_equal(encode(va)[key], encode(vb)[key])
        self.assertFalse(np.array_equal(a.training_state(), b.training_state()))
        self.assertFalse(hasattr(va, 'depths'))
        self.assertFalse(hasattr(va.agents[0], 'truth'))
        for array in (va.bed, va.agents[0].labels, va.agents[0].observed_s):
            with self.assertRaises(ValueError):
                array.flat[0] = 1
        # Observed dry and unknown both have wet=0 but different known masks.
        image = encode(va)['images'][0]
        self.assertEqual(image.shape, (16, 7, 7))
        self.assertEqual(image[0, 3, 3], 0)
        self.assertEqual(image[1, 3, 3], 1)
        self.assertEqual(image[7, 0, 0], 0)  # outside domain explicitly masked


if __name__ == '__main__':
    unittest.main()
