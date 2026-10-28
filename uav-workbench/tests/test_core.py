"""Acceptance: truth isolation, deterministic motion, temporal sampling and metrics."""
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
from hyra_uav.core import Config, FloodProcess, Memory, Pose, Task, footprint, map_metrics
from hyra_uav.policies import policy_for

TEST_TEMP = Path(__file__).resolve().parents[1] / "generated" / "test-temp"
TEST_TEMP.mkdir(parents=True, exist_ok=True)

def bundle(path, labels, times=None, mutate=None):
    labels = np.asarray(labels)
    shape = list(labels.shape[1:])
    times = list(range(len(labels))) if times is None else times
    grid = dict(shape=shape, axis_order=["z", "x"], origin_blocks=[100, 200], dx_m=1)
    process = dict(grid=grid, config=dict(metres_per_block=2), frame_fields=["depth", "u", "v", "maximum_depth"],
                   frame_dtype="<f4", frames=[dict(index=i, time_s=t, url=f"/generated/scene/frame-{i}.f32") for i, t in enumerate(times)])
    if mutate:
        mutate(process)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("process.json", json.dumps(process))
        z.writestr("data_dictionary.json", json.dumps(dict(frames=dict(order="[field,z,x]"))))
        z.writestr("map_model.json", json.dumps(dict(grid=grid, M_base=dict(z=np.zeros(shape).tolist()))))
        maximum = np.zeros(shape)
        for i, layer in enumerate(labels):
            depth = layer * .1
            maximum = np.maximum(maximum, depth)
            z.writestr(f"frame-{i}.f32", np.stack((depth, np.zeros(shape), np.zeros(shape), maximum)).astype("<f4").tobytes())
    return FloodProcess(path)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=TEST_TEMP)
        labels = np.zeros((9, 5, 9), int)
        labels[1:, 2, 3] = 1
        labels[2:5, 0, 0] = 1
        self.process = bundle(Path(self.temp.name)/"p.zip", labels)
        self.config = Config(speed_m_s=2, angular_speed_deg_s=90, range_m=1.1, fov_deg=60,
                             action_s=1, sample_s=1, duration_s=8)

    def tearDown(self):
        self.temp.cleanup()

    def task(self):
        return Task(self.process, self.config, Pose(2, 2, 0))

    def test_determinism_hover_and_clock(self):
        a, b = self.task(), self.task()
        for action in ["hover", "forward", "right", "right", "forward", "left", "hover", "hover"]:
            a.step(action)
            b.step(action)
        self.assertEqual(a.trajectory, b.trajectory)
        self.assertEqual(a.observations, b.observations)
        self.assertEqual(a.trajectory[0]["x_m"], a.trajectory[1]["x_m"])
        self.assertEqual(a.time_s, 8)

    def test_turn_changes_view_without_observing_behind(self):
        task = self.task()
        self.assertEqual(task.memory.labels[2, 3], 0)
        self.assertEqual(task.memory.labels[2, 1], -1)
        old_visible = task.snapshots[-1]["visible"]
        task.step("right")
        self.assertNotEqual(old_visible, task.snapshots[-1]["visible"])
        self.assertEqual(task.pose, Pose(2, 2, 90))
        self.assertEqual(task.memory.labels[2, 1], -1)

    def test_inflight_sampling_keeps_intermediate_pose_and_offset(self):
        cfg = replace(self.config, action_s=2)
        task = Task(self.process, cfg, Pose(2, 2))
        task.step("forward")
        self.assertEqual([p["x_m"] for p in task.trajectory], [2, 4, 6])
        self.assertEqual([o["sample_offset_s"] for o in task.observations], [0, 1, 2])
        self.assertEqual([o["frame_index"] for o in task.observations], [0, 1, 2])

    def test_out_of_view_change_stays_old_until_revisit(self):
        task = self.task()
        task.step("forward")  # x=4; changed cell x=3 is behind
        self.assertTrue(self.process.labels[1, 2, 3])
        self.assertEqual(task.memory.labels[2, 3], 0)
        self.assertEqual(task.memory.observed_s[2, 3], 0)
        task.step("right")
        self.assertEqual(task.memory.labels[2, 3], 0)
        task.step("right")
        self.assertEqual(task.memory.labels[2, 3], 1)
        self.assertEqual(task.memory.observed_s[2, 3], 3)
        self.assertEqual(task.memory.labels[0, 0], -1)
        self.assertEqual(task.memory.history[2*9+3][-1][2], 1)

    def test_observation_footprints_and_timestamps_are_legal(self):
        task = self.task()
        for action in ("forward", "left", "hover", "right"):
            task.step(action)
        for point, obs in zip(task.trajectory, task.observations):
            visible = footprint(Pose(point["x_m"], point["z_m"], point["heading_deg"]), self.config, self.process.shape, self.process.dx)
            self.assertEqual(obs["cells"], np.flatnonzero(visible).tolist())
            self.assertEqual(obs["observed_s"], point["time_s"])
            self.assertEqual(obs["arrived_s"], point["time_s"])
            np.testing.assert_array_equal(obs["labels"], self.process.labels[obs["frame_index"]].ravel()[obs["cells"]])

    def test_old_duplicate_future_and_not_arrived_messages(self):
        memory = Memory((2, 2), 2)
        obs = dict(cells=[0], labels=[1], observed_s=2, arrived_s=3)
        memory.accept(obs, 3)
        memory.accept(dict(obs, labels=[0], observed_s=1), 3)
        memory.accept(dict(obs, labels=[0]), 3)
        self.assertEqual(memory.labels[0, 0], 1)
        self.assertEqual(len(memory.history[0]), 1)
        for bad in [dict(obs, arrived_s=4), dict(obs, observed_s=4), dict(obs, cells=[0, 0], labels=[1, 1])]:
            with self.assertRaises(ValueError):
                memory.accept(bad, 3)

    def test_policy_view_is_immutable_and_has_no_truth(self):
        task = self.task()
        view = task.view()
        self.assertEqual(set(asdict(view)), {"pose", "time_s", "shape", "dx", "labels", "observed_s", "quality", "history"})
        with self.assertRaises(ValueError):
            view.labels[0, 0] = 1
        for method in ("R0", "R1", "R2"):
            policy = policy_for(method, view, self.config)
            self.assertIn(policy.choose(view, self.config), ("forward", "left", "right", "hover"))

    def test_change_segments_keep_undiscovered_and_repeated_changes(self):
        task = self.task()
        for action in ("forward", "right", "right", "hover", "hover", "hover", "hover", "hover"):
            task.step(action)
        result = task.result()
        found = [c for c in result["changes"] if c["cell"] == 2*9+3]
        self.assertEqual(found[0]["delay_s"], 2)
        missed = [c for c in result["changes"] if c["cell"] == 0]
        self.assertEqual(len(missed), 2)
        self.assertTrue(all(c["detected_s"] is None for c in missed))
        self.assertAlmostEqual(result["summary"]["undiscovered_change_fraction"], 2/3)

    def test_time_integral_uses_physical_intervals_and_not_endpoint_average(self):
        task = self.task()
        task.step("forward")
        result = task.result()
        self.assertEqual(result["summary"]["time_average_correct_coverage"], task.snapshots[0]["metrics"]["correct_coverage"])

    def test_boundary_motion_stays_in_domain_and_charges_actual_distance(self):
        task = Task(self.process, self.config, Pose(8, 2))
        task.step("forward")
        self.assertLess(task.pose.x_m, 8.5)
        self.assertAlmostEqual(task.distance_m, .5, places=6)
        task.step("forward")
        self.assertAlmostEqual(task.distance_m, .5, places=6)

    def test_actual_frame_access_refuses_future_or_interpolation(self):
        for t in (-1, .5, 9, float("nan")):
            with self.assertRaises(ValueError):
                self.process.frame_index(t)

    def test_time_config_prevents_skipping_changes_and_partial_actions(self):
        for cfg in [replace(self.config, sample_s=.5), replace(self.config, sample_s=2, action_s=2),
                    replace(self.config, duration_s=9), replace(self.config, speed_m_s=float("nan")),
                    replace(self.config, history_length=1.5), replace(self.config, fov_deg=0)]:
            with self.assertRaises(ValueError):
                cfg.validate(self.process)

    def test_invalid_pose_and_action_do_not_mutate_task(self):
        with self.assertRaises(ValueError):
            Task(self.process, self.config, Pose(-2, 0))
        task = self.task()
        with self.assertRaises(ValueError):
            task.step("teleport")
        self.assertEqual(task.time_s, 0)
        self.assertFalse(task.actions)


class MetricTests(unittest.TestCase):
    def test_unknown_correct_opposite_and_single_class(self):
        truth = np.array([[0, 0], [0, 1]], bool)
        self.assertEqual(map_metrics(truth, np.full((2, 2), -1))["correct_coverage"], 0)
        self.assertIsNone(map_metrics(truth, np.full((2, 2), -1))["observed_error_rate"])
        self.assertEqual(map_metrics(truth, truth.astype(int))["correct_coverage"], 1)
        self.assertEqual(map_metrics(truth, (~truth).astype(int))["correct_coverage"], 0)
        self.assertEqual(map_metrics(np.zeros((2, 2), bool), np.zeros((2, 2), int))["correct_coverage"], 1)

    def test_classes_are_balanced_and_unknowns_stay_in_denominator(self):
        truth = np.array([[0, 0], [0, 1]], bool)
        labels = np.array([[-1, -1], [-1, 1]])
        self.assertEqual(map_metrics(truth, labels)["correct_coverage"], .5)


class ProcessValidationTests(unittest.TestCase):
    def test_malformed_metadata_bytes_and_nonfinite_frames_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as tmp:
            path = Path(tmp)/"bad.zip"
            labels = np.zeros((3, 3, 3))
            for mutate in [lambda p:p.update(frame_dtype=">f4"), lambda p:p["frames"][1].update(time_s=0),
                           lambda p:p["frames"][1].update(url=p["frames"][0]["url"])]:
                with self.assertRaises(ValueError):
                    bundle(path, labels, mutate=mutate)
            for value in (np.nan, -1):
                labels[1, 0, 0] = value
                with self.assertRaises(ValueError):
                    bundle(path, labels)
            bundle(path, np.zeros((3, 3, 3)))
            with zipfile.ZipFile(path) as archive:
                entries = {name:archive.read(name) for name in archive.namelist()}
            entries["frame-1.f32"] = b"short"
            with zipfile.ZipFile(path, "w") as archive:
                for name, raw in entries.items():
                    archive.writestr(name, raw)
            with self.assertRaises(ValueError):
                FloodProcess(path)


if __name__ == "__main__":
    unittest.main()
