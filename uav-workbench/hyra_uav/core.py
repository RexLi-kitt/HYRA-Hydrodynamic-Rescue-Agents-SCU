"""Deterministic single-UAV task. Truth never appears in a PolicyView."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

ACTIONS = ("forward", "left", "right", "hover")


class FloodProcess:
    """Read calculated channel-first frames; reject interpolation and malformed data.

    x/z are local metres from the first cell centre, matching the map exporter.
    The physical domain extends half a cell beyond the outermost centres.
    """

    def __init__(self, path, threshold_m=0.05):
        path = Path(path)
        self.path = path.resolve()
        self.digest = sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as archive:
            self.process = json.loads(archive.read("process.json"))
            self.dictionary = json.loads(archive.read("data_dictionary.json"))
            model = json.loads(archive.read("map_model.json"))
            grid = self.process["grid"]
            if grid.get("axis_order") != ["z", "x"]:
                raise ValueError("过程轴顺序必须是 [z,x]")
            raw_shape = grid["shape"]
            if (len(raw_shape) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in raw_shape)):
                raise ValueError("网格形状必须是两个至少为2的整数")
            self.shape = tuple(raw_shape)
            self.dx = float(grid["dx_m"])
            self.origin_blocks = grid.get("origin_blocks", [0, 0])
            self.metres_per_block = float(self.process["config"]["metres_per_block"])
            if not math.isfinite(self.dx) or self.dx <= 0 or not math.isfinite(self.metres_per_block) or self.metres_per_block <= 0:
                raise ValueError("格距与尺度必须为有限正数")
            if self.process.get("frame_fields") != ["depth", "u", "v", "maximum_depth"] or self.process.get("frame_dtype") != "<f4":
                raise ValueError("仅支持 depth/u/v/maximum_depth 小端float32实际帧")
            if self.dictionary.get("frames", {}).get("order") != "[field,z,x]":
                raise ValueError("数据字典缺少可确认的帧排列")
            if model.get("grid") != grid:
                raise ValueError("静态地图与过程网格不一致")
            self.bed = np.asarray(model["M_base"]["z"], dtype=float)
            if self.bed.shape != self.shape or not np.isfinite(self.bed).all():
                raise ValueError("静态高程尺寸或数值错误")
            entries = self.process["frames"]
            self.times = np.asarray([f["time_s"] for f in entries], dtype=float)
            if len(entries) < 2 or not np.isfinite(self.times).all() or self.times[0] != 0 or np.any(np.diff(self.times) <= 0):
                raise ValueError("实际帧时间必须从0开始且严格递增")
            expected = 4 * math.prod(self.shape) * 4
            frames = []
            names = []
            for index, entry in enumerate(entries):
                name = entry["url"].replace("\\", "/").rsplit("/", 1)[-1]
                if entry.get("index") != index or name in names:
                    raise ValueError("帧索引或文件名重复")
                names.append(name)
                raw = archive.read(name)
                if len(raw) != expected:
                    raise ValueError(f"{name} 长度错误：应为 {expected} 字节")
                frame = np.frombuffer(raw, dtype="<f4").reshape((4, *self.shape))
                if not np.isfinite(frame).all() or frame[0].min() < -1e-7:
                    raise ValueError(f"{name} 存在非法数值或负水深")
                if np.any(frame[3] + 1e-6 < frame[0]):
                    raise ValueError(f"{name} 最大水深小于当前水深")
                frames.append(frame.copy())
            self.depths = np.stack(frames)[:, 0]
        self.threshold = float(threshold_m)
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("淹没阈值必须为有限正数")
        self.labels = self.depths >= self.threshold
        self.bed.setflags(write=False)
        self.labels.setflags(write=False)

    def frame_index(self, time_s):
        matches = np.flatnonzero(np.isclose(self.times, time_s, rtol=0, atol=1e-7))
        if not len(matches):
            raise ValueError(f"{time_s}s 没有实际计算帧；禁止使用未来帧或插值感知")
        return int(matches[0])

    def audit(self):
        changed = self.labels[1:] != self.labels[:-1]
        probes = [(0, 0), (self.shape[0] // 2, self.shape[1] // 2), (self.shape[0] - 1, self.shape[1] - 1)]
        return dict(schema="hyra-uav-p0-v1", source=str(self.path), sha256=self.digest,
                    structural_checks_passed=True, physical_validation="requires_review_not_established_by_structural_checks",
                    grid=self.process["grid"], threshold_m=self.threshold,
                    first_cell_centre_absolute_m=[float(v*self.metres_per_block) for v in self.origin_blocks],
                    frame_count=len(self.times), time_range_s=[float(self.times[0]), float(self.times[-1])],
                    frame_intervals_s=np.unique(np.diff(self.times)).tolist(),
                    bed_range_m=[float(self.bed.min()), float(self.bed.max())],
                    depth_range_m=[float(self.depths.min()), float(self.depths.max())],
                    flooded_area_m2=(self.labels.sum(axis=(1, 2)) * self.dx ** 2).tolist(),
                    changed_cells_per_interval=changed.sum(axis=(1, 2)).tolist(),
                    changed_cells_ever=int(np.any(changed, axis=0).sum()),
                    state_disagreement=[dict(lag_s=float(self.times[k] - self.times[0]),
                                             fraction=float((self.labels[k:] != self.labels[:-k]).mean()))
                                        for k in (1, 6, 12) if k < len(self.times)],
                    probes=[dict(row=r, col=c, local_x_m=c*self.dx, local_z_m=r*self.dx,
                                 depths_m=[float(self.depths[i, r, c]) for i in (0, len(self.times)//2, len(self.times)-1)])
                            for r, c in probes],
                    initial_conditions=self.process.get("initial_conditions", {}),
                    forcing=self.process["config"],
                    permissions=dict(environment=["current_hidden_depth", "full_process", "future_forcing"],
                                     policy=["grid_geometry", "own_pose", "arrived_observation_history"],
                                     evaluator=["current_truth", "change_segments"]))


@dataclass(frozen=True)
class Config:
    speed_m_s: float = 2.5
    angular_speed_deg_s: float = 9.0
    range_m: float = 25.0
    fov_deg: float = 90.0
    action_s: float = 10.0
    sample_s: float = 5.0
    duration_s: float = 600.0
    history_length: int = 4
    age_weight: float = 0.5

    def validate(self, process):
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values()):
            raise ValueError("参数必须为有限数")
        if any(values[k] <= 0 for k in ("speed_m_s", "angular_speed_deg_s", "range_m", "action_s", "sample_s", "duration_s")):
            raise ValueError("速度、视距及时间必须大于0")
        if not 0 < self.fov_deg <= 360 or self.age_weight < 0:
            raise ValueError("张角必须在(0,360]，年龄权重不能为负")
        if isinstance(self.history_length, bool) or not isinstance(self.history_length, int) or not 1 <= self.history_length <= 32:
            raise ValueError("短历史长度必须是1至32整数")
        for n in (self.action_s / self.sample_s, self.duration_s / self.action_s):
            if not math.isclose(n, round(n), abs_tol=1e-8):
                raise ValueError("动作必须包含整数次采样，任务必须包含整数个动作")
        if not np.allclose(np.diff(process.times), self.sample_s, rtol=0, atol=1e-7):
            raise ValueError("首版采样周期必须等于均匀实际帧间隔，避免遗漏帧间变化")
        for t in np.arange(0, self.duration_s + self.sample_s/2, self.sample_s):
            process.frame_index(float(t))
        if len(np.arange(0, self.duration_s + self.sample_s/2, self.sample_s)) > 5000:
            raise ValueError("MVP单次运行最多5000个采样时刻")


@dataclass(frozen=True)
class Pose:
    x_m: float
    z_m: float
    heading_deg: float = 0.0


def move(pose, action, dt, config, shape, dx):
    if action not in ACTIONS:
        raise ValueError(f"未知动作 {action}")
    x, z, heading = pose.x_m, pose.z_m, pose.heading_deg
    distance = 0.0
    if action == "forward":
        angle = math.radians(heading)
        x += config.speed_m_s * dt * math.cos(angle)
        z += config.speed_m_s * dt * math.sin(angle)
        x = float(np.clip(x, -dx/2, (shape[1]-.5)*dx - 1e-9))
        z = float(np.clip(z, -dx/2, (shape[0]-.5)*dx - 1e-9))
        distance = math.hypot(x-pose.x_m, z-pose.z_m)
    elif action in ("left", "right"):
        heading += config.angular_speed_deg_s * dt * (-1 if action == "left" else 1)
    return Pose(x, z, heading % 360), distance


def footprint(pose, config, shape, dx):
    zz, xx = np.indices(shape, dtype=float)
    delta_x, delta_z = xx * dx - pose.x_m, zz * dx - pose.z_m
    angles = (np.degrees(np.arctan2(delta_z, delta_x)) - pose.heading_deg + 180) % 360 - 180
    visible = (np.hypot(delta_x, delta_z) <= config.range_m + 1e-9) & (np.abs(angles) <= config.fov_deg/2 + 1e-9)
    col = int(np.clip(math.floor(pose.x_m / dx + .5), 0, shape[1]-1))
    row = int(np.clip(math.floor(pose.z_m / dx + .5), 0, shape[0]-1))
    visible[row, col] = True
    return visible


class Memory:
    def __init__(self, shape, history_length=4):
        self.labels = np.full(shape, -1, dtype=np.int8)
        self.observed_s = np.full(shape, -1.0)
        self.arrived_s = np.full(shape, -1.0)
        self.quality = np.zeros(shape)
        self.history = [[] for _ in range(math.prod(shape))]
        self.history_length = history_length

    def accept(self, observation, now):
        t, arrival = observation["observed_s"], observation["arrived_s"]
        if not 0 <= t <= arrival <= now or not math.isfinite(now):
            raise ValueError("观测或到达时间不合法")
        cells, labels = observation["cells"], observation["labels"]
        if len(cells) != len(labels) or len(set(cells)) != len(cells):
            raise ValueError("观测长度或格点去重错误")
        if any(not isinstance(c, int) or not 0 <= c < self.labels.size for c in cells) or any(v not in (0, 1) for v in labels):
            raise ValueError("观测格点或标签非法")
        quality = observation.get("quality", 1.0)
        if not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("质量必须在[0,1]")
        for c, label in zip(cells, labels):
            # Equal timestamps keep the first arrival deterministically. Old messages cannot overwrite.
            if t <= self.observed_s.flat[c]:
                continue
            self.labels.flat[c] = label
            self.observed_s.flat[c] = t
            self.arrived_s.flat[c] = arrival
            self.quality.flat[c] = quality
            self.history[c].append([t, arrival, label, quality])
            self.history[c] = self.history[c][-self.history_length:]


@dataclass(frozen=True)
class PolicyView:
    pose: Pose
    time_s: float
    shape: tuple
    dx: float
    labels: np.ndarray
    observed_s: np.ndarray
    quality: np.ndarray
    history: tuple


def map_metrics(truth, labels):
    known = labels >= 0
    correct = known & (truth == labels)
    classes = [float(correct[truth == c].mean()) for c in (False, True) if np.any(truth == c)]
    return dict(correct_coverage=float(np.mean(classes)), unknown_fraction=float((~known).mean()),
                observed_error_rate=float((truth[known] != labels[known]).mean()) if known.any() else None,
                first_coverage=float(known.mean()))


class Task:
    def __init__(self, process, config=Config(), pose=None):
        config.validate(process)
        self._process = process
        self.config = config
        self.pose = pose or Pose(0, 0)
        if not all(math.isfinite(v) for v in asdict(self.pose).values()):
            raise ValueError("初始姿态必须有限")
        if not (-process.dx/2 <= self.pose.x_m < (process.shape[1]-.5)*process.dx and
                -process.dx/2 <= self.pose.z_m < (process.shape[0]-.5)*process.dx):
            raise ValueError("初始位置在地图外")
        self.time_s = 0.0
        self.memory = Memory(process.shape, config.history_length)
        self.observations, self.trajectory, self.actions, self.snapshots = [], [], [], []
        self.distance_m = self.turn_s = 0.0
        self._sample("initial", 0)

    def view(self):
        def readonly(a):
            result = a.copy()
            result.setflags(write=False)
            return result
        return PolicyView(self.pose, self.time_s, self._process.shape, self._process.dx,
                          readonly(self.memory.labels), readonly(self.memory.observed_s), readonly(self.memory.quality),
                          tuple(tuple(tuple(v) for v in h) for h in self.memory.history))

    def _sample(self, action, offset):
        p = self._process
        index = p.frame_index(self.time_s)
        visible = footprint(self.pose, self.config, p.shape, p.dx)
        cells = np.flatnonzero(visible).tolist()
        obs = dict(agent_id=0, frame_index=index, observed_s=self.time_s, arrived_s=self.time_s,
                   quality=1.0, cells=cells, labels=p.labels[index].flat[np.asarray(cells)].astype(int).tolist(),
                   action_index=len(self.actions)-1, sample_offset_s=offset)
        self.memory.accept(obs, self.time_s)
        self.observations.append(obs)
        point = dict(time_s=self.time_s, **asdict(self.pose), action=action, sample_offset_s=offset,
                     distance_m=self.distance_m, turn_s=self.turn_s)
        self.trajectory.append(point)
        self.snapshots.append(dict(**point, frame_index=index, visible=cells,
                                   labels=self.memory.labels.ravel().tolist(),
                                   observed_s=self.memory.observed_s.ravel().tolist(),
                                   metrics=map_metrics(p.labels[index], self.memory.labels)))

    def step(self, action):
        if self.time_s + self.config.action_s > self.config.duration_s + 1e-7:
            raise ValueError("任务已到截止时间")
        if action not in ACTIONS:
            raise ValueError("动作必须是 forward/left/right/hover")
        self.actions.append(dict(time_s=self.time_s, action=action))
        for k in range(round(self.config.action_s / self.config.sample_s)):
            self.pose, distance = move(self.pose, action, self.config.sample_s, self.config, self._process.shape, self._process.dx)
            self.distance_m += distance
            if action in ("left", "right"):
                self.turn_s += self.config.sample_s
            self.time_s = round(self.time_s + self.config.sample_s, 9)
            self._sample(action, (k+1)*self.config.sample_s)
        return self.view(), self.time_s >= self.config.duration_s - 1e-7

    def result(self):
        p = self._process
        times = np.array([s["time_s"] for s in self.snapshots])
        scores = np.array([s["metrics"]["correct_coverage"] for s in self.snapshots])
        duration = times[-1] - times[0]
        # Left rectangles: states at t describe [t,t_next); endpoint has zero duration.
        average = float(np.dot(scores[:-1], np.diff(times))/duration) if duration else float(scores[0])
        truth = p.labels[[s["frame_index"] for s in self.snapshots]]
        records = np.asarray([s["labels"] for s in self.snapshots]).reshape(truth.shape)
        arrivals = np.asarray([s["observed_s"] for s in self.snapshots]).reshape(truth.shape)
        changes = []
        for row in range(p.shape[0]):
            for col in range(p.shape[1]):
                starts = np.flatnonzero(truth[1:, row, col] != truth[:-1, row, col]) + 1
                for n, start in enumerate(starts):
                    stop = int(starts[n+1]) if n+1 < len(starts) else len(times)
                    detected = np.flatnonzero((arrivals[start:stop, row, col] >= times[start]) &
                                             (records[start:stop, row, col] == truth[start, row, col]))
                    found = float(times[start+detected[0]]) if len(detected) else None
                    changes.append(dict(cell=row*p.shape[1]+col, segment=n, changed_s=float(times[start]),
                                        segment_end_s=float(times[stop]) if stop < len(times) else float(times[-1]),
                                        detected_s=found, delay_s=found-float(times[start]) if found is not None else None,
                                        restricted_delay_s=(found if found is not None else float(times[min(stop, len(times)-1)]))-float(times[start])))
        revisit, all_sample_gaps = [], []
        for c in range(records.shape[1]*records.shape[2]):
            visits = [o["observed_s"] for o in self.observations if c in o["cells"]]
            gaps = np.diff(visits)
            all_sample_gaps.extend(gaps.tolist())
            revisit.extend(gaps[gaps > self.config.sample_s + 1e-7].tolist())
        reached = next((s["time_s"] for s in self.snapshots if s["metrics"]["first_coverage"] >= .9), None)
        summary = dict(time_average_correct_coverage=average, distance_m=self.distance_m, turn_s=self.turn_s,
                       first_90_percent_coverage_s=reached, cutoff_s=self.time_s,
                       change_count=len(changes), undiscovered_change_fraction=float(np.mean([c["detected_s"] is None for c in changes])) if changes else None,
                       restricted_change_delay_s=float(np.mean([c["restricted_delay_s"] for c in changes])) if changes else None,
                       revisit_median_s=float(np.median(revisit)) if revisit else None,
                       revisit_p90_s=float(np.quantile(revisit, .9)) if revisit else None,
                       continuous_sample_gap_median_s=float(np.median(all_sample_gaps)) if all_sample_gaps else None,
                       final=self.snapshots[-1]["metrics"])
        return dict(schema="hyra-uav-run-v1", config=asdict(self.config), source_sha256=p.digest,
                    source_name=p.path.name, grid=p.process["grid"], threshold_m=p.threshold,
                    evaluation=dict(time_integration="left_rectangle_actual_sample_intervals",
                                    change_segments="end_at_next_state_change_or_cutoff; undiscovered retained",
                                    revisit="last observation to next return, only gaps longer than sample period; continuous dwell excluded"),
                    summary=summary, actions=self.actions, trajectory=self.trajectory,
                    observations=self.observations, snapshots=self.snapshots, changes=changes,
                    truth=truth.astype(int).reshape(len(times), -1).tolist(), audit=p.audit())
