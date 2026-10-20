"""Published forecast support and complete-service windows."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def service_start_intervals(times, depths, duration=12., threshold=.8):
    """Exact safe-start intervals for a piecewise-linear saved depth curve."""
    times, depths = np.asarray(times, float), np.asarray(depths, float)
    if times.ndim != 1 or depths.shape != times.shape or len(times) < 2:
        raise ValueError("Expected a time/depth curve")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(depths)) or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid forecast curve")
    if duration <= 0:
        raise ValueError("Service duration must be positive")
    wet_intervals = []
    for a, b, ha, hb in zip(times[:-1], times[1:], depths[:-1], depths[1:]):
        if ha <= threshold and hb <= threshold:
            wet_intervals.append((float(a), float(b)))
        elif ha <= threshold < hb:
            wet_intervals.append((float(a), float(a + (b - a) * (threshold - ha) / (hb - ha))))
        elif hb <= threshold < ha:
            wet_intervals.append((float(a + (b - a) * (threshold - ha) / (hb - ha)), float(b)))
    merged = []
    for a, b in wet_intervals:
        if merged and a <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return tuple((a, b - duration) for a, b in merged if b - a >= duration - 1e-9)


@dataclass(frozen=True)
class WindowEstimate:
    cell: tuple[int, int]
    issued_at: float
    version: int
    future_times: tuple[float, ...]
    future_depths: tuple[float, ...]
    service_intervals: tuple[tuple[float, float], ...]
    evidence_supported: bool
    valid_until: float
    evidence: str = "independent_nominal_only"
    service_duration_s: float = 12.

    @property
    def query_cell(self):
        return self.cell

    def can_start(self, start):
        if start < self.issued_at - 1e-9 or start + self.service_duration_s > self.valid_until + 1e-9:
            return None
        return any(a - 1e-9 <= start <= b + 1e-9 for a, b in self.service_intervals)

    def earliest_start(self, arrival):
        if arrival < self.issued_at - 1e-9:
            return None
        for a, b in self.service_intervals:
            start = max(float(arrival), a)
            if start <= b + 1e-9:
                return start
        return None


class NominalPredictor:
    """Engineering adapter, not a trained H/E/P information module.

    It only receives an independently simulated nominal forecast and already
    arrived observations; it validates observation causality but does not learn
    or assimilate them. Publication horizon is explicitly capped.
    """
    def __init__(self, times, depths, *, horizon_s=60., service_duration_s=12., threshold_m=.8):
        self.times = np.asarray(times, float)
        self.depths = np.asarray(depths, float)
        if self.depths.ndim != 3 or self.depths.shape[0] != len(self.times) or np.any(np.diff(self.times) <= 0):
            raise ValueError("Expected nominal depths[T,H,W] with increasing time")
        if not np.all(np.isfinite(self.depths)):
            raise ValueError("Nonfinite nominal field")
        self.horizon_s, self.service_duration_s, self.threshold_m = horizon_s, service_duration_s, threshold_m
        self.issued_at, self.version, self.valid_until = None, 0, None

    def issue(self, now, arrived_observations):
        if self.issued_at is not None and now < self.issued_at:
            raise ValueError("Forecast clock moved backwards")
        for row in arrived_observations:
            if row["sample_time"] > row["arrival_time"] + 1e-9 or row["arrival_time"] > now + 1e-9:
                raise ValueError("Future or not-yet-arrived observation")
        self.issued_at = float(now)
        self.valid_until = min(float(self.times[-1]), now + self.horizon_s)
        self.version += 1

    def estimate(self, cell, now):
        if self.issued_at is None or now < self.issued_at - 1e-9:
            raise ValueError("Forecast not published yet")
        cell = tuple(map(int, cell))
        curve = self.depths[:, cell[0], cell[1]]
        inside = self.times[(self.times > self.issued_at) & (self.times < self.valid_until)]
        times = np.unique(np.r_[self.issued_at, inside, self.valid_until])
        depths = np.interp(times, self.times, curve)
        intervals = service_start_intervals(times, depths, self.service_duration_s, self.threshold_m) if len(times) >= 2 else ()
        return WindowEstimate(cell, self.issued_at, self.version, tuple(map(float, times)),
                              tuple(map(float, depths)), intervals, False, self.valid_until,
                              service_duration_s=self.service_duration_s)

    def estimate_many(self, cells, now):
        """Same published WindowEstimate contract, batched without extrapolation.

        Public grid locations may be queried before a target is discovered.
        This API accepts locations only, never a hidden target map.
        """
        if self.issued_at is None or now < self.issued_at - 1e-9:
            raise ValueError("Forecast not published yet")
        cells = np.asarray(cells, int).reshape(-1, 2)
        if not len(cells):
            return []
        if np.any(cells < 0) or np.any(cells[:, 0] >= self.depths.shape[1]) or np.any(cells[:, 1] >= self.depths.shape[2]):
            raise ValueError("Forecast query outside public grid")
        inside = self.times[(self.times > self.issued_at) & (self.times < self.valid_until)]
        times = np.unique(np.r_[self.issued_at, inside, self.valid_until])
        upper = np.clip(np.searchsorted(self.times, times, side="right"), 1, len(self.times) - 1)
        lower = upper - 1
        weight = ((times - self.times[lower]) / (self.times[upper] - self.times[lower]))[:, None]
        values = ((1 - weight) * self.depths[lower[:, None], cells[None, :, 0], cells[None, :, 1]] +
                  weight * self.depths[upper[:, None], cells[None, :, 0], cells[None, :, 1]])
        safe = values <= self.threshold_m
        changes = np.diff(np.pad(safe.astype(np.int8), ((1, 1), (0, 0))), axis=0)
        first_t, first_c = np.nonzero(changes == 1)
        last_t, last_c = np.nonzero(changes == -1)
        intervals = [[] for _ in cells]
        starts = {(int(t), int(col)): float(times[t]) for t, col in zip(first_t, first_c)}
        # Iterate over safe runs (usually one per cell), rather than T*N segments.
        begin_by_col = [[] for _ in cells]
        for t, col in zip(first_t, first_c):
            begin_by_col[col].append(int(t))
        end_by_col = [[] for _ in cells]
        for t, col in zip(last_t, last_c):
            end_by_col[col].append(int(t) - 1)
        for col in range(len(cells)):
            for first, last in zip(begin_by_col[col], end_by_col[col]):
                a, b = starts[first, col], float(times[last])
                if first > 0:
                    a = float(times[first - 1] + (times[first] - times[first - 1]) *
                              (self.threshold_m - values[first - 1, col]) / (values[first, col] - values[first - 1, col]))
                if last < len(times) - 1:
                    b = float(times[last] + (times[last + 1] - times[last]) *
                              (self.threshold_m - values[last, col]) / (values[last + 1, col] - values[last, col]))
                if b - a >= self.service_duration_s - 1e-9:
                    intervals[col].append((a, b - self.service_duration_s))
        stamp = tuple(map(float, times))
        return [WindowEstimate(tuple(map(int, cell)), self.issued_at, self.version, stamp,
                               tuple(map(float, values[:, i])), tuple(intervals[i]), False, self.valid_until,
                               service_duration_s=self.service_duration_s) for i, cell in enumerate(cells)]
