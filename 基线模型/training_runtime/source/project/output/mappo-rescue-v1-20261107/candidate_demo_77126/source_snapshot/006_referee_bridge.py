"""Thin, non-training bridge to an explicitly supplied 20261106.1 referee.

The caller constructs the official DeliveryMission after setting its radius to
one metre. This module does not import or execute the distributed referee.
"""
from types import SimpleNamespace

import numpy as np

OBS_FIELDS = ("t", "pos", "heading", "last", "measured", "seen_xy", "seen_v",
              "seen_at", "visible", "active", "owner", "new_cells", "observed_cells")
# Local PROPOSED_DIRECTIONS to referee MOVES; axes must be shared x, row/y.
LOCAL_TO_REFEREE = np.array([1, 0, 2, 3, 5, 4, 7, 6, 8], dtype=np.int64)
METRICS = ("ST", "ACC", "RCR", "FRT", "CRT", "coverage")


def public_view(env):
    raw = env.observation()
    fields = {}
    for name in OBS_FIELDS:
        value = getattr(raw, name)
        if isinstance(value, np.ndarray):
            value = value.copy()
            value.flags.writeable = False
        fields[name] = value
    return SimpleNamespace(**fields)


def _integers(values, lower, upper, label):
    result = np.asarray(values)
    if (result.shape != (4,) or result.dtype.kind not in "iu"
            or np.any(result < lower) or np.any(result > upper)):
        raise ValueError(f"{label} must be four integers in [{lower}, {upper}]")
    return result.astype(np.int64, copy=True)


class RefereeBridge:
    """Validate actions and retain the referee's own transitions and metrics.

    No planning, target assignment, action masking, reward or extra physics is
    introduced. This is not yet a learner-specific observation encoder.
    """
    def __init__(self, env):
        if (getattr(env, "count", None), getattr(env, "window", None),
                getattr(env, "duration", None), getattr(env, "radius", None)) != (64, 720, 2400, 1.0):
            raise ValueError("expected 64 people, 720 s window, 2400 s and 1 m radius")
        # The referee uses judge.within_radius, whose global radius can differ
        # from env.radius if initialization/import order was altered.
        step_globals = getattr(getattr(env.step, '__func__', env.step), '__globals__', {})
        within = step_globals.get('within_radius')
        judge_globals = getattr(within, '__globals__', {})
        if not callable(within) or judge_globals.get('DELIVERY_RADIUS_M') != 1.0:
            raise ValueError("referee within_radius must actually use a 1 m radius")
        if env.t != 0:
            raise ValueError("referee must be at initial time")
        self.env = env
        self.done = False

    def observation(self):
        return public_view(self.env)

    def step(self, actions, targets, *, local_action_order=True):
        if self.done:
            raise ValueError("episode ended")
        actions = _integers(actions, 0, 8, "actions")
        targets = _integers(targets, -1, 63, "targets")
        if local_action_order:
            actions = LOCAL_TO_REFEREE[actions]
        self.env.step(actions, targets)
        self.done = bool(self.env.t >= self.env.duration or
                         (np.all(self.env.supplied >= 0) and np.all(self.env.last >= 0)))
        metrics = self.env.metrics() if self.done else None
        return self.observation(), self.done, metrics
