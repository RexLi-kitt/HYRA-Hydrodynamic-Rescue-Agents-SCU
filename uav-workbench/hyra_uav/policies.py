"""Rule baselines receive only immutable legal observation histories."""
import math
import numpy as np
from .core import ACTIONS, footprint, move


class ScanPolicy:
    """Predefined lawnmower waypoints, with explicit paid rotations and looping."""
    def __init__(self, view, config):
        step = config.speed_m_s * config.action_s
        # One action advances one waypoint; lane spacing bounded by sensing width.
        xs = np.arange(0, (view.shape[1]-1)*view.dx + step/2, step)
        xs = np.clip(xs, 0, (view.shape[1]-1)*view.dx)
        if xs[-1] < (view.shape[1]-1)*view.dx:
            xs = np.append(xs, (view.shape[1]-1)*view.dx)
        spacing = max(step, config.range_m * math.sin(math.radians(config.fov_deg/2)))
        zs = list(np.arange(0, (view.shape[0]-1)*view.dx, spacing)) + [(view.shape[0]-1)*view.dx]
        self.waypoints = [(float(x), float(z)) for i, z in enumerate(zs) for x in (xs if i % 2 == 0 else xs[::-1])]
        # Return along the same fixed path, never teleport from its end to start.
        self.waypoints += self.waypoints[-2:0:-1]
        self.index = min(range(len(self.waypoints)), key=lambda i: math.hypot(self.waypoints[i][0]-view.pose.x_m, self.waypoints[i][1]-view.pose.z_m))

    def choose(self, view, config):
        step = config.speed_m_s * config.action_s
        for _ in self.waypoints:
            x, z = self.waypoints[self.index]
            if math.hypot(x-view.pose.x_m, z-view.pose.z_m) > max(step*.51, view.dx*.5):
                break
            self.index = (self.index+1) % len(self.waypoints)
        desired = math.degrees(math.atan2(z-view.pose.z_m, x-view.pose.x_m))
        error = (desired-view.pose.heading_deg+180) % 360 - 180
        turn = config.angular_speed_deg_s * config.action_s
        if abs(error) > max(turn*.51, 5):
            return "right" if error > 0 else "left"
        return "forward"


class GreedyPolicy:
    """Same two-action geometric lookahead for R1/R2; no hidden map access."""
    def __init__(self, use_age=False):
        self.use_age = use_age

    def choose(self, view, config):
        known = view.labels >= 0
        age = np.maximum(0, view.time_s-view.observed_s) / config.duration_s
        utility = np.where(known, age*config.age_weight if self.use_age else 0, 1.0)
        targets = np.argwhere(~known)
        best = (-math.inf, "forward")
        for first in ACTIONS:
            for second in ACTIONS:
                pose, seen, score = view.pose, np.zeros(view.shape, bool), 0.0
                for depth, action in enumerate((first, second)):
                    for _ in range(round(config.action_s/config.sample_s)):
                        pose, _ = move(pose, action, config.sample_s, config, view.shape, view.dx)
                        visible = footprint(pose, config, view.shape, view.dx)
                        score += float(utility[visible & ~seen].sum()) * (.8 ** depth)
                        seen |= visible
                if len(targets):
                    # A small geometry-only tie breaker escapes already-covered pockets.
                    distance = np.hypot(targets[:, 1]*view.dx-pose.x_m, targets[:, 0]*view.dx-pose.z_m).min()
                    score -= .025 * distance / view.dx
                # Deterministic tie order prefers progress, then turning, then hovering.
                if score > best[0] + 1e-10:
                    best = (score, first)
        return best[1]


def policy_for(name, view, config):
    if name == "R0":
        return ScanPolicy(view, config)
    if name in ("R1", "R2"):
        return GreedyPolicy(name == "R2")
    raise ValueError("策略必须是 R0、R1 或 R2")
