"""Synchronous actual-frame multi-UAV sensing with reliable shared records.

Actors receive an immutable TeamView. Hidden hydraulic fields remain in the
environment and are available only via the explicit centralized-training seam.
"""
from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace

import numpy as np

from hyra_uav.core import ACTIONS, Config, Memory, PolicyView, Pose, Task, footprint, map_metrics, move
from hyra_uav.interfaces import merge_arrived


@dataclass(frozen=True)
class TeamConfig:
    separation_m: float = 2.
    distance_cost_per_m: float = .00002
    turn_cost_per_s: float = .00002
    collision_cost: float = .005

    def validate(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()) or self.separation_m <= 0:
            raise ValueError('Team distances and costs must be finite and nonnegative')


@dataclass(frozen=True)
class TeamView:
    agents: tuple
    bed: np.ndarray


def readonly(value):
    result = np.asarray(value).copy()
    result.setflags(write=False)
    return result


def path_separation(a, b, next_a, next_b):
    relative = np.array([a.x_m-b.x_m, a.z_m-b.z_m])
    delta = np.array([next_a.x_m-next_b.x_m, next_a.z_m-next_b.z_m])-relative
    fraction = float(np.clip(-relative@delta/(delta@delta), 0, 1)) if delta@delta else 0.
    return float(np.linalg.norm(relative+fraction*delta))


class TeamEnv:
    def __init__(self, process, config=Config(), team=TeamConfig(), poses=None):
        config.validate(process)
        team.validate()
        self._process, self.config, self.team = process, config, team
        self.initial_poses = tuple(poses or (Pose(0, 0), Pose(0, (process.shape[0]-1)*process.dx)))
        if len(self.initial_poses) < 2:
            raise ValueError('A team needs at least two agents')
        for pose in self.initial_poses:
            if (not all(math.isfinite(v) for v in asdict(pose).values()) or
                not -process.dx/2 <= pose.x_m < (process.shape[1]-.5)*process.dx or
                not -process.dx/2 <= pose.z_m < (process.shape[0]-.5)*process.dx):
                raise ValueError('Initial pose outside the physical domain')
        for i, a in enumerate(self.initial_poses):
            for b in self.initial_poses[i+1:]:
                if math.hypot(a.x_m-b.x_m, a.z_m-b.z_m) < team.separation_m:
                    raise ValueError('Initial poses violate minimum separation')
        self.n_agents = len(self.initial_poses)
        self._static_bed = readonly(process.bed)
        self.time_s = None

    def reset(self, seed=None):
        self.poses = list(self.initial_poses)
        self.time_s = self.distance_m = self.turn_s = self.integrated_quality = 0.
        self.collision_events = 0
        self.memory = Memory(self._process.shape, self.config.history_length)
        self.actions, self.observations, self.snapshots, self.trajectory = [], [], [], []
        self._sample(['initial']*self.n_agents, 0.)
        return self.view(), dict(seed=seed, time_s=0., source_sha256=self._process.digest)

    def view(self):
        if self.time_s is None:
            raise ValueError('Call reset first')
        history = tuple(tuple(tuple(v) for v in h) for h in self.memory.history)
        agents = tuple(PolicyView(pose, self.time_s, self._process.shape, self._process.dx,
                                 readonly(self.memory.labels), readonly(self.memory.observed_s),
                                 readonly(self.memory.quality), history) for pose in self.poses)
        return TeamView(agents, readonly(self._static_bed))

    def _sample(self, actions, offset):
        index = self._process.frame_index(self.time_s)
        observations = []
        for agent_id, pose in enumerate(self.poses):
            cells = np.flatnonzero(footprint(pose, self.config, self._process.shape, self._process.dx)).tolist()
            observations.append(dict(agent_id=agent_id, frame_index=index, observed_s=self.time_s,
                                     arrived_s=self.time_s, quality=1., cells=cells,
                                     labels=self._process.labels[index].ravel()[cells].astype(int).tolist(),
                                     action_index=len(self.actions)-1, sample_offset_s=offset))
        # All agents observe the SAME actual hydraulic frame before merging.
        merge_arrived(self.memory, observations, self.time_s)
        self.observations.extend(observations)
        point = dict(time_s=self.time_s, poses=[asdict(p) for p in self.poses],
                     actions=list(actions), sample_offset_s=offset, distance_m=self.distance_m,
                     turn_s=self.turn_s, collision_events=self.collision_events)
        self.trajectory.append(point)
        self.snapshots.append(dict(**point, frame_index=index, labels=self.memory.labels.ravel().tolist(),
                                   observed_s=self.memory.observed_s.ravel().tolist(),
                                   metrics=map_metrics(self._process.labels[index], self.memory.labels)))

    def _joint_motion(self, actions, dt):
        proposed = [move(p, a, dt, self.config, self._process.shape, self._process.dx)
                    for p, a in zip(self.poses, actions)]
        blocked, conflicts = set(), set()
        # Cancel conflicting translations, then recheck because one cancelled
        # path can obstruct a third agent. Rotations retain their paid duration.
        for _ in range(self.n_agents+1):
            end = [self.poses[i] if i in blocked else value[0] for i, value in enumerate(proposed)]
            more = set()
            for i in range(self.n_agents):
                for j in range(i+1, self.n_agents):
                    if path_separation(self.poses[i], self.poses[j], end[i], end[j]) < self.team.separation_m-1e-9:
                        conflicts.add((i, j))
                        more.update(k for k in (i, j) if actions[k] == 'forward' and k not in blocked)
            if not more:
                break
            blocked |= more
        else:
            raise RuntimeError('Collision cancellation did not converge')
        self.poses = [self.poses[i] if i in blocked else value[0] for i, value in enumerate(proposed)]
        distance = sum(value[1] for i, value in enumerate(proposed) if i not in blocked)
        turns = sum(a in ('left', 'right') for a in actions)*dt
        self.distance_m += distance
        self.turn_s += turns
        self.collision_events += len(conflicts)
        return distance, turns, len(conflicts)

    def step(self, joint_actions):
        if self.time_s is None or self.time_s >= self.config.duration_s-1e-7:
            raise ValueError('Environment not reset or already truncated')
        if len(joint_actions) != self.n_agents:
            raise ValueError('Expected one action per agent')
        actions = []
        for value in joint_actions:
            if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) and 0 <= value < len(ACTIONS):
                value = ACTIONS[value]
            if value not in ACTIONS:
                raise ValueError('Invalid joint action')
            actions.append(value)
        self.actions.append(dict(time_s=self.time_s, actions=actions))
        reward = 0.
        for k in range(round(self.config.action_s/self.config.sample_s)):
            # Same left-rectangle physical-time objective as the existing evaluator.
            dt = self.config.sample_s
            quality = self.snapshots[-1]['metrics']['correct_coverage']
            self.integrated_quality += quality*dt
            distance, turns, collisions = self._joint_motion(actions, dt)
            reward += quality*dt/self.config.duration_s-self.team.distance_cost_per_m*distance-self.team.turn_cost_per_s*turns-self.team.collision_cost*collisions
            self.time_s = round(self.time_s+dt, 9)
            self._sample(actions, (k+1)*dt)
        return self.view(), float(reward), False, self.time_s >= self.config.duration_s-1e-7, dict(time_s=self.time_s)

    def training_state(self):
        """Explicit CTDE-only access. Never included in TeamView or actor args."""
        index = self._process.frame_index(self.time_s)
        return readonly(self._process.depths[index])

    def result(self):
        # Reuse the existing audited evaluator on shared UNION records. The
        # adapter supplies no actor inputs; truth stays in the offline result.
        adapter = SimpleNamespace(_process=self._process, config=self.config, time_s=self.time_s,
                                  snapshots=self.snapshots, observations=self.observations,
                                  distance_m=self.distance_m, turn_s=self.turn_s,
                                  actions=self.actions, trajectory=self.trajectory)
        result = Task.result(adapter)
        result.update(schema='hyra-team-run-v1', team=asdict(self.team), n_agents=self.n_agents,
                      initial_poses=[asdict(p) for p in self.initial_poses])
        result['summary']['collision_events'] = self.collision_events
        return result
