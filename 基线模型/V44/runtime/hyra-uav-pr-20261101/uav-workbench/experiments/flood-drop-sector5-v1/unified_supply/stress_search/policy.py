"""Team search from shared causal observations, with a belief over undetected people."""
import numpy as np

from planner import CENTERS, Planner, pool, smooth
from population import EPS, MOVES, RULES, response
from supply import OFFSETS

DIRECTIONS = MOVES[:8]


def advect(mass, dx, dy):
    """Conservative bilinear translation with open boundaries (cell units)."""
    for shift, axis in ((dx, 1), (dy, 0)):
        if abs(shift) < 1e-12:
            continue
        lower = int(np.floor(shift))
        fraction = shift - lower
        translated = np.zeros_like(mass)
        for offset, weight in ((lower, 1 - fraction), (lower + 1, fraction)):
            if weight == 0 or abs(offset) >= mass.shape[axis]:
                continue
            source, dest = [slice(None)] * 2, [slice(None)] * 2
            if offset > 0:
                source[axis], dest[axis] = slice(None, -offset), slice(offset, None)
            elif offset < 0:
                source[axis], dest[axis] = slice(-offset, None), slice(None, offset)
            translated[tuple(dest)] += weight * mass[tuple(source)]
        mass = translated
    return mass


def candidate_footprints(ends, heads):
    cells = np.floor(ends / 10).astype(int)[..., None, :] + OFFSETS
    delta = cells * 10 + 5 - ends[..., None, :]
    covers = ((cells >= 0) & (cells < 100)).all(-1)
    covers &= (delta * delta).sum(-1) <= 400 + EPS
    covers &= (delta * heads[..., None, :]).sum(-1) >= -EPS
    xy = cells.clip(0, 99)
    return 100 * xy[..., 1] + xy[..., 0], covers


def approach(pos, current, velocity, visible, ends):
    delta = current - pos
    if visible and np.linalg.norm(delta) <= 10 + EPS:
        return 8
    along = DIRECTIONS @ delta
    side2 = np.square(delta - along[:, None] * DIRECTIONS).sum(-1)
    entry = along - np.sqrt(np.maximum(0, 25 - side2))
    hit = visible & (side2 <= 25 + EPS) & (entry >= 0) & (entry <= np.linalg.norm(ends-pos, axis=1) + EPS)
    future = np.where(hit[:, None], pos + entry[:, None] * DIRECTIONS, ends)
    target = current + 5 * velocity
    diff = target - future
    dist = np.linalg.norm(diff, axis=1)
    sight = (dist <= 20 + EPS) & ((DIRECTIONS * diff).sum(-1) >= -EPS)
    ready = sight & (dist <= 10 + EPS)
    if ready.any():
        return int(np.argmin(np.where(ready, dist, np.inf)))
    # Prefer an approach from which a second legal move can intercept.
    next_along = diff @ DIRECTIONS.T
    next_side2 = np.maximum(0, (diff * diff).sum(-1)[:,None] - next_along**2)
    next_hit = sight[:,None] & (next_side2 <= 25) & (next_along >= 0) & (next_along <= 50)
    next_ends = np.clip(future[:,None] + 50 * DIRECTIONS, 0, 1000)
    next_diff = target + 5 * velocity - next_ends
    next_dist = np.linalg.norm(next_diff,axis=-1)
    next_ready = (next_dist <= 10) & ((next_diff * DIRECTIONS).sum(-1) >= 0)
    return int(np.argmin(dist + .65 * np.min(next_dist, axis=1) -
                         75 * (next_hit | next_ready).any(axis=1)))


class StressSearch:
    def __init__(self, theta, params):
        self.original = Planner("mlp", 10, theta)
        self.transport, self.reach, self.memory = params
        self.use_belief = None
        self.standing = np.ones((100, 100))
        self.drifting = np.zeros_like(self.standing)
        self.invalid = {}
        self.last_speed = np.zeros(2)
        self.switch_time = None

    def act(self, obs):
        if self.use_belief is None:
            known = obs.last >= 0
            if known.sum() >= 40 or obs.t >= 60:
                self.use_belief = np.mean(obs.measured[0][known] >= .01) >= .8
                if self.use_belief:
                    self.standing[known] = 0
                    self.switch_time = obs.t
        if not self.use_belief:
            return self.original.act(obs)
        ids = obs.observed_cells
        if len(ids):
            h, vx, vy = np.median(obs.measured.reshape(3,-1)[:,ids], axis=1)
        else:
            h, vx, vy = 0., 0., 0.
        # Use previous measurements for elapsed transport, current data for future forecasts.
        self.drifting = advect(self.drifting, *(self.last_speed * .5))
        self.last_speed = self.transport * np.array([vx,vy])
        fall = -np.expm1(-5 * RULES["hazard_k"] * response(h, np.hypot(vx,vy))) * self.transport
        self.drifting += self.standing * fall
        self.standing *= 1 - fall
        self.standing.ravel()[ids] = 0
        self.drifting.ravel()[ids] = 0
        belief = self.standing + self.drifting
        ends = np.clip(obs.pos[:,None] + 50 * DIRECTIONS, 0, 1000)
        travel = ends - obs.pos[:,None]
        length = np.linalg.norm(travel, axis=-1)
        heads = travel / np.maximum(length[...,None], 1e-12)
        index, visible = candidate_footprints(ends, heads)
        live = max(.05, float(belief.mean()))
        local = (belief.ravel()[index] * visible).sum(-1)
        local += .12 * live * (heads * obs.heading[:,None]).sum(-1)
        local[length < 49.999] -= 20 * live
        coarse = pool(belief)
        utility = .45 * coarse + .55 * smooth(coarse)
        reserved = np.ones_like(utility)
        assigned, targets = self.assign(obs)
        actions = np.empty(4,dtype=int)
        for i, j in enumerate(assigned):
            if j >= 0:
                actions[i] = approach(obs.pos[i], targets[i], obs.seen_v[j], obs.visible[j], ends[i])
                reserved *= np.clip(np.linalg.norm(CENTERS - targets[i], axis=-1)/160, .1, 1)
        for i in np.flatnonzero(assigned < 0):
            dist = np.linalg.norm(CENTERS - obs.pos[i], axis=-1)
            interest = utility * reserved / (1 + dist / 180)
            interest[dist < 55] *= .25
            goal = CENTERS[np.unravel_index(interest.argmax(),interest.shape)]
            progress = (np.linalg.norm(goal-obs.pos[i]) - np.linalg.norm(goal-ends[i],axis=-1)) / 50
            sep = np.linalg.norm(ends[i,:,None] - obs.pos[None,:],axis=-1)
            sep[:,i] = np.inf
            actions[i] = np.argmax(local[i] + self.reach * live * progress -
                                   4 * live * np.clip(1 - sep.min(axis=1)/90,0,1))
            reserved *= np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/180,.05,1)
        return actions, assigned

    def assign(self, obs):
        ages = obs.t - obs.seen_at
        candidates = np.flatnonzero(obs.active & (obs.seen_at >= 0) & (ages <= self.memory))
        assigned = np.full(4, -1, int)
        targets = obs.pos.copy()
        claims = []
        for j in candidates:
            prediction = obs.seen_xy[j] + ages[j] * obs.seen_v[j]
            if not obs.visible[j]:
                ray = prediction - obs.pos
                if ((np.linalg.norm(ray,axis=1) <= 20) & ((ray*obs.heading).sum(-1) >= 0)).any():
                    self.invalid[j] = obs.seen_at[j]
                if self.invalid.get(j,-1) == obs.seen_at[j]:
                    continue
            distance = np.linalg.norm(obs.pos - prediction,axis=1)
            for i in np.flatnonzero(distance <= 180):
                claims.append((obs.owner[j] != i, not(obs.visible[j] and distance[i] <= 10+EPS),
                               distance[i] + .25*ages[j], j, i, prediction))
        used = set()
        for _, _, _, j, i, xy in sorted(claims,key=lambda item:item[:5]):
            if assigned[i] < 0 and j not in used:
                used.add(j)
                assigned[i] = j
                targets[i] = xy
        return assigned, targets
