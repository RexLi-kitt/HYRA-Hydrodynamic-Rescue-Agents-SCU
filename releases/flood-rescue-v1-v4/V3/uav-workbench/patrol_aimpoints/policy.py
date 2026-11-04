"""Within-zone patrol only. All arguments are available before takeoff."""
import numpy as np
from dynamic_drop_rebuild.planner import CENTERS, GRID, ZONE
from dynamic_drop_rebuild.scenarios import T
from flood_dqn.env import close

OFFSETS = np.array([[0, 0], [-25, -25], [-25, 25], [25, -25], [25, 25]])
CELLS = np.stack([np.flatnonzero(ZONE == z) for z in range(100)])
COORD = GRID[CELLS]


def routes(pos, goals, horizon):
    delta = goals - pos
    distance = np.linalg.norm(delta, axis=-1)
    fraction = np.minimum(1, 50 * np.arange(1, horizon + 1) / np.maximum(distance[:, None], 1e-12))
    return pos + delta[:, None, :] * fraction[:, :, None]


def first_view(paths, cells, radius):
    dx = paths[..., 0, None] - cells[:, 0]
    dz = paths[..., 1, None] - cells[:, 1]
    visible = dx * dx + dz * dz <= radius ** 2
    return np.where(visible.any(-2), visible.argmax(-2) + 1, 99)


def conflict(p, end, q, other):
    d = end - p
    e = other - q
    a, b = np.linalg.norm(d), np.linalg.norm(e)
    return close(p, d / max(a, 1e-12), a, q, e / max(b, 1e-12), b)


def select(group, stamp, p, t, zones, base, radius):
    end = base.copy()
    indices = np.where(zones >= 0, 0, -1)
    if group == 'K0':
        return end, indices
    coordinated = group == 'K2'
    published = {i: base[i:i+1] for i in range(4) if zones[i] < 0}
    horizon = min(4, (T-t)//5)
    for k in range(4):
        i = (t//20 + k) % 4
        z = zones[i]
        if z < 0:
            continue
        cells = COORD[z]
        goals = CENTERS[z] + OFFSETS
        paths = routes(p[i], goals, horizon)
        first = first_view(paths, cells, radius)
        seen = first < 99
        feasible = np.ones(5, dtype=bool)
        if coordinated:
            deadline = np.full(len(cells), 99)
            for j, route in published.items():
                deadline = np.minimum(deadline, first_view(route, cells, radius))
                for a in np.flatnonzero(feasible):
                    feasible[a] = not conflict(p[i], paths[a,0], p[j], route[0])
            eligible = seen & (first < deadline)
        else:
            eligible = seen
        times = stamp[CELLS[z]]
        weights = np.where(times < 0, 1., np.minimum((t-times)/240., 1.))
        score = (eligible * weights / first).sum(1)
        choice = 0
        if feasible.any() and seen.any():
            dist = np.linalg.norm(goals - p[i], axis=1)
            choice = min(np.flatnonzero(feasible), key=lambda a: (-score[a], a != 0, dist[a], a))
        indices[i] = choice
        if choice:
            end[i] = paths[choice, 0]
        if coordinated:
            published[i] = paths[choice]
    return end, indices


def footprint(pos, radius):
    x,z = pos
    lo = np.maximum(0, np.ceil((pos-radius-5)/10).astype(int))
    hi = np.minimum(99, np.floor((pos+radius-5)/10).astype(int))
    xs = np.arange(lo[0],hi[0]+1)
    zs = np.arange(lo[1],hi[1]+1)[:,None]
    ids = xs + 100*zs
    return set(ids[(xs*10+5-x)**2 + (zs*10+5-z)**2 <= radius**2].tolist())

