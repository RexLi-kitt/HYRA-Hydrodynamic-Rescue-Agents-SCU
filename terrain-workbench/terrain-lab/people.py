"""Rescue start and trapped-person placement on the hydraulic grid.

Coordinates follow the structure layer: local metres from the displayed grid
outer corner, positive +X to the right and +Z downward. Legality uses the
fixed environment only (buildings, sealed wall faces, active levees); flood
water is deliberately not a barrier because it changes during the run.
"""
from __future__ import annotations
import math
import numpy as np

SCHEMA = 'hyra-people-v1'
MAX_PEOPLE = 200
PERSON_KINDS = ('trapped', 'injured')
COORDINATE_SYSTEM = 'local x_m/z_m from displayed grid outer corner, positive right/down; independent of grid_factor'


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name}需要是有限数值')
    return float(value)


def _cell(x_m, z_m, shape, dx):
    nz, nx = shape
    x = int(math.floor(x_m / dx + 1e-9))
    z = int(math.floor(z_m / dx + 1e-9))
    if not 0 <= x < nx or not 0 <= z < nz:
        raise ValueError('人员坐标超出地图范围')
    return z, x


def ocean_mask(terrain_class, terrain_ids=None):
    """Cells whose land-cover name contains "ocean"; people are not placed there."""
    classes = np.asarray(terrain_class)
    codes = set()
    if isinstance(terrain_ids, dict):
        for code, name in terrain_ids.items():
            try:
                number = int(code)
            except (TypeError, ValueError):
                continue
            if 'ocean' in str(name).lower():
                codes.add(number)
    if not codes:
        return np.zeros(classes.shape, bool)
    return np.isin(classes, sorted(codes))


def barrier_masks(shape, wall_x, wall_y, levee_edges=()):
    """Faces that a walking person cannot cross: sealed walls plus active levees."""
    nz, nx = shape
    blocked_x = np.asarray(wall_x, bool).copy()
    blocked_y = np.asarray(wall_y, bool).copy()
    for axis, z, x in levee_edges:
        if axis == 1:
            if 0 <= z < nz and 0 <= x <= nx:
                blocked_x[z, x] = True
        else:
            if 0 <= z <= nz and 0 <= x < nx:
                blocked_y[z, x] = True
    return blocked_x, blocked_y


def _open_faces(z, x, closed, blocked_x, blocked_y):
    """Number of traversable 4-neighbours of a cell."""
    nz, nx = closed.shape
    count = 0
    if x > 0 and not closed[z, x - 1] and not blocked_x[z, x]:
        count += 1
    if x + 1 < nx and not closed[z, x + 1] and not blocked_x[z, x + 1]:
        count += 1
    if z > 0 and not closed[z - 1, x] and not blocked_y[z, x]:
        count += 1
    if z + 1 < nz and not closed[z + 1, x] and not blocked_y[z + 1, x]:
        count += 1
    return count


def reachable_from(start, closed, blocked_x, blocked_y):
    """Cells connected to ``start`` through open cells and faces."""
    nz, nx = closed.shape
    seen = np.zeros((nz, nx), bool)
    z0, x0 = start
    if closed[z0, x0]:
        return seen
    seen[z0, x0] = True
    stack = [(z0, x0)]
    while stack:
        z, x = stack.pop()
        for nz_, nx_ in ((z, x - 1), (z, x + 1), (z - 1, x), (z + 1, x)):
            if not (0 <= nz_ < nz and 0 <= nx_ < nx) or seen[nz_, nx_] or closed[nz_, nx_]:
                continue
            if nz_ == z:
                if blocked_x[z, max(x, nx_)]:
                    continue
            elif blocked_y[max(z, nz_), x]:
                continue
            seen[nz_, nx_] = True
            stack.append((nz_, nx_))
    return seen


def default_start(closed, bed, blocked_x, blocked_y):
    """Lowest eligible boundary cell; the rescue team enters from the map rim."""
    nz, nx = closed.shape
    candidates = []
    for z in range(nz):
        for x in range(nx):
            if z not in (0, nz - 1) and x not in (0, nx - 1):
                continue
            if closed[z, x] or _open_faces(z, x, closed, blocked_x, blocked_y) == 0:
                continue
            candidates.append((float(bed[z, x]), z, x))
    if not candidates:
        for z in range(nz):
            for x in range(nx):
                if not closed[z, x] and _open_faces(z, x, closed, blocked_x, blocked_y):
                    candidates.append((float(bed[z, x]), z, x))
    if not candidates:
        raise ValueError('地图上没有可站立的人员起点')
    _, z, x = min(candidates)
    return z, x


def normalize(people, *, shape, dx, solid, bed, wall_x, wall_y, levee_edges=(), terrain_class=None, terrain_ids=None):
    """Validate and compile a people layer for one hydraulic grid.

    Raises ``ValueError`` with a user-facing reason when a person sits inside a
    building, the rescue start is sealed, or a target is not reachable.
    """
    nz, nx = shape
    solid = np.asarray(solid, bool)
    ocean = ocean_mask(terrain_class, terrain_ids) if terrain_class is not None else np.zeros(shape, bool)
    closed = solid | ocean
    blocked_x, blocked_y = barrier_masks(shape, wall_x, wall_y, levee_edges)
    raw = people if isinstance(people, dict) else {}
    mode = raw.get('mode', 'manual')
    if mode not in ('uniform', 'cluster', 'cluster_isolated', 'manual'):
        raise ValueError('人员放置模式不正确')
    seed = raw.get('seed', 0)
    if isinstance(seed, bool) or not isinstance(seed, (int, float)) or not math.isfinite(seed):
        raise ValueError('随机种子需要是数值')
    parameters = raw.get('parameters') if isinstance(raw.get('parameters'), dict) else {}

    items = raw.get('items') or []
    if not isinstance(items, list) or len(items) > MAX_PEOPLE:
        raise ValueError(f'人员列表需要是数组，最多{MAX_PEOPLE}个')
    start_raw = raw.get('start')
    if start_raw is None and not items:
        return dict(schema=SCHEMA, coordinate_system=COORDINATE_SYSTEM, mode=mode, seed=float(seed),
                    parameters=dict(parameters), start=None, items=[],
                    counts=dict(total=0, reachable=0, connected_cells=0))
    if start_raw is None:
        start_cell = default_start(closed, bed, blocked_x, blocked_y)
        start = dict(x_m=(start_cell[1] + .5) * dx, z_m=(start_cell[0] + .5) * dx, cell=list(start_cell), automatic=True)
    else:
        if not isinstance(start_raw, dict):
            raise ValueError('救援起点需要包含坐标')
        sx = _finite(start_raw.get('x_m'), '救援起点X坐标')
        sz = _finite(start_raw.get('z_m'), '救援起点Z坐标')
        if not (0 <= sx < nx * dx and 0 <= sz < nz * dx):
            raise ValueError('救援起点超出地图范围')
        cell = _cell(sx, sz, shape, dx)
        if solid[cell[0], cell[1]]:
            raise ValueError('救援起点位于建筑或封闭占地内')
        if ocean[cell[0], cell[1]]:
            raise ValueError('救援起点位于海域，请移到陆地')
        start = dict(x_m=sx, z_m=sz, cell=list(cell), automatic=False)
        if _open_faces(cell[0], cell[1], closed, blocked_x, blocked_y) == 0:
            raise ValueError('救援起点被完全封闭，四条边均不可通行')
    z0, x0 = start['cell']
    if closed[z0, x0]:
        raise ValueError('救援起点位于建筑或海域内')
    if _open_faces(z0, x0, closed, blocked_x, blocked_y) == 0:
        raise ValueError('救援起点被完全封闭，四条边均不可通行')
    reachable = reachable_from((z0, x0), closed, blocked_x, blocked_y)
    if int(reachable.sum()) <= 1 and solid.size > 1:
        raise ValueError('救援起点被完全封闭，无法到达任何相邻格')

    compiled = []
    unreachable = []
    for i, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError('人员条目需要是对象')
        x_m = _finite(entry.get('x_m'), '人员X坐标')
        z_m = _finite(entry.get('z_m'), '人员Z坐标')
        if not (0 <= x_m < nx * dx and 0 <= z_m < nz * dx):
            raise ValueError(f'人员 {i + 1} 超出地图范围')
        z, x = _cell(x_m, z_m, shape, dx)
        if solid[z, x]:
            raise ValueError(f'人员 {i + 1} 位于建筑或封闭占地内')
        if ocean[z, x]:
            raise ValueError(f'人员 {i + 1} 位于海域，请移到陆地')
        if not reachable[z, x]:
            unreachable.append(i + 1)
        kind = entry.get('kind', 'trapped')
        if kind not in PERSON_KINDS:
            raise ValueError('人员类型不正确')
        group = entry.get('group', 0)
        if isinstance(group, bool) or not isinstance(group, int) or not -1 <= group <= 99:
            raise ValueError('人员分组需要是 -1 到 99 的整数')
        compiled.append(dict(id=str(entry.get('id') or f'person-{i + 1}')[:80], x_m=x_m, z_m=z_m,
                             cell=[z, x], group=group, kind=kind))
    if len({item['id'] for item in compiled}) != len(compiled):
        raise ValueError('人员编号需要唯一')
    if unreachable:
        listed = '、'.join(str(v) for v in unreachable[:8])
        more = '等' if len(unreachable) > 8 else ''
        raise ValueError(f'人员 {listed}{more} 与救援起点不连通，请移动人员或改选起点')

    counts = dict(total=len(compiled),
                  reachable=len(compiled),
                  connected_cells=int(reachable.sum()))
    return dict(schema=SCHEMA, coordinate_system=COORDINATE_SYSTEM, mode=mode, seed=float(seed),
                parameters=dict(parameters), start=start, items=compiled, counts=counts)


def cell_counts(people, shape):
    """Number of people per base cell, for the exported array layer."""
    counts = np.zeros(shape, np.int16)
    for item in (people or {}).get('items', []):
        z, x = item['cell']
        counts[z, x] += 1
    return counts
