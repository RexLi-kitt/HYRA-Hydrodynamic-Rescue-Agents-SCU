"""Authored buildings and finite-crest barriers on a uniform Cartesian grid.

weir_flux is a Python translation of ANUGA's gpu_adjust_edgeflux_with_weir
(e2e7bf67b5ed); see third_party/anuga/ and docs/structures.md. The Cartesian
mapping, timed activation and hydrostatic coupling are HYRA implementations.
"""
import math
import numpy as np


def weir_flux(mass, advx, advz, left, right, height, qfactor=1.):
    """ANUGA Villemonte blend; discharge per unit edge width (m²/s).

    Pressure is deliberately separate, as in the original ANUGA flux kernel.
    Defaults s1=.9,s2=.95,h1=1,h2=1.5 are fixed for this first integration.
    """
    speed = 0.
    if left <= 0. and right <= 0.:
        return mass, advx, advz, speed
    lo, hi = min(left, right), max(left, right)
    raw = qfactor*(2./3.)*hi*math.sqrt((2./3.)*9.81*hi)
    tail = qfactor*(2./3.)*lo*math.sqrt((2./3.)*9.81*lo)
    ratio = tail/max(raw, 1.e-100)
    sub = lo/max(hi, 1.e-100)
    deep = lo/max(height, 1.e-100)
    ideal = raw*max(0., 1.-ratio)**.385
    if right > left:
        ideal = -ideal
    if sub < .95 and deep < 1.5:
        w1 = min(max(sub-.9, 0.)/.05, 1.)
        w2 = min(max(deep-1., 0.)/.5, 1.)
        new = (ideal*(1.-w1)+w1*mass)*(1.-w2)+w2*mass
        scale = max(new/mass if abs(mass) > 1.e-100 else 0., 0.)
        mass = new
        advx *= min(scale, 10.)
        advz *= min(scale, 10.)
    if abs(mass) > 0.:
        speed = math.sqrt(9.81*(hi+height))+abs(mass/(hi+1.e-12))
    return mass, advx, advz, speed


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{name}需要在 {low:g} 到 {high:g} 之间')
    return float(value)


def normalize(items, shape, dx):
    """Coordinates are local metres from the displayed grid's outer corner."""
    if not isinstance(items, list) or len(items) > 1000:
        raise ValueError('建筑与堤防需要是列表，最多1000个对象')
    nz, nx = shape
    result, ids = [], set()
    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise ValueError('建筑与堤防对象无效')
        kind = raw.get('kind')
        if kind not in ('building', 'permanent_levee', 'temporary_levee'):
            raise ValueError('未知建筑／堤防类型')
        sid = raw.get('id', f'structure-{i+1}')
        if not isinstance(sid, str) or not 1 <= len(sid) <= 80 or sid in ids:
            raise ValueError('建筑／堤防编号需要唯一且长度不超过80')
        ids.add(sid)
        name = raw.get('name', sid)
        if not isinstance(name, str) or len(name) > 120:
            raise ValueError('建筑／堤防名称过长')
        obj = dict(id=sid, kind=kind, name=name)
        if kind == 'building':
            obj['x_m'] = number(raw.get('x_m'), '建筑X坐标', 0, nx*dx)
            obj['z_m'] = number(raw.get('z_m'), '建筑Z坐标', 0, nz*dx)
            obj['width_m'] = number(raw.get('width_m'), '建筑宽度', .01, nx*dx)
            obj['length_m'] = number(raw.get('length_m'), '建筑长度', .01, nz*dx)
            obj['height_m'] = number(raw.get('height_m', 6), '建筑高度', .1, 300)
            if obj['x_m']+obj['width_m'] > nx*dx+1e-8 or obj['z_m']+obj['length_m'] > nz*dx+1e-8:
                raise ValueError('建筑轮廓超出地图')
        else:
            points = raw.get('points_m')
            if not isinstance(points, list) or not 2 <= len(points) <= 200:
                raise ValueError('堤防折线需要2到200个点')
            obj['points_m'] = []
            for p in points:
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise ValueError('堤防点需要[X米,Z米]')
                obj['points_m'].append([number(p[0], '堤防X坐标', 0, nx*dx), number(p[1], '堤防Z坐标', 0, nz*dx)])
            mode = raw.get('crest_mode', 'relative')
            if mode not in ('relative', 'absolute'):
                raise ValueError('堤顶需要选择沿地形增高或绝对高程')
            obj.update(crest_mode=mode,
                       crest_m=number(raw.get('crest_m', 1), '堤高／堤顶高程', .01 if mode == 'relative' else -10000, 10000),
                       qfactor=number(raw.get('qfactor', 1), '堰流系数倍数', .05, 2))
            obj['start_s'] = number(raw.get('start_s', 0), '启用时间', 0, 1e7) if kind == 'temporary_levee' else 0.
            end = raw.get('end_s') if kind == 'temporary_levee' else None
            obj['end_s'] = None if end is None else number(end, '撤除时间', 0, 1e7)
            if obj['end_s'] is not None and obj['end_s'] <= obj['start_s']:
                raise ValueError('临时堤撤除时间必须晚于启用时间')
        result.append(obj)
    return result


def snap_path(points, shape, dx):
    """Connected staircase on grid edges. Original metre geometry is retained.

    Use half-up rounding (also used by the JS preview), and a monotone path
    whose segment centres are closest to the original segment.
    """
    nz, nx = shape
    vertices = [(min(nx, max(0, int(math.floor(x/dx+.5)))),
                 min(nz, max(0, int(math.floor(z/dx+.5))))) for x, z in points]
    faces = set()
    for (x, z), (ex, ez) in zip(vertices, vertices[1:]):
        if (x,z) > (ex,ez):
            x,z,ex,ez = ex,ez,x,z
        startx, startz = x, z
        vx, vz = ex-x, ez-z
        while (x, z) != (ex, ez):
            sx, sz = int(np.sign(ex-x)), int(np.sign(ez-z))
            if not sx:
                vertical = True
            elif not sz:
                vertical = False
            else:
                # Compare perpendicular distance of each candidate edge centre.
                horizontal_error = abs(vx*(z-startz)-vz*(x+.5*sx-startx))
                vertical_error = abs(vx*(z+.5*sz-startz)-vz*(x-startx))
                vertical = vertical_error <= horizontal_error
            if vertical:
                faces.add((1, min(z, z+sz), x)); z += sz
            else:
                faces.add((0, z, min(x, x+sx))); x += sx
    for axis, z, x in faces:
        if (axis == 1 and not 0 < x < nx) or (axis == 0 and not 0 < z < nz):
            raise ValueError('堤防需位于地图内部；端点可以接到边界，不能沿外边界布置')
    if not faces:
        raise ValueError('堤防短于当前计算格，请延长折线')
    return sorted(faces)


class Structures:
    def __init__(self, items, bed, dx):
        self.items = normalize(items, bed.shape, dx)
        self.bed, self.dx = bed, dx
        self.buildings = np.zeros_like(bed, bool)
        self.barriers = []
        self.compiled = []
        self.last_time = None
        self.masks = {1: np.zeros((bed.shape[0], bed.shape[1]+1), bool),
                      0: np.zeros((bed.shape[0]+1, bed.shape[1]), bool)}
        for obj in self.items:
            record = dict(obj)
            if obj['kind'] == 'building':
                x0, z0 = [int(math.floor(obj[k]/dx+1e-10)) for k in ('x_m', 'z_m')]
                x1 = int(math.ceil((obj['x_m']+obj['width_m'])/dx-1e-10))
                z1 = int(math.ceil((obj['z_m']+obj['length_m'])/dx-1e-10))
                self.buildings[z0:z1, x0:x1] = True
                record.update(cell_bounds=[x0, z0, x1, z1],
                              foundation_elevation_m=float(bed[z0:z1, x0:x1].max()))
                record['roof_elevation_m'] = record['foundation_elevation_m']+obj['height_m']
            else:
                edges = []
                for axis, z, x in snap_path(obj['points_m'], bed.shape, dx):
                    a, b = (bed[z, x-1], bed[z, x]) if axis == 1 else (bed[z-1, x], bed[z, x])
                    crest = max(a, b)+obj['crest_m'] if obj['crest_mode'] == 'relative' else obj['crest_m']
                    if crest < max(a, b)-1e-8:
                        raise ValueError('堤顶低于沿线地面，请提高绝对堤顶高程或使用沿地形增高')
                    edges.append(dict(axis=axis, z=z, x=x, crest_m=float(crest)))
                record['edges'] = edges
                self.barriers.append(record)
            self.compiled.append(record)

    def active(self, obj, time, warmup=False):
        return obj['kind'] == 'permanent_levee' or (not warmup and obj['start_s'] <= time+1e-10
                and (obj['end_s'] is None or time < obj['end_s']-1e-10))

    def faces(self, time, warmup=False):
        stamp = (time, warmup)
        if stamp == self.last_time:
            return self.current
        self.last_time = stamp
        merged = {0: {}, 1: {}}
        for obj in self.barriers:
            if not self.active(obj, time, warmup):
                continue
            for e in obj['edges']:
                z,x,axis = e['z'],e['x'],e['axis']
                ground = max(self.bed[z,x-1],self.bed[z,x]) if axis == 1 else max(self.bed[z-1,x],self.bed[z,x])
                # Upstream also applies weir adjustment only above both beds.
                if e['crest_m'] <= ground:
                    continue
                key = (e['z'], e['x']); prior = merged[e['axis']].get(key)
                if prior is None or e['crest_m'] > prior[0]:
                    merged[e['axis']][key] = (e['crest_m'], obj['qfactor'])
        for axis, mask in self.masks.items():
            mask.fill(False)
            for z, x in merged[axis]:
                mask[z, x] = True
        self.current = merged
        return merged

    def next_event(self, time):
        events = [v for o in self.barriers if o['kind'] == 'temporary_levee'
                  for v in (o['start_s'], o['end_s']) if v is not None and v > time+1e-10]
        return min(events, default=float('inf'))
