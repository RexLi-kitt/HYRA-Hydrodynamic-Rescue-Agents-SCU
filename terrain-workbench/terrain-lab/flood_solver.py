"""2-D shallow-water finite volumes: Rusanov flux + hydrostatic reconstruction.

Reference: Audusse et al. (2004), doi:10.1137/S1064827503431090.
Cell centres, SI units, first-order explicit transport, semi-implicit Manning drag.
"""
from __future__ import annotations
import numpy as np

G = 9.81
DRY = 1e-8


class FloodSolver:
    def __init__(self, bed, depth, dx, *, boundary='open', manning=0.035,
                 sea_level=None, sea_edges=None):
        self.bed = np.asarray(bed, dtype=np.float64).copy()
        self.h = np.asarray(depth, dtype=np.float64).copy()
        if (self.bed.ndim != 2 or min(self.bed.shape) < 2 or self.h.shape != self.bed.shape
                or not np.isfinite(self.bed).all() or not np.isfinite(self.h).all()
                or (self.h < 0).any() or not np.isfinite(dx) or dx <= 0):
            raise ValueError('Invalid terrain, depth or grid spacing')
        if boundary not in ('ocean', 'open', 'closed') or not np.isfinite(manning) or manning < 0:
            raise ValueError('Invalid boundary or Manning coefficient')
        self.dx, self.boundary, self.manning = float(dx), boundary, float(manning)
        nz, nx = self.h.shape
        self.sea_edges = {name: np.zeros(n, dtype=bool) for name, n in
                          (('west', nz), ('east', nz), ('north', nx), ('south', nx))}
        self.sea_level = None
        self.sea_wave_speed = 0.0
        if boundary == 'ocean':
            if sea_level is None or not np.isfinite(sea_level) or not isinstance(sea_edges, dict):
                raise ValueError('Ocean boundary requires a finite sea level and edge masks')
            self.sea_level = float(sea_level)
            for name, mask in self.sea_edges.items():
                supplied = np.asarray(sea_edges.get(name, mask), dtype=bool)
                if supplied.shape != mask.shape:
                    raise ValueError('Invalid ocean edge mask')
                self.sea_edges[name] = supplied.copy()
            edge_beds = (self.bed[:, 0], self.bed[:, -1], self.bed[0, :], self.bed[-1, :])
            for mask, edge_bed in zip(self.sea_edges.values(), edge_beds):
                depth_at_sea = np.maximum(0, self.sea_level-edge_bed[mask])
                if depth_at_sea.size:
                    self.sea_wave_speed = max(self.sea_wave_speed, float(np.sqrt(G*depth_at_sea).max()))
        self.mx = np.zeros_like(self.h)
        self.mz = np.zeros_like(self.h)
        self.initial = self.h.copy()
        self.maximum = self.h.copy()
        self.time = 0.0
        self.steps = 0
        self.balance = dict(initial_m3=float(self.h.sum() * dx**2), rainfall_m3=0.0,
                            inflow_m3=0.0, infiltration_m3=0.0, outflow_m3=0.0,
                            sea_inflow_m3=0.0, sea_outflow_m3=0.0, numerical_adjustment_m3=0.0)

    def velocity(self):
        wet = self.h > DRY
        return (np.divide(self.mx, self.h, out=np.zeros_like(self.h), where=wet),
                np.divide(self.mz, self.h, out=np.zeros_like(self.h), where=wet))

    def timestep(self, limit=1.0, rain=0.0, inflow=0.0, source=None):
        u, v = self.velocity()
        # Include this step's potential source water in the wave-speed bound.
        prospective = self.h + rain * limit
        if source is not None:
            prospective = prospective.copy()
            prospective[source] += inflow * limit / self.dx**2
        wave = np.sqrt(G * prospective)
        speed = float(np.max(np.abs(u) + wave) + np.max(np.abs(v) + wave))
        # A dry initial map can still receive deep water from the external sea.
        speed = max(speed, 2*self.sea_wave_speed)
        return min(limit, 0.45 * self.dx / max(speed, 1e-9))

    def _flux(self, axis):
        pads = ((0, 0), (1, 1)) if axis == 1 else ((1, 1), (0, 0))
        h, b, mx, mz = [np.pad(a, pads, mode='edge') for a in (self.h, self.bed, self.mx, self.mz)]
        normal = mx if axis == 1 else mz
        edge_names = ('west', 'east') if axis == 1 else ('north', 'south')
        if self.boundary == 'ocean':
            # Ghost cells represent a calm external reservoir at the map's sea level.
            # Only ocean faces receive a stage boundary; inland water is never reset.
            for edge, name in zip((0, -1), edge_names):
                s = (slice(None), edge) if axis == 1 else (edge, slice(None))
                mask = self.sea_edges[name]
                h[s][mask] = np.maximum(0, self.sea_level-b[s][mask])
                mx[s][mask] = 0; mz[s][mask] = 0
        if self.boundary == 'closed':
            if axis == 1:
                normal[:, 0] *= -1; normal[:, -1] *= -1
            else:
                normal[0, :] *= -1; normal[-1, :] *= -1
        left = (slice(None), slice(None, -1)) if axis == 1 else (slice(None, -1), slice(None))
        right = (slice(None), slice(1, None)) if axis == 1 else (slice(1, None), slice(None))
        hl, hr, bl, br = h[left], h[right], b[left], b[right]
        top = np.maximum(bl, br)
        # Avoid subtracting large absolute elevations from a small depth.
        dl = np.maximum(0, hl - (top - bl))
        dr = np.maximum(0, hr - (top - br))
        ul = np.divide(mx[left], hl, out=np.zeros_like(hl), where=hl > DRY)
        ur = np.divide(mx[right], hr, out=np.zeros_like(hr), where=hr > DRY)
        vl = np.divide(mz[left], hl, out=np.zeros_like(hl), where=hl > DRY)
        vr = np.divide(mz[right], hr, out=np.zeros_like(hr), where=hr > DRY)
        nl, nr = (ul, ur) if axis == 1 else (vl, vr)
        speed = np.maximum(np.abs(nl) + np.sqrt(G*dl), np.abs(nr) + np.sqrt(G*dr))
        mass = .5*(dl*nl + dr*nr) - .5*speed*(dr-dl)
        fx = .5*(dl*nl*ul + dr*nr*ur) - .5*speed*(dr*ur-dl*ul)
        fz = .5*(dl*nl*vl + dr*nr*vr) - .5*speed*(dr*vr-dl*vl)
        pressure = .25*G*(dl**2 + dr**2)
        if axis == 1: fx += pressure
        else: fz += pressure
        if self.boundary in ('open', 'ocean'):
            # Transmissive outflow only. A boundary is not an infinite water source.
            for edge, sign, name in zip((0, -1), (-1, 1), edge_names):
                s = (slice(None), edge) if axis == 1 else (edge, slice(None))
                q = sign * np.maximum(0, sign*normal[s])
                ue = np.divide(mx[s], h[s], out=np.zeros_like(q), where=h[s] > DRY)
                ve = np.divide(mz[s], h[s], out=np.zeros_like(q), where=h[s] > DRY)
                ordinary = ~self.sea_edges[name]
                mass[s][ordinary] = q[ordinary]
                fx[s][ordinary] = (q*ue + (.5*G*h[s]**2 if axis == 1 else 0))[ordinary]
                fz[s][ordinary] = (q*ve + (.5*G*h[s]**2 if axis == 0 else 0))[ordinary]
        return mass, fx, fz, .5*G*(hl**2-dl**2), .5*G*(hr**2-dr**2)

    def step(self, dt, *, rain=0.0, infiltration=0.0, inflow=0.0, source=None):
        if not all(np.isfinite(v) and v >= 0 for v in (dt, rain, infiltration, inflow)) or dt == 0:
            raise ValueError('Invalid step or forcing')
        if inflow and (source is None or len(source) != 2 or not
                       all(isinstance(v, (int, np.integer)) and 0 <= v < n for v, n in zip(source, self.h.shape))):
            raise ValueError('Inflow requires a valid source cell')
        area = self.dx**2
        self.h += rain*dt
        self.balance['rainfall_m3'] += rain*dt*self.h.size*area
        if inflow:
            self.h[source] += inflow*dt/area
            self.balance['inflow_m3'] += inflow*dt
        removed = np.minimum(self.h, infiltration*dt)
        retained = np.divide(self.h-removed, self.h, out=np.zeros_like(self.h), where=self.h > 0)
        self.mx *= retained; self.mz *= retained; self.h -= removed
        self.balance['infiltration_m3'] += float(removed.sum()*area)
        ax, bx, cx, lx, rx = self._flux(1)
        az, bz, cz, lz, rz = self._flux(0)
        f = dt/self.dx
        self.h -= f*(ax[:, 1:]-ax[:, :-1] + az[1:, :]-az[:-1, :])
        self.mx -= f*(bx[:, 1:]+lx[:, 1:] - bx[:, :-1]-rx[:, :-1] + bz[1:, :]-bz[:-1, :])
        self.mz -= f*(cx[:, 1:]-cx[:, :-1] + cz[1:, :]+lz[1:, :] - cz[:-1, :]-rz[:-1, :])
        for name, outward in zip(self.sea_edges, (-ax[:, 0], ax[:, -1], -az[0, :], az[-1, :])):
            sea = self.sea_edges[name]
            self.balance['outflow_m3'] += float(dt*self.dx*outward[~sea].sum())
            self.balance['sea_inflow_m3'] += float(dt*self.dx*np.maximum(0, -outward[sea]).sum())
            self.balance['sea_outflow_m3'] += float(dt*self.dx*np.maximum(0, outward[sea]).sum())
        if not all(np.isfinite(a).all() for a in (self.h, self.mx, self.mz)) or self.h.min() < -1e-7:
            raise RuntimeError('水动力计算不稳定，请检查网格与场景参数')
        adjustment = float(-np.minimum(self.h, 0).sum()*area)
        self.balance['numerical_adjustment_m3'] += adjustment
        np.maximum(self.h, 0, out=self.h)
        wet = self.h > DRY
        self.mx[~wet] = 0; self.mz[~wet] = 0
        drag = 1 + dt*G*self.manning**2*np.hypot(self.mx, self.mz)/np.maximum(self.h, DRY)**(7/3)
        self.mx /= drag; self.mz /= drag
        np.maximum(self.maximum, self.h, out=self.maximum)
        self.time += dt
        self.steps += 1

    def advance(self, end, *, rain=0.0, rain_until=float('inf'), infiltration=0.0,
                inflow=0.0, source=None, cancelled=lambda: False):
        while self.time < end - 1e-10:
            if cancelled(): raise InterruptedError('模拟已取消')
            raining = rain if self.time < rain_until - 1e-10 else 0.0
            limit = min(1.0, end-self.time)
            if raining: limit = min(limit, rain_until-self.time)
            dt = self.timestep(limit, raining, inflow, source)
            if dt < 1e-9 or self.steps >= 500000:
                raise RuntimeError('场景需要过多计算步，请缩短时间或降低计算分辨率')
            self.step(dt, rain=raining, infiltration=infiltration, inflow=inflow, source=source)

    def summary(self):
        u, v = self.velocity()
        current = float(self.h.sum()*self.dx**2)
        b = self.balance
        expected = (b['initial_m3'] + b['rainfall_m3'] + b['inflow_m3'] + b['sea_inflow_m3']
                    - b['infiltration_m3'] - b['outflow_m3'] - b['sea_outflow_m3'])
        return dict(time_s=self.time, steps=self.steps, max_depth_m=float(self.h.max()),
                    max_speed_m_s=float(np.hypot(u, v).max()),
                    wet_area_m2=int((self.h >= .01).sum())*self.dx**2,
                    newly_flooded_area_m2=int(((self.initial < .01) & (self.h >= .01)).sum())*self.dx**2,
                    balance=dict(b, current_m3=current, residual_m3=current-expected))
