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
        n = np.broadcast_to(np.asarray(manning, dtype=np.float64), self.h.shape)
        if boundary not in ('ocean', 'open', 'closed') or not np.isfinite(n).all() or (n < 0).any():
            raise ValueError('Invalid boundary or Manning coefficient')
        self.dx, self.boundary, self.manning = float(dx), boundary, n.copy()
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
        self.baseflow_rate = np.zeros_like(self.h)
        self.baseflow_m3_s = 0.0
        self.solid = np.zeros_like(self.h, dtype=bool)
        self.wall_x = np.zeros((nz,nx+1),dtype=bool)
        self.wall_y = np.zeros((nz+1,nx),dtype=bool)
        self.drain_capacity = np.zeros_like(self.h)
        self.qx = np.zeros((nz,nx+1))
        self.qy = np.zeros((nz+1,nx))
        self.source_rate = np.zeros_like(self.h)
        self.source_schedule = []
        self.external_schedule = []
        self.external_rate = np.zeros_like(self.h)
        self.external_volume_by_id = {}
        self.last_boundary_inflows = []
        self.boundary_sample_time = None
        self.balance = dict(initial_m3=float(self.h.sum() * dx**2), rainfall_m3=0.0,
                            inflow_m3=0.0, baseflow_m3=0.0, infiltration_m3=0.0, outflow_m3=0.0,
                            sea_inflow_m3=0.0, sea_outflow_m3=0.0, drainage_m3=0.0,
                            scheduled_inflow_m3=0.0, external_inflow_m3=0.0, numerical_adjustment_m3=0.0)

    def set_environment(self, *, solid=None, wall_x=None, wall_y=None, drain_capacity=None):
        for name,value in (('solid',solid),('wall_x',wall_x),('wall_y',wall_y),('drain_capacity',drain_capacity)):
            if value is None: continue
            a = np.asarray(value,dtype=bool if name!='drain_capacity' else float)
            if a.shape!=getattr(self,name).shape or (name=='drain_capacity' and
                    (not np.isfinite(a).all() or (a<0).any())): raise ValueError('Invalid environment layer')
            setattr(self,name,a.copy())
        if np.any(self.solid & (self.baseflow_rate>0)):
            raise ValueError('River sources cannot be inside a solid cell')
        self.h[self.solid]=0; self.mx[self.solid]=0; self.mz[self.solid]=0
        self.initial=self.h.copy(); self.maximum=self.h.copy()
        self.balance['initial_m3']=float(self.h.sum()*self.dx**2)
        if self.boundary=='ocean':
            self.sea_wave_speed=0.0
            for name,b,solid in (('west',self.bed[:,0],self.solid[:,0]),('east',self.bed[:,-1],self.solid[:,-1]),
                                ('north',self.bed[0,:],self.solid[0,:]),('south',self.bed[-1,:],self.solid[-1,:])):
                h=np.maximum(0,self.sea_level-b[self.sea_edges[name]&~solid])
                if h.size:self.sea_wave_speed=max(self.sea_wave_speed,float(np.sqrt(G*h).max()))

    def set_source_schedule(self, schedule):
        # Each event is a piecewise-constant hydrograph shared by source_id cells.
        self.source_schedule = schedule

    def _scheduled_rate(self, time):
        rate = np.zeros_like(self.h)
        for source in self.source_schedule:
            points=source['hydrograph']; q=0.0
            for t,value in points:
                if t>time+1e-10: break
                q=value
            if q:
                cells=source['cells']; each=q/(len(cells)*self.dx**2)
                for z,x in cells: rate[z,x]+=each
        return rate

    def set_external_inflows(self,entries):
        self.external_schedule=[e for e in entries if e['enabled']]
        self.external_volume_by_id={e['id']:0. for e in entries}
        self.external_rate=self._external_rate(self.time)

    def _external_rate(self,time):
        rate=np.zeros_like(self.h)
        for entry in self.external_schedule:
            if entry.get('delivery')=='boundary_discharge_with_momentum':continue
            q=self.external_discharge(entry,time)
            for cell in entry['cells']:rate[cell['z'],cell['x']]+=q*cell['weight']/self.dx**2
        return rate

    def boundary_forcing(self):
        """Exact inward discharge on native-width fractions of outer faces.

        Add advected normal momentum; the pressure closure extrapolates the
        interior depth rather than prescribing both stage and discharge.
        """
        edges={name:dict(fraction=np.zeros(len(mask)),q=np.zeros(len(mask)),
            advective=np.zeros(len(mask)),wave=0.) for name,mask in self.sea_edges.items()}
        states=[]
        for entry in self.external_schedule:
            if entry.get('delivery')!='boundary_discharge_with_momentum':continue
            q=self.external_discharge(entry,self.time);faces=[]
            for face in entry['boundary_faces']:
                flow=q*face['weight'];width=face['width_m'];z,x=face['z'],face['x']
                critical=(flow/width)**(2/3)/G**(1/3) if flow else 0.
                depth=max(float(self.h[z,x]),critical,DRY)
                velocity=entry['inlet_velocity_m_s'] or flow/(width*depth)
                if not flow:velocity=0.
                e=edges[face['edge']];i=face['offset'];e['fraction'][i]+=width/self.dx
                e['q'][i]+=flow/self.dx;e['advective'][i]+=flow/self.dx*velocity
                if flow:e['wave']=max(e['wave'],velocity+np.sqrt(G*max(depth,flow/(width*max(velocity,DRY)))))
                faces.append(dict(face,discharge_m3_s=flow,inward_velocity_m_s=float(velocity),
                    reference_depth_m=float(flow/(width*velocity)) if velocity else float(self.h[z,x])))
            states.append(dict(id=entry['id'],discharge_m3_s=q,faces=faces))
        for e in edges.values():
            if (e['fraction']>1+1e-10).any():raise ValueError('河口占用的边界宽度超过网格面宽')
        return edges,states

    def forcing_rate(self):
        """Equivalent depth rate for diagnostics/CFL; only manual sources add h."""
        rate=self._external_rate(self.time)
        for state in self.boundary_forcing()[1]:
            for face in state['faces']:rate[face['z'],face['x']]+=face['discharge_m3_s']/self.dx**2
        return rate

    @staticmethod
    def external_discharge(entry,time):
        q=0.
        for t,value in entry['hydrograph']:
            if t>time+1e-10:break
            q=value
        return q

    def set_baseflow(self, sources):
        rate = np.zeros_like(self.h)
        for source in sources:
            z, x, q = source['z'], source['x'], source['discharge_m3_s']
            if (not isinstance(z, (int, np.integer)) or not isinstance(x, (int, np.integer))
                    or not 0 <= z < self.h.shape[0] or not 0 <= x < self.h.shape[1]
                    or not np.isfinite(q) or q < 0):
                raise ValueError('Invalid river baseflow source')
            rate[z, x] += q/self.dx**2
        self.baseflow_rate = rate
        self.baseflow_m3_s = float(rate.sum()*self.dx**2)

    def start_scenario(self):
        """Use the warmed state as t=0 without losing momentum or source rates."""
        self.initial = self.h.copy(); self.maximum = self.h.copy()
        self.time = 0.0; self.steps = 0
        for name in self.balance: self.balance[name] = 0.0
        self.balance['initial_m3'] = float(self.h.sum()*self.dx**2)
        for key in self.external_volume_by_id:self.external_volume_by_id[key]=0.
        self.last_boundary_inflows=[];self.boundary_sample_time=None

    def velocity(self):
        wet = self.h > DRY
        return (np.divide(self.mx, self.h, out=np.zeros_like(self.h), where=wet),
                np.divide(self.mz, self.h, out=np.zeros_like(self.h), where=wet))

    def timestep(self, limit=1.0, rain=0.0, inflow=0.0, source=None):
        u, v = self.velocity()
        # Include this step's potential source water in the wave-speed bound.
        prospective = self.h + (rain*(~self.solid)+self.baseflow_rate+self.source_rate+self.forcing_rate()) * limit
        if source is not None:
            prospective = prospective.copy()
            prospective[source] += inflow * limit / self.dx**2
        wave = np.sqrt(G * prospective)
        speed = float(np.max(np.abs(u) + wave) + np.max(np.abs(v) + wave))
        # A dry initial map can still receive deep water from the external sea.
        speed = max(speed, 2*self.sea_wave_speed,2*max(e['wave'] for e in self.boundary_forcing()[0].values()))
        return min(limit, 0.45 * self.dx / max(speed, 1e-9))

    def _flux(self, axis, inlet_edges=None):
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
        inlet_edges=inlet_edges if inlet_edges is not None else self.boundary_forcing()[0]
        for face,sign,name,cell in zip((0,-1),(1,-1),edge_names,(0,-1)):
            s=(slice(None),face) if axis==1 else (face,slice(None))
            interior=(slice(None),cell) if axis==1 else (cell,slice(None))
            e=inlet_edges[name];fraction=np.minimum(1,e['fraction']);keep=1-fraction
            mass[s]=mass[s]*keep+sign*e['q']
            pressure=.5*G*self.h[interior]**2*fraction
            fx[s]=fx[s]*keep+(e['advective']+pressure if axis==1 else 0)
            fz[s]=fz[s]*keep+(e['advective']+pressure if axis==0 else 0)
        # Reflect each side of a blocked face separately: equal/opposite water
        # exchange is zero, while each fluid cell retains its own pressure force.
        cells=np.pad(self.solid,pads,mode='edge')
        blocked=(self.wall_x if axis==1 else self.wall_y) | cells[left] | cells[right]
        mass[blocked]=0; fx[blocked]=0; fz[blocked]=0
        corr_l=.5*G*(hl**2-dl**2); corr_r=.5*G*(hr**2-dr**2)
        corr_l[blocked]=.5*G*hl[blocked]**2
        corr_r[blocked]=.5*G*hr[blocked]**2
        return mass, fx, fz, corr_l, corr_r

    def step(self, dt, *, rain=0.0, infiltration=0.0, inflow=0.0, source=None):
        if not all(np.isfinite(v) and v >= 0 for v in (dt, rain, infiltration, inflow)) or dt == 0:
            raise ValueError('Invalid step or forcing')
        if inflow and (source is None or len(source) != 2 or not
                       all(isinstance(v, (int, np.integer)) and 0 <= v < n for v, n in zip(source, self.h.shape))):
            raise ValueError('Inflow requires a valid source cell')
        if inflow and self.solid[source]:raise ValueError('Inflow cannot be inside a solid cell')
        area = self.dx**2
        self.h += rain*dt*(~self.solid)
        self.balance['rainfall_m3'] += rain*dt*int((~self.solid).sum())*area
        self.h += self.baseflow_rate*dt
        self.balance['baseflow_m3'] += self.baseflow_m3_s*dt
        self.h += self.source_rate*dt
        self.balance['scheduled_inflow_m3'] += float(self.source_rate.sum()*dt*area)
        self.h += self.external_rate*dt
        self.balance['external_inflow_m3'] += float(self.external_rate.sum()*dt*area)
        for entry in self.external_schedule:
            self.external_volume_by_id[entry['id']]+=self.external_discharge(entry,self.time)*dt
        if inflow:
            self.h[source] += inflow*dt/area
            self.balance['inflow_m3'] += inflow*dt
        removed = np.minimum(self.h, infiltration*dt)
        retained = np.divide(self.h-removed, self.h, out=np.zeros_like(self.h), where=self.h > 0)
        self.mx *= retained; self.mz *= retained; self.h -= removed
        self.balance['infiltration_m3'] += float(removed.sum()*area)
        drained=np.minimum(self.h,self.drain_capacity*dt/area)
        retained=np.divide(self.h-drained,self.h,out=np.zeros_like(self.h),where=self.h>0)
        self.mx*=retained; self.mz*=retained; self.h-=drained
        self.balance['drainage_m3']+=float(drained.sum()*area)
        inlet_edges,states=self.boundary_forcing()
        self.last_boundary_inflows=states;self.boundary_sample_time=self.time
        self.balance['external_inflow_m3']+=dt*sum(s['discharge_m3_s'] for s in states)
        ax, bx, cx, lx, rx = self._flux(1,inlet_edges)
        az, bz, cz, lz, rz = self._flux(0,inlet_edges)
        self.qx=ax.copy(); self.qy=az.copy()
        f = dt/self.dx
        self.h -= f*(ax[:, 1:]-ax[:, :-1] + az[1:, :]-az[:-1, :])
        self.mx -= f*(bx[:, 1:]+lx[:, 1:] - bx[:, :-1]-rx[:, :-1] + bz[1:, :]-bz[:-1, :])
        self.mz -= f*(cx[:, 1:]-cx[:, :-1] + cz[1:, :]+lz[1:, :] - cz[:-1, :]-rz[:-1, :])
        for name, outward in zip(self.sea_edges, (-ax[:, 0], ax[:, -1], -az[0, :], az[-1, :])):
            # Remove prescribed incoming Q before classifying the remaining
            # open/sea exchange, including mixed river/ocean coarse faces.
            outward=outward+inlet_edges[name]['q']
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
        self.mx[self.solid]=0; self.mz[self.solid]=0; self.h[self.solid]=0
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
            self.source_rate=self._scheduled_rate(self.time)
            self.external_rate=self._external_rate(self.time)
            for entry in self.source_schedule+self.external_schedule:
                for t,_ in entry['hydrograph']:
                    if t>self.time+1e-10: limit=min(limit,t-self.time);break
            dt = self.timestep(limit, raining, inflow, source)
            if dt < 1e-9 or self.steps >= 500000:
                raise RuntimeError('场景需要过多计算步，请缩短时间或降低计算分辨率')
            self.step(dt, rain=raining, infiltration=infiltration, inflow=inflow, source=source)

    def summary(self):
        u, v = self.velocity()
        current = float(self.h.sum()*self.dx**2)
        b = self.balance
        expected = (b['initial_m3'] + b['rainfall_m3'] + b['inflow_m3'] + b['baseflow_m3'] + b['sea_inflow_m3']
                    + b['scheduled_inflow_m3'] + b['external_inflow_m3'] - b['drainage_m3']
                    - b['infiltration_m3'] - b['outflow_m3'] - b['sea_outflow_m3'])
        return dict(time_s=self.time, steps=self.steps, max_depth_m=float(self.h.max()),
                    max_speed_m_s=float(np.hypot(u, v).max()),
                    wet_area_m2=int((self.h >= .01).sum())*self.dx**2,
                    newly_flooded_area_m2=int(((self.initial < .01) & (self.h >= .01)).sum())*self.dx**2,
                    external_inflow_by_id_m3=dict(self.external_volume_by_id),
                    boundary_inflows=self.last_boundary_inflows,boundary_sample_time_s=self.boundary_sample_time,
                    balance=dict(b, current_m3=current, residual_m3=current-expected))
