"""Cartesian shallow-water finite volumes with MUSCL and SSP-RK2 transport.

Reference: Audusse et al. (2004), doi:10.1137/S1064827503431090.
KNP/normal rotation/low-Froude ideas: ANUGA e2e7bf67b5ed, see docs.
Native square cells, SI units, hydrostatic reconstruction, RK-weighted water
sources/sinks and split exact fixed-depth Manning drag. Bounded dry sinks and
discontinuous forcing do not imply smooth global second-order convergence.
"""
from __future__ import annotations
import os
import time
import numpy as np
from flood_numerics import face_flux, minmod_slope

G = 9.81
DRY = 1e-8


class FloodSolver:
    def __init__(self, bed, depth, dx, *, boundary='open', manning=0.035,
                 sea_level=None, sea_edges=None, flux_method='rusanov',
                 spatial_order=2, time_integrator='rk2', low_froude=0,
                 cfl=.4, limiter_theta=1.5, velocity_depth=1e-6,
                 max_timestep=1., max_steps=500000, open_boundary_mode='outflow_only'):
        self.bed = np.asarray(bed, dtype=np.float64).copy()
        self.h = np.asarray(depth, dtype=np.float64).copy()
        if (self.bed.ndim != 2 or min(self.bed.shape) < 2 or self.h.shape != self.bed.shape
                or not np.isfinite(self.bed).all() or not np.isfinite(self.h).all()
                or (self.h < 0).any() or not np.isfinite(dx) or dx <= 0):
            raise ValueError('Invalid terrain, depth or grid spacing')
        n = np.broadcast_to(np.asarray(manning, dtype=np.float64), self.h.shape)
        if boundary not in ('ocean', 'open', 'closed', 'periodic') or not np.isfinite(n).all() or (n < 0).any():
            raise ValueError('Invalid boundary or Manning coefficient')
        if (flux_method not in ('rusanov', 'knp') or spatial_order not in (1, 2)
                or isinstance(spatial_order, bool) or time_integrator not in ('euler', 'rk2')
                or low_froude not in (0, 1, 2) or isinstance(low_froude, bool)
                or not np.isfinite(cfl) or not 0 < cfl <= .45
                or not np.isfinite(limiter_theta) or not 1 <= limiter_theta <= 2
                or not np.isfinite(velocity_depth) or velocity_depth < DRY
                or not np.isfinite(max_timestep) or max_timestep <= 0
                or not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1):
            raise ValueError('Invalid numerical configuration')
        self.flux_method, self.spatial_order = flux_method, int(spatial_order)
        self.time_integrator, self.low_froude = time_integrator, int(low_froude)
        self.cfl, self.limiter_theta = float(cfl), float(limiter_theta)
        self.velocity_depth, self.max_timestep = float(velocity_depth), float(max_timestep)
        self.max_steps = max_steps
        if open_boundary_mode not in ('outflow_only', 'extrapolated'):
            raise ValueError('Invalid open_boundary_mode')
        if open_boundary_mode == 'extrapolated' and boundary != 'open':
            raise ValueError('Experimental extrapolation requires boundary=open')
        self.open_boundary_mode = open_boundary_mode
        self.acceleration = None
        if open_boundary_mode == 'outflow_only' and os.environ.get('HYRA_FLOOD_BACKEND', 'auto') != 'numpy':
            try:
                import flood_acceleration
                self.acceleration = flood_acceleration
            except ImportError:
                if os.environ.get('HYRA_FLOOD_BACKEND') == 'numba':
                    raise
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
        self.structures = None
        self.structure_warmup = False
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
        self.boundary_state_time = None
        self.last_timestep = {}
        self.diagnostics = dict(min_dt_s=None, max_dt_s=0., limited_faces=0,
                                max_stage_courant=0., minimum_raw_depth_m=0.)
        self.balance = dict(initial_m3=float(self.h.sum() * dx**2), rainfall_m3=0.0,
                            inflow_m3=0.0, baseflow_m3=0.0, infiltration_m3=0.0, outflow_m3=0.0,
                            sea_inflow_m3=0.0, sea_outflow_m3=0.0, drainage_m3=0.0,
                            scheduled_inflow_m3=0.0, external_inflow_m3=0.0, numerical_adjustment_m3=0.0)

    def set_environment(self, *, solid=None, wall_x=None, wall_y=None, drain_capacity=None):
        pending = {}
        for name,value in (('solid',solid),('wall_x',wall_x),('wall_y',wall_y),('drain_capacity',drain_capacity)):
            if value is None: continue
            a = np.asarray(value,dtype=bool if name!='drain_capacity' else float)
            if a.shape!=getattr(self,name).shape or (name=='drain_capacity' and
                    (not np.isfinite(a).all() or (a<0).any())): raise ValueError('Invalid environment layer')
            pending[name] = a.copy()
        if np.any(pending.get('solid', self.solid) & (self.baseflow_rate>0)):
            raise ValueError('River sources cannot be inside a solid cell')
        self._validated_source_schedule(self.source_schedule, pending.get('solid', self.solid))
        for name, a in pending.items():
            setattr(self, name, a)
        self.h[self.solid]=0; self.mx[self.solid]=0; self.mz[self.solid]=0
        self.initial=self.h.copy(); self.maximum=self.h.copy()
        self.balance['initial_m3']=float(self.h.sum()*self.dx**2)
        if self.boundary=='ocean':
            self.sea_wave_speed=0.0
            for name,b,solid in (('west',self.bed[:,0],self.solid[:,0]),('east',self.bed[:,-1],self.solid[:,-1]),
                                ('north',self.bed[0,:],self.solid[0,:]),('south',self.bed[-1,:],self.solid[-1,:])):
                h=np.maximum(0,self.sea_level-b[self.sea_edges[name]&~solid])
                if h.size:self.sea_wave_speed=max(self.sea_wave_speed,float(np.sqrt(G*h).max()))

    def _validated_source_schedule(self, schedule, solid=None):
        """Validate and detach scheduled sources before committing any state."""
        solid = self.solid if solid is None else solid
        if not isinstance(schedule, (list, tuple)):
            raise ValueError('Source schedule must be a sequence')
        clean = []
        for source in schedule:
            if not isinstance(source, dict):
                raise ValueError('Invalid scheduled source')
            cells = source.get('cells', [])
            points = source.get('hydrograph', [])
            if not isinstance(cells, (list, tuple)) or not cells:
                raise ValueError('Scheduled source requires cells')
            checked = []
            for cell in cells:
                if (not isinstance(cell, (list, tuple)) or len(cell) != 2
                        or not all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                                   and 0 <= v < n for v, n in zip(cell, self.h.shape))):
                    raise ValueError('Invalid scheduled source cell')
                z, x = map(int, cell)
                if solid[z, x]:
                    raise ValueError('Scheduled sources cannot be inside a solid cell')
                checked.append((z, x))
            if len(set(checked)) != len(checked):
                raise ValueError('Duplicate scheduled source cells')
            if not isinstance(points, (list, tuple)) or not points:
                raise ValueError('Scheduled source requires a hydrograph')
            graph = []; previous = -1.
            for point in points:
                if (not isinstance(point, (list, tuple)) or len(point) != 2
                        or not all(isinstance(v, (int, float, np.integer, np.floating))
                                   and not isinstance(v, (bool, np.bool_)) and np.isfinite(v) for v in point)):
                    raise ValueError('Invalid source hydrograph point')
                t, q = map(float, point)
                if t < 0 or t <= previous or q < 0:
                    raise ValueError('Hydrograph times must increase and flows must be nonnegative')
                previous = t; graph.append([t, q])
            clean.append(dict(source, cells=checked, hydrograph=graph))
        return clean

    def set_source_schedule(self, schedule):
        # Each event is piecewise constant; use a detached validated snapshot.
        self.source_schedule = self._validated_source_schedule(schedule)

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

        Add both components of advected momentum; the pressure closure extrapolates the
        interior depth rather than prescribing both stage and discharge.
        """
        edges={name:dict(fraction=np.zeros(len(mask)),q=np.zeros(len(mask)),
            advective=np.zeros(len(mask)),advective_x=np.zeros(len(mask)),advective_z=np.zeros(len(mask)),
            wave=0.) for name,mask in self.sea_edges.items()}
        states=[]
        for entry in self.external_schedule:
            if entry.get('delivery')!='boundary_discharge_with_momentum':continue
            q=self.external_discharge(entry,self.time);faces=[]
            for face in entry['boundary_faces']:
                flow=q*face['weight'];width=face['width_m'];z,x=face['z'],face['x']
                critical=(flow/width)**(2/3)/G**(1/3) if flow else 0.
                depth=max(float(self.h[z,x]),critical,DRY)
                normal={'west':(1.,0.),'east':(-1.,0.),'north':(0.,1.),'south':(0.,-1.)}[face['edge']]
                direction=face.get('momentum_direction_xz',normal)
                projection=direction[0]*normal[0]+direction[1]*normal[1]
                velocity=entry['inlet_velocity_m_s'] or flow/(width*depth*projection)
                if not flow:velocity=0.
                ux,uz=velocity*direction[0],velocity*direction[1]
                normal_velocity=velocity*projection
                e=edges[face['edge']];i=face['offset'];e['fraction'][i]+=width/self.dx
                signed_flow=flow/self.dx*(1 if face['edge'] in ('west','north') else -1)
                e['q'][i]+=flow/self.dx;e['advective'][i]+=flow/self.dx*normal_velocity
                e['advective_x'][i]+=signed_flow*ux;e['advective_z'][i]+=signed_flow*uz
                if flow:e['wave']=max(e['wave'],max(abs(ux),abs(uz))+np.sqrt(G*max(depth,flow/(width*max(normal_velocity,DRY)))))
                faces.append(dict(face,discharge_m3_s=flow,inward_velocity_m_s=float(normal_velocity),
                    velocity_m_s=float(velocity),velocity_x_m_s=float(ux),velocity_z_m_s=float(uz),
                    reference_depth_m=float(flow/(width*normal_velocity)) if normal_velocity else float(self.h[z,x])))
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
        if np.any(self.solid & (rate > 0)):
            raise ValueError('River sources cannot be inside a solid cell')
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
        self.boundary_state_time=None;self.last_timestep={}
        self.diagnostics=dict(min_dt_s=None,max_dt_s=0.,limited_faces=0,
                              max_stage_courant=0.,minimum_raw_depth_m=0.)

    def velocity(self):
        wet = self.h > self.velocity_depth
        return (np.divide(self.mx, self.h, out=np.zeros_like(self.h), where=wet),
                np.divide(self.mz, self.h, out=np.zeros_like(self.h), where=wet))

    def numerical_config(self):
        return dict(schema='hyra-swe-v2', flux_method=self.flux_method,
                    open_boundary_mode=self.open_boundary_mode,
                    structures='hydrostatic_crest_ANUGA_Villemonte_shared_RK_flux' if self.structures and self.structures.barriers else 'none',
                    compute_backend='numba' if self.acceleration else 'numpy',
                    spatial_order=self.spatial_order, time_integrator=self.time_integrator,
                    low_froude=self.low_froude, cfl=self.cfl,
                    limiter='generalised_minmod', limiter_theta=self.limiter_theta,
                    dry_depth_m=DRY, velocity_depth_m=self.velocity_depth,
                    max_timestep_s=self.max_timestep, max_steps=self.max_steps,
                    source_integration='RK-weighted_water_sources; Strang_Manning; piecewise_constant_events',
                    dry_sink_completion='unused_per_cell_capacity_after_RK_average',
                    positivity_limiter='shared_donor_factor_on_mass_and_advective_momentum',
                    bed_source='hydrostatic_reconstruction_and_within_cell_bed_gradient',
                    face_flux_sampling='RK-weighted mean over last accepted transport step')

    def _blocked_faces(self, axis):
        pads = ((0, 0), (1, 1)) if axis == 1 else ((1, 1), (0, 0))
        cells = np.pad(self.solid, pads, mode='wrap' if self.boundary == 'periodic' else 'edge')
        if axis == 1:
            return self.wall_x | cells[:, :-1] | cells[:, 1:]
        return self.wall_y | cells[:-1, :] | cells[1:, :]

    def _reconstruction_walls(self):
        if self.structures is None or not self.structures.barriers:
            return self.wall_x, self.wall_y
        self.structures.faces(self.time, self.structure_warmup)
        return self.wall_x | self.structures.masks[1], self.wall_y | self.structures.masks[0]

    def _structure_flux(self, axis, fields):
        """Finite crests: hydrostatic reconstruction then ANUGA weir blend.

        Both adjacent cell slopes are zeroed at a barrier. One shared mass and
        advective flux enters the existing donor limiter/RK stages. Hydrostatic
        wall pressure remains separate, including on a fully dry crest.
        """
        if self.structures is None:
            return fields
        entries = self.structures.faces(self.time, self.structure_warmup)[axis]
        if not entries:
            return fields
        blocked = self._blocked_faces(axis)
        entries = [(z,x,c,q) for (z,x),(c,q) in entries.items() if not blocked[z,x]]
        if not entries:
            return fields
        from hydraulic_structures import weir_flux
        z, x, crest, coefficients = map(np.asarray, zip(*entries))
        z, x = z.astype(int), x.astype(int)
        li = (z, x-1) if axis == 1 else (z-1, x)
        ri = (z, x)
        hl, hr = self.h[li], self.h[ri]
        bl, br = self.bed[li], self.bed[ri]
        top = np.maximum(crest, np.maximum(bl, br))
        dl = np.maximum(0., hl-(top-bl)); dr = np.maximum(0., hr-(top-br))
        ul = np.divide(self.mx[li],hl,out=np.zeros_like(hl),where=hl>self.velocity_depth)
        ur = np.divide(self.mx[ri],hr,out=np.zeros_like(hr),where=hr>self.velocity_depth)
        vl = np.divide(self.mz[li],hl,out=np.zeros_like(hl),where=hl>self.velocity_depth)
        vr = np.divide(self.mz[ri],hr,out=np.zeros_like(hr),where=hr>self.velocity_depth)
        mass, ax, az, pressure, speed = face_flux(dl,dr,ul,vl,ur,vr,
            normal=(1.,0.) if axis == 1 else (0.,1.),method=self.flux_method,low_froude=self.low_froude)
        for i in range(len(entries)):
            mass[i],ax[i],az[i],wall_speed = weir_flux(mass[i],ax[i],az[i],dl[i],dr[i],
                max(0.,top[i]-min(bl[i],br[i])),coefficients[i])
            speed[i] = max(speed[i],wall_speed)
        for key, value in dict(mass=mass,advx=ax,advz=az,pressure=pressure,
                corr_l=.5*G*(hl-dl)*(hl+dl),corr_r=.5*G*(hr-dr)*(hr+dr),speed=speed).items():
            fields[key][z,x] = value
        return fields

    def _structure_speed(self):
        if self.structures is None:
            return 0.
        speed = 0.
        # Conservative bound on the corrected crest flow; include it BEFORE
        # choosing dt, rather than only reporting it after the step.
        for axis, entries in self.structures.faces(self.time, self.structure_warmup).items():
            for (z,x),(crest,qfactor) in entries.items():
                li = (z,x-1) if axis == 1 else (z-1,x)
                ri = (z,x)
                if self.solid[li] or self.solid[ri]:continue
                head = max(0.,self.h[li]+self.bed[li]-crest,self.h[ri]+self.bed[ri]-crest)
                if not head:continue
                height = max(0.,crest-min(self.bed[li],self.bed[ri]))
                velocity = max(abs(self.mx[li])/max(self.h[li],DRY),abs(self.mx[ri])/max(self.h[ri],DRY),
                               abs(self.mz[li])/max(self.h[li],DRY),abs(self.mz[ri])/max(self.h[ri],DRY))
                speed = max(speed,velocity+np.sqrt(G*(head+height))+
                            max(2.,qfactor*2/3)*np.sqrt(2/3*G*head))
        return 2*speed

    def _slope(self, values, axis, *, differences=None, blocked=None):
        if self.spatial_order == 1:
            return np.zeros_like(values)
        if differences is None:
            differences = (values-np.roll(values, 1, axis), np.roll(values, -1, axis)-values)
        slope = minmod_slope(*differences, self.limiter_theta)
        if blocked is None:
            wx,wy = self._reconstruction_walls()
            blocked = self._blocked_faces(axis) | (wx if axis == 1 else wy)
        unavailable = (blocked[:, :-1] | blocked[:, 1:]) if axis == 1 else (blocked[:-1] | blocked[1:])
        slope[unavailable] = 0.
        if self.boundary != 'periodic':
            if axis == 1:
                slope[:, 0] = 0.; slope[:, -1] = 0.
            else:
                slope[0, :] = 0.; slope[-1, :] = 0.
        return slope

    def _reconstruction(self, h=None, velocities=None):
        h = self.h if h is None else h
        u, v = self.velocity() if velocities is None else velocities
        if self.acceleration:
            wx,wy = self._reconstruction_walls()
            a = self.acceleration.reconstruction(h,self.bed,u,v,self.solid,wx,wy,
                self.boundary=='periodic',self.spatial_order,self.limiter_theta,self.velocity_depth)
            return {axis:tuple(a[axis,k] for k in range(4)) for axis in (0,1)}
        result = {}
        wx,wy = self._reconstruction_walls()
        for axis in (0, 1):
            blocked = self._blocked_faces(axis) | (wx if axis == 1 else wy)
            hm, hp = np.roll(h, 1, axis), np.roll(h, -1, axis)
            dh = (h-hm, hp-h)
            sh = self._slope(h, axis, differences=dh, blocked=blocked)
            sh *= np.minimum(1., 2*np.maximum(h, 0.)/np.maximum(np.abs(sh), 1e-100))
            # Difference h and bed separately: a large absolute elevation must
            # not erase a millimetre-scale water-surface gradient.
            deta = (dh[0] + (self.bed-np.roll(self.bed, 1, axis)),
                    dh[1] + (np.roll(self.bed, -1, axis)-self.bed))
            seta = self._slope(h, axis, differences=deta, blocked=blocked)
            su, sv = self._slope(u, axis, blocked=blocked), self._slope(v, axis, blocked=blocked)
            near_dry = ((h <= self.velocity_depth)
                        | (hm <= self.velocity_depth) | (hp <= self.velocity_depth))
            # Fall back to first order beside a dry cell, as in ANUGA's
            # wet/dry limiter. A dry neighbour's bed is not a water surface
            # from which to extrapolate a wet cell's free-surface gradient.
            sh[near_dry] = 0.; seta[near_dry] = 0.
            su[near_dry] = 0.; sv[near_dry] = 0.
            # Reconstruct stage and depth together; derive the bed slope.
            result[axis] = (sh, seta-sh, su, sv)
        return result

    def _faces(self, values, slope, axis):
        pads = ((0, 0), (1, 1)) if axis == 1 else ((1, 1), (0, 0))
        mode = 'wrap' if self.boundary == 'periodic' else 'edge'
        left = np.pad(values+.5*slope, pads, mode=mode)
        right = np.pad(values-.5*slope, pads, mode=mode)
        if axis == 1:
            return left[:, :-1], right[:, 1:]
        return left[:-1, :], right[1:, :]

    def timestep(self, limit=1.0, rain=0.0, inflow=0.0, source=None):
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError('Timestep limit must be finite and positive')
        u, v = self.velocity()
        prospective = self.h + (rain*(~self.solid)+self.baseflow_rate
                               + self.source_rate+self.forcing_rate())*limit
        if source is not None:
            prospective = prospective.copy()
            prospective[source] += inflow*limit/self.dx**2
        bounds = []
        wx,wy = self._reconstruction_walls()
        for axis, velocity in (() if self.acceleration else ((1, u), (0, v))):
            # CFL needs only depth and normal-velocity slopes, not the full
            # eight-field transport reconstruction (including bed/tangential slopes).
            blocked = self._blocked_faces(axis) | (wx if axis == 1 else wy)
            hm, hp = np.roll(prospective, 1, axis), np.roll(prospective, -1, axis)
            sh = self._slope(prospective, axis, differences=(prospective-hm, hp-prospective), blocked=blocked)
            sh *= np.minimum(1., 2*np.maximum(prospective, 0.)/np.maximum(np.abs(sh), 1e-100))
            velocity_slope = self._slope(velocity, axis, blocked=blocked)
            near_dry = ((prospective <= self.velocity_depth)
                        | (hm <= self.velocity_depth) | (hp <= self.velocity_depth))
            sh[near_dry] = 0.; velocity_slope[near_dry] = 0.
            hv = prospective+.5*np.abs(sh)
            bounds.append(np.abs(velocity)+.5*np.abs(velocity_slope)+np.sqrt(G*np.maximum(hv, 0.)))
        if self.acceleration:
            internal_speed,z,x = self.acceleration.timestep_bounds(prospective,u,v,self.solid,
                wx,wy,self.boundary=='periodic',self.spatial_order,
                self.limiter_theta,self.velocity_depth)
            index=(z,x)
        else:
            cell_bound = bounds[0]+bounds[1]
            index = np.unravel_index(int(np.argmax(cell_bound)), self.h.shape)
            # Global directional maxima also bound waves coming from a neighbour.
            internal_speed = float(bounds[0].max()+bounds[1].max())
        inlet_speed = 2*max(e['wave'] for e in self.boundary_forcing()[0].values())
        boundary_speed = max(2*self.sea_wave_speed, inlet_speed)
        structure_speed = self._structure_speed()
        speed = max(internal_speed, boundary_speed, structure_speed, 1e-100)
        cfl_limit = self.cfl*self.dx/speed
        dt = min(limit, self.max_timestep, cfl_limit)
        cause = ('cfl_boundary' if boundary_speed > internal_speed else 'cfl_grid') if dt == cfl_limit else (
                 'max_timestep' if dt == self.max_timestep else 'event_or_output')
        if dt == cfl_limit and structure_speed >= max(internal_speed,boundary_speed):cause='cfl_structure'
        self.last_timestep = dict(dt_s=float(dt), cause=cause, requested_limit_s=float(limit),
            cfl_limit_s=float(cfl_limit), limiting_cell=[int(index[0]), int(index[1])],
            cell_depth_m=float(self.h[index]), cell_u_m_s=float(u[index]), cell_v_m_s=float(v[index]),
            grid_wave_bound_m_s=internal_speed, boundary_wave_bound_m_s=float(boundary_speed),
            structure_wave_bound_m_s=float(structure_speed))
        return dt

    def _flux(self, axis, inlet_edges=None, reconstruction=None, details=False, velocities=None):
        u, v = self.velocity() if velocities is None else velocities
        reconstruction = self._reconstruction(velocities=(u, v)) if reconstruction is None else reconstruction
        sh, sb, su, sv = reconstruction[axis]
        if self.acceleration:
            names=('west','east') if axis==1 else ('north','south')
            inlets=self.boundary_forcing()[0] if inlet_edges is None else inlet_edges
            a,b=(inlets[name] for name in names)
            fields=self.acceleration.flux_axis(self.h,self.bed,u,v,sh,sb,su,sv,self._blocked_faces(axis),
                axis,{'closed':0,'open':1,'ocean':2,'periodic':3}[self.boundary],
                self.sea_level if self.sea_level is not None else 0.,
                self.sea_edges[names[0]],self.sea_edges[names[1]],
                a['fraction'],b['fraction'],a['q'],b['q'],a['advective_x'],b['advective_x'],a['advective_z'],b['advective_z'],
                0 if self.flux_method=='rusanov' else 1,self.low_froude)
            mass,advx,advz,pressure,cl,cr,speed=fields
            face = self._structure_flux(axis,dict(zip(('mass','advx','advz','pressure','corr_l','corr_r','speed'),fields)))
            if details:
                return face
            return mass,advx+(pressure if axis==1 else 0.),advz+(pressure if axis==0 else 0.),cl,cr
        hl, hr = self._faces(self.h, sh, axis)
        zeros = np.zeros_like(self.h)
        bl, br = self._faces(self.bed, zeros, axis)
        offset_l, offset_r = self._faces(zeros, sb, axis)
        ul, ur = self._faces(u, su, axis)
        vl, vr = self._faces(v, sv, axis)
        edge_names = ('west', 'east') if axis == 1 else ('north', 'south')
        if self.boundary != 'periodic':
            for face, name, side in ((0, edge_names[0], 0), (-1, edge_names[1], 1)):
                index = (slice(None), face) if axis == 1 else (face, slice(None))
                gh, gb = (hl, bl) if side == 0 else (hr, br)
                gu, gv = (ul, vl) if side == 0 else (ur, vr)
                if self.boundary == 'ocean':
                    mask = self.sea_edges[name]
                    gh[index][mask] = np.maximum(0., self.sea_level-gb[index][mask])
                    gu[index][mask] = 0.; gv[index][mask] = 0.
                elif self.boundary == 'closed':
                    normal_velocity = gu if axis == 1 else gv
                    normal_velocity[index] *= -1.
        # Keep reconstruction offsets separate from absolute bed elevations.
        # Forming (bed+offset) first loses small offsets on high terrain.
        jump = (br-bl)+(offset_r-offset_l)
        dl = np.maximum(0., hl-np.maximum(jump, 0.))
        dr = np.maximum(0., hr-np.maximum(-jump, 0.))
        mass, advx, advz, pressure, speed = face_flux(
            dl, dr, ul, vl, ur, vr, normal=(1., 0.) if axis == 1 else (0., 1.),
            method=self.flux_method, low_froude=self.low_froude)
        # Extrapolated ghost states already passed through the selected face flux.
        # Keep that flux, including inward transport; no outflow-only override.
        if self.boundary in ('open', 'ocean') and self.open_boundary_mode == 'outflow_only':
            for face, sign, name, cell in zip((0, -1), (-1, 1), edge_names, (0, -1)):
                index = (slice(None), face) if axis == 1 else (face, slice(None))
                ci = (slice(None), cell) if axis == 1 else (cell, slice(None))
                normal_velocity = u[ci] if axis == 1 else v[ci]
                q = sign*np.maximum(0., sign*self.h[ci]*normal_velocity)
                ordinary = ~self.sea_edges[name]
                mass[index][ordinary] = q[ordinary]
                advx[index][ordinary] = (q*u[ci])[ordinary]
                advz[index][ordinary] = (q*v[ci])[ordinary]
                pressure[index][ordinary] = (.5*G*self.h[ci]**2)[ordinary]
        inlet_edges = self.boundary_forcing()[0] if inlet_edges is None else inlet_edges
        for face, sign, name, cell in zip((0, -1), (1, -1), edge_names, (0, -1)):
            index = (slice(None), face) if axis == 1 else (face, slice(None))
            ci = (slice(None), cell) if axis == 1 else (cell, slice(None))
            e = inlet_edges[name]; fraction = np.minimum(1., e['fraction']); keep = 1.-fraction
            mass[index] = mass[index]*keep+sign*e['q']
            advx[index] = advx[index]*keep+e['advective_x']
            advz[index] = advz[index]*keep+e['advective_z']
            pressure[index] = pressure[index]*keep+.5*G*self.h[ci]**2*fraction
        blocked = self._blocked_faces(axis)
        mass[blocked] = 0.; advx[blocked] = 0.; advz[blocked] = 0.; pressure[blocked] = 0.
        corr_l = .5*G*(hl-dl)*(hl+dl)
        corr_r = .5*G*(hr-dr)*(hr+dr)
        corr_l[blocked] = .5*G*hl[blocked]**2
        corr_r[blocked] = .5*G*hr[blocked]**2
        self._structure_flux(axis,dict(mass=mass,advx=advx,advz=advz,pressure=pressure,
                                      corr_l=corr_l,corr_r=corr_r,speed=speed))
        if details:
            return dict(mass=mass, advx=advx, advz=advz, pressure=pressure,
                        corr_l=corr_l, corr_r=corr_r, speed=speed)
        return (mass, advx+(pressure if axis == 1 else 0.),
                advz+(pressure if axis == 0 else 0.), corr_l, corr_r)

    def _transport_euler(self, dt, inlet_edges):
        velocities = self.velocity()
        recon = self._reconstruction(velocities=velocities)
        fx = self._flux(1, inlet_edges, recon, details=True, velocities=velocities)
        fz = self._flux(0, inlet_edges, recon, details=True, velocities=velocities)
        if self.acceleration:
            keys=('mass','advx','advz','pressure','corr_l','corr_r','speed')
            h,mx,mz,adjustment,minimum,limited,finite=self.acceleration.transport(
                self.h,self.mx,self.mz,recon[1][1],recon[0][1],
                tuple(fx[k] for k in keys),tuple(fz[k] for k in keys),dt,self.dx,self.boundary=='periodic')
            if not finite or minimum < -1e-9:
                raise RuntimeError(f'Water transport unstable: t={self.time:g}, min(h)={minimum:g}, dt={dt:g}')
            d=self.diagnostics;d['limited_faces']+=limited
            d['minimum_raw_depth_m']=min(d['minimum_raw_depth_m'],minimum)
            courant=dt/self.dx*(float(fx['speed'].max())+float(fz['speed'].max()))
            d['max_stage_courant']=max(d['max_stage_courant'],courant)
            return h,mx,mz,fx,fz,adjustment
        # Conservative draining limiter: share one donor factor at each face.
        # Scale transported mass/advection, leaving hydrostatic pressure intact.
        outgoing = (np.maximum(fx['mass'][:, 1:], 0.)+np.maximum(-fx['mass'][:, :-1], 0.)
                    + np.maximum(fz['mass'][1:], 0.)+np.maximum(-fz['mass'][:-1], 0.))
        available, requested = self.h*self.dx, dt*outgoing
        theta = np.divide(available, requested, out=np.ones_like(self.h),
                          where=requested > available)
        limited = 0
        for axis, face in ((1, fx), (0, fz)):
            pads = ((0, 0), (1, 1)) if axis == 1 else ((1, 1), (0, 0))
            if self.boundary == 'periodic':
                padded = np.pad(theta, pads, mode='wrap')
            else:
                padded = np.pad(theta, pads, mode='constant', constant_values=1.)
            donor_l = padded[:, :-1] if axis == 1 else padded[:-1]
            donor_r = padded[:, 1:] if axis == 1 else padded[1:]
            factor = np.where(face['mass'] >= 0., donor_l, donor_r)
            limited += int(np.count_nonzero((factor < 1.-1e-12) & (face['mass'] != 0.)))
            for name in ('mass', 'advx', 'advz'):
                face[name] *= factor
        ax, az = fx['mass'], fz['mass']
        bx, bz = fx['advx']+fx['pressure'], fz['advx']
        cx, cz = fx['advz'], fz['advz']+fz['pressure']
        factor = dt/self.dx
        h = self.h-factor*(ax[:, 1:]-ax[:, :-1]+az[1:]-az[:-1])
        mx = self.mx-factor*(bx[:, 1:]+fx['corr_l'][:, 1:]-bx[:, :-1]-fx['corr_r'][:, :-1]
                             +bz[1:]-bz[:-1])
        mz = self.mz-factor*(cx[:, 1:]-cx[:, :-1]
                             +cz[1:]+fz['corr_l'][1:]-cz[:-1]-fz['corr_r'][:-1])
        # Within-cell bed source, consistent with the two reconstructed depths.
        # Their average is h_i; this cancels the pressure difference at rest.
        mx -= factor*G*self.h*recon[1][1]
        mz -= factor*G*self.h*recon[0][1]
        minimum = float(h.min())
        if not all(np.isfinite(a).all() for a in (h, mx, mz)) or minimum < -1e-9:
            raise RuntimeError(f'Water transport unstable: t={self.time:g}, min(h)={minimum:g}, dt={dt:g}')
        adjustment = float(-np.minimum(h, 0.).sum()*self.dx**2)
        np.maximum(h, 0., out=h)
        self.diagnostics['limited_faces'] += limited
        self.diagnostics['minimum_raw_depth_m'] = min(self.diagnostics['minimum_raw_depth_m'], minimum)
        courant = dt/self.dx*(float(fx['speed'].max())+float(fz['speed'].max()))
        self.diagnostics['max_stage_courant'] = max(self.diagnostics['max_stage_courant'], courant)
        return h, mx, mz, fx, fz, adjustment

    def _boundary_budget(self, fx, fz, inlet_edges, dt):
        for name, outward in zip(self.sea_edges, (-fx[:, 0], fx[:, -1], -fz[0], fz[-1])):
            outward = outward+inlet_edges[name]['q']
            sea = self.sea_edges[name]
            self.balance['outflow_m3'] += float(dt*self.dx*outward[~sea].sum())
            self.balance['sea_inflow_m3'] += float(dt*self.dx*np.maximum(0., -outward[sea]).sum())
            self.balance['sea_outflow_m3'] += float(dt*self.dx*np.maximum(0., outward[sea]).sum())

    def _water_sources(self, dt, rain, infiltration, inflow, source):
        area = self.dx**2
        if self.acceleration:
            sz,sx=source if source is not None else (-1,-1)
            a,b,removed_i,removed_d,fluid=self.acceleration.water_sources(
                self.h,self.mx,self.mz,self.solid,self.baseflow_rate,self.source_rate,self.external_rate,
                self.drain_capacity,dt,rain,infiltration,self.dx,inflow,sz,sx)
            self.balance['rainfall_m3']+=rain*dt*fluid*area
            self.balance['baseflow_m3']+=self.baseflow_m3_s*dt
            self.balance['scheduled_inflow_m3']+=float(self.source_rate.sum()*dt*area)
            self.balance['external_inflow_m3']+=float(self.external_rate.sum()*dt*area)
            self.balance['inflow_m3']+=inflow*dt
            self.balance['infiltration_m3']+=removed_i;self.balance['drainage_m3']+=removed_d
            return [a,b]
        self.h += dt*(rain*(~self.solid)+self.baseflow_rate+self.source_rate+self.external_rate)
        self.balance['rainfall_m3'] += rain*dt*int((~self.solid).sum())*area
        self.balance['baseflow_m3'] += self.baseflow_m3_s*dt
        self.balance['scheduled_inflow_m3'] += float(self.source_rate.sum()*dt*area)
        self.balance['external_inflow_m3'] += float(self.external_rate.sum()*dt*area)
        if inflow:
            self.h[source] += inflow*dt/area
            self.balance['inflow_m3'] += inflow*dt
        removals = []
        for amount, key in ((infiltration*dt, 'infiltration_m3'),
                            (self.drain_capacity*dt/area, 'drainage_m3')):
            removed = np.minimum(self.h, amount)
            retained = np.divide(self.h-removed, self.h, out=np.zeros_like(self.h), where=self.h > 0.)
            self.mx *= retained; self.mz *= retained; self.h -= removed
            self.balance[key] += float(removed.sum()*area)
            removals.append(removed)
        return removals

    def _complete_dry_sinks(self, dt, infiltration, removals):
        """Spend unused sink capacity after SSP averaging at a drying event.

        Averaging a fully drained Euler stage with the initial wet state would
        otherwise leave half the water indefinitely. Only unused per-cell
        capacity can remove this remainder; smooth, unsaturated sinks need no
        correction and keep their RK weights.
        """
        if self.acceleration:
            a,b=self.acceleration.complete_sinks(self.h,self.mx,self.mz,self.drain_capacity,
                removals[0],removals[1],dt,infiltration,self.dx)
            self.balance['infiltration_m3']+=a;self.balance['drainage_m3']+=b
            return
        for requested, applied, key in zip(
                (infiltration*dt, self.drain_capacity*dt/self.dx**2), removals,
                ('infiltration_m3', 'drainage_m3')):
            extra = np.minimum(self.h, np.maximum(requested-applied, 0.))
            retained = np.divide(self.h-extra, self.h, out=np.zeros_like(self.h), where=self.h > 0.)
            self.mx *= retained; self.mz *= retained; self.h -= extra
            self.balance[key] += float(extra.sum()*self.dx**2)

    def _drag(self, dt):
        if self.acceleration:
            self.acceleration.drag(self.h,self.mx,self.mz,self.manning,dt)
            return
        drag = 1.+dt*G*self.manning**2*np.hypot(self.mx, self.mz)/np.maximum(self.h, DRY)**(7./3.)
        self.mx /= drag; self.mz /= drag

    def step(self, dt, *, rain=0.0, infiltration=0.0, inflow=0.0, source=None):
        if not all(np.isfinite(v) and v >= 0 for v in (dt, rain, infiltration, inflow)) or dt == 0:
            raise ValueError('Invalid step or forcing')
        if inflow and (source is None or len(source) != 2 or not
                       all(isinstance(v, (int, np.integer)) and 0 <= v < n for v, n in zip(source, self.h.shape))):
            raise ValueError('Inflow requires a valid source cell')
        if inflow and self.solid[source]:
            raise ValueError('Inflow cannot be inside a solid cell')
        self.source_rate = self._scheduled_rate(self.time)
        self.external_rate = self._external_rate(self.time)
        # Piecewise-constant forcing uses its left limit for BOTH RK stages.
        self._drag(.5*dt)
        initial = (self.h.copy(), self.mx.copy(), self.mz.copy())
        balance_before = dict(self.balance)
        inlet0, states0 = self.boundary_forcing()
        h1, mx1, mz1, fx0, fz0, adjustment0 = self._transport_euler(dt, inlet0)
        # Only mass fluxes survive to RK averaging and boundary accounting.
        # Release the other face arrays before constructing the second stage.
        qx0, qy0 = fx0['mass'], fz0['mass']
        del fx0, fz0
        self.h, self.mx, self.mz = h1, mx1, mz1
        removals = self._water_sources(dt, rain, infiltration, inflow, source)
        if self.time_integrator == 'rk2':
            inlet1, states1 = self.boundary_forcing()
            h2, mx2, mz2, fx1, fz1, adjustment1 = self._transport_euler(dt, inlet1)
            self.h, self.mx, self.mz = h2, mx2, mz2
            second_removals = self._water_sources(dt, rain, infiltration, inflow, source)
            removals = [.5*(a+b) for a,b in zip(removals, second_removals)]
            self.h = .5*(initial[0]+self.h)
            self.mx = .5*(initial[1]+self.mx)
            self.mz = .5*(initial[2]+self.mz)
            # Each Euler stage has applied the full source duration. Match the
            # same SSP weights used for the accepted conserved state.
            for name in self.balance:
                self.balance[name] = balance_before[name]+.5*(self.balance[name]-balance_before[name])
            self.qx = .5*(qx0+fx1['mass'])
            self.qy = .5*(qy0+fz1['mass'])
            self._boundary_budget(qx0, qy0, inlet0, .5*dt)
            self._boundary_budget(fx1['mass'], fz1['mass'], inlet1, .5*dt)
            self.balance['numerical_adjustment_m3'] += .5*(adjustment0+adjustment1)
            self.last_boundary_inflows = states1
            self.boundary_state_time = self.time+dt
        else:
            self.qx, self.qy = qx0, qy0
            self._boundary_budget(self.qx, self.qy, inlet0, dt)
            self.balance['numerical_adjustment_m3'] += adjustment0
            self.last_boundary_inflows = states0
            self.boundary_state_time = self.time
        self.boundary_sample_time = self.time
        self.balance['external_inflow_m3'] += dt*sum(s['discharge_m3_s'] for s in states0)
        for entry in self.external_schedule:
            self.external_volume_by_id[entry['id']] += self.external_discharge(entry, self.time)*dt
        self._drag(.5*dt)
        self._complete_dry_sinks(dt, infiltration, removals)
        wet = (self.h > self.velocity_depth) & (~self.solid)
        self.mx[~wet] = 0.; self.mz[~wet] = 0.; self.h[self.solid] = 0.
        if not all(np.isfinite(a).all() for a in (self.h, self.mx, self.mz)):
            raise RuntimeError('Non-finite hydraulic state after source update')
        np.maximum(self.maximum, self.h, out=self.maximum)
        self.time += dt; self.steps += 1
        d = self.diagnostics
        d['min_dt_s'] = dt if d['min_dt_s'] is None else min(d['min_dt_s'], dt)
        d['max_dt_s'] = max(d['max_dt_s'], dt)

    def advance(self, end, *, rain=0.0, rain_until=float('inf'), infiltration=0.0,
                inflow=0.0, source=None, cancelled=lambda: False, progress=None):
        if not np.isfinite(end) or end < self.time:
            raise ValueError('End time must be finite and cannot go backwards')
        if (not all(np.isfinite(v) and v >= 0 for v in (rain, infiltration, inflow))
                or np.isnan(rain_until) or rain_until < 0):
            raise ValueError('Invalid forcing')
        if inflow and (source is None or len(source) != 2 or not all(
                isinstance(v, (int, np.integer)) and 0 <= v < n for v, n in zip(source, self.h.shape))):
            raise ValueError('Inflow requires a valid source cell')
        last_report=time.perf_counter()
        while self.time < end:
            if cancelled():
                raise InterruptedError('模拟已取消')
            remaining = float(end-self.time)
            if remaining <= 8*np.finfo(float).eps*max(1., abs(end)):
                self.time = float(end)
                break
            raining = rain if self.time < rain_until-1e-10 else 0.
            limit = min(self.max_timestep, remaining)
            if raining:
                limit = min(limit, rain_until-self.time)
            if self.structures is not None and not self.structure_warmup:
                limit = min(limit,self.structures.next_event(self.time)-self.time)
            self.source_rate = self._scheduled_rate(self.time)
            self.external_rate = self._external_rate(self.time)
            for entry in self.source_schedule+self.external_schedule:
                for t, _ in entry['hydrograph']:
                    if t > self.time+1e-10:
                        limit = min(limit, t-self.time)
                        break
            dt = self.timestep(limit, raining, inflow, source)
            if dt < 1e-10 or self.steps >= self.max_steps:
                raise RuntimeError('水动力步长过小或超过步数上限: '+str(self.last_timestep))
            self.step(dt, rain=raining, infiltration=infiltration, inflow=inflow, source=source)
            if progress is not None and time.perf_counter()-last_report>=1.:
                progress(self)
                last_report=time.perf_counter()

    def summary(self):
        u, v = self.velocity()
        current = float(self.h.sum()*self.dx**2)
        b = self.balance
        expected = (b['initial_m3'] + b['rainfall_m3'] + b['inflow_m3'] + b['baseflow_m3'] + b['sea_inflow_m3']
                    + b['scheduled_inflow_m3'] + b['external_inflow_m3'] - b['drainage_m3']
                    - b['infiltration_m3'] - b['outflow_m3'] - b['sea_outflow_m3'])
        return dict(time_s=self.time, steps=self.steps, max_depth_m=float(self.h.max()),
                    numerics=self.numerical_config(), diagnostics=dict(self.diagnostics),
                    last_timestep=dict(self.last_timestep), boundary_state_time_s=self.boundary_state_time,
                    max_speed_m_s=float(np.hypot(u, v).max()),
                    wet_area_m2=int((self.h >= .01).sum())*self.dx**2,
                    newly_flooded_area_m2=int(((self.initial < .01) & (self.h >= .01)).sum())*self.dx**2,
                    external_inflow_by_id_m3=dict(self.external_volume_by_id),
                    boundary_inflows=self.last_boundary_inflows,boundary_sample_time_s=self.boundary_sample_time,
                    balance=dict(b, current_m3=current, residual_m3=current-expected,
                                 corrected_residual_m3=current-expected-b['numerical_adjustment_m3']))
