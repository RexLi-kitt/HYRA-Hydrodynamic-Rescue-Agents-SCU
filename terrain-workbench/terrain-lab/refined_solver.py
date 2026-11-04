"""Conservative static 2:1 leaf-cell SWE solver with synchronized SSP-RK2 stages.

Coarse/fine interfaces are split into physical subfaces; each Riemann flux is
computed once and enters both neighbors with opposite signs and its true length.
The base grid is an area/volume projection for editing and legacy display only.
Subcells inherit parent topography by default; optional aligned fine elevations
are supported only for fresh dry interior terrain (no obstacle-mask changes).
"""
from copy import deepcopy
import numpy as np
from flood_solver import FloodSolver,G,DRY
from flood_numerics import face_flux
from refined_mesh import LeafMesh


class RefinedFloodSolver(FloodSolver):
    @classmethod
    def from_solver(cls,solver,mask,*,fine_bed=None,fine_dx=None):
        if solver.time!=0 or solver.steps!=0:raise ValueError('加密网格必须在模拟开始前建立')
        mask=np.asarray(mask,bool)
        if mask.shape!=solver.h.shape:raise ValueError('Refinement mask must match parent grid')
        if (fine_bed is None)!=(fine_dx is None):raise ValueError('Fine terrain requires both elevations and spacing')
        fine=None
        if fine_bed is not None:
            fine=np.asarray(fine_bed,dtype=float)
            if (fine.shape!=tuple(2*n for n in solver.h.shape) or not np.isfinite(fine).all()
                    or not isinstance(fine_dx,(int,float,np.integer,np.floating))
                    or isinstance(fine_dx,(bool,np.bool_)) or not np.isfinite(fine_dx)
                    or not np.isclose(fine_dx,solver.dx/2,rtol=1e-12,atol=0)):
                raise ValueError('Fine terrain must be finite, aligned 2:1, with half parent spacing')
            if any(np.any(getattr(solver,k)!=0) for k in ('h','mx','mz','initial','maximum')):
                raise ValueError('Fine terrain currently requires a fresh dry state')
            if solver.structures is not None and solver.structures.compiled:
                raise ValueError('Fine terrain with hydraulic structures is not supported')
            # Boundary forcing still uses parent geometry; keep it unchanged.
            changed=(fine!=np.repeat(np.repeat(solver.bed,2,axis=0),2,axis=1)).reshape(
                solver.h.shape[0],2,solver.h.shape[1],2).any(axis=(1,3)) & mask
            if changed[0].any() or changed[-1].any() or changed[:,0].any() or changed[:,-1].any():
                raise ValueError('Fine terrain changes must stay away from outer boundary parents')
        # These experimental closures exist only in the native-grid flux path.
        if solver.open_boundary_mode != 'outflow_only' or solver.wall_flux_mode != 'hydrostatic':
            raise ValueError('Refined mesh does not support experimental open_boundary_mode or wall_flux_mode; use the native grid')
        obj=cls.__new__(cls)
        # Each simulation owns its mutable state, forcing and audit histories.
        # The acceleration module is stateless code and cannot be deep-copied.
        obj.__dict__=deepcopy(solver.__dict__, {id(solver.acceleration): solver.acceleration})
        obj.mesh=m=LeafMesh(mask,solver.dx,solver.boundary)
        p=m.parent
        obj.leaf={key:getattr(solver,base).ravel()[p].copy() for key,base in
            [('h','h'),('mx','mx'),('mz','mz'),('bed','bed'),('maximum','maximum'),('initial','initial')]}
        obj.terrain_representation='inherited_parent'
        if fine is not None:
            selected=m.child>=0
            zz=2*(p[selected]//m.shape[1])+m.child[selected]//2
            xx=2*(p[selected]%m.shape[1])+m.child[selected]%2
            obj.leaf['bed'][selected]=fine[zz,xx]
            obj.bed=m.project(obj.leaf['bed'])
            obj.terrain_representation='aligned_fine_bed_dry_start_experimental'
        obj.face_mass_flux=np.zeros(len(m.left))
        obj._setup_geometry();obj._sync()
        return obj

    def _setup_geometry(self):
        m=self.mesh
        self.leaf_solid=self.solid.ravel()[m.parent]
        blocked=self.leaf_solid[m.l]|self.leaf_solid[m.r]
        for axis,wall in ((1,self.wall_x),(0,self.wall_y)):
            selected=(m.axis==axis)&(m.baseface>=0)
            blocked[selected]|=wall.ravel()[m.baseface[selected]]
            if self.boundary=='periodic':
                selected=m.periodic&(m.axis==axis)
                opposite=wall[:,0] if axis==1 else wall[0]
                index=m.parent[m.l[selected]]//m.shape[1] if axis==1 else m.parent[m.l[selected]]%m.shape[1]
                blocked[selected]|=opposite[index]
        self.blocked=blocked
        self.neighbors=m.internal&~blocked
        self.il=m.l[self.neighbors];self.ir=m.r[self.neighbors]
        dx=m.x[self.ir]-m.x[self.il];dz=m.z[self.ir]-m.z[self.il]
        periodic=m.periodic[self.neighbors]
        dx[periodic&(m.axis[self.neighbors]==1)]+=m.shape[1]*self.dx
        dz[periodic&(m.axis[self.neighbors]==0)]+=m.shape[0]*self.dx
        w=m.length[self.neighbors]/np.maximum(dx*dx+dz*dz,1e-100)
        self.wx=w*dx;self.wz=w*dz
        def both(value):return np.bincount(self.il,weights=value,minlength=m.count)+np.bincount(self.ir,weights=value,minlength=m.count)
        a,b,c=both(w*dx*dx),both(w*dx*dz),both(w*dz*dz)
        determinant=a*c-b*b
        self.invxx=np.divide(c,determinant,out=np.zeros_like(c),where=determinant>1e-100)
        self.invxz=np.divide(-b,determinant,out=np.zeros_like(b),where=determinant>1e-100)
        self.invzz=np.divide(a,determinant,out=np.zeros_like(a),where=determinant>1e-100)

    def _sync(self):
        for key in ('h','mx','mz','maximum'):setattr(self,key,self.mesh.project(self.leaf[key]))

    def leaf_velocity(self):
        h=self.leaf['h'];wet=h>self.velocity_depth
        return tuple(np.divide(self.leaf[key],h,out=np.zeros_like(h),where=wet) for key in ('mx','mz'))

    def _gradient(self,values,difference=None):
        m=self.mesh
        difference=values[self.ir]-values[self.il] if difference is None else difference
        def rhs(weights):
            v=weights*difference
            return np.bincount(self.il,weights=v,minlength=m.count)+np.bincount(self.ir,weights=v,minlength=m.count)
        bx,bz=rhs(self.wx),rhs(self.wz)
        gx,gz=self.invxx*bx+self.invxz*bz,self.invxz*bx+self.invzz*bz
        lower=np.zeros(m.count);upper=np.zeros(m.count)
        np.minimum.at(lower,self.il,difference);np.minimum.at(lower,self.ir,-difference)
        np.maximum.at(upper,self.il,difference);np.maximum.at(upper,self.ir,-difference)
        factor=np.ones(m.count)
        for valid,cell,ox,oz in ((m.lv,m.l,m.ox_l,m.oz_l),(m.rv,m.r,m.ox_r,m.oz_r)):
            delta=gx[cell]*ox+gz[cell]*oz
            bound=np.where(delta>0,upper[cell],lower[cell])*.5*self.limiter_theta
            ratio=np.divide(bound,delta,out=np.ones_like(delta),where=np.abs(delta)>1e-100)
            np.minimum.at(factor,cell[valid],np.clip(ratio[valid],0,1))
        return gx*factor,gz*factor

    def _reconstruct(self):
        m=self.mesh;h=self.leaf['h'];u,v=self.leaf_velocity()
        if self.spatial_order==1:
            gradients=[(np.zeros(m.count),np.zeros(m.count)) for _ in range(4)]
        else:
            dh=h[self.ir]-h[self.il]
            gradients=[self._gradient(h,dh),self._gradient(h,dh+(self.leaf['bed'][self.ir]-self.leaf['bed'][self.il])),
                self._gradient(u),self._gradient(v)]
            near=h<=self.velocity_depth
            dry_neighbor=(h[self.il]<=self.velocity_depth)|(h[self.ir]<=self.velocity_depth)
            near[self.il[dry_neighbor]]=True;near[self.ir[dry_neighbor]]=True
            for gx,gz in gradients:gx[near]=0;gz[near]=0
            # Positivity at every subface, including hanging-node interfaces.
            factor=np.ones(m.count);gx,gz=gradients[0]
            for valid,cell,ox,oz in ((m.lv,m.l,m.ox_l,m.oz_l),(m.rv,m.r,m.ox_r,m.oz_r)):
                delta=gx[cell]*ox+gz[cell]*oz
                ratio=np.divide(h[cell],-delta,out=np.ones_like(delta),where=delta<0)
                np.minimum.at(factor,cell[valid],np.clip(ratio[valid],0,1))
            gradients[0]=(gx*factor,gz*factor)
        gh,geta,gu,gv=gradients
        gb=(geta[0]-gh[0],geta[1]-gh[1])
        sides=[]
        for cell,ox,oz in ((m.l,m.ox_l,m.oz_l),(m.r,m.ox_r,m.oz_r)):
            extrapolate=lambda pair:pair[0][cell]*ox+pair[1][cell]*oz
            sides.append([np.maximum(0,h[cell]+extrapolate(gh)),extrapolate(gb),
                u[cell]+extrapolate(gu),v[cell]+extrapolate(gv)])
        return sides

    def _fluxes(self,inlets):
        m=self.mesh;(hl,bl,ul,vl),(hr,br,ur,vr)=self._reconstruct()
        for edge,name in enumerate(('west','east','north','south')):
            selected=m.edge==edge
            ghost=(hl,ul,vl) if edge in (0,2) else (hr,ur,vr)
            if self.boundary=='closed':ghost[1 if edge<2 else 2][selected]*=-1
            elif self.boundary=='ocean':
                sea=selected.copy();sea[selected]=self.sea_edges[name][m.offset[selected]]
                bed=self.leaf['bed'][m.l if edge in (0,2) else m.r]
                offset=bl if edge in (0,2) else br
                ghost[0][sea]=np.maximum(0,self.sea_level-bed[sea]-offset[sea])
                ghost[1][sea]=0;ghost[2][sea]=0
        jump=(self.leaf['bed'][m.r]-self.leaf['bed'][m.l])+(br-bl)
        dl=np.maximum(0,hl-np.maximum(jump,0));dr=np.maximum(0,hr-np.maximum(-jump,0))
        mass=np.zeros(len(m.l));advx=mass.copy();advz=mass.copy();pressure=mass.copy();speed=mass.copy()
        for axis in (1,0):
            choose=m.axis==axis
            fields=face_flux(dl[choose],dr[choose],ul[choose],vl[choose],ur[choose],vr[choose],
                normal=(1.,0.) if axis==1 else (0.,1.),method=self.flux_method,low_froude=self.low_froude)
            for result,field in zip((mass,advx,advz,pressure,speed),fields):result[choose]=field
        for edge,name in enumerate(('west','east','north','south')):
            choose=m.edge==edge;indices=np.flatnonzero(choose);offset=m.offset[choose]
            h,u,v=(hr,ur,vr) if edge in (0,2) else (hl,ul,vl)
            sign=-1 if edge in (0,2) else 1
            if self.boundary in ('open','ocean'):
                ordinary=indices[~self.sea_edges[name][offset]]
                q=sign*np.maximum(0,sign*h[ordinary]*(u[ordinary] if edge<2 else v[ordinary]))
                mass[ordinary]=q;advx[ordinary]=q*u[ordinary];advz[ordinary]=q*v[ordinary]
                pressure[ordinary]=.5*G*h[ordinary]**2
            e=inlets[name];fraction=np.minimum(1,e['fraction'][offset]);keep=1-fraction
            mass[choose]=mass[choose]*keep-sign*e['q'][offset]
            advx[choose]=advx[choose]*keep+e['advective_x'][offset]
            advz[choose]=advz[choose]*keep+e['advective_z'][offset]
            pressure[choose]=pressure[choose]*keep+.5*G*h[choose]**2*fraction
        corr_l=.5*G*(hl-dl)*(hl+dl);corr_r=.5*G*(hr-dr)*(hr+dr)
        for a in (mass,advx,advz,pressure):a[self.blocked]=0
        corr_l[self.blocked]=.5*G*hl[self.blocked]**2;corr_r[self.blocked]=.5*G*hr[self.blocked]**2
        return mass,advx,advz,pressure,corr_l,corr_r,speed,hl,hr,bl,br

    def _transport(self,dt,inlets):
        m=self.mesh;h=self.leaf['h']
        mass,ax,az,pressure,cl,cr,speed,hl,hr,bl,br=self._fluxes(inlets)
        outgoing=m.sum_left(np.maximum(mass,0)*m.length)+m.sum_right(np.maximum(-mass,0)*m.length)
        available=h*m.area;requested=dt*outgoing
        theta=np.divide(available,requested,out=np.ones_like(h),where=requested>available)
        donor=np.where(mass>=0,np.where(m.lv,theta[m.l],1),np.where(m.rv,theta[m.r],1))
        for a in (mass,ax,az):a*=donor
        new_h=h-dt*m.divergence(mass)/m.area
        momenta=[]
        for axis,adv,key in ((1,ax,'mx'),(0,az,'mz')):
            normal=(m.axis==axis).astype(float)
            force=m.divergence(adv+pressure*normal)
            force+=m.sum_left(cl*normal*m.length)-m.sum_right(cr*normal*m.length)
            # Cell/face bed source quadrature: at a constant stage this cancels
            # the reconstructed pressure exactly, also across coarse/fine faces.
            bed_source=-.5*G*(m.sum_left((hl+h[m.l])*bl*normal*m.length)
                -m.sum_right((hr+h[m.r])*br*normal*m.length))
            momenta.append(self.leaf[key]+dt*(bed_source-force)/m.area)
        minimum=float(new_h.min())
        if minimum < -1e-9 or not all(np.isfinite(a).all() for a in (new_h,*momenta)):
            raise RuntimeError(f'加密网格水动力不稳定: t={self.time:g}, min(h)={minimum:g}, dt={dt:g}')
        adjustment=float((-np.minimum(new_h,0)*m.area).sum());np.maximum(new_h,0,out=new_h)
        self.leaf['h'],self.leaf['mx'],self.leaf['mz']=new_h,*momenta
        d=self.diagnostics;d['limited_faces']+=int(np.count_nonzero(donor<1-1e-12))
        d['minimum_raw_depth_m']=min(d['minimum_raw_depth_m'],minimum)
        courant=float((dt*(m.sum_left(speed*m.length)+m.sum_right(speed*m.length))/m.area).max())
        d['max_stage_courant']=max(d['max_stage_courant'],courant)
        return mass,adjustment

    def _sources(self,dt,rain,infiltration,inflow,source):
        m=self.mesh;p=m.parent;h=self.leaf['h'];solid=self.leaf_solid
        h+=dt*(rain*(~solid)+(self.baseflow_rate+self.source_rate+self.external_rate).ravel()[p])
        for key,value in [('rainfall_m3',rain*float(m.area[~solid].sum())),('baseflow_m3',self.baseflow_m3_s),
                ('scheduled_inflow_m3',float(self.source_rate.sum()*self.dx**2)),
                ('external_inflow_m3',float(self.external_rate.sum()*self.dx**2))]:self.balance[key]+=dt*value
        if inflow:
            cell=source[0]*m.shape[1]+source[1];start=m.start[cell];count=4 if m.mask.ravel()[cell] else 1
            h[start:start+count]+=inflow*dt/self.dx**2;self.balance['inflow_m3']+=inflow*dt
        removals=[]
        for requested,key in ((infiltration*dt,'infiltration_m3'),(self.drain_capacity.ravel()[p]*dt/self.dx**2,'drainage_m3')):
            removed=np.minimum(h,requested)
            retained=np.divide(h-removed,h,out=np.zeros_like(h),where=h>0)
            self.leaf['mx']*=retained;self.leaf['mz']*=retained;h-=removed
            self.balance[key]+=float((removed*m.area).sum());removals.append(removed)
        return removals

    def _leaf_drag(self,dt):
        h=self.leaf['h'];mx=self.leaf['mx'];mz=self.leaf['mz']
        drag=1+dt*G*self.manning.ravel()[self.mesh.parent]**2*np.hypot(mx,mz)/np.maximum(h,DRY)**(7/3)
        mx/=drag;mz/=drag

    def _mesh_boundary_budget(self,flux,inlets,dt):
        m=self.mesh
        for edge,name in enumerate(('west','east','north','south')):
            choose=m.edge==edge;offset=m.offset[choose];sign=-1 if edge in (0,2) else 1
            outward=(sign*flux[choose]+inlets[name]['q'][offset])*m.length[choose]
            sea=self.sea_edges[name][offset]
            self.balance['outflow_m3']+=dt*float(outward[~sea].sum())
            self.balance['sea_inflow_m3']+=dt*float(np.maximum(-outward[sea],0).sum())
            self.balance['sea_outflow_m3']+=dt*float(np.maximum(outward[sea],0).sum())

    def timestep(self,limit=1.,rain=0.,inflow=0.,source=None):
        if not np.isfinite(limit) or limit<=0:raise ValueError('Timestep limit must be finite and positive')
        u,v=self.leaf_velocity();m=self.mesh
        inlets=self.boundary_forcing()[0]
        rate=rain*(~self.leaf_solid)+(self.baseflow_rate+self.source_rate+self.external_rate).ravel()[m.parent]
        for edge,name in enumerate(('west','east','north','south')):
            choose=m.edge==edge;cell=m.r[choose] if edge in (0,2) else m.l[choose]
            rate+=np.bincount(cell,weights=inlets[name]['q'][m.offset[choose]]*m.length[choose],minlength=m.count)/m.area
        prospective=self.leaf['h']+limit*rate
        if inflow and source is not None:prospective[m.parent==source[0]*m.shape[1]+source[1]]+=limit*inflow/self.dx**2
        speed=float(np.abs(u).max()+np.abs(v).max()+2*np.sqrt(G*np.maximum(prospective,0)).max())
        boundary=max(2*self.sea_wave_speed,2*max(e['wave'] for e in inlets.values()))
        bound=max(speed,boundary,1e-100)
        cfl_limit=.5*self.cfl*float(m.width.min())/bound
        dt=min(limit,self.max_timestep,cfl_limit)
        self.last_timestep=dict(dt_s=dt,cause='cfl_refined_mesh' if dt==cfl_limit else 'event_or_output',
            cfl_limit_s=cfl_limit,minimum_cell_width_m=float(m.width.min()),leaf_cell_count=m.count)
        return dt

    def step(self,dt,*,rain=0.,infiltration=0.,inflow=0.,source=None):
        if not all(np.isfinite(v) and v>=0 for v in (dt,rain,infiltration,inflow)) or dt==0:raise ValueError('Invalid step or forcing')
        if inflow and (source is None or len(source)!=2 or not all(isinstance(v,(int,np.integer)) and 0<=v<n for v,n in zip(source,self.h.shape)) or self.solid[source]):
            raise ValueError('Inflow requires a valid non-solid source cell')
        if self.steps == 0: self._record_speed_peak()
        self.source_rate=self._scheduled_rate(self.time);self.external_rate=self._external_rate(self.time)
        self._leaf_drag(.5*dt);initial={k:self.leaf[k].copy() for k in ('h','mx','mz')};before=dict(self.balance)
        inlet0,states0=self.boundary_forcing();flux0,adjust0=self._transport(dt,inlet0)
        removals=self._sources(dt,rain,infiltration,inflow,source)
        if self.time_integrator=='rk2':
            self._sync();inlet1,states1=self.boundary_forcing();flux1,adjust1=self._transport(dt,inlet1)
            second=self._sources(dt,rain,infiltration,inflow,source)
            removals=[.5*(a+b) for a,b in zip(removals,second)]
            for k in initial:self.leaf[k]=.5*(initial[k]+self.leaf[k])
            for k in self.balance:self.balance[k]=before[k]+.5*(self.balance[k]-before[k])
            self.face_mass_flux=.5*(flux0+flux1)
            self._mesh_boundary_budget(flux0,inlet0,.5*dt);self._mesh_boundary_budget(flux1,inlet1,.5*dt)
            self.balance['numerical_adjustment_m3']+=.5*(adjust0+adjust1)
            self.last_boundary_inflows=states1;self.boundary_state_time=self.time+dt
        else:
            self.face_mass_flux=flux0;self._mesh_boundary_budget(flux0,inlet0,dt)
            self.balance['numerical_adjustment_m3']+=adjust0
            self.last_boundary_inflows=states0;self.boundary_state_time=self.time
        self.boundary_sample_time=self.time
        self.balance['external_inflow_m3']+=dt*sum(s['discharge_m3_s'] for s in states0)
        for entry in self.external_schedule:self.external_volume_by_id[entry['id']]+=self.external_discharge(entry,self.time)*dt
        self._leaf_drag(.5*dt)
        h=self.leaf['h'];m=self.mesh
        for capacity,applied,key in zip((infiltration*dt,self.drain_capacity.ravel()[m.parent]*dt/self.dx**2),removals,('infiltration_m3','drainage_m3')):
            extra=np.minimum(h,np.maximum(capacity-applied,0));keep=np.divide(h-extra,h,out=np.zeros_like(h),where=h>0)
            self.leaf['mx']*=keep;self.leaf['mz']*=keep;h-=extra;self.balance[key]+=float((extra*m.area).sum())
        dry=(h<=self.velocity_depth)|self.leaf_solid
        self.leaf['mx'][dry]=0;self.leaf['mz'][dry]=0;h[self.leaf_solid]=0
        np.maximum(self.leaf['maximum'],h,out=self.leaf['maximum'])
        if not all(np.isfinite(self.leaf[k]).all() for k in ('h','mx','mz')):raise RuntimeError('Non-finite refined hydraulic state')
        self.time+=dt;self.steps+=1;d=self.diagnostics
        self._record_speed_peak()
        d['min_dt_s']=dt if d['min_dt_s'] is None else min(d['min_dt_s'],dt);d['max_dt_s']=max(d['max_dt_s'],dt)
        self.qx=m.face_projection(self.face_mass_flux,1);self.qy=m.face_projection(self.face_mass_flux,0);self._sync()

    def start_scenario(self):
        self.leaf['initial']=self.leaf['h'].copy();self.leaf['maximum']=self.leaf['h'].copy();self.face_mass_flux.fill(0)
        self._sync();super().start_scenario()

    def numerical_config(self):
        result=super().numerical_config()
        result.update(terrain_representation=self.terrain_representation,mesh='static_2_to_1_leaf_squares',limiter='least_squares_Barth_Jespersen',
            source_integration='synchronized_SSP_stages; area_weighted_sources; Strang_Manning',
            bed_source='cell_face_trapezoid_and_hydrostatic_face_corrections')
        return result

    def summary(self):
        result=super().summary();u,v=self.leaf_velocity();h=self.leaf['h'];area=self.mesh.area
        current=float((h*area).sum());balance=result['balance'];expected=balance['current_m3']-balance['residual_m3']
        balance.update(current_m3=current,residual_m3=current-expected,corrected_residual_m3=current-expected-balance['numerical_adjustment_m3'])
        result.update(max_depth_m=float(h.max()),max_speed_m_s=float(np.hypot(u,v).max()),
            wet_area_m2=float(area[h>=.01].sum()),newly_flooded_area_m2=float(area[(h>=.01)&(self.leaf['initial']<.01)].sum()),
            leaf_cell_count=self.mesh.count,refined_parent_cells=int(self.mesh.mask.sum()))
        return result

    def mesh_snapshot(self,compact=False):
        m=self.mesh;u,v=self.leaf_velocity()
        result=dict(schema='hyra-static-leaf-v1',parent=m.parent,child=m.child,cell_count=m.count,
            h=self.leaf['h'],u=u,v=v,maximum_depth=self.leaf['maximum'])
        if not compact:result.update(area_m2=m.area,x_m=m.x,z_m=m.z,width_m=m.width,bed=self.leaf['bed'],
            mx=self.leaf['mx'],mz=self.leaf['mz'],initial_depth=self.leaf['initial'],
            face_left=m.left,face_right=m.right,face_axis=m.axis,face_length_m=m.length,
            face_mass_flux=self.face_mass_flux)
        return result
