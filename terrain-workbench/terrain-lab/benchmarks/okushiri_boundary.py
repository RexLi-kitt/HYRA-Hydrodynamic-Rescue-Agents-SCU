"""Experimental wet west wavemaker; three reflective walls. Not production API.
Stage is linearly interpolated without extrapolation. Ghost normal conserved
momentum is copied from the interior; tangential momentum is zero.
"""
import numpy as np
from flood_solver import FloodSolver, G
from flood_numerics import face_flux

class OkushiriBoundarySolver(FloodSolver):
    def __init__(self, bed, depth, dx, *, wave, **kwargs):
        self.wave=np.asarray(wave,dtype=float).copy()
        if self.wave.ndim!=2 or self.wave.shape[1]!=2 or len(self.wave)<2 or not np.isfinite(self.wave).all() or not (np.diff(self.wave[:,0])>0).all() or self.wave[0,0]!=0:
            raise ValueError('wave must start at zero with increasing finite time/stage pairs')
        if any(k in kwargs for k in ('boundary','sea_level','sea_edges','time_integrator')):
            raise ValueError('This experiment requires closed walls and RK2')
        super().__init__(bed,depth,dx,boundary='closed',time_integrator='rk2',**kwargs)
        self._ghost_min=self.wave[:,1].min()-self.bed[:,0]
        self._ghost_max=self.wave[:,1].max()-self.bed[:,0]
        if (self._ghost_min<=self.velocity_depth).any():
            raise ValueError('Experimental wavemaker must stay wet for entire input series')
        self.sea_edges['west'][:]=True # Account actual inlet and returning wave flux separately.
        self._wave_time=0.;self._rk_stage=0
        self.last_wave_stage_times=[]

    def stage(self,t):
        if t<0 or t>self.wave[-1,0]+1e-12:
            raise ValueError('Wave time outside supplied observations')
        return float(np.interp(t,self.wave[:,0],self.wave[:,1]))

    def timestep(self,limit=1.,rain=0.,inflow=0.,source=None):
        # Bound ghost velocity at minimum depth and gravity speed at maximum
        # depth, over the entire input. Conservative, no guessed next stage.
        self.sea_wave_speed=float((abs(self.mx[:,0])/self._ghost_min+np.sqrt(G*self._ghost_max)).max())
        future=self.wave[self.wave[:,0]>self.time+1e-12,0]
        if len(future):limit=min(limit,float(future[0]-self.time))
        return super().timestep(limit,rain,inflow,source)

    def advance(self,end,**kwargs):
        self.stage(end) # Reject extrapolation before any state mutation.
        return super().advance(end,**kwargs)

    def step(self,dt,**kwargs):
        self.stage(self.time+dt)
        if self.structures is not None or self.external_schedule or self.source_schedule or self.solid[:,0].any() or self.wall_x[:,0].any():
            raise ValueError('Unsupported structure/source/blocked wavemaker')
        self._rk_stage=0;self.last_wave_stage_times=[]
        try:return super().step(dt,**kwargs)
        finally:self._wave_time=self.time

    def _transport_euler(self,dt,inlet_edges):
        self._wave_time=self.time+self._rk_stage*dt
        self.last_wave_stage_times.append(float(self._wave_time));self._rk_stage+=1
        return super()._transport_euler(dt,inlet_edges)

    def _flux(self,axis,inlet_edges=None,reconstruction=None,details=False,velocities=None):
        fields=super()._flux(axis,inlet_edges,reconstruction,True,velocities)
        if axis==1:
            hr=self.h[:,0];hg=self.stage(self._wave_time)-self.bed[:,0]
            u,v=self.velocity() if velocities is None else velocities
            ug=self.mx[:,0]/hg
            f=face_flux(hg,hr,ug,np.zeros_like(ug),u[:,0],v[:,0],normal=(1.,0.),method=self.flux_method,low_froude=self.low_froude)
            for key,value in zip(('mass','advx','advz','pressure','speed'),f):fields[key][:,0]=value
            # Native boundary-cell slopes are zero; ghost bed equals cell bed.
            fields['corr_l'][:,0]=0.;fields['corr_r'][:,0]=0.
        if details:return fields
        return fields['mass'],fields['advx']+(fields['pressure'] if axis==1 else 0.),fields['advz']+(fields['pressure'] if axis==0 else 0.),fields['corr_l'],fields['corr_r']
