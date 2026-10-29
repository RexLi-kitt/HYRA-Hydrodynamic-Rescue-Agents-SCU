"""Sparse-stencil Gaussian SAR: (I-C)(X-mu)=epsilon (paper equations 6-14).

Nonnegative C has max row sum <= coupling < 1, so its spectral radius is
strictly below one. Cutting edges never increases an existing coefficient.
No dense N-by-N inverse or covariance is constructed, including on large grids.
"""
from dataclasses import dataclass
import numpy as np

OFFSETS=((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1))

def slices(shape,dz,dx):
    nz,nx=shape
    return ((slice(max(0,-dz),min(nz,nz-dz)),slice(max(0,-dx),min(nx,nx-dx))),
            (slice(max(0,dz),min(nz,nz+dz)),slice(max(0,dx),min(nx,nx+dx))))

@dataclass(frozen=True)
class SARConfig:
    coupling:float=.35
    noise_std:float=.1
    tolerance:float=1e-8
    max_iterations:int=256
    signal_decay_cells:float=5.

class SAR:
    def __init__(self,shape,config=SARConfig(),blocked=None,sources=()):
        self.shape=tuple(shape);self.config=config
        if len(self.shape)!=2 or any(type(n) is not int or n<2 for n in self.shape):raise ValueError('Invalid SAR grid')
        if not 0<config.coupling<1 or config.noise_std<0 or config.tolerance<=0 or config.max_iterations<1 or config.signal_decay_cells<=0:raise ValueError('Invalid SAR parameters')
        blocked=np.zeros(shape,dtype=bool) if blocked is None else np.asarray(blocked,dtype=bool)
        if blocked.shape!=self.shape:raise ValueError('Invalid blocked grid')
        proximity=np.ones(shape)
        if len(sources):
            z,x=np.indices(shape);d2=np.minimum.reduce([(z-r)**2+(x-c)**2 for r,c in sources])
            proximity+=.5*np.exp(-np.sqrt(d2))
        weights=[];denom=np.zeros(shape)
        for dz,dx in OFFSETS:
            dst,src=slices(shape,dz,dx);w=np.zeros(shape)
            w[dst]=proximity[src]*(~blocked[dst])*(~blocked[src]);weights.append(w);denom+=w
        self.weights=np.asarray(weights)*config.coupling/np.maximum(denom,1)[None]
        self.row_sum_bound=float(self.weights.sum(axis=0).max())

    def apply(self,values,cut=None):
        out=np.zeros(self.shape)
        keep=np.ones(self.shape) if cut is None else ~np.asarray(cut,dtype=bool)
        for w,(dz,dx) in zip(self.weights,OFFSETS):
            dst,src=slices(self.shape,dz,dx);out[dst]+=w[dst]*keep[dst]*keep[src]*values[src]
        return out

    def solve(self,mean,noise,cut=None):
        mean,noise=np.asarray(mean,dtype=float),np.asarray(noise,dtype=float)
        if mean.shape!=self.shape or noise.shape!=self.shape or not np.isfinite(mean).all() or not np.isfinite(noise).all():raise ValueError('Invalid SAR forcing')
        deviation=noise.copy()
        for iteration in range(1,self.config.max_iterations+1):
            next_=self.apply(deviation,cut)+noise
            bound=float(np.max(np.abs(next_-deviation)))/(1-self.config.coupling)
            deviation=next_
            if bound<=self.config.tolerance:break
        else:raise ArithmeticError('SAR fixed-point solver did not converge')
        residual=float(np.max(np.abs(deviation-self.apply(deviation,cut)-noise)))
        return mean+deviation,dict(iterations=iteration,residual_inf=residual,error_bound_inf=bound,row_sum_bound=self.row_sum_bound)

class DualGMRF:
    def __init__(self,shape,blocked,targets,seed,config=SARConfig()):
        self.shape=tuple(shape);self.config=config;self.environment=SAR(shape,config)
        self.signal=SAR(shape,config,blocked,targets)
        rng=np.random.default_rng(seed)
        self.environment_noise=rng.normal(0,config.noise_std,shape)
        noise=rng.normal(0,config.noise_std,shape)
        raw,self.signal_diagnostics=self.signal.solve(np.ones(shape),noise)
        decay=np.zeros(shape)
        if len(targets):
            z,x=np.indices(shape);d2=np.minimum.reduce([(z-r)**2+(x-c)**2 for r,c in targets])
            decay=np.exp(-d2/config.signal_decay_cells**2);decay[np.asarray(blocked,dtype=bool)]=0
        self._physical_signal=raw*decay

    def sample_signal(self,cells):
        # Sensor-only seam: caller is the environment, not the actor.
        return self._physical_signal.ravel()[cells].copy()

    def observed_environment(self,explored,observed_obstacles):
        mean=np.where(explored,np.where(observed_obstacles,-1.,1.),0.)
        return self.environment.solve(mean,self.environment_noise,cut=explored)
