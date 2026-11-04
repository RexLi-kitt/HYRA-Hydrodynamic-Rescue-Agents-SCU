"""Synthetic dynamic drop scenarios. Coordinates x,z and velocity u,v are in SI units."""
from functools import lru_cache
from pathlib import Path
import hashlib
import numpy as np
from flood_dqn import maps
OUT=maps.ROOT/'generated'/'dynamic-drop-rebuild-v1'
T=1320
LIMIT=240
P_REFLOAT=-np.expm1(np.log(.95)/60)
@lru_cache(maxsize=1)
def fields(parent):
    p=maps.DATA/'maps'/f'{parent}.npz'
    if not p.exists():p=OUT/'new-water'/'maps'/f'{parent}.npz'
    with np.load(p) as f:return f['fields']
def sample_water(f,t,pos):
    a=np.clip(pos/10-.5,0,99);i=a.astype(int);j=np.minimum(i+1,99)
    dx,dz=(a-i).T
    def frame(k):
        v=f[k]
        return ((1-dz)*((1-dx)*v[:,i[:,1],i[:,0]]+dx*v[:,i[:,1],j[:,0]])+
                dz*((1-dx)*v[:,j[:,1],i[:,0]]+dx*v[:,j[:,1],j[:,0]])).T
    k=t//5;frac=t%5/5
    return frame(k) if not frac else (1-frac)*frame(k)+frac*frame(k+1)
def release(h,speed,u):
    return (h>=.01)&((h>=2)|(speed>=1)|(u<P_REFLOAT))
def contact(a,b,bollards):
    v=b-a;d=a[:,None]-bollards
    vv=(v*v).sum(1)[:,None];dv=(d*v[:,None]).sum(2);c=(d*d).sum(2)-16
    disc=dv*dv-vv*c
    t=(-dv-np.sqrt(np.maximum(0,disc)))/np.maximum(vv,1e-25)
    return np.where(c<=0,0.,np.where((vv>0)&(disc>=0)&(t>=0)&(t<=1),t,np.inf)).min(1)
@lru_cache(maxsize=2)
def scenario(parent,layout,refloat=True):
    f=fields(parent);rng=np.random.default_rng([20261103,parent,layout,0]);events=[]
    for _ in range(10000):
        at=rng.integers(0,T-LIMIT+1,4096);cell=rng.integers(10000,size=len(at))
        v=f[at//5,:,cell//100,cell%100]*(1-at[:,None]%5/5)+f[at//5+1,:,cell//100,cell%100]*(at[:,None]%5/5)
        strong=(v[:,0]>=2)|(np.hypot(v[:,1],v[:,2])>=1)
        ok=(v[:,0]>=.01)&(strong|(rng.random(len(at))<.1))
        events.extend(zip(at[ok],cell[ok],strong[ok]))
        if len(events)>=40:break
    if len(events)<40:raise ValueError('insufficient wet exposures')
    events=sorted(events[:40],key=lambda a:a[0]);birth,cells,strong=map(np.array,zip(*events))
    hooks=np.random.default_rng([20261103,parent,layout,1]).uniform(0,1000,(12,2))
    coins=np.random.default_rng([20261103,parent,layout,2]).random((T+1,40))
    xy=np.zeros((T+1,40,2));initial=np.c_[cells%100,cells//100]*10.+5
    state=np.zeros((T+1,40),np.uint8) # 0 pending, 1 floating, 2 stranded, 3 snagged, 4 outside
    for t in range(T+1):
        if t:
            xy[t]=xy[t-1];state[t]=state[t-1]
            active=(state[t-1]==1)&(t<=birth+LIMIT);stranded=(state[t-1]==2)&(t<=birth+LIMIT)
            idx=np.flatnonzero(active)
            if len(idx):
                a=xy[t-1,idx];wv=sample_water(f,t-1,a);dry=wv[:,0]<.01
                state[t,idx[dry]]=2
                idx=idx[~dry];a=a[~dry];v=wv[~dry,1:]
                if len(idx):
                    b=a+v;hit=contact(a,b,hooks);snap=np.isfinite(hit)
                    xy[t,idx]=a+v*np.minimum(hit,1)[:,None];state[t,idx[snap]]=3
                    outside=((xy[t,idx]<0)|(xy[t,idx]>1000)).any(1);state[t,idx[outside]]=4
            idx=np.flatnonzero(stranded)
            if refloat and len(idx):
                w=sample_water(f,t-1,xy[t-1,idx]);state[t,idx[release(w[:,0],np.hypot(w[:,1],w[:,2]),coins[t,idx])]]=1
        new=birth==t;xy[t,new]=initial[new];state[t,new]=1
    digest=hashlib.sha256(birth.tobytes()+xy.tobytes()+state.tobytes()).hexdigest()
    return birth,xy,state,digest,int(strong.sum())
