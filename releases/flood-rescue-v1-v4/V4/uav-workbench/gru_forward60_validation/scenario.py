"""Conditioned external events shared by all agents. Distances and times in SI units."""
from functools import lru_cache
import hashlib
import numpy as np
from dynamic_drop_rebuild.scenarios import sample_water, contact, release, P_REFLOAT
from flood_dqn import maps

T=1500
LIMIT=180
OUT=maps.ROOT/'generated'/'gru-forward60-validation-v1'


@lru_cache(maxsize=1)
def load_fields(parent):
    name=f'{parent}.npz'
    p=OUT/'water'/'maps'/name if 188001<=parent<=188024 else maps.DATA/'maps'/name
    if not p.exists():p=maps.DATA/'maps'/name
    with np.load(p) as data:return data['fields']


def conditioned(prob,rng,k=40):
    """Poisson-binomial suffix recurrence and exact conditional selection."""
    n=len(prob)
    suffix=np.zeros((n+1,k+1))
    suffix[n,0]=1
    for j in reversed(range(n)):
        suffix[j]=(1-prob[j])*suffix[j+1]
        suffix[j,1:]+=prob[j]*suffix[j+1,:-1]
    probability=float(suffix[0,k]);selected=[]
    if probability<=0:raise ValueError('zero probability of requested count')
    for j,p in enumerate(prob):
        if not k:break
        if rng.random()<p*suffix[j+1,k-1]/suffix[j,k]:
            selected.append(j);k-=1
    assert not k
    return np.array(selected),probability


def generate(parent,layout):
    f=load_fields(parent)
    rng=np.random.default_rng([20261104,parent,layout,188])
    waves=320*np.arange(3)+rng.integers(0,261,3)
    locations=[];risks=[];forced=[]
    for start in waves:
        depth=(1-start%5/5)*f[start//5,0]+(start%5/5)*f[start//5+1,0]
        wet=np.flatnonzero(depth>=.01)
        if not len(wet):raise ValueError(f'no wet sites for parent {parent}, layout {layout}, time {start}')
        cells=rng.choice(wet,40,replace=True)
        pos=np.column_stack([cells%100,cells//100])*10.+5
        values=np.stack([sample_water(f,t,pos) for t in range(start,start+60)],axis=1)
        strong=(values[:,:,0]>=2)|(np.linalg.norm(values[:,:,1:],axis=2)>=1)
        hazards=np.where(strong,1.,(values[:,:,0]>=.01)*P_REFLOAT)
        risks.extend(hazards*np.cumprod(np.c_[np.ones(40),1-hazards[:,:-1]],axis=1))
        forced.extend(strong)
        locations.extend(pos)
    risks=np.array(risks)
    ids,pr=conditioned(risks.sum(1).clip(0,1),rng)
    rng.shuffle(ids)
    delay=np.array([rng.choice(60,p=row/row.sum()) for row in risks[ids]])
    birth=waves[ids//40]+delay
    first=np.asarray(locations)[ids]
    hooks=rng.uniform(0,1000,(12,2));coins=rng.random((T+1,40))
    xy=np.zeros((T+1,40,2));st=np.zeros((T+1,40),np.uint8)
    for t in range(T+1):
        if t:
            xy[t]=xy[t-1];st[t]=st[t-1]
            for status in (1,2):
                j=np.flatnonzero((st[t-1]==status)&(t<=birth+LIMIT))
                if not len(j):continue
                a=xy[t-1,j];water=sample_water(f,t-1,a)
                if status==2:
                    st[t,j[release(water[:,0],np.linalg.norm(water[:,1:],axis=1),coins[t,j])]]=1
                    continue
                dry=water[:,0]<.01
                st[t,j[dry]]=2
                j=j[~dry];a=a[~dry];vel=water[~dry,1:]
                if len(j):
                    hit=contact(a,a+vel,hooks)
                    xy[t,j]=a+vel*np.minimum(1,hit)[:,None]
                    st[t,j[np.isfinite(hit)]]=3
                    st[t,j[((xy[t,j]<0)|(xy[t,j]>1000)).any(1)]]=4
        new=birth==t;st[t,new]=1;xy[t,new]=first[new]
    meta=dict(parent=parent,layout=layout,wave_starts=waves.tolist(),candidate_count=120,conditional_probability=pr,
              births_per_wave=np.bincount(ids//40,minlength=3).tolist(),strong=int(np.array(forced)[ids,delay].sum()),
              refloats=int(((st[:-1]==2)&(st[1:]==1)).sum()),hooks=hooks.tolist(),map_failures=0)
    return f,birth,xy,st,meta


def transform(pos,orientation):
    p=pos.copy()
    if orientation=='rot':
        p[...,0]=pos[...,1];p[...,1]=1000-pos[...,0]
    elif orientation=='mirror':p[...,0]=1000-pos[...,0]
    elif orientation!='main':raise ValueError(orientation)
    return p


@lru_cache(maxsize=1)
def make(parent,layout,orientation='main'):
    f,b,xy,st,meta=generate(parent,layout)
    if orientation=='rot':
        f=np.rot90(f,axes=(2,3)).copy()
        old_u=f[:,1].copy();f[:,1]=f[:,2];f[:,2]=-old_u
    elif orientation=='mirror':
        f=f[:,:,:,::-1].copy();f[:,1]*=-1
    if orientation!='main':
        xy=transform(xy,orientation)
        xy[st==0]=0
        meta['hooks']=transform(np.array(meta['hooks']),orientation).tolist()
    sha=hashlib.sha256(b.tobytes()+xy.tobytes()+st.tobytes()+np.array(meta['hooks']).tobytes()).hexdigest()
    meta.update(orientation=orientation,scenario_hash=sha,sha256=sha)
    return f,b,xy,st,meta

