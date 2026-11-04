import hashlib
import json
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt
from dynamic_drop_rebuild.scenarios import fields, sample_water, contact, release
from dynamic_drop_rebuild.planner import GRID
from flood_dqn import maps

OUT=maps.ROOT/'generated'/'guizhou-mixed-rescue-v1'
ORIGINAL_DATA=maps.DATA


def block(pos):
    x,z=(pos//200).astype(int).clip(0,4)*2
    return [int(x+10*z+j) for j in (0,1,10,11)]


def make(parent,layout):
    rng=np.random.default_rng([20261104,parent,layout])
    f=fields(parent)[180:241]
    h,u,v=f[0]
    wet=h>=.01
    edge=np.flatnonzero(wet & ~binary_erosion(wet))
    speed=np.hypot(u,v).ravel()
    cells=np.flatnonzero(wet)
    anchor=rng.choice(cells,p=(w:=speed[cells]+.001)/w.sum())
    dist=np.linalg.norm(GRID-GRID[anchor],axis=1)
    reach=cells[np.argsort(dist[cells])[:48]]
    hooks=GRID[rng.choice(edge[np.argsort(dist[edge])[:30]],12,replace=len(edge)<12)]+rng.uniform(-3,3,(12,2))
    births=np.r_[rng.integers(0,61,16),np.full(8,30),np.full(16,60)]
    first=np.empty((40,2))
    for i in range(16):
        hh=f[births[i]//5,0].flatten()
        choices=reach[hh[reach]>=.01]
        first[i]=GRID[rng.choice(choices)]
    # A bar at the wet boundary; villages on nearby land in distinct report blocks.
    bar=GRID[rng.choice(edge[np.argsort(dist[edge])[-min(len(edge),30):]])]
    bank=np.flatnonzero((~wet)&(distance_transform_edt(~wet)<=16))
    bank=bank[rng.permutation(len(bank))]
    villages=[]
    for c in bank:
        point=GRID[c]
        if all(block(point)!=block(q) and np.linalg.norm(point-q)>=120 for q in [bar,*villages]):
            villages.append(point)
        if len(villages)==2:break
    assert len(villages)==2
    for j,center in enumerate([bar,*villages]):
        first[16+8*j:24+8*j]=(center+rng.uniform(-9,9,(8,2))).clip(0,999.99)
    # Report boxes cover all eight positions near their center, including block boundaries.
    def box(pts):
        lo=np.clip(np.ceil(pts.max(0)/100).astype(int)-2,0,8)
        assert (pts.min(0)>=lo*100).all()
        return [int(lo[0]+10*lo[1]+j) for j in (0,1,10,11)]
    upstream=box(first[:16]) if (first[:16].max(0)//100-first[:16].min(0)//100<=1).all() else None
    if upstream is None:
        upstream=block(GRID[anchor])
        mask=np.isin((GRID[reach,0]//100+10*(GRID[reach,1]//100)).astype(int),upstream)
        for i in range(16):
            ids=reach[mask & (f[births[i]//5,0].ravel()[reach]>=.01)]
            first[i]=GRID[rng.choice(ids)]
    reports=[(0,upstream),(30,box(first[16:24])),(60,box(first[24:32])),(60,box(first[32:]))]
    # Nominal flood route from the mapped inflow at row 97 into the domain.
    from large_flood_routes.dataset import parent_parameters
    terrain=parent_parameters(parent)
    corridor=[]
    for row in range(min(z//10 for z in upstream)-1,-1,-1):
        z=100*row+50
        x=500+5*(terrain['center_cells']-120)+5*terrain['meander_cells']*np.sin(z/(5*terrain['wavelength_cells'])+terrain['phase'])
        corridor.append(int(np.clip(x//100,0,9))+10*row)
    corridor=corridor[:3]
    coords=np.zeros((301,40,2));state=np.zeros((301,40),dtype='u1')
    coins=rng.random((301,16))
    for t in range(301):
        if t:
            coords[t]=coords[t-1];state[t]=state[t-1]
            for st in (1,2):
                ids=np.flatnonzero((state[t-1,:16]==st)&(t<=births[:16]+240))
                if not len(ids):continue
                a=coords[t-1,ids];water=sample_water(f,t-1,a)
                if st==2:
                    state[t,ids[release(water[:,0],np.linalg.norm(water[:,1:],axis=1),coins[t,ids])]]=1
                    continue
                dry=water[:,0]<.01
                state[t,ids[dry]]=2
                ids=ids[~dry];a=a[~dry];vel=water[~dry,1:]
                if len(ids):
                    hit=contact(a,a+vel,hooks)
                    coords[t,ids]=a+vel*np.minimum(1,hit)[:,None]
                    state[t,ids[np.isfinite(hit)]]=3
                    outside=((coords[t,ids]<0)|(coords[t,ids]>1000)).any(1)
                    state[t,ids[outside]]=4
        born=births==t
        coords[t,born]=first[born]
        state[t,born]=np.where(np.flatnonzero(born)<16,1,5)
    meta=dict(parent=parent,layout=layout,hooks=hooks.tolist(),reports=reports,corridor=corridor)
    sha=hashlib.sha256(births.tobytes()+coords.tobytes()+state.tobytes()+json.dumps(meta,sort_keys=True).encode()).hexdigest()
    tags=[]
    for i,b in enumerate(births):
        tr=coords[b:b+241,i][state[b:b+241,i]!=4]
        move=float(np.linalg.norm(tr-first[i],axis=1).max())
        tags.append(dict(kind=['落水','浅滩','村落一','村落二'][0 if i<16 else 1+(i-16)//8],
                         displacement=move,crossed=bool(np.any(tr//100!=first[i]//100)),
                         snag=bool(np.any(state[b:b+241,i]==3))))
    return (births,coords,state,sha,0),f,meta,tags

