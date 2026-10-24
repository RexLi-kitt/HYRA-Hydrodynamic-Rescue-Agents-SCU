"""Read-only diagnostics of clear corridors between authored rectangular buildings.

Connectivity refers only to the building raster inside each corridor. Terrain,
walls, levees, water levels and routes around the corridor are outside this audit.
"""
from collections import deque
import math
import numpy as np

def audit_channels(structures, max_candidates=2000, cell_budget=2000000):
    buildings=[b for b in structures.compiled if b['kind']=='building']
    dx=structures.dx
    rects=np.asarray([[b['x_m'],b['z_m'],b['x_m']+b['width_m'],
                       b['z_m']+b['length_m']] for b in buildings],float).reshape(-1,4)
    result=dict(schema='hyra-channel-audit-v1',scope='authored_rectangle_buildings_only',
                dx_m=dx,building_count=len(buildings),channels=[],excluded_obstructed=0,
                complete=True,warning_threshold_fraction=.2)
    candidates=0
    for i,a in enumerate(rects):
        for j in range(i+1,len(rects)):
            b=rects[j]
            for axis in (0,1):
                along=1-axis
                left,right=(a,b) if a[axis]<b[axis] else (b,a)
                lo,hi=left[axis+2],right[axis]
                start,end=max(a[along],b[along]),min(a[along+2],b[along+2])
                if hi-lo<=1e-9 or end-start<=1e-9:continue
                bounds=[0.,0.,0.,0.]
                bounds[axis],bounds[axis+2]=lo,hi
                bounds[along],bounds[along+2]=start,end
                candidates+=1
                if candidates>max_candidates:
                    result['complete']=False
                    return result
                # A third physical building means this is not a clear corridor.
                overlap=(rects[:,0]<bounds[2]-1e-9)&(rects[:,2]>bounds[0]+1e-9)&(rects[:,1]<bounds[3]-1e-9)&(rects[:,3]>bounds[1]+1e-9)
                overlap[[i,j]]=False
                if overlap.any():
                    result['excluded_obstructed']+=1
                    continue
                lateral0=math.ceil(lo/dx-1e-10)
                lateral1=math.floor(hi/dx+1e-10)
                longitudinal0=math.floor(start/dx+1e-10)
                longitudinal1=math.ceil(end/dx-1e-10)
                width=max(0,lateral1-lateral0)
                length=longitudinal1-longitudinal0
                entry=dict(building_ids=[buildings[i]['id'],buildings[j]['id']],
                           bounds_m=bounds,flow_axis='z' if axis==0 else 'x',
                           physical_width_m=float(hi-lo),grid_min_width_m=None,
                           connected=None,status='not_evaluated')
                result['channels'].append(entry)
                if width*length>cell_budget:
                    result['complete']=False
                    continue
                cell_budget-=width*length
                mask=structures.buildings if axis==0 else structures.buildings.T
                free=~mask[longitudinal0:longitudinal1,lateral0:lateral1]
                min_width=float(free.sum(axis=1).min()*dx) if width else 0.
                connected=False
                if width and free.shape[0]:
                    seen=np.zeros_like(free)
                    q=deque((0,int(x)) for x in np.flatnonzero(free[0]))
                    seen[0]=free[0]
                    while q:
                        z,x=q.popleft()
                        if z==len(free)-1:
                            connected=True
                            break
                        for zz,xx in ((z-1,x),(z+1,x),(z,x-1),(z,x+1)):
                            if 0<=zz<free.shape[0] and 0<=xx<free.shape[1] and free[zz,xx] and not seen[zz,xx]:
                                seen[zz,xx]=True;q.append((zz,xx))
                loss=max(0.,1-min_width/(hi-lo))
                entry.update(grid_min_width_m=min_width,connected=connected,width_loss_fraction=loss,
                             status='closed' if not connected else 'narrowed' if loss>=.2-1e-9 else 'open')
    return result
