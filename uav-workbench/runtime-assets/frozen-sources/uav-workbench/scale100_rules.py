"""Water-aware baseline using arrived records only, with team conflict avoidance."""
from collections import deque
from itertools import product
import math
import numpy as np
from sur_rescue.environment import MOVES,joint_motion

def water_aware(env):
    view=env.view();actions=[];reserved=set();positions=set(view.positions);nz,nx=view.explored.shape
    for agent,start in enumerate(view.positions):
        other=positions-{start};todo=deque([start]);paths={start:(0,None)}
        while todo:
            r,c=todo.popleft();distance,first=paths[(r,c)]
            for a,(dz,dx) in enumerate(MOVES):
                p=(r+dz,c+dx)
                if p in paths or p in other or p in reserved or not 0<=p[0]<nz or not 0<=p[1]<nx or view.obstacles[p]:continue
                if dz and dx and(view.obstacles[r,p[1]] or view.obstacles[p[0],c]):continue
                paths[p]=(distance+1,a if first is None else first);todo.append(p)
        candidates=[]
        for r,c,status in view.targets:
            p=(r,c)
            if status!='discovered' or p not in paths or paths[p][1] is None:continue
            h,u,v=env.water[:,r,c];age=env.time_s-env.observed_s[p]
            safe=h<=env.config.service_depth_limit_m and math.hypot(u,v)<=env.config.service_speed_limit_m_s
            if not safe and age<30:continue
            distances,first=paths[p];history=[];cell=r*nx+c
            for obs in reversed(env.observations):
                if cell in obs['cells'] and (not history or obs['observed_s']!=history[-1][0]):
                    history.append((obs['observed_s'],obs['depth_m'][obs['cells'].index(cell)]))
                    if len(history)==2:break
            rate=max(0.,(history[0][1]-history[1][1])/(history[0][0]-history[1][0])) if len(history)==2 else 0.
            remaining=(env.config.service_depth_limit_m-h)/rate if rate>1e-6 else float('inf')
            slack=remaining-distances*env.config.step_s-env.config.service_s
            # Prioritize reachable, currently serviceable targets with a shrinking observed window.
            candidates.append(((not safe,slack<0,slack if math.isfinite(slack) else 1e9,distances),first))
        if candidates:a=min(candidates)[1]
        else:
            unseen=[(d,p,first) for p,(d,first) in paths.items() if first is not None and not view.explored[p]]
            if unseen:a=min(unseen)[2]
            else:
                refresh=[((env.time_s-env.observed_s[p])/(1+d),-d,p,first) for p,(d,first) in paths.items() if first is not None and view.explored[p]]
                if refresh:a=max(refresh)[3]
                else:
                    legal=np.flatnonzero(view.available[agent]);a=int(legal[0]) if len(legal) else 0
        actions.append(a);dz,dx=MOVES[a];reserved.add((start[0]+dz,start[1]+dx))
    # Endpoint reservations alone do not prevent diagonal segment crossings.
    busy=[bool(b) for b in view.busy]
    if joint_motion(view.positions,actions,view.obstacles,busy)[2]:
        choices=[np.flatnonzero(mask).tolist() or [0] for mask in view.available];best=None
        for candidate in product(*choices):
            end,reasons,conflicts=joint_motion(view.positions,candidate,view.obstacles,busy)
            if conflicts:continue
            score=sum(a!=b for a,b in zip(actions,candidate))
            if best is None or score<best[0]:best=(score,candidate)
        if best is not None:actions=list(best[1])
    return actions
