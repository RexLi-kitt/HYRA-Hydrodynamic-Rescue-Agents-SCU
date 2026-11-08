"""Predict physically executable meetings with a moving recipient from one legal sighting."""
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/"uav-combined-close-v25"))
from combined import Combined,Routed,quota,read,save,sha,runtime,finished,load_scene,s
from beam16 import DIRECTIONS,material,NoveltyBeam
from lattice import bins,directions,route
from population import segment_circle,EPS
from local_route import endpoints
from windows import cloned_mission

DIR=np.array(DIRECTIONS)
HOLD=8
FOLLOW=17


def inside(x):
    return bool((x>=0).all() and (x<1000).all())


def sight(pos,head,x):
    delta=x-pos
    return bool(delta@delta<=400+1e-9 and delta@head>=-1e-9)


def projection(pos,head,x,v,visible,code):
    """All codes project the judge's five-second command; a true follow keeps the heading."""
    action=code%9
    if action==8:
        velocity=v*min(1.,10/max(1e-12,np.linalg.norm(v))) if code==FOLLOW and visible else np.zeros(2)
    else:
        velocity=10*DIR[action]
    length=5.
    if code>=9 and action<8 and visible and np.linalg.norm(x-pos)>.5+EPS:
        hit=segment_circle(pos,5*velocity,x,.5)
        if hit is not None:
            length=min(length,5*hit)
    positive=velocity>0;negative=velocity<0
    if positive.any():
        length=min(length,float(((1000-pos[positive])/velocity[positive]).min()))
    if negative.any():
        length=min(length,float((-pos[negative]/velocity[negative]).min()))
    end=np.clip(pos+max(0,length)*velocity,0,1000)
    delta=end-pos;dist=np.linalg.norm(delta)
    return end,delta/dist if action!=8 and dist>1e-8 else head


@lru_cache(maxsize=1)
def catalogue():
    points=[];counts=[];steps=[]
    for (n,_,_),values in bins().items():
        for disp,count in values:
            points.append(disp);counts.append(count);steps.append(n)
    return np.asarray(points),counts,np.asarray(steps)


def approach_route(p,h,x,stage,count,time,initial_view):
    """Reach the stage with a heading that sees x. Delays consume real steps."""
    direct=directions(count);n=len(direct);ray=x-stage
    if ray@ray>400+1e-9 or time<n:
        return None
    if not direct and ((time==0 and initial_view) or ray@h>=-1e-9):
        return [HOLD]*time
    for last in sorted(set(direct)):
        if ray@DIR[last]<-1e-9:
            continue
        trial=route(p,direct,last)
        if trial is not None:
            return [a for a,_ in trial]+[HOLD]*(time-n)
    if time<n+2:
        return None
    prefix=[] if not n else None
    for last in sorted(set(direct)):
        if prefix is not None:
            break
        trial=route(p,direct,last)
        if trial is not None:
            prefix=[a for a,_ in trial]
    if prefix is None:
        return None
    for last,heading in enumerate(DIR):
        turn=stage-50*heading
        if ray@heading>=-1e-9 and (turn>=0).all() and (turn<=1000).all():
            opposite=int(np.argmin(DIR@heading))
            return prefix+[opposite,last]+[HOLD]*(time-n-2)
    return None


def terminal(p,h,x,v,vis,limit):
    """Enumerate moving-frame intersections in time order; no time step is rounded away."""
    if limit<0:
        return None
    ds,counts,ns=catalogue()
    # Only displacements within one footprint of the target's 50-second path can matter.
    radius=np.linalg.norm(x-p)+5*limit*np.linalg.norm(v)+20.000001
    candidates=np.flatnonzero((ns<=limit)&((ds*ds).sum(1)<=radius**2))
    ends=p+ds[candidates]
    valid=((ends>=0)&(ends<=1000)).all(1)
    candidates=candidates[valid];ends=ends[valid];n=ns[candidates]
    best=None
    for k in range(limit+1):
        if best is not None and len(best)<=k:
            break
        y=x+5*k*v
        if not inside(y):
            break
        ray=y-ends
        r2=np.einsum("ij,ij->i",ray,ray)
        for z in np.flatnonzero((n<=k)&(r2<=400+1e-9)):
            stage=ends[z];d=ray[z]
            finishing=None
            if r2[z]<=1+1e-9 and inside(y+10*v) and ((stage+10*v>=0)&(stage+10*v<=1000)).all():
                finishing=[]
            elif k<limit and r2[z]>(.5+EPS)**2:
                along=DIR@d;perp=np.maximum(0.,r2[z]-along*along)
                for a in np.flatnonzero((along>=0)&(perp<=.25+1e-12)):
                    last=stage+max(0,float(along[a]-np.sqrt(max(0,.25-perp[a]))))*DIR[a]
                    if not ((last>=0)&(last<=1000)&(last+10*v>=0)&(last+10*v<=1000)).all():
                        continue
                    for wait in range(limit-k):
                        if finishing is not None and 1+wait>=len(finishing):
                            break
                        contact=y+5*(1+wait)*v
                        g=contact-last
                        if g@g<=1+1e-9 and g@DIR[a]>=-1e-9 and inside(contact) and inside(contact+10*v):
                            finishing=[9+int(a)]+[HOLD]*wait
                            break
            if finishing is None or best is not None and k+len(finishing)>=len(best):
                continue
            leg=approach_route(p,h,y,stage,counts[candidates[z]],k,vis)
            if leg is not None:
                best=leg+finishing
    return best


def fractional_starts(p,h,x,v,vis):
    states=[(p,h,[],vis)];used=set()
    for k in range(3):
        nxt={};y=x+5*k*v
        if not inside(y):
            break
        for loc,heading,prefix,observed in states:
            if observed and np.linalg.norm(loc-y)>.5+EPS:
                for a in range(8):
                    ray=segment_circle(loc,50*DIR[a],y,.5)
                    if ray is None:
                        continue
                    end,last=projection(loc,heading,y,v,True,a+9)
                    key=(k,*np.round(end,8),*np.round(last,8))
                    if key not in used and np.linalg.norm(end-loc)>1e-8:
                        used.add(key)
                        yield end,last,prefix+[a+9]
            if k<2:
                for a in range(9):
                    end,last=projection(loc,heading,y,v,observed,a)
                    # A first segment is allowed to stop at the domain boundary.
                    key=(*np.round(end,8),*np.round(last,8))
                    nxt.setdefault(key,(end,last,prefix+[a],sight(end,last,y+5*v)))
        states=nxt.values()


def solve(p,h,x,v,vis,limit,extended):
    base=terminal(p,h,x,v,vis,limit)
    best=base;fractional=False
    if extended:
        for loc,heading,prefix in fractional_starts(p,h,x,v,vis):
            n=len(prefix);cap=limit-n if best is None else min(limit,len(best)-1)-n
            if cap<0:
                continue
            y=x+5*n*v
            after=terminal(loc,heading,y,v,sight(loc,heading,y),cap)
            if after is not None:
                best=prefix+after;fractional=True
    return best,fractional


def prepare(p,h,x,v,vis,seq):
    """Replay every proposed action before accepting it, including both follow steps."""
    poses=[]
    for k,code in enumerate([*seq,FOLLOW,FOLLOW]):
        loc=x+5*k*v
        if not inside(loc) or code>=9 and not vis:
            return None
        if k>=len(seq) and (np.linalg.norm(loc-p)>1+1e-9 or not inside(loc+5*v)):
            return None
        poses.append((p.copy(),h.copy(),loc.copy(),int(code)))
        end,heading=projection(p,h,loc,v,vis,code)
        if k>=len(seq) and np.linalg.norm(loc+5*v-end)>1+1e-9:
            return None
        p,h=end,heading;vis=sight(p,h,loc+5*v)
    return poses
