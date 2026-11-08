from pathlib import Path
from types import SimpleNamespace
import os,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
V13=ROOT.parent/"uav-strong-control-v13"
sys.path.insert(0,str(V13))
os.environ["HYRA_STUDY_RADIUS"]="1"
from trial import Switcher
from support import runtime,s,read,save,sha,quota
from lattice import prepare_plan
from local_route import endpoints
from approach_policy import DIRECTIONS,candidate_footprints
from planner import pool,smooth
from population import segment_circle

def covariates(o,poses,head):
    shape=poses.shape[:-1]
    p=poses.reshape(-1,2);hd=np.broadcast_to(head,poses.shape).reshape(-1,2)
    idx,mask=candidate_footprints(p,hd)
    raw=o.last.ravel()[idx];seen=(raw>=0)&mask;fresh=mask&~seen
    xy=np.stack((idx%100,idx//100),axis=-1)*10+5
    offset=xy-p[:,None,:]
    projections=offset@DIRECTIONS.T
    perp2=np.maximum(0,np.sum(offset**2,-1)[...,None]-projections**2)
    ray=((perp2<=.25+1e-9)&(projections>=0)).any(-1) | (np.sum(offset**2,-1)<=1)
    short=(fresh&ray).sum(-1)
    coverage=pool(o.last>=0)
    block=(p//50).astype(int).clip(0,19);ix=(block[:,1],block[:,0])
    h,u,v=o.measured
    current=o.observed_cells
    flow=np.median(o.measured.reshape(3,-1)[:,current],axis=1) if len(current) else np.zeros(3)
    axis=flow[1:]/max(.05,np.linalg.norm(flow[1:]))
    velocity=np.hypot(u,v)
    def near(field):
        return (smooth(pool(np.where(o.last>=0,field,0)))/(smooth(coverage)+.02))[ix]
    phase=(p-5)*2*np.pi/10
    local=np.zeros((20,20))
    people=np.flatnonzero(o.seen_at>=0)
    if len(people):
        block2=(o.seen_xy[people]//50).astype(int).clip(0,19)
        np.add.at(local,(block2[:,1],block2[:,0]),1.)
    full=lambda x:np.full(len(p),x)
    x=np.stack([mask.sum(-1)/10,fresh.sum(-1)/10,short/6,(mask&ray).sum(-1)/6,
        np.where(seen,np.minimum(1,(o.t-raw)/400.),0.).sum(-1)/10,coverage[ix],
        near(np.log1p(np.maximum(h,0))),near(np.log1p(velocity)),near(h>=.01),
        np.log1p(smooth(local)[ix]*4),np.log1p(smooth(smooth(local))[ix]*8),
        np.cos(phase[:,0]),np.cos(phase[:,1]),np.sin(phase[:,0]),np.sin(phase[:,1]),
        np.cos(phase[:,0])*np.cos(phase[:,1]),hd[:,0]*hd[:,1],abs(hd[:,0]*hd[:,1]),
        (p-500)@axis/500,hd@axis,np.minimum(p,1000-p).min(-1)/500,
        full(o.t/2400),full(np.linalg.norm(flow[1:])),full(flow[0]/3),full(len(people)/64)],-1)
    return x.reshape(*shape,-1)

def in_view(pos,h,xy):
    delta=xy-pos
    return delta@delta<=400+1e-9 and delta@h>=-1e-9

def short_service(p,j,t,pos,heading,deadline):
    # Simulate a feasible isolated approach against the recorded path. Do not grant an
    # in-step correction or any access to future truth to the route chooser.
    start=t;pos=pos.copy();h=heading.copy();route=None;progress=0
    while t+5<=min(start+40,deadline,2400):
        if not p["available"][t,j]:return 0.
        xy=p["points"][t,j]
        visible=in_view(pos,h,xy)
        if visible and np.linalg.norm(xy-pos)<=1+1e-9:
            v=p["velocity"][t//5,j]
            track=pos+np.arange(6)[:,None]*v
            if ((track<0)|(track>1000)).any():return 0.
            if not p["available"][t:t+6,j].all() or np.max(np.linalg.norm(p["points"][t:t+6,j]-track,axis=1))>1+1e-9:return 0.
            pos=track[-1];progress+=5;t+=5
            if progress==10:return float(np.exp(-(t-start-10)/28.))
            continue
        if progress:return 0.
        if route is None:
            if not visible:return 0.
            route=prepare_plan(pos,h,xy,p["velocity"][t//5,j],True,6)
        if not route:return 0.
        a,transit=route.pop(0)
        if a!=8:
            move=DIRECTIONS[a]*50
            fraction=1.
            if not transit and visible and np.linalg.norm(xy-pos)>.5+1e-7:
                hit=segment_circle(pos,move,xy,.5)
                if hit is not None:fraction=hit
            new_pos=pos+fraction*move
            if ((new_pos<-1e-7)|(new_pos>1000+1e-7)).any():return 0.
            if fraction>1e-9:h=DIRECTIONS[a]
            pos=new_pos
        t+=5
    return 0.

MODES={"opportunity":(.65,"service",False),"assertive":(1.5,"service",False),
       "stress":(1.0,"service",True),"discovery":(1.0,"discovery",False),
       "geometry":(1.0,"geometry",False),"geometry_stress":(1.0,"geometry",True)}
class Search(Switcher):
    def __init__(self,mode,weights=None):
        super().__init__("persist")
        self.strength,self.signal,self.stress_only=MODES[mode]
        self.model=weights;self.fresh_search=True
        self.choices=0;self.changes=0
    def rank(self,o,ends,length,assigned,base,active):
        nums=np.flatnonzero(assigned<0)
        if not len(nums) or (self.stress_only and not active):return base
        x=covariates(o,ends[nums],DIRECTIONS)
        if self.signal=="geometry":scores=1.3*x[:,:,2]+.65*x[:,:,1]+.25*x[:,:,4]
        else:
            w=self.model
            z=np.clip((x-w["mean"])/w["std"],-4,4)
            scores=(np.tanh(z@w["w"]+w["b"])@w["out"]+w["bias"])[...,0 if self.signal=="discovery" else 1]
        scores=np.clip(scores-scores.max(1,keepdims=True),-5.,0.)*self.strength
        result=base.copy();booked=[]
        for k,i in enumerate(nums):
            heading=base[i]
            if heading==8:continue
            ranked=scores[k]+.5*(DIRECTIONS@DIRECTIONS[heading])
            ranked[heading]+=.75
            for prev in booked:
                ranked-=.8*np.clip(1-np.linalg.norm(ends[i]-prev,axis=1)/100,0,1)
            ranked[length[i]<1]=-1e9
            result[i]=int(ranked.argmax());booked.append(ends[i,result[i]])
        self.choices+=len(nums);self.changes+=int((result[nums]!=base[nums]).sum())
        return result
    def search(self,o,ends,length,assigned,targets):
        raw=super().search(o,ends,length,assigned,targets)
        if self.stress_only:return raw
        return np.eye(8)[self.rank(o,ends,length,assigned,raw.argmax(1),False)]
    def pressure_search(self,o,ends,length,assigned,targets):
        proxy=SimpleNamespace(**{**vars(o),"active":np.zeros_like(o.active)})
        actions,_=self.legacy.act(proxy)
        return self.rank(o,ends,length,assigned,actions,True)
