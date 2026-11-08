from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
OUT=Path(__file__).resolve().parent
V14=OUT.parent/"uav-service-search-v14"
sys.path.insert(0,str(V14))
from service14 import covariates,Switcher,DIRECTIONS,endpoints,candidate_footprints,pool,smooth
from bound_search import Cohorts

# Rotate each manoeuvre through all eight physically allowed headings.
ANGLE=np.argsort(np.arctan2(DIRECTIONS[:,1],DIRECTIONS[:,0]))
TURNS=np.array([[0,0,0,0,0,0],[0,0,0,1,1,1],[0,0,0,-1,-1,-1],
 [0,0,2,2,2,2],[0,0,-2,-2,-2,-2],[0,0,2,4,4,4],[0,0,-2,-4,-4,-4],
 [0,1,0,1,0,1],[0,-1,0,-1,0,-1],[0,0,1,2,2,3],[0,0,-1,-2,-2,-3]])
ACTIONS=ANGLE[(TURNS[None,:,:]+np.arange(8)[:,None,None])%8].reshape(-1,6)
OFFSET=(DIRECTIONS[ACTIONS]*50).cumsum(1)
# Reuse identical waypoints among routes; keep heading as part of each viewed position.
all_keys=np.c_[np.round(OFFSET.reshape(-1,2),7),ACTIONS.ravel()]
_,FIRST,INDEX=np.unique(all_keys,axis=0,return_index=True,return_inverse=True)
INDEX=INDEX.reshape(ACTIONS.shape)
POINTS=OFFSET.reshape(-1,2)[FIRST]
HEAD_IDS=ACTIONS.ravel()[FIRST]
HEADS=DIRECTIONS[HEAD_IDS]
HORIZON=6
FIXED=np.array([.45,.30,1.0,.8,0.,.7])

def path_inputs(o,free,mass):
    poses=o.pos[free,None,:]+POINTS
    ok=((poses>=-1e-7)&(poses<=1000+1e-7)).all(-1)
    valid=ok[:,INDEX].all(-1)
    safe=np.clip(poses,0,1000)
    features=covariates(o,safe,HEADS)
    ids,visible=candidate_footprints(safe,HEADS)
    unseen=(mass.ravel()[ids]*visible).sum(-1)/8
    # One spatial field per waypoint. Time sequence and displacements are retained for the learned scorer.
    point=np.concatenate([features,unseen[...,None]],axis=-1).astype("f4")
    return point,valid,safe

def heuristic(point):
    # Expected usable new views, with partial credit for revisiting a sector swept long ago.
    return .25*point[...,0]+point[...,1]+.4*point[...,2]+.2*point[...,4]+.8*point[...,-1]

def compress(point):
    traj=point[:,INDEX]
    # Preserve features at early/middle/late portions of the route rather than collapsing to one heading.
    early=traj[:,:,:2].mean(2);middle=traj[:,:,2:4].mean(2);late=traj[:,:,4:].mean(2)
    return np.concatenate((early,middle,late,traj.std(2)[:,:,[1,2,5,9,18,19,20,25]]),-1)

def predicted(point,weights):
    x=compress(point)
    z=np.clip((x-weights["mean"])/weights["std"],-5,5)
    return np.tanh(z@weights["w"]+weights["b"])@weights["out"]

class FleetPaths(Switcher):
    def __init__(self,theta,weights=None,scope="pressure",joint=True):
        super().__init__("persist")
        self.fresh_search=True
        self.theta=np.asarray(theta);self.weights=weights;self.scope=scope;self.joint=joint
        self.belief=Cohorts(8);self.belief.personal_window_seconds=720
        self.cache={};self.replans=0;self.flown=0
    def act(self,o):
        self.belief.observe(o)
        return super().act(o)
    def search(self,o,ends,length,assigned,targets):
        raw=super().search(o,ends,length,assigned,targets)
        if self.scope=="pressure":return raw
        return np.eye(8)[self.team(o,assigned,targets,raw.argmax(1))]
    def pressure_search(self,o,ends,length,assigned,targets):
        # Keep the historical baselines' observation-history update and obtain their first-action reference.
        masked=SimpleNamespace(**{**vars(o),"active":np.zeros_like(o.active)})
        base,_=self.legacy.act(masked)
        return self.team(o,assigned,targets,base)
    def team(self,o,assigned,targets,base):
        free=np.flatnonzero(assigned<0)
        for i in np.flatnonzero(assigned>=0):self.cache.pop(i,None)
        if not len(free):return base
        ready=all(i in self.cache and self.cache[i][0]==o.t and np.linalg.norm(self.cache[i][1]-o.pos[i])<.01 and self.cache[i][2] for i in free)
        if not ready:
            mass=self.belief.mass(o.t,1.)
            point,valid,pos=path_inputs(o,free,mass)
            geometric=heuristic(point)[:,INDEX].mean(-1)
            def standardize(v):
                count=np.maximum(1,valid.sum(1,keepdims=True))
                center=np.where(valid,v,0).sum(1,keepdims=True)/count
                spread=np.sqrt(np.where(valid,(v-center)**2,0).sum(1,keepdims=True)/count)
                return np.clip((v-center)/np.maximum(.12,spread),-4,4)
            local=standardize(geometric)
            learned=standardize(predicted(point,self.weights)) if self.weights is not None else local
            neural,anchor,repel,discount,tilt,commit=self.theta
            score=(1-neural)*local+neural*learned
            score+=anchor*(DIRECTIONS[ACTIONS[:,0]]@DIRECTIONS[base[free]].T).T
            # Domain-exit urgency from measured flow only.
            axis=self.belief.axis
            score+=tilt*(OFFSET[:,-1]@axis)/300*max(0,1-o.t/1440)
            score+=discount*standardize(heuristic(point)[:,INDEX[:,:2]].mean(-1))
            for i in np.flatnonzero(assigned>=0):
                proximity=np.linalg.norm(pos[:,INDEX[:,3]]-targets[i],axis=-1)
                score-=repel*.45*np.maximum(0,1-proximity/110)
            score[~valid]=-1.e6
            choices=[];utility=[]
            for k in range(len(free)):
                # Keep distinct initial headings plus the highest-scoring full paths.
                ordered=np.argsort(-score[k],kind="stable")
                best=list(ordered[:5])
                for h in range(8):
                    group=ordered[ACTIONS[ordered,0]==h]
                    if len(group) and valid[k,group[0]]:best.append(group[0])
                take=np.unique(best);choices.append(take);utility.append(score[k,take])
            grid=np.ix_(*[np.arange(len(x)) for x in choices])
            summed=sum(u[g] for u,g in zip(utility,grid))
            if self.joint:
                for a in range(len(free)):
                    for b in range(a):
                        pa=pos[a,INDEX[choices[a]]];pb=pos[b,INDEX[choices[b]]]
                        # Charge spatial overlap on a 30 s horizon (not just final waypoint).
                        delta=pa[:,None,:,None,:]-pb[None,:,None,:,:]
                        sep=np.sqrt((delta*delta).sum(-1))
                        close=(np.maximum(0,1-sep.min(3)/100).mean(2)+np.maximum(0,1-sep.min(2)/100).mean(2))/2
                        summed-=repel*close[grid[a],grid[b]]
            pick=np.unravel_index(np.argmax(summed),summed.shape)
            for k,i in enumerate(free):
                selected=choices[k][pick[k]]
                seq=list(map(int,ACTIONS[selected,:max(1,min(6,round(1+5*commit)))])) if valid[k,selected] else [int(base[i])]
                self.cache[i]=(o.t,o.pos[i].copy(),seq)
            self.replans+=1
        result=base.copy()
        for i in free:
            seq=self.cache[i][2];action=seq[0];result[i]=action
            self.cache[i]=(o.t+5,np.clip(o.pos[i]+DIRECTIONS[action]*50,0,1000),seq[1:])
            self.flown+=1
        return result
