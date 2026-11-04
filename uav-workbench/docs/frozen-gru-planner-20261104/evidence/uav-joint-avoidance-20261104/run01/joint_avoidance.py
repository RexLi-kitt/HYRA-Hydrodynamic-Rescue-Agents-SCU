"""Public-state local collision resolution above unchanged physical safety checks."""
import itertools
import numpy as np
from train import FineEnv,FlowFogEnv
from long_environment import MOVES,first_hit
JOINT4=np.array(list(itertools.product(range(9),repeat=4)),dtype=np.int8)

def swept_distances(start,ends):
    out=[]
    for i in range(len(start)):
        for j in range(i+1,len(start)):
            x=start[i]-start[j];d=(ends[i]-start[i])-(ends[j]-start[j])
            f=np.clip(-x@d/max(float(d@d),1e-12),0,1)
            out.append(float(np.linalg.norm(x+f*d)))
    return np.array(out)

def resolve_joint(start,ends,desired,scores,separation):
    """4 UAVs x9 existing actions; exhaustive local legal joint-action selection."""
    start=np.asarray(start);ends=np.asarray(ends);desired=np.asarray(desired)
    if start.shape!=(4,2) or ends.shape!=(4,9,2) or scores.shape!=(4,9):raise ValueError('Isolated4UAVresearchcontroller')
    if not np.isfinite(start).all() or not np.isfinite(ends).all() or not np.isfinite(scores).all():raise ValueError('Nonfinitegeometry')
    original=ends[np.arange(4),desired]
    if swept_distances(start,original).min()>=separation-1e-8:return desired.copy(),False
    valid=np.ones(len(JOINT4),bool)
    for i in range(4):
        for j in range(i+1,4):
            x=start[i]-start[j]
            d=(ends[i]-start[i])[:,None,:]-(ends[j]-start[j])[None,:,:]
            f=np.clip(-np.sum(x*d,axis=-1)/np.maximum(np.sum(d*d,axis=-1),1e-12),0,1)
            distances=np.linalg.norm(x+f[...,None]*d,axis=-1)
            valid &= distances[JOINT4[:,i],JOINT4[:,j]]>=separation-1e-8
    if not valid.any():return desired.copy(),False # unchangedenginewillstillstopunsafeactions
    values=scores[np.arange(4)[None,:],JOINT4].sum(1)
    values[~valid]=-np.inf
    chosen=JOINT4[int(np.argmax(values))].astype(int)
    assert swept_distances(start,ends[np.arange(4),chosen]).min()>=separation-1e-8
    return chosen,True

class SafeControl:
    def control(self):
        mask,desired,assigned=super().control()
        if self._orders is None:
            _,expert,options,_=self.choices();orders=[options[i][expert[i]] for i in range(4)]
        else:orders=self._orders
        dt=min(self.cfg.step,self.cfg.duration-self.t)
        if dt<=0:return mask,desired,assigned
        start=self.pos
        ends=np.clip(start[:,None,:]+MOVES[None,:,:]*self.cfg.speed*dt,0,1000)
        scores=np.zeros((4,9))
        for i,j in enumerate(assigned):
            # Replicate public-information endpoint calculations inEnv.step.
            if j>=0 and self.visible[j]:
                velocity=self.seen_v[j]
                ends[i,8]=np.clip(start[i]+velocity*min(dt,self.cfg.speed*dt/max(np.linalg.norm(velocity),1e-9)),0,1000)
                for a in range(8):
                    hit=first_hit(start[i],ends[i,a],self.seen_xy[j:j+1],self.cfg.drop_radius*.5)
                    if hit is not None:ends[i,a]=start[i]+(ends[i,a]-start[i])*hit
            delta=orders[i]['xy']-start[i];norm=np.linalg.norm(delta)
            direction=delta/max(float(norm),1e-12)
            scores[i]=(ends[i]-start[i])@direction
            scores[i,desired[i]]+=1e-7
            if j>=0 and self.visible[j] and np.linalg.norm(self.seen_xy[j]-start[i])<=self.cfg.drop_radius+1e-8:
                scores[i,8]+=10000. # preserveongoingservicewhensafe; safetyremainsmandatory
        chosen,changed=resolve_joint(start,ends,desired,scores,self.cfg.separation)
        if changed:
            if not hasattr(self,'avoidance_events'):self.avoidance_events=[]
            if not self.avoidance_events or self.avoidance_events[-1]['time']!=self.t:
                self.avoidance_events.append(dict(time=int(self.t),desired=desired.tolist(),chosen=chosen.tolist(),min_clearance=float(swept_distances(start,ends[np.arange(4),chosen]).min())))
            mask=np.zeros_like(mask);mask[np.arange(4),chosen]=True
        return mask,chosen,assigned

class SafeFineEnv(SafeControl,FineEnv):pass
class SafeFlowEnv(SafeControl,FlowFogEnv):pass
