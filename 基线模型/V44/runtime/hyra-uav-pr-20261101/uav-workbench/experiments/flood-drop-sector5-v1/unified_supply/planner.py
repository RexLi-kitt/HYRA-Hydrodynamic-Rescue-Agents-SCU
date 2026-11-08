"""A shared search planner, with an optional sensor-trained 4-8-1 occupancy model."""
from types import SimpleNamespace
import numpy as np
from population import MOVES
from supply import OFFSETS, EPS, assign

AXIS=np.arange(20)*50+25
CENTERS=np.stack(np.meshgrid(AXIS,AXIS),axis=-1)
# Bin local sensor values so the network can be fitted on counts rather than hidden maps.
H=np.array([0,.005,.01,.025,.05,.1,.2,.4,.8,1.5,3.])
V=np.array([0,.01,.03,.06,.1,.2,.35,.5,.75,1.,1.5,2.5])
def bins(h,v):
    return (np.searchsorted(H[1:],h,side="right")*len(V)+np.searchsorted(V[1:],v,side="right"))
def features():
    h,v=np.meshgrid(H,V,indexing="ij")
    return np.column_stack((np.log1p(5*h.ravel()),np.log1p(4*v.ravel()),(h.ravel()>=.01).astype(float),
                            np.log1p(5*h.ravel()*v.ravel())))

def record(obs,stats):
    ids=obs.new_cells
    if not len(ids):return
    h,u,v=obs.measured.reshape(3,-1)[:,ids]
    b=bins(h,np.hypot(u,v))
    stats[0]+=np.bincount(b,minlength=stats.shape[1])
    coords=np.floor(obs.seen_xy[obs.visible]/10).astype(int).clip(0,99)
    person_cells=100*coords[:,1]+coords[:,0]
    at=np.searchsorted(ids,person_cells)
    eligible=(at<len(ids))&(ids[at.clip(0,len(ids)-1)]==person_cells)
    stats[1]+=np.bincount(b[at[eligible]],minlength=stats.shape[1])

def intensity(theta):
    x=features()
    hidden=np.tanh(x@theta[:32].reshape(4,8)+theta[32:40])
    return np.exp(hidden@theta[40:48]+theta[48])

def train(stats,seed):
    samples,events=stats
    if events.sum()==0:raise ValueError("No observed positive labels")
    rng=np.random.default_rng(seed)
    x=features()
    theta=np.r_[rng.normal(0,.12,32),np.zeros(8),rng.normal(0,.08,8),
                np.log(events.sum()/samples.sum())]
    first=np.zeros_like(theta);second=np.zeros_like(theta)
    trace=[]
    for step in range(1,401):
        hidden=np.tanh(x@theta[:32].reshape(4,8)+theta[32:40])
        rate=intensity(theta)
        error=(samples*rate-events)/events.sum()
        delta=(error[:,None]*theta[40:48])*(1-hidden**2)
        grad=np.r_[(x.T@delta).ravel(),delta.sum(0),hidden.T@error,error.sum()]
        grad[:-1]+=.003*theta[:-1]
        first=.9*first+.1*grad
        second=.999*second+.001*grad**2
        theta-=.018*(first/(1-.9**step))/(np.sqrt(second/(1-.999**step))+1e-8)
        if step==1 or step%50==0:
            loss=(samples*rate-events*np.log(rate)).sum()/events.sum()+.0015*(theta[:-1]@theta[:-1])
            trace.append([step,float(loss)])
    if not np.isfinite(theta).all():raise ValueError("Nonfinite model")
    return theta,trace

def pool(a):
    return a.reshape(20,5,20,5).mean(axis=(1,3))
def smooth(a):
    p=np.pad(a,1,mode='edge')
    return sum(p[y:y+20,x:x+20]*wy*wx for y,wy in enumerate((.25,.5,.25))
               for x,wx in enumerate((.25,.5,.25)))

class Planner:
    def __init__(self,method,radius,theta=None):
        self.method=method
        self.radius=radius
        self.lookup=intensity(theta) if theta is not None else None
        self.cell=np.zeros(10000)
        self.last_block=None
    def act(self,o):
        pos=o.pos
        ends=np.clip(pos[:,None]+50*MOVES[:8],0,1000)
        travel=ends-pos[:,None]
        length=np.linalg.norm(travel,axis=-1)
        head=travel/np.maximum(length[:,:,None],1e-12)
        idx=np.floor(ends/10).astype(int)[:,:,None,:]+OFFSETS
        rel=idx*10+5-ends[:,:,None]
        view=((idx>=0)&(idx<100)).all(-1)&((rel*rel).sum(-1)<=400+1e-9)&((rel*head[:,:,None]).sum(-1)>=-1e-9)
        col,row=np.moveaxis(idx.clip(0,99),-1,0)
        last=o.last[row,col]
        gain=np.where(last>=0,np.clip((o.t-last)/660.,0,1),1)
        score=(gain*view).sum(-1)+.5*(head*o.heading[:,None]).sum(-1)
        near=np.linalg.norm(ends[:,:,None]-pos[None,None,:],axis=-1)
        for i in range(4):near[i,:,i]=np.inf
        score-=np.clip((100-near.min(-1))/25,0,4)
        score[length<50-1e-6]=-100
        if self.method!='endpoint':
            ids=o.observed_cells
            h,u,v=o.measured.reshape(3,-1)[:,ids]
            value=(h>=.01).astype(float) if self.lookup is None else self.lookup[bins(h,np.hypot(u,v))]
            self.cell[ids]=value
            known=o.last>=0
            cov=pool(known)
            qty=pool(self.cell.reshape(100,100))
            prior=(self.cell.sum()+(.5 if self.lookup is None else self.lookup.mean()))/(known.sum()+1)
            density=(qty+.6*smooth(qty)+.06*prior)/(cov+.6*smooth(cov)+.06)
            if self.lookup is None:density+=.12
            if density.max()>1.35*density.min():
                age=np.where(known,np.clip((o.t-o.last)/660.,0,1),1.)
                mass=density*pool(age)
                reserved=[]
                for i,p in enumerate(pos):
                    dist=np.linalg.norm(CENTERS-p,axis=-1)
                    utility=mass/(1+dist/125.)
                    for goal in reserved:
                        utility*=.2+.8*np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/125.,0,1)
                    goal=CENTERS[np.unravel_index(utility.argmax(),utility.shape)]
                    reserved.append(goal)
                    before=np.linalg.norm(goal-p)
                    if before>=60:score[i]+=4.5*(before-np.linalg.norm(ends[i]-goal,axis=-1))/50.
        chosen=score.argmax(axis=1)
        assigned,goals=assign(o,self.radius)
        for i,j in enumerate(assigned):
            if j<0:continue
            d=goals[i]-pos[i]
            if o.visible[j] and np.linalg.norm(d)<=self.radius+EPS:
                chosen[i]=8
                continue
            # Follow a legal ray. Prefer one that actually intercepts the observed
            # target's approach disk this step, then its earliest interception.
            along=MOVES[:8]@d
            rem=d-along[:,None]*MOVES[:8]
            side=np.linalg.norm(rem,axis=1)
            inner=self.radius*.5
            hit=(side<=inner+EPS)&(along>0)&(along-np.sqrt(np.maximum(0,inner**2-side**2))<=length[i]+EPS)
            if hit.any():
                chosen[i]=np.argmin(np.where(hit,along-np.sqrt(np.maximum(0,inner**2-side**2)),np.inf))
            else:
                chosen[i]=np.argmin(np.linalg.norm(ends[i]-goals[i],axis=1))
        return chosen,assigned

