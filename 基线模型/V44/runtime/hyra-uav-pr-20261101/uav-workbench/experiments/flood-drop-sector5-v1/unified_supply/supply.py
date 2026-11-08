"""Common physical simulator and a restricted sensor interface for the paired pilot."""
from itertools import combinations
from types import SimpleNamespace
import numpy as np
from population import START, COUNT, DURATION, EPS, MOVES, RULES, sector, segment_circle

OFFSETS = np.stack(np.meshgrid(np.arange(-2,3),np.arange(-2,3)),axis=-1).reshape(-1,2)
PAIRS = np.array(list(combinations(range(4),2))).T

def sample_cells(pos, headings):
    indices = np.floor(pos/10).astype(int)[:,None,:] + OFFSETS
    delta = indices*10+5-pos[:,None,:]
    seen = ((indices>=0)&(indices<100)).all(-1)
    seen &= ((delta*delta).sum(-1)<=400+1e-9) & ((delta*headings[:,None,:]).sum(-1)>=-1e-9)
    return np.unique(indices[:,:,1][seen]*100+indices[:,:,0][seen])

def raw_conflicts(track):
    a,b=PAIRS
    separation=track[:,a]-track[:,b]
    change=np.diff(separation,axis=0)
    fraction=np.clip(-(separation[:-1]*change).sum(-1)/np.maximum((change*change).sum(-1),1e-18),0,1)
    closest=separation[:-1]+fraction[:,:,None]*change
    return np.flatnonzero(((closest*closest).sum(-1)<(5-1e-8)**2).any(axis=0))

def intentions(pos, speed, seconds):
    """Straight rays with an exact stop time; include all stops in collision knots."""
    knots=np.unique(np.r_[np.arange(6),seconds])
    track=pos+np.minimum(knots[:,None],seconds)[...,None]*speed
    pairs=raw_conflicts(track)
    blocked=np.zeros(4,bool)
    conflicts=pairs
    while conflicts.size:
        before=int(blocked.sum())
        blocked[np.unique(PAIRS[:,conflicts])]=True
        if int(blocked.sum())==before:raise ValueError("Unsafe coincident starts")
        track[:,blocked]=pos[blocked]
        conflicts=raw_conflicts(track)
    return track[np.searchsorted(knots,np.arange(6))],blocked,len(pairs)

def align_service(drone_path,person_path,radius):
    offset=drone_path-person_path
    return (np.linalg.norm(offset[0])<=radius+EPS and
            np.linalg.norm(offset-offset[0],axis=1).max()<=EPS)

def assign(obs,radius):
    eligible=np.flatnonzero(obs.active & (obs.seen_at>=0) & (obs.t-obs.seen_at<=45))
    assigned=np.full(4,-1,int)
    goals=obs.pos.copy()
    if not len(eligible):return assigned,goals
    forecast=obs.seen_xy[eligible]+obs.seen_v[eligible]*np.minimum(15,obs.t-obs.seen_at[eligible])[:,None]
    forecast=np.clip(forecast,0,1000)
    distances=np.linalg.norm(obs.pos[:,None]-forecast,axis=2)
    claims=[]
    for k,j in enumerate(eligible):
        for i in range(4):
            claims.append((obs.owner[j]!=i,not (obs.visible[j] and distances[i,k]<=radius+EPS),
                           distances[i,k],int(j),i,k))
    used=set()
    for _,_,_,j,i,k in sorted(claims):
        if assigned[i]<0 and j not in used:
            assigned[i]=j
            goals[i]=forecast[k]
            used.add(j)
    return assigned,goals

class Supply:
    def __init__(self,water,population,radius):
        if radius not in (0,10):raise ValueError("Unregistered alignment")
        self.water=water
        self.truth=population
        self.radius=radius
        self.t=0
        self.pos=START.copy()
        self.heading=500-self.pos
        self.heading/=np.linalg.norm(self.heading,axis=1)[:,None]
        self.last=np.full((100,100),-1,int)
        self.measured=np.zeros((3,100,100))
        self.seen_xy=np.zeros((COUNT,2))
        self.seen_v=np.zeros((COUNT,2))
        self.seen_at=np.full(COUNT,-1,int)
        self.visible=np.zeros(COUNT,bool)
        self.active=np.zeros(COUNT,bool)
        self.owner=np.full(COUNT,-1,int)
        self.progress=np.zeros(COUNT,int)
        self.supplied=np.full(COUNT,-1,int)
        self.deadline=np.minimum(DURATION,population['fall']+RULES['personal_window_after_fall_s'])
        self.conflicts=0
        self.boundary=0
        self.sample()

    def valid(self,t):
        return self.truth['available'][t] & (self.supplied<0) & (t<=self.deadline)

    def sample(self):
        t=self.t
        ids=sample_cells(self.pos,self.heading)
        fresh=ids[self.last.ravel()[ids]<0]
        self.last.ravel()[ids]=t
        self.measured.reshape(3,-1)[:,ids]=self.water[t//5].reshape(3,-1)[:,ids]
        xy=self.truth['points'][t]
        view=np.logical_or.reduce([sector(xy,p,h,20.) for p,h in zip(self.pos,self.heading)])
        self.visible=view & self.valid(t) & (t<self.deadline)
        # A known target's expiry/out-of-domain status is not reported remotely.
        self.active[view & ~self.valid(t)]=False
        self.active[self.visible]=True
        self.active[self.supplied>=0]=False
        self.seen_at[self.visible]=t
        self.seen_xy[self.visible]=xy[self.visible]
        self.seen_v[self.visible]=self.truth['velocity'][t//5,self.visible]
        self.new_cells=fresh
        self.observed_cells=ids

    def observation(self):
        return SimpleNamespace(t=self.t,pos=self.pos.copy(),heading=self.heading.copy(),last=self.last,
                     measured=self.measured,seen_xy=self.seen_xy,seen_v=self.seen_v,
                     seen_at=self.seen_at,visible=self.visible,active=self.active,owner=self.owner,
                     new_cells=self.new_cells,observed_cells=self.observed_cells)

    def step(self,actions,assigned):
        if self.t>=DURATION:raise ValueError("Mission ended")
        actions=np.asarray(actions)
        if actions.shape!=(4,) or not np.issubdtype(actions.dtype,np.integer) or np.any((actions<0)|(actions>8)):
            raise ValueError("Invalid joint action")
        pos=self.pos
        speed=MOVES[actions].copy()*10
        duration=np.full(4,5.)
        for i,(j,a) in enumerate(zip(assigned,actions)):
            if j<0 or not self.visible[j]:continue
            if a==8:
                current=self.seen_v[j]
                speed[i]=current*min(1.,10/max(np.linalg.norm(current),1e-12))
            else:
                target=self.seen_xy[j]
                inner=self.radius*.5
                if np.linalg.norm(pos[i]-target)>inner+EPS:
                    hit=segment_circle(pos[i],speed[i]*5,target,inner)
                    if hit is not None:duration[i]=min(duration[i],hit*5)
        bound=np.full((4,2),np.inf)
        np.divide(1000-pos,speed,out=bound,where=speed>0)
        np.divide(-pos,speed,out=bound,where=speed<0)
        b=bound.min(axis=1)
        self.boundary+=int((b<duration-1e-9).sum())
        track,stopped,conflicts=intentions(pos,speed,np.maximum(0,np.minimum(b,duration)))
        self.conflicts+=conflicts
        next_owner=np.full(COUNT,-1,int)
        next_progress=np.zeros(COUNT,int)
        for i,j in enumerate(assigned):
            if j<0 or actions[i]!=8 or stopped[i] or not self.visible[j]:continue
            end=self.t+5
            if not self.truth['available'][self.t:end+1,j].all() or self.deadline[j]<end:continue
            path=self.truth['points'][self.t:end+1,j]
            if not align_service(track[:,i],path,self.radius):continue
            next_owner[j]=i
            next_progress[j]=5+(self.progress[j] if self.owner[j]==i else 0)
            if next_progress[j]>=10:
                self.supplied[j]=end
                next_owner[j]=-1
                next_progress[j]=0
        self.progress,self.owner=next_progress,next_owner
        self.pos=np.clip(track[-1],0,1000)
        delta=self.pos-pos
        length=np.linalg.norm(delta,axis=1)
        move=(actions!=8)&(length>1e-8)
        self.heading[move]=delta[move]/length[move,None]
        self.t+=5
        self.sample()

    def metrics(self):
        ok=self.supplied>=0
        fall=self.truth['fall']
        fell=(fall<=self.t)&(~ok|(fall<=self.supplied))
        return dict(delivered=int(ok.sum()),total=COUNT,at_least_95=bool(ok.sum()>=76),all=bool(ok.all()),
            ST=self.t//5,ACC=self.conflicts/max(1,self.t//5),RCR=float(ok.mean()),
            FRT=int(self.supplied[ok].min()//5) if ok.any() else None,
            CRT=int(self.supplied.max()//5) if ok.all() else None,coverage=float((self.last>=0).mean()),
            discovered=int((self.seen_at>=0).sum()),conflicts=self.conflicts,boundary=self.boundary,
            actual_falls=int(fell.sum()),delivered_standing=int((ok&~fell).sum()),delivered_after_fall=int((ok&fell).sum()),
            censored_window=int((fell&(fall+RULES['personal_window_after_fall_s']>DURATION)).sum()),
            timeout=not bool(ok.all() and np.all(self.last>=0)),delivery_s=self.supplied.tolist())

