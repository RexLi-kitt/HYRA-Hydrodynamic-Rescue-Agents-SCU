"""Common physical task and observed-history interface. No learner receives truth."""
from dataclasses import dataclass, asdict
import numpy as np

MOVES = np.array([[0,1],[0,-1],[-1,0],[1,0],[-1,1],[-1,-1],[1,1],[1,-1],[0,0]], float)
MOVES[:8] /= np.linalg.norm(MOVES[:8], axis=1, keepdims=True)

@dataclass(frozen=True)
class Config:
    side: float = 1000.
    dx: float = 10.
    step_s: float = 5.
    speed: float = 10.
    radius: float = 100.
    service_radius: float = 10.
    service_s: float = 10.
    separation: float = 5.
    depth_limit: float = .08
    speed_limit: float = .3
    max_steps: int = 300
    targets: int = 20

def sector(points, position, heading, radius):
    d = np.asarray(points) - position
    return (np.sum(d*d, axis=-1) <= radius*radius+1e-9) & (d @ heading >= -1e-9)

def segment_circle(p, d, center, radius):
    """First intersection in [0,1], including an initially occupied service disk."""
    offset = p-center
    if offset@offset <= radius*radius+1e-9: return 0.
    aa = d@d
    if aa < 1e-15: return None
    bb = 2*(offset@d); cc = offset@offset-radius*radius
    disc = bb*bb-4*aa*cc
    if disc < -1e-9: return None
    x = (-bb-np.sqrt(max(0.,disc)))/(2*aa)
    return float(np.clip(x,0,1)) if -1e-9 <= x <= 1+1e-9 else None

def position_at(start, end, time_s, speed=10.):
    delta=end-start; length=np.linalg.norm(delta)
    return start+delta*min(1., time_s*speed/max(length,1e-12))

def min_separation(a0,a1,b0,b1,duration=5.,speed=10.):
    """Exact closest approach for straight motion followed by a short-step stop."""
    knots=sorted(set([0.,duration,min(duration,np.linalg.norm(a1-a0)/speed),min(duration,np.linalg.norm(b1-b0)/speed)]))
    best=np.inf
    for lo,hi in zip(knots[:-1],knots[1:]):
        x=position_at(a0,a1,lo,speed)-position_at(b0,b1,lo,speed)
        y=position_at(a0,a1,hi,speed)-position_at(b0,b1,hi,speed)
        d=y-x; f=np.clip(-x@d/max(d@d,1e-15),0,1)
        best=min(best,float(np.linalg.norm(x+f*d)))
    return best

class Env:
    def __init__(self, water, seed=0, cfg=None, people=None):
        self.cfg=cfg or Config(); self.water=water
        if water.shape != (301,3,100,100): raise ValueError('Need 0..1500s h,u,v at 5s on 100x100 cells')
        self.rng=np.random.default_rng(seed)
        self.start=np.array([[5.,5.],[995.,5.],[5.,995.],[995.,995.]])
        yy,xx=np.indices((100,100)); self.centers=np.stack((xx*10+5,yy*10+5),-1)
        if people is None:
            valid=np.ones(10000,bool); valid[[0,99,9900,9999]]=False
            ids=self.rng.choice(np.flatnonzero(valid),20,replace=False)
            people=self.centers.reshape(-1,2)[ids]
        self.people=np.asarray(people,float).copy()
        if self.people.shape!=(20,2) or len(np.unique(self.people,axis=0))!=20: raise ValueError('20 distinct people required')
        if (self.people<0).any() or (self.people>=1000).any(): raise ValueError('Person outside domain')
        self.reset()

    def reset(self):
        self.pos=self.start.copy(); self.heading=500-self.pos
        self.heading/=np.linalg.norm(self.heading,axis=1,keepdims=True)
        self.t=0; self.known=np.zeros((100,100),bool); self.last=np.full((100,100),-1.,np.float32)
        self.measured=np.zeros((3,100,100),np.float32)
        self.status=np.zeros(20,np.int8) # hidden=0, discovered=1, rescued=2
        self.owner=np.full(20,-1,int); self.progress=np.zeros(20,float)
        self.last_actions=np.full(4,8); self.conflicts=0; self.boundaries=0; self.first=None; self.complete=None
        self.events=[]; self.sample(); return self.observe()

    def sample(self):
        prior=int(self.known.sum()); found=int(np.count_nonzero(self.status))
        mask=np.zeros((100,100),bool)
        for i in range(4):
            mask |= sector(self.centers,self.pos[i],self.heading[i],self.cfg.radius)
            seen=sector(self.people,self.pos[i],self.heading[i],self.cfg.radius)
            self.status[seen & (self.status==0)]=1
        self.known |= mask; self.last[mask]=self.t*5
        self.measured[:,mask]=self.water[self.t][:,mask]
        return int(self.known.sum())-prior,int(np.count_nonzero(self.status))-found

    def observe(self):
        """Explicit memory is identical for all methods; no hidden bed or people."""
        pmap=np.zeros((100,100),np.float32)
        records=np.zeros((20,6),np.float32)
        for k,j in enumerate(np.flatnonzero(self.status)):
            x,y=self.people[j]; pmap[int(y//10),int(x//10)]=self.status[j]/2
            records[k]=[x/1000,y/1000,self.status[j]/2,self.progress[j]/10,(self.owner[j]+1)/4,1]
        age=np.where(self.known,(self.t*5-self.last)/1500,0)
        fields=np.concatenate((self.known[None],self.measured,np.asarray(age)[None],pmap[None])).astype(np.float32)
        fields[1:4]=np.clip(fields[1:4],-5,5)/5
        global_map=fields.reshape(6,10,10,10,10).mean((2,4)).ravel()
        state=np.concatenate((self.pos/1000,self.heading,(self.last_actions/8)[:,None]),axis=1).ravel()
        padded=np.pad(fields,((0,0),(5,5),(5,5)))
        out=[]
        for i,(x,y) in enumerate(self.pos):
            c,r=min(99,int(x//10)),min(99,int(y//10))
            local=padded[:,r:r+11,c:c+11].ravel()
            out.append(np.concatenate((local,global_map,state,records.ravel(),[self.t/300,i/3])))
        return np.asarray(out,np.float32)

    def available(self):
        # The bounds are public. A partially clipped step remains legal.
        result=np.ones((4,9),bool)
        for i,p in enumerate(self.pos):
            for a,d in enumerate(MOVES[:8]):
                result[i,a]=not any((p[k]<=1e-8 and d[k]<0) or (p[k]>=1000-1e-8 and d[k]>0) for k in (0,1))
        return result

    def intent(self,i,a):
        p=self.pos[i]; d=MOVES[a]*50; fraction=1.
        for k in (0,1):
            if d[k]>0: fraction=min(fraction,(1000-p[k])/d[k])
            elif d[k]<0: fraction=min(fraction,-p[k]/d[k])
        boundary=fraction<1-1e-10
        for j in np.flatnonzero(self.status==1):
            # Inside a service circle: permit departure, rather than trap movement.
            if np.linalg.norm(p-self.people[j]) <= self.cfg.service_radius+1e-8: continue
            hit=segment_circle(p,d,self.people[j],self.cfg.service_radius)
            if hit is not None: fraction=min(fraction,hit)
        return np.clip(p+d*max(0,fraction),0,1000),boundary

    def pairs(self,start,end):
        return {(i,j) for i in range(4) for j in range(i+1,4)
                if min_separation(start[i],end[i],start[j],end[j]) < self.cfg.separation-1e-8}

    def service_safe(self,j):
        x,y=self.people[j]; cell=(int(y//10),int(x//10))
        f=self.water[self.t:self.t+2,:,cell[0],cell[1]]
        # Linear h,u,v between frames: maximum h and speed over this fixed-position
        # interval occur at an endpoint (convex norm); thus this covers the whole 5s.
        return bool((f[:,0]<=self.cfg.depth_limit).all() and
                    (np.linalg.norm(f[:,1:],axis=1)<=self.cfg.speed_limit).all())

    def step(self,actions):
        a=np.asarray(actions)
        if self.t>=300 or a.shape!=(4,) or not np.issubdtype(a.dtype,np.integer) or ((a<0)|(a>8)).any(): raise ValueError('Invalid joint action')
        old=self.pos.copy(); intent=[self.intent(i,int(a[i])) for i in range(4)]
        end=np.array([x[0] for x in intent]); boundary=sum(x[1] for x in intent)
        raw_pairs=self.pairs(old,end); stopped=set()
        for _ in range(5):
            conflicts=self.pairs(old,end)
            if not conflicts: break
            stopped.update(k for pair in conflicts for k in pair)
            for k in stopped: end[k]=old[k]
        if self.pairs(old,end): raise RuntimeError('Collision resolution failed')
        self.pos=end; self.conflicts+=len(raw_pairs); self.boundaries+=boundary
        for i in range(4):
            delta=end[i]-old[i]
            if np.linalg.norm(delta)>1e-8: self.heading[i]=delta/np.linalg.norm(delta)
        rescued=0; used=set()
        # Existing service precedes new requests. Each UAV can serve one person.
        order=sorted(np.flatnonzero(self.status==1),key=lambda j:(self.owner[j]<0,int(j)))
        for j in order:
            eligible=[i for i in range(4) if a[i]==8 and i not in used and i not in stopped
                      and np.linalg.norm(self.pos[i]-self.people[j])<=self.cfg.service_radius+1e-8]
            safe=self.service_safe(j)
            if not safe or not eligible:
                self.owner[j]=-1; self.progress[j]=0.; continue
            previous=int(self.owner[j])
            winner=previous if previous in eligible else min(eligible,key=lambda i:(np.linalg.norm(self.pos[i]-self.people[j]),i))
            self.progress[j]=self.progress[j]+5 if winner==previous else 5.
            self.owner[j]=winner; used.add(winner)
            if self.progress[j]>=10:
                self.status[j]=2; rescued+=1; self.owner[j]=-1
                self.events.append({'type':'rescued','target':int(j),'agent':winner,'step':self.t+1})
        self.t+=1; self.last_actions=a.copy(); new,found=self.sample()
        if rescued and self.first is None: self.first=self.t
        if (self.status==2).all() and self.complete is None: self.complete=self.t
        success=bool(self.known.all() and (self.status==2).all()); timeout=self.t==300
        reward=10*rescued+found+20*new/10000-.01-2*len(raw_pairs)-.2*boundary+5*success
        info=dict(new_cells=new,found=found,rescued=rescued,conflicts=len(raw_pairs),boundary=boundary,
                  completed=success,time_limit=timeout and not success)
        return self.observe(),float(reward),success or timeout,info

    def metrics(self):
        return dict(ST=self.t,ACC=self.conflicts/max(1,self.t),RCR=float(np.mean(self.status==2)),FRT=self.first,
                    CRT=self.complete,coverage=float(self.known.mean()),rescued=int((self.status==2).sum()),
                    discovered=int((self.status>0).sum()),conflicts=int(self.conflicts),boundaries=int(self.boundaries),
                    task_complete=bool(self.known.all() and (self.status==2).all()),
                    timed_out=bool(self.t==300 and not(self.known.all() and (self.status==2).all())))

def rule_action(env):
    """Partitioned sweep with greedy unique assignments from observed targets only."""
    actions=[]; claimed=set()
    for i,p in enumerate(env.pos):
        known=[j for j in np.flatnonzero(env.status==1) if j not in claimed]
        if known:
            j=min(known,key=lambda j:np.linalg.norm(p-env.people[j])); claimed.add(j); goal=env.people[j]
            if np.linalg.norm(p-goal)<=10+1e-8: actions.append(8); continue
        else:
            region=np.zeros((100,100),bool)
            region[(i//2)*50:(i//2+1)*50,(i%2)*50:(i%2+1)*50]=True
            unseen=np.argwhere(region & ~env.known)
            if not len(unseen): unseen=np.argwhere(~env.known)
            if not len(unseen): actions.append(8); continue
            coords=unseen[:,::-1]*10+5
            goal=coords[np.argmin(np.sum((coords-p)**2,axis=1))]
        d=goal-p; scores=MOVES[:8]@d
        scores[~env.available()[i,:8]]=-np.inf
        actions.append(int(np.argmax(scores)))
    return np.array(actions)
