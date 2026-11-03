"""Synthetic dynamic-water supply-drop task. No training on import."""
from dataclasses import dataclass, asdict
import numpy as np

MOVES=np.array([[0,1],[0,-1],[-1,0],[1,0],[-1,1],[-1,-1],[1,1],[1,-1],[0,0]],float)
MOVES[:8]/=np.linalg.norm(MOVES[:8],axis=1,keepdims=True)

@dataclass(frozen=True)
class Config:
    duration: int=1500
    window: int=360
    people: int=20
    step: int=5
    speed: float=10.
    sensor: float=100.
    drop_radius: float=10.
    drop_seconds: int=10
    wet_depth: float=.01
    trigger_depth: float=.10
    trigger_speed: float=.20
    random_per_minute: float=.05
    traps: int=12
    trap_radius: float=4.

    def __post_init__(self):
        if not (0<self.window<self.duration<=1500 and self.window%5==0 and self.duration%5==0):raise ValueError('Invalid time support')
        if self.step!=5 or self.people!=20:raise ValueError('Pilot uses 5s steps and 20 targets')
        if not (0<self.random_per_minute<1 and 0<self.drop_seconds<=self.window):raise ValueError('Invalid rates')

def sample_water(water,second,points):
    """Bilinear in space, linear in time; h,u,v in SI units."""
    p=np.asarray(points,float);q=np.clip(p/10-.5,0,99);lo=np.floor(q).astype(int);hi=np.minimum(lo+1,99);f=q-lo
    a=min(300,int(second//5));b=min(300,a+1);alpha=(second-a*5)/5
    def at(t):
        def cell(x,y):return water[t,:,y,x]
        return (cell(lo[:,0],lo[:,1])*(1-f[:,0,None])*(1-f[:,1,None])+
                cell(hi[:,0],lo[:,1])*f[:,0,None]*(1-f[:,1,None])+
                cell(lo[:,0],hi[:,1])*(1-f[:,0,None])*f[:,1,None]+
                cell(hi[:,0],hi[:,1])*f[:,0,None]*f[:,1,None])
    return at(a)*(1-alpha)+at(b)*alpha

def first_hit(start,end,centers,radius):
    """Swept segment/circle interception: no jumping over a narrow trap."""
    if not len(centers):return None
    d=end-start;o=start-centers;c=np.sum(o*o,axis=1)-radius**2
    if (c<=0).any():return 0.
    aa=float(d@d)
    if aa<1e-18:return None
    bb=2*(o@d);disc=bb*bb-4*aa*c
    roots=(-bb-np.sqrt(np.maximum(disc,0)))/(2*aa)
    ok=(disc>=0)&(roots>=0)&(roots<=1)
    return float(roots[ok].min()) if ok.any() else None

def make_scenario(water,seed,cfg):
    """Condition on N accepted exposure events, independently of any agent.

    Candidate exposure time is uniform on [0,T-L], location uniform on wet cells.
    High water/speed always accepts; remaining wet exposures use a time-scaled
    probability. Rejected exposures are non-falls, never scored people removed
    after simulation. Accepted-time distribution need not be uniform.
    """
    rng=np.random.default_rng(seed);last=cfg.duration-cfg.window;wet={}
    for t in range(0,last+1,5):
        ids=np.argwhere(water[t//5,0]>=cfg.wet_depth)
        if len(ids):wet[t]=ids
    if not wet:raise ValueError('No wet exposure locations; do not fabricate water')
    middle=np.argwhere(water[min(150,last)//5,0]>=cfg.wet_depth)
    if not len(middle):middle=next(iter(wet.values()))
    centers=middle[rng.choice(len(middle),min(cfg.traps,len(middle)),replace=False)][:,::-1]*10+5
    times=list(wet);events=[];attempts=0;p=1-(1-cfg.random_per_minute)**(cfg.step/60)
    while len(events)<cfg.people and attempts<200000:
        attempts+=1;t=int(rng.choice(times));cell=wet[t][rng.integers(len(wet[t]))];xy=cell[::-1]*10+5
        if len(centers) and (np.linalg.norm(centers-xy,axis=1)<=cfg.trap_radius).any():continue
        field=sample_water(water,t,np.asarray([xy]))[0]
        forced=bool(field[0]>=cfg.trigger_depth or np.linalg.norm(field[1:])>=cfg.trigger_speed)
        if not forced and rng.random()>=p:continue
        if any(e['fall']==t and e['xy']==xy.tolist() for e in events):continue
        events.append(dict(fall=t,deadline=t+cfg.window,xy=xy.tolist(),trigger='threshold' if forced else 'random',field=field.tolist()))
    if len(events)!=cfg.people:raise ValueError('Event generation cap reached; register a different protocol')
    events.sort(key=lambda e:e['fall'])
    return dict(seed=seed,events=events,traps=centers.tolist(),attempted_exposures=attempts,
                conditioning='N actual falls from accepted wet exposure events; not a population fall-rate estimate',config=asdict(cfg))

class Env:
    def __init__(self,water,scenario,cfg=None):
        self.cfg=cfg or Config(**scenario['config']);self.water=water;self.scenario=scenario
        if water.shape!=(301,3,100,100):raise ValueError('Expected 301x3x100x100 water')
        self.traps=np.asarray(scenario['traps'],float).reshape(-1,2)
        self.fall=np.array([e['fall'] for e in scenario['events']]);self.deadline=np.array([e['deadline'] for e in scenario['events']])
        self.person=np.array([e['xy'] for e in scenario['events']],float)
        if len(self.person)!=self.cfg.people:raise ValueError('Roster changed')
        self.status=np.zeros(20,int) # 0 future,1 pending,2 supplied,3 expired,4 outside
        self.stuck=np.zeros(20,bool);self.progress=np.zeros(20,float);self.owner=np.full(20,-1,int)
        self.pos=np.array([[5.,5.],[995.,5.],[5.,995.],[995.,995.]])
        self.t=0;self.known=np.zeros((100,100),bool);self.last=np.full((100,100),-1.)
        self.measured=np.zeros((3,100,100));self.discovered=np.zeros(20,bool)
        self.seen_xy=np.zeros((20,2));self.seen_v=np.zeros((20,2));self.seen_at=np.full(20,-1.)
        self.first_seen=np.full(20,-1.);self.belief=np.zeros(20,int);self.visible=np.zeros(20,bool)
        self.delivered_at=np.full(20,-1.);self.expired_unseen=0;self.conflicts=0;self.events=[]
        yy,xx=np.indices((100,100));self.cells=np.stack((xx*10+5,yy*10+5),-1)
        self._activate();self.observe_world()

    def _activate(self):self.status[(self.status==0)&(self.fall<=self.t)]=1

    def observe_world(self):
        before=int(self.known.sum());found=int(self.discovered.sum())
        # Common circular sensor is an explicit V2 change; no hidden truth input.
        mask=np.zeros((100,100),bool)
        for p in self.pos:mask|=np.sum((self.cells-p)**2,axis=-1)<=self.cfg.sensor**2
        self.known|=mask;self.last[mask]=self.t;self.measured[:,mask]=self.water[self.t//5][:,mask]
        self.visible=(self.status>0)&(self.status!=4)&(np.linalg.norm(self.person[:,None]-self.pos[None],axis=-1).min(1)<=self.cfg.sensor)
        ids=np.flatnonzero(self.visible)
        self.first_seen[ids[self.first_seen[ids]<0]]=self.t
        self.discovered[ids]=True;self.seen_xy[ids]=self.person[ids];self.seen_at[ids]=self.t
        if len(ids):
            self.seen_v[ids]=sample_water(self.water,self.t,self.person[ids])[:,1:]
            self.seen_v[ids[self.stuck[ids]]]=0
        self.belief[ids]=np.where(self.status[ids]==1,1,2)
        return int(self.known.sum())-before,int(self.discovered.sum())-found

    def observe(self):
        age=np.where(self.known,(self.t-self.last)/self.cfg.duration,1.)
        fields=np.concatenate((self.known[None],np.clip(self.measured,-3,3)/3,age[None]))
        pooled=fields.reshape(5,10,10,10,10).mean((2,4)).ravel()
        records=np.zeros((20,8))
        for j in np.flatnonzero(self.discovered):
            records[j]=[* (self.seen_xy[j]/1000),self.belief[j]/2,(self.t-self.seen_at[j])/self.cfg.duration,
                        *self.seen_v[j],float(self.visible[j]),1.]
        shared=np.concatenate((pooled,records.ravel(),(self.pos/1000).ravel(),[self.t/self.cfg.duration]))
        return np.array([np.concatenate((shared,[i/3])) for i in range(4)],np.float32)

    def control(self):
        """Observation-only common task allocator; returns masks, not actions.

        Learners choose among up to three forward actions. A singleton tracking
        command near a visible person ensures the promised 10s supply service.
        Pure planner uses the same goals, selecting the highest progress action.
        """
        mask=np.zeros((4,9),bool);preferred=np.full(4,8,int);claimed=set();assigned=np.full(4,-1,int)
        stale=np.where(self.known,(self.t-self.last)/120,2.)
        risk=np.clip(self.measured[0]/self.cfg.trigger_depth,0,2)+np.clip(np.linalg.norm(self.measured[1:],axis=0)/self.cfg.trigger_speed,0,2)
        score=stale*(1+risk)
        for i,p in enumerate(self.pos):
            eligible=[j for j in np.flatnonzero((self.belief==1)&self.discovered) if j not in claimed and self.t-self.seen_at[j]<=120]
            if eligible:
                j=min(eligible,key=lambda j:np.linalg.norm(self.seen_xy[j]-p));claimed.add(j);assigned[i]=j
                goal=np.clip(self.seen_xy[j]+self.seen_v[j]*min(30,self.t-self.seen_at[j]),0,1000)
                if self.visible[j] and np.linalg.norm(self.seen_xy[j]-p)<=self.cfg.drop_radius+1e-8:
                    mask[i,8]=True;continue
            else:
                # Region preference, not restriction: old regions must be revisited.
                region=np.ones((100,100))*.5;region[(i//2)*50:(i//2+1)*50,(i%2)*50:(i%2+1)*50]=1
                utility=score*region/(1+np.linalg.norm(self.cells-p,axis=-1)/250)
                r,c=np.unravel_index(np.argmax(utility),utility.shape);goal=np.array([c*10+5,r*10+5])
            scores=MOVES[:8]@(goal-p)
            legal=((p+MOVES[:8]*.01>=0)&(p+MOVES[:8]*.01<=1000)).all(1)
            scores[~legal]=-np.inf
            ids=np.argsort(scores)[-3:];ids=ids[np.isfinite(scores[ids])&(scores[ids]>0)]
            if len(ids):mask[i,ids]=True;preferred[i]=int(np.argmax(scores))
            else:mask[i,8]=True
        return mask,preferred,assigned

    def step(self,actions):
        actions=np.asarray(actions)
        mask,_,assigned=self.control()
        if self.t>=self.cfg.duration or actions.shape!=(4,) or not np.issubdtype(actions.dtype,np.integer) or ((actions<0)|(actions>8)).any():raise ValueError('Invalid actions')
        if not mask[np.arange(4),actions].all():raise ValueError('Unavailable action')
        start=self.pos.copy();end=np.clip(start+MOVES[actions]*self.cfg.speed*5,0,1000)
        for i,j in enumerate(assigned):
            if j<0:continue
            if actions[i]==8 and self.visible[j]:
                velocity=self.seen_v[j];end[i]=np.clip(start[i]+velocity*min(5,self.cfg.speed*5/max(np.linalg.norm(velocity),1e-9)),0,1000)
            elif self.visible[j]:
                # Aim inside the delivery disk: stopping exactly on its edge can
                # chase a drifting person forever without entering service range.
                hit=first_hit(start[i],end[i],self.seen_xy[j:j+1],self.cfg.drop_radius*.5)
                if hit is not None:end[i]=start[i]+(end[i]-start[i])*hit
        # Resolve swept drone conflicts by stopping involved drones. Shared by all methods.
        stopped=set();pairs=set()
        for _ in range(5):
            conflicts=[]
            for i in range(4):
                for j in range(i+1,4):
                    x=start[i]-start[j];d=(end[i]-start[i])-(end[j]-start[j]);f=np.clip(-x@d/max(d@d,1e-12),0,1)
                    if np.linalg.norm(x+f*d)<5-1e-8:conflicts.append((i,j))
            if not conflicts:break
            pairs.update(conflicts);stopped.update(k for pair in conflicts for k in pair)
            for i in stopped:end[i]=start[i]
        if conflicts:raise RuntimeError('Unresolved drone collision')
        self.conflicts+=len(pairs);delivered=failed=0
        for sub in range(5):
            now=self.t+sub;old=self.person.copy();active=np.flatnonzero((self.status==1)&~self.stuck)
            if len(active):
                values=sample_water(self.water,now,self.person[active]);self.stuck[active[values[:,0]<self.cfg.wet_depth]]=True
                for k,j in enumerate(active):
                    if self.stuck[j]:continue
                    nxt=self.person[j]+values[k,1:];hit=first_hit(self.person[j],nxt,self.traps,self.cfg.trap_radius)
                    if hit is not None:nxt=self.person[j]+(nxt-self.person[j])*hit;self.stuck[j]=True
                    if ((nxt<0)|(nxt>=1000)).any():self.status[j]=4;failed+=1;self.events.append(dict(type='outside',id=int(j),time=now+1))
                    self.person[j]=nxt
            drone0=start+(end-start)*(sub/5);drone1=start+(end-start)*((sub+1)/5)
            for j in np.flatnonzero(self.status==1):
                eligible=[i for i in range(4) if i not in stopped and actions[i]==8 and assigned[i]==j
                          and np.linalg.norm(drone0[i]-old[j])<=self.cfg.drop_radius+1e-8
                          and np.linalg.norm(drone1[i]-self.person[j])<=self.cfg.drop_radius+1e-8]
                if eligible:
                    owner=min(eligible);self.progress[j]=self.progress[j]+1 if self.owner[j]==owner else 1;self.owner[j]=owner
                else:self.progress[j]=0;self.owner[j]=-1
                if self.progress[j]>=self.cfg.drop_seconds and now+1<=self.deadline[j]:
                    self.status[j]=2;self.delivered_at[j]=now+1;self.belief[j]=2;delivered+=1;self.events.append(dict(type='delivered',id=int(j),time=now+1))
                elif now+1>=self.deadline[j]:
                    self.status[j]=3;failed+=1;self.expired_unseen+=int(not self.discovered[j]);self.events.append(dict(type='expired',id=int(j),time=now+1))
        self.t+=5;self.pos=end;self._activate();new,found=self.observe_world()
        if not np.isfinite(self.pos).all() or not np.isfinite(self.person).all():raise FloatingPointError('Nonfinite state')
        done=bool(self.t>=self.cfg.duration or (self.status>=2).all())
        reward=10*delivered+found+5*new/10000-5*failed-.01-2*len(pairs)
        return self.observe(),float(reward),done,dict(delivered=delivered,failed=failed,new_cells=new,found=found)

    def metrics(self):
        supplied=self.status==2;failed=(self.status==3)|(self.status==4)
        return dict(scene_success=bool(supplied.all()),supplied=int(supplied.sum()),supply_fraction=float(supplied.mean()),
                    expired=int((self.status==3).sum()),outside=int((self.status==4).sum()),
                    expired_unseen=int(self.expired_unseen),failed_after_discovery=int((failed&self.discovered).sum()),
                    discovered=int(self.discovered.sum()),stuck=int(self.stuck.sum()),coverage=float(self.known.mean()),
                    conflicts=int(self.conflicts),duration_s=self.t,
                    mean_supply_delay_s=float(np.mean(self.delivered_at[supplied]-self.fall[supplied])) if supplied.any() else None,
                    max_drift_m=float(np.linalg.norm(self.person-np.array([e['xy'] for e in self.scenario['events']]),axis=1).max()))
