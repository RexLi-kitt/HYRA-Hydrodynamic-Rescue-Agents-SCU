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
    uavs: int=4
    step: int=5
    speed: float=10.
    sensor: float=100.
    drop_radius: float=10.
    drop_seconds: int=10
    separation: float=5.
    wet_depth: float=.01
    trigger_depth: float=.10
    trigger_speed: float=.20
    random_per_minute: float=.05
    traps: int=12
    trap_radius: float=4.
    fall_pattern: str='uniform'
    burst_width: int=60
    spatial_clusters: int=0
    cluster_radius: float=100.

    def __post_init__(self):
        for name in ('duration','window','people','uavs','step','drop_seconds','traps','burst_width','spatial_clusters'):
            if type(getattr(self,name)) is not int:raise ValueError('Integer required: '+name)
        if not 0<self.window<self.duration<=1500:raise ValueError('Time exceeds available hydraulics or window')
        if not (1<=self.people<=200 and 1<=self.uavs<=8 and 1<=self.step<=30):raise ValueError('Unsupported population/team/decision interval')
        if not (0<self.drop_seconds<=self.window and 0<self.random_per_minute<1):raise ValueError('Invalid service/rate')
        for name in ('speed','sensor','drop_radius','separation','wet_depth','trigger_depth','trigger_speed','trap_radius','cluster_radius'):
            if not np.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError('Positive finite required: '+name)
        if not (self.sensor<=1000 and self.speed<=50 and self.separation<=100 and self.traps>=0):raise ValueError('Unsupported geometry')
        if self.trigger_depth<self.wet_depth:raise ValueError('Fall depth below numerical wet threshold')
        if self.fall_pattern not in ('uniform','three_bursts') or self.burst_width<1 or self.spatial_clusters<0:raise ValueError('Invalid event distribution')

def start_positions(n):
    if n==4:return np.array([[5.,5.],[995.,5.],[5.,995.],[995.,995.]])
    corners=np.array([[5.,5.],[995.,5.],[995.,995.],[5.,995.],[5.,5.]])
    return np.array([corners[int(q)]+(q-int(q))*(corners[int(q)+1]-corners[int(q)]) for q in np.arange(n)*4/n])

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

def make_scenario(water,seed,cfg,fall_end=None,pool_size=None):
    """Fixed actual-fall count; common latent exposures, independent ID shuffle.

    Changing trigger/distribution can change the accepted roster; those are
    distribution interventions, not a fixed-person deadline comparison.
    """
    end=cfg.duration-cfg.window if fall_end is None else int(fall_end)
    if not 0<=end<=cfg.duration-cfg.window:raise ValueError('Incomplete individual window')
    count=cfg.people if pool_size is None else int(pool_size)
    if count<cfg.people or count>200:raise ValueError('Invalid pool')
    streams=np.random.SeedSequence(seed).spawn(4)
    exposure=np.random.default_rng(streams[0]);trng=np.random.default_rng(streams[1])
    irng=np.random.default_rng(streams[2]);crng=np.random.default_rng(streams[3])
    mid=min(end//2,1500);wet_mid=np.argwhere(water[mid//5,0]>=cfg.wet_depth)
    if not len(wet_mid):
        wet_mid=np.argwhere(np.max(water[:end//5+1,0],axis=0)>=cfg.wet_depth)
    if not len(wet_mid):raise ValueError('No wet exposure support')
    # Prefix-stable traps; they do not change the exposure RNG or hydraulics.
    order=trng.permutation(len(wet_mid))
    centers=wet_mid[order[:min(cfg.traps,len(order))]][:,::-1]*10+5
    cluster_centers=wet_mid[crng.integers(len(wet_mid),size=max(1,cfg.spatial_clusters))][:,::-1]*10+5
    probability=1-(1-cfg.random_per_minute)**(1/60)
    events=[];wet={};attempts=0;identities=set()
    while len(events)<count and attempts<1000000:
        attempts+=1;u=exposure.random(7)
        if cfg.fall_pattern=='uniform':t=int(u[0]*(end+1))
        else:t=int(np.clip((int(u[1]*3)+.5)*end/3+(u[0]-.5)*cfg.burst_width,0,end))
        # Exact interpolated wet support at the exposure second.
        if t not in wet:
            a=t//5;b=min(a+1,300);alpha=t%5/5
            wet[t]=np.argwhere((1-alpha)*water[a,0]+alpha*water[b,0]>=cfg.wet_depth)
        ids=wet[t]
        if not len(ids):continue
        if cfg.spatial_clusters:
            center=cluster_centers[int(u[2]*cfg.spatial_clusters)]
            radius=cfg.cluster_radius*np.sqrt(u[3]);angle=2*np.pi*u[4]
            point=center+radius*np.array([np.cos(angle),np.sin(angle)])
            xy=ids[np.argmin(np.sum((ids[:,::-1]*10+5-point)**2,axis=1))][::-1]*10+5
        else:xy=ids[int(u[2]*len(ids))][::-1]*10+5
        key=(t,*xy)
        if key in identities:continue
        field=sample_water(water,t,np.asarray([xy]))[0]
        forced=bool(field[0]>=cfg.trigger_depth or np.linalg.norm(field[1:])>=cfg.trigger_speed)
        if not forced and u[5]>=probability:continue
        identities.add(key)
        events.append(dict(fall=t,deadline=t+cfg.window,xy=xy.tolist(),trigger='threshold' if forced else 'random',field=field.tolist(),exposure_id=attempts))
    if len(events)!=count:raise ValueError('Exposure generation cap; no target deletion')
    # Selection is exposure order, never fall-time order; random public IDs.
    selected=[events[i] for i in irng.permutation(count) if i<cfg.people]
    return dict(seed=seed,events=selected,traps=centers.tolist(),attempted_exposures=attempts,
                pool_size=count,fall_end=end,config=asdict(cfg),id_assignment='independent shuffle',
                conditioning='Fixed N accepted actual falls, not a population fall-risk estimate; 1-second exposure probability')

class Env:
    def __init__(self,water,scenario,cfg=None):
        self.cfg=cfg or Config(**scenario['config']);self.water=water;self.scenario=scenario
        if water.shape!=(301,3,100,100):raise ValueError('Expected 301x3x100x100 water')
        self.traps=np.asarray(scenario['traps'],float).reshape(-1,2)
        self.fall=np.array([e['fall'] for e in scenario['events']]);self.deadline=np.array([e['deadline'] for e in scenario['events']])
        self.person=np.array([e['xy'] for e in scenario['events']],float)
        if len(self.person)!=self.cfg.people:raise ValueError('Roster changed')
        if (not np.isfinite(self.person).all() or ((self.person<0)|(self.person>=1000)).any()
            or (self.fall<0).any() or (self.deadline>self.cfg.duration).any()
            or not np.array_equal(self.deadline-self.fall,np.full(self.cfg.people,self.cfg.window))
            or not np.array_equal(self.fall,np.floor(self.fall))):raise ValueError('Invalid event support')
        self.status=np.zeros(self.cfg.people,int) # 0 future,1 pending,2 supplied,3 expired,4 outside
        self.stuck=np.zeros(self.cfg.people,bool);self.progress=np.zeros(self.cfg.people,float);self.owner=np.full(self.cfg.people,-1,int)
        self.pos=start_positions(self.cfg.uavs)
        self.t=0;self.known=np.zeros((100,100),bool);self.last=np.full((100,100),-1.)
        self.measured=np.zeros((3,100,100));self.discovered=np.zeros(self.cfg.people,bool)
        self.seen_xy=np.zeros((self.cfg.people,2));self.seen_v=np.zeros((self.cfg.people,2));self.seen_at=np.full(self.cfg.people,-1.)
        self.first_seen=np.full(self.cfg.people,-1.);self.belief=np.zeros(self.cfg.people,int);self.visible=np.zeros(self.cfg.people,bool)
        self.delivered_at=np.full(self.cfg.people,-1.);self.expired_unseen=0;self.conflicts=0;self.events=[]
        yy,xx=np.indices((100,100));self.cells=np.stack((xx*10+5,yy*10+5),-1)
        self._activate();self.observe_world()

    def _activate(self):self.status[(self.status==0)&(self.fall<=self.t)]=1

    def observe_world(self):
        before=int(self.known.sum());found=int(self.discovered.sum())
        # Common circular sensor is an explicit V2 change; no hidden truth input.
        mask=np.zeros((100,100),bool)
        for p in self.pos:mask|=np.sum((self.cells-p)**2,axis=-1)<=self.cfg.sensor**2
        self.known|=mask;self.last[mask]=self.t
        a=self.t//5;b=min(a+1,300);alpha=self.t%5/5
        self.measured[:,mask]=(1-alpha)*self.water[a][:,mask]+alpha*self.water[b][:,mask]
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
        records=np.zeros((self.cfg.people,8))
        for j in np.flatnonzero(self.discovered):
            records[j]=[* (self.seen_xy[j]/1000),self.belief[j]/2,(self.t-self.seen_at[j])/self.cfg.duration,
                        *self.seen_v[j],float(self.visible[j]),1.]
        shared=np.concatenate((pooled,records.ravel(),(self.pos/1000).ravel(),[self.t/self.cfg.duration]))
        return np.array([np.concatenate((shared,[i/max(1,self.cfg.uavs-1)])) for i in range(self.cfg.uavs)],np.float32)

    def control(self):
        """Observation-only common task allocator; returns masks, not actions.

        Learners choose among up to three forward actions. A singleton tracking
        command near a visible person ensures the promised 10s supply service.
        Pure planner uses the same goals, selecting the highest progress action.
        """
        mask=np.zeros((self.cfg.uavs,9),bool);preferred=np.full(self.cfg.uavs,8,int);claimed=set();assigned=np.full(self.cfg.uavs,-1,int)
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
                region=np.ones((100,100))*.5;region[((i%4)//2)*50:((i%4)//2+1)*50,(i%2)*50:(i%2+1)*50]=1
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
        if self.t>=self.cfg.duration or actions.shape!=(self.cfg.uavs,) or not np.issubdtype(actions.dtype,np.integer) or ((actions<0)|(actions>8)).any():raise ValueError('Invalid actions')
        if not mask[np.arange(self.cfg.uavs),actions].all():raise ValueError('Unavailable action')
        dt=min(self.cfg.step,self.cfg.duration-self.t)
        start=self.pos.copy();end=np.clip(start+MOVES[actions]*self.cfg.speed*dt,0,1000)
        for i,j in enumerate(assigned):
            if j<0:continue
            if actions[i]==8 and self.visible[j]:
                velocity=self.seen_v[j];end[i]=np.clip(start[i]+velocity*min(dt,self.cfg.speed*dt/max(np.linalg.norm(velocity),1e-9)),0,1000)
            elif self.visible[j]:
                # Aim inside the delivery disk: stopping exactly on its edge can
                # chase a drifting person forever without entering service range.
                hit=first_hit(start[i],end[i],self.seen_xy[j:j+1],self.cfg.drop_radius*.5)
                if hit is not None:end[i]=start[i]+(end[i]-start[i])*hit
        # Resolve swept drone conflicts by stopping involved drones. Shared by all methods.
        stopped=set();pairs=set()
        for _ in range(self.cfg.uavs+1):
            conflicts=[]
            for i in range(self.cfg.uavs):
                for j in range(i+1,self.cfg.uavs):
                    x=start[i]-start[j];d=(end[i]-start[i])-(end[j]-start[j]);f=np.clip(-x@d/max(d@d,1e-12),0,1)
                    if np.linalg.norm(x+f*d)<self.cfg.separation-1e-8:conflicts.append((i,j))
            if not conflicts:break
            pairs.update(conflicts);stopped.update(k for pair in conflicts for k in pair)
            for i in stopped:end[i]=start[i]
        if conflicts:raise RuntimeError('Unresolved drone collision')
        self.conflicts+=len(pairs);delivered=failed=0
        for sub in range(dt):
            now=self.t+sub
            self.status[(self.status==0)&(self.fall<=now)]=1
            old=self.person.copy();active=np.flatnonzero((self.status==1)&~self.stuck)
            if len(active):
                values=sample_water(self.water,now,self.person[active]);self.stuck[active[values[:,0]<self.cfg.wet_depth]]=True
                for k,j in enumerate(active):
                    if self.stuck[j]:continue
                    nxt=self.person[j]+values[k,1:];hit=first_hit(self.person[j],nxt,self.traps,self.cfg.trap_radius)
                    if hit is not None:nxt=self.person[j]+(nxt-self.person[j])*hit;self.stuck[j]=True
                    if ((nxt<0)|(nxt>=1000)).any():self.status[j]=4;failed+=1;self.events.append(dict(type='outside',id=int(j),time=now+1))
                    self.person[j]=nxt
            drone0=start+(end-start)*(sub/dt);drone1=start+(end-start)*((sub+1)/dt)
            for j in np.flatnonzero(self.status==1):
                eligible=[i for i in range(self.cfg.uavs) if i not in stopped and actions[i]==8 and assigned[i]==j
                          and np.linalg.norm(drone0[i]-old[j])<=self.cfg.drop_radius+1e-8
                          and np.linalg.norm(drone1[i]-self.person[j])<=self.cfg.drop_radius+1e-8]
                if eligible:
                    owner=min(eligible);self.progress[j]=self.progress[j]+1 if self.owner[j]==owner else 1;self.owner[j]=owner
                else:self.progress[j]=0;self.owner[j]=-1
                if self.progress[j]>=self.cfg.drop_seconds and now+1<=self.deadline[j]:
                    self.status[j]=2;self.delivered_at[j]=now+1;self.belief[j]=2;delivered+=1;self.events.append(dict(type='delivered',id=int(j),time=now+1))
                elif now+1>=self.deadline[j]:
                    self.status[j]=3;failed+=1;self.expired_unseen+=int(not self.discovered[j]);self.events.append(dict(type='expired',id=int(j),time=now+1))
        self.t+=dt;self.pos=end;self._activate();new,found=self.observe_world()
        if not np.isfinite(self.pos).all() or not np.isfinite(self.person).all():raise FloatingPointError('Nonfinite state')
        done=bool(self.t>=self.cfg.duration or (self.status>=2).all())
        reward=10*delivered+found+5*new/10000-5*failed-.01-2*len(pairs)
        return self.observe(),float(reward),done,dict(delivered=delivered,failed=failed,new_cells=new,found=found)

    def metrics(self):
        supplied=self.status==2;failed=(self.status==3)|(self.status==4)
        return dict(scene_success=bool(supplied.all()),supplied=int(supplied.sum()),supply_fraction=float(supplied.mean()),
                    people_total=self.cfg.people,threshold95=(95*self.cfg.people+99)//100,
                    scene95_success=bool(int(supplied.sum())>=(95*self.cfg.people+99)//100),
                    expired=int((self.status==3).sum()),outside=int((self.status==4).sum()),
                    expired_unseen=int(self.expired_unseen),failed_after_discovery=int((failed&self.discovered).sum()),
                    discovered=int(self.discovered.sum()),stuck=int(self.stuck.sum()),coverage=float(self.known.mean()),
                    conflicts=int(self.conflicts),duration_s=self.t,
                    mean_supply_delay_s=float(np.mean(self.delivered_at[supplied]-self.fall[supplied])) if supplied.any() else None,
                    max_drift_m=float(np.linalg.norm(self.person-np.array([e['xy'] for e in self.scenario['events']]),axis=1).max()))
