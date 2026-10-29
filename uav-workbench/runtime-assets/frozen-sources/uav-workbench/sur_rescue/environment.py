"""Eight-neighbour grid rescue over an actual flood background.

Synthetic targets/no-fly cells are explicit scenario data, not inferred people
or obstacles in a flood. Flood depth does not currently decide rescue success.
Actor view is built solely from arrived sensor records and known team poses.
"""
from collections import deque
from dataclasses import dataclass,asdict
import math
import numpy as np
from .gmrf import DualGMRF
from hyra_uav.core import map_metrics

ACTIONS=('west','east','north','south','northwest','southwest','northeast','southeast')
MOVES=((0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1))

@dataclass(frozen=True)
class RescueConfig:
    step_s:float=5.
    duration_s:float=250.
    sense_width:int=5
    service_s:float=10.
    n_targets:int=5
    obstacle_fraction:float=.1
    seed:int=11
    def validate(self,source):
        if not math.isfinite(self.step_s) or not math.isfinite(self.duration_s) or self.step_s<=0 or self.duration_s<=0 or self.service_s<=0:raise ValueError('Invalid rescue time scale')
        if not math.isclose(self.duration_s/self.step_s,round(self.duration_s/self.step_s)) or self.duration_s/self.step_s>240:raise ValueError('Use an integer number of at most 240 decisions')
        if type(self.sense_width) is not int or not 3<=self.sense_width<=15 or self.sense_width%2!=1:raise ValueError('Use an odd sensor window')
        if type(self.n_targets) is not int or not 1<=self.n_targets<=50 or not 0<=self.obstacle_fraction<.5 or type(self.seed) is not int or not 0<=self.seed<2**32:raise ValueError('Invalid synthetic scenario')
        for t in np.arange(round(self.duration_s/self.step_s)+1)*self.step_s:source.index(float(t))

def frozen(a):
    a=np.asarray(a).copy();a.setflags(write=False);return a

def pooled(a,size=5):
    nz,nx=a.shape[-2:];out=np.empty((*a.shape[:-2],size,size),dtype=np.float32)
    for z in range(size):
        for x in range(size):out[...,z,x]=a[...,math.floor(z*nz/size):math.ceil((z+1)*nz/size),math.floor(x*nx/size):math.ceil((x+1)*nx/size)].mean(axis=(-1,-2))
    return out

@dataclass(frozen=True)
class RescueView:
    images:np.ndarray
    available:np.ndarray
    positions:tuple
    explored:np.ndarray
    obstacles:np.ndarray
    signal:np.ndarray
    targets:tuple
    busy:tuple
    step:int

def joint_motion(positions,actions,blocked,busy):
    old=np.asarray(positions,dtype=int);end=old.copy();nz,nx=blocked.shape;reasons=['move']*len(old)
    for i,action in enumerate(actions):
        if busy[i]:reasons[i]='rescuing';continue
        dz,dx=MOVES[action];r,c=old[i]+[dz,dx]
        if not (0<=r<nz and 0<=c<nx):reasons[i]='boundary';continue
        if blocked[r,c] or (dz and dx and (blocked[old[i,0],c] or blocked[r,old[i,1]])):
            reasons[i]='obstacle';continue
        end[i]=[r,c]
    conflicts=set()
    for _ in range(len(old)+1):
        rejected=set()
        for i in range(len(old)):
            for j in range(i):
                relative=old[i]-old[j];delta=(end[i]-end[j])-relative
                denom=float(delta@delta);t=float(np.clip(-(relative@delta)/denom,0,1)) if denom else 0
                if np.linalg.norm(relative+t*delta)<1e-8:
                    conflicts.add((j,i));rejected.update(k for k in (i,j) if np.any(end[k]!=old[k]))
        if not rejected:break
        for i in rejected:end[i]=old[i];reasons[i]='collision'
    else:raise ArithmeticError('Joint collision cancellation failed')
    return end,reasons,len(conflicts)

class RescueEnv:
    def __init__(self,source,config=RescueConfig(),scenario=None):
        config.validate(source);self.source,self.config=source,config;self.shape=source.shape;self.dx=source.dx
        self.initial_positions=np.array([[0,0],[self.shape[0]-1,0]],dtype=int)
        rng=np.random.default_rng(config.seed)
        if scenario is None:
            available=np.ones(self.shape,dtype=bool)
            for r,c in self.initial_positions:available[r,c]=False
            ids=np.flatnonzero(available);obstacles=np.zeros(self.shape,dtype=bool)
            obstacles.flat[rng.choice(ids,int(available.sum()*config.obstacle_fraction),replace=False)]=True
            ids=np.flatnonzero(available&~obstacles)
            if len(ids)<config.n_targets:raise ValueError('Not enough free target cells')
            targets=[divmod(int(i),self.shape[1]) for i in rng.choice(ids,config.n_targets,replace=False)]
        else:obstacles=np.asarray(scenario['obstacles'],dtype=bool);targets=[tuple(p) for p in scenario['targets']]
        if obstacles.shape!=self.shape or len(set(targets))!=len(targets) or not targets:raise ValueError('Invalid scenario grid or targets')
        if any(not 0<=r<self.shape[0] or not 0<=c<self.shape[1] or obstacles[r,c] for r,c in targets):raise ValueError('Invalid target location')
        if any(obstacles[r,c] for r,c in self.initial_positions):raise ValueError('Blocked start')
        self._blocked=obstacles.copy();self._targets=targets
        self.fields=DualGMRF(self.shape,self._blocked,self._targets,config.seed)

    def reset(self):
        self.positions=self.initial_positions.copy();self.time_s=0.;self.step_count=0
        self.explored=np.zeros(self.shape,dtype=bool);self.obstacles=np.zeros(self.shape,dtype=bool);self.signal=np.zeros(self.shape)
        self.labels=np.full(self.shape,-1,dtype=np.int8);self.observed_s=np.full(self.shape,-1.)
        self.target_status={p:'undiscovered' for p in self._targets};self.busy=[None,None]
        self.distance_m=0.;self.collision_events=0;self.actions=[];self.observations=[];self.snapshots=[];self.trajectory=[];self.events=[]
        self._sample(['initial','initial'],['initial','initial']);return self.view()

    def _sample(self,actions,reasons):
        truth=self.source.labels(self.time_s);half=self.config.sense_width//2;new=[]
        for agent,(r,c) in enumerate(self.positions):
            rr,cc=np.mgrid[max(0,r-half):min(self.shape[0],r+half+1),max(0,c-half):min(self.shape[1],c+half+1)]
            cells=(rr*self.shape[1]+cc).ravel();new.append(int((~self.explored.ravel()[cells]).sum()))
            self.explored.ravel()[cells]=True;self.obstacles.ravel()[cells]=self._blocked.ravel()[cells]
            values=self.fields.sample_signal(cells);self.signal.ravel()[cells]=values
            self.labels.ravel()[cells]=truth.ravel()[cells];self.observed_s.ravel()[cells]=self.time_s
            for p in self._targets:
                if max(abs(p[0]-r),abs(p[1]-c))<=half and self.target_status[p]=='undiscovered':
                    self.target_status[p]='discovered';self.events.append(dict(type='discovered',agent=agent,cell=list(p),time_s=self.time_s))
            self.observations.append(dict(agent_id=agent,observed_s=self.time_s,cells=cells.tolist(),labels=truth.ravel()[cells].astype(int).tolist(),signal=values.tolist(),obstacles=self._blocked.ravel()[cells].astype(int).tolist()))
        self.environment_field,self.environment_diagnostics=self.fields.observed_environment(self.explored,self.obstacles)
        self._current_truth=truth
        point=dict(time_s=self.time_s,poses=[dict(x_m=float(c*self.dx),z_m=float(r*self.dx),heading_deg=self._heading(agent)) for agent,(r,c) in enumerate(self.positions)],
            actions=actions,motion_status=reasons,distance_m=self.distance_m,turn_s=0.,collision_events=self.collision_events)
        self.trajectory.append(point)
        self.snapshots.append(dict(**point,labels=self.labels.ravel().tolist(),observed_s=self.observed_s.ravel().tolist(),metrics=map_metrics(truth,self.labels),
            explored=self.explored.ravel().astype(int).tolist(),obstacles=self.obstacles.ravel().astype(int).tolist(),gmrf_environment=np.round(self.environment_field,6).ravel().tolist(),gmrf_signal=np.round(self.signal,6).ravel().tolist(),
            targets=[dict(row=r,col=c,status=s) for (r,c),s in self.target_status.items() if s!='undiscovered'],rescued=sum(s=='rescued' for s in self.target_status.values()),busy=[b['remaining'] if b else 0 for b in self.busy]))
        return new

    def _heading(self,agent):
        if not self.actions:return 0.
        dz,dx=MOVES[ACTIONS.index(self.actions[-1]['actions'][agent])]
        return math.degrees(math.atan2(dz,dx))%360

    def view(self):
        pos=np.zeros(self.shape)
        known=[]
        for (r,c),status in self.target_status.items():
            if status!='undiscovered':pos[r,c]=3 if status=='rescued' else 2;known.append((r,c,status))
        for r,c in self.positions:pos[r,c]=1
        base=np.stack((self.explored&~self.obstacles,self.obstacles,self.environment_field,self.signal)).astype(np.float32)
        size=5;pad=np.pad(base,((0,0),(2,2),(2,2)));global_=pooled(base,size);positions=pooled(pos[None],size)
        images=np.stack([np.concatenate((pad[:,r:r+size,c:c+size],global_,positions)) for r,c in self.positions])
        available=np.ones((2,8),dtype=bool)
        for i,(r,c) in enumerate(self.positions):
            for a,(dz,dx) in enumerate(MOVES):
                rr,cc=r+dz,c+dx
                available[i,a]=0<=rr<self.shape[0] and 0<=cc<self.shape[1]
                if available[i,a]:available[i,a]=not self.obstacles[rr,cc] and not (dz and dx and (self.obstacles[r,cc] or self.obstacles[rr,c]))
        return RescueView(frozen(images),frozen(available),tuple(map(tuple,self.positions)),frozen(self.explored),frozen(self.obstacles),frozen(self.signal),tuple(known),tuple(b['remaining'] if b else 0 for b in self.busy),self.step_count)

    def step(self,actions):
        if self.time_s>=self.config.duration_s-1e-8:raise ValueError('Rescue episode ended')
        if len(actions)!=2:raise ValueError('Expected two eight-direction actions')
        actions=[ACTIONS.index(a) if isinstance(a,str) and a in ACTIONS else a for a in actions]
        if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<8 for a in actions):raise ValueError('Invalid eight-direction action')
        old=self.positions.copy();before=int(self.explored.sum());busy=[b is not None for b in self.busy]
        self.positions,reasons,conflicts=joint_motion(old,actions,self._blocked,busy)
        self.distance_m+=float(np.linalg.norm(self.positions-old,axis=1).sum()*self.dx);self.collision_events+=conflicts
        self.actions.append(dict(time_s=self.time_s,actions=[ACTIONS[a] for a in actions]));self.step_count+=1;self.time_s=round(self.step_count*self.config.step_s,9)
        rescued=0
        for agent,(r,c) in enumerate(self.positions):
            p=(int(r),int(c));b=self.busy[agent]
            if b:
                b['remaining']=max(0,b['remaining']-self.config.step_s)
                if b['remaining']==0:
                    self.target_status[b['cell']]='rescued';rescued+=5/max(b['entry_distance'],1.);self.busy[agent]=None
                    self.events.append(dict(type='rescued',agent=agent,cell=list(b['cell']),time_s=self.time_s))
            if self.busy[agent] is None and p in self.target_status and self.target_status[p] not in ('rescued','rescuing'):
                self.target_status[p]='rescuing';self.busy[agent]=dict(cell=p,remaining=self.config.service_s,entry_distance=float(np.linalg.norm(self.positions[agent]-old[agent])))
                self.events.append(dict(type='rescuing',agent=agent,cell=list(p),time_s=self.time_s))
        new=self._sample([ACTIONS[a] for a in actions],reasons)
        reward=sum(.1*(1+before/math.prod(self.shape))*n if reason=='move' else -1. if reason in ('boundary','obstacle','collision') else 0. for n,reason in zip(new,reasons))+rescued
        reward-=min(10.,.1*math.log(max(self.step_count,1))*math.log(math.prod(self.shape)-int(self.explored.sum())+1))
        complete=all(s=='rescued' for s in self.target_status.values()) and self.explored.all()
        done=self.time_s>=self.config.duration_s-1e-8 or complete
        if done and not complete:reward-=10*sum(s!='rescued' for s in self.target_status.values())/len(self._targets)
        return self.view(),float(reward),bool(done)

    def training_state(self):
        # Centralized training only, never included in RescueView.
        reference=np.where(self._blocked,-1.,1.)
        for p,s in self.target_status.items():reference[p]=3 if s=='rescued' else 2
        positions=np.asarray(self.positions,dtype=float)/np.asarray(self.shape)
        return np.r_[pooled(reference[None],5).ravel(),pooled(self._current_truth[None],5).ravel(),positions.ravel(),self.time_s/self.config.duration_s].astype(np.float32)

    def result(self):
        quality=float(np.mean([s['metrics']['correct_coverage'] for s in self.snapshots[:-1]]))
        count=sum(s=='rescued' for s in self.target_status.values())
        return dict(schema='hyra-dual-gmrf-rescue-v1',n_agents=2,grid=self.source.process['grid'],source_sha256=self.source.digest,
            config=dict(duration_s=self.config.duration_s,range_m=(self.config.sense_width//2)*self.dx,fov_deg=360.,action_s=self.config.step_s,sample_s=self.config.step_s),rescue_config=asdict(self.config),
            summary=dict(time_average_correct_coverage=quality,undiscovered_change_fraction=None,distance_m=self.distance_m,turn_s=0.,collision_events=self.collision_events,final=self.snapshots[-1]['metrics'],rescued=count,total_targets=len(self._targets),rescue_fraction=count/len(self._targets),explored_fraction=float(self.explored.mean())),
            snapshots=self.snapshots,trajectory=self.trajectory,actions=self.actions,observations=self.observations,changes=[],events=self.events,
            truth=[self.source.labels(s['time_s']).astype(int).ravel().tolist() for s in self.snapshots],
            scenario_truth=dict(obstacles=self._blocked.ravel().astype(int).tolist(),targets=[list(p) for p in self._targets],scope='offline evaluator only; synthetic overlay'),
            audit=dict(gmrf_environment=self.environment_diagnostics,gmrf_signal=self.fields.signal_diagnostics,hydraulic_cache_max_frames=2,actual_frames_only=True,
                execution_permissions='9-channel local/pooled arrived exploration,observed obstacle/environment GMRF/sampled personnel signal,known poses/targets; no hidden target coordinates or hydraulic truth',
                scope='engineering rescue overlay, not full paper reproduction; flood does not constrain personnel survival or rescue feasibility'))

def rule_actions(view):
    """Legal-view BFS toward known victims, then unobserved frontiers."""
    choices=[];reserved=[];nz,nx=view.explored.shape
    for agent,start in enumerate(view.positions):
        goals={(r,c) for r,c,status in view.targets if status=='discovered'}
        queue=deque([(start,None)]);visited={start};best=None;frontier=None
        while queue:
            (r,c),first=queue.popleft()
            if first is not None:
                if (r,c) in goals:best=first;break
                if frontier is None and not view.explored[r,c]:frontier=first
            for a,(dz,dx) in enumerate(MOVES):
                p=(r+dz,c+dx)
                if p in visited or not 0<=p[0]<nz or not 0<=p[1]<nx or view.obstacles[p] or p in reserved:continue
                if dz and dx and (view.obstacles[r,p[1]] or view.obstacles[p[0],c]):continue
                visited.add(p);queue.append((p,a if first is None else first))
            if not goals and frontier is not None:break
        a=best if best is not None else frontier
        if a is None:
            legal=np.flatnonzero(view.available[agent]);a=int(legal[0]) if len(legal) else 0
        choices.append(a);dz,dx=MOVES[a];reserved.append((start[0]+dz,start[1]+dx))
    return choices
