"""Independent 100m flood task contract; never changes the registered pilot code."""
import io,json,math,zipfile
from dataclasses import dataclass,replace
from pathlib import Path
import numpy as np
from sur_rescue.environment import RescueEnv,RescueConfig,RescueView,ACTIONS,MOVES,joint_motion,pooled,frozen
from sur_rescue.hydraulics import HydraulicFrames
from sur_rescue.gmrf import DualGMRF

class FloodCase(HydraulicFrames):
    def __init__(self,path):
        super().__init__(path)
        with zipfile.ZipFile(path) as a:
            self.bed=np.load(io.BytesIO(a.read('bed.npy')),allow_pickle=False)
            self.case=json.loads(a.read('case.json'))
            self.probes={k:np.load(io.BytesIO(a.read('probe_'+k+'.npy')),allow_pickle=False) for k in ('before','after','dt','source')}
        self._values={}
    def values(self,t):
        i=self.index(t)
        if i not in self._values:
            name=self.process['frames'][i]['url']
            with zipfile.ZipFile(self.path) as a:v=np.frombuffer(a.read(name),dtype='<f4').reshape(4,*self.shape).copy()
            v.setflags(write=False);self._values[i]=v
            if len(self._values)>2:self._values.pop(next(iter(self._values)))
        return self._values[i]
    def labels(self,t):return self.values(t)[0]>=.05

@dataclass(frozen=True)
class ScaleConfig(RescueConfig):
    duration_s:float=900.
    n_agents:int=2
    task:str='E1'
    representation:str='gmrf'
    service_depth_limit_m:float=.08
    service_speed_limit_m_s:float=.3
    def validate(self,source):
        if self.task not in ('E1','E2') or self.representation not in ('gmrf','raw'):raise ValueError('Invalid task or representation')
        if self.n_agents not in (2,4):raise ValueError('Supported fleet sizes: 2 or 4')
        if source.shape!=(20,20) or source.dx!=5:raise ValueError('100m training requires original 20x20 cells, dx=5m')
        if self.task=='E1' and self.n_targets!=0:raise ValueError('E1 must have no people')
        if self.task=='E2' and self.n_targets!=5:raise ValueError('E2 requires five personnel')
        if self.service_depth_limit_m<=0 or self.service_speed_limit_m_s<=0:raise ValueError('Invalid synthetic service limits')
        RescueConfig(**{k:getattr(self,k) for k in RescueConfig.__dataclass_fields__}|dict(n_targets=max(1,self.n_targets))).validate(source)

class ScaleEnv(RescueEnv):
    def __init__(self,source,config):
        config.validate(source);self.source,self.config=source,config;self.shape=source.shape;self.dx=source.dx
        n=config.n_agents;self.initial_positions=np.array([(0,0),(19,0),(0,19),(19,19)][:n]);rng=np.random.default_rng(config.seed)
        self._blocked=np.zeros(self.shape,dtype=bool);valid=np.ones(self.shape,dtype=bool)
        for p in self.initial_positions:valid[tuple(p)]=False
        ids=np.flatnonzero(valid);self._blocked.flat[rng.choice(ids,int(valid.sum()*config.obstacle_fraction),replace=False)]=True
        # Every free cell is reachable; reject isolated obstacle draws deterministically.
        for _ in range(100):
            seen={tuple(self.initial_positions[0])};todo=list(seen)
            while todo:
                r,c=todo.pop()
                for dz,dx in MOVES[:4]:
                    p=(r+dz,c+dx)
                    if 0<=p[0]<20 and 0<=p[1]<20 and not self._blocked[p] and p not in seen:seen.add(p);todo.append(p)
            if len(seen)==int((~self._blocked).sum()):break
            self._blocked[:]=False;self._blocked.flat[rng.choice(ids,int(valid.sum()*config.obstacle_fraction),replace=False)]=True
        else:raise ValueError('Unable to generate reachable scenario')
        safe=valid&~self._blocked&(source.values(0)[0]<.2);ids=np.flatnonzero(safe)
        self._targets=[divmod(int(i),20) for i in rng.choice(ids,config.n_targets,replace=False)] if config.n_targets else []
        self.fields=DualGMRF(self.shape,self._blocked,self._targets,config.seed)
    def reset(self):
        self.positions=self.initial_positions.copy();self.time_s=0.;self.step_count=0;n=self.config.n_agents
        self.explored=np.zeros(self.shape,dtype=bool);self.obstacles=np.zeros(self.shape,dtype=bool);self.signal=np.zeros(self.shape)
        self.labels=np.full(self.shape,-1,dtype=np.int8);self.observed_s=np.full(self.shape,-1.)
        self.water=np.zeros((3,*self.shape));self.target_status={p:'undiscovered' for p in self._targets};self.busy=[None]*n
        self.failed=set();self.distance_m=0.;self.collision_events=0;self.actions=[];self.observations=[];self.snapshots=[];self.trajectory=[];self.events=[]
        self.dynamic_gain=0.;self._sample(['initial']*n,['initial']*n);return self.view()
    def _sample(self,actions,reasons):
        truth=self.source.labels(self.time_s);old_correct=(self.labels==truth)&self.explored;start=len(self.observations)
        new=super()._sample(actions,reasons);fields=self.source.values(self.time_s)
        for obs in self.observations[start:]:
            cells=obs['cells'];self.water.reshape(3,-1)[:,cells]=fields[:3].reshape(3,-1)[:,cells]
            obs['depth_m']=fields[0].ravel()[cells].tolist();obs['u_m_s']=fields[1].ravel()[cells].tolist();obs['v_m_s']=fields[2].ravel()[cells].tolist()
        # Evaluator reward uses truth; actor sees only measurements taken now or earlier.
        self.dynamic_gain=float((((self.labels==truth)&self.explored)&~old_correct).sum()-sum(new))
        return new
    def view(self):
        pos=np.zeros(self.shape);known=[]
        for p,s in self.target_status.items():
            if s!='undiscovered':pos[p]=3 if s=='rescued' else 2;known.append((*p,s))
        for p in self.positions:pos[tuple(p)]=1
        raw=np.where(self.explored,np.where(self.obstacles,-1.,1.),0.)
        base=np.stack((self.explored&~self.obstacles,self.obstacles,self.environment_field if self.config.representation=='gmrf' else raw,self.signal)).astype(np.float32)
        age=np.where(self.explored,(self.time_s-self.observed_s)/self.config.duration_s,0.)
        # h,u,v, validity, age: all groups get exactly the same arrived raw water records.
        hydro=np.concatenate((self.water*np.array([1.,1.,1.])[:,None,None],self.explored[None],age[None])).astype(np.float32)
        bed=(self.source.bed-self.source.bed.mean())[None].astype(np.float32)
        pad=np.pad(base,((0,0),(2,2),(2,2)));hp=np.pad(hydro,((0,0),(2,2),(2,2)));bp=np.pad(bed,((0,0),(2,2),(2,2)))
        images=np.stack([np.concatenate((pad[:,r:r+5,c:c+5],pooled(base),pooled(pos[None]),hp[:,r:r+5,c:c+5],pooled(hydro),bp[:,r:r+5,c:c+5],pooled(bed))) for r,c in self.positions])
        available=np.ones((self.config.n_agents,8),dtype=bool)
        for i,(r,c) in enumerate(self.positions):
            for a,(dz,dx) in enumerate(MOVES):
                rr,cc=r+dz,c+dx;available[i,a]=0<=rr<20 and 0<=cc<20
                if available[i,a]:available[i,a]=not self.obstacles[rr,cc] and not(dz and dx and(self.obstacles[r,cc] or self.obstacles[rr,c]))
        return RescueView(frozen(images.astype(np.float32)),frozen(available),tuple(map(tuple,self.positions)),frozen(self.explored),frozen(self.obstacles),frozen(self.signal),tuple(known),tuple(b['remaining'] if b else 0 for b in self.busy),self.step_count)
    def step(self,actions):
        if self.time_s>=self.config.duration_s:raise ValueError('Episode ended')
        if len(actions)!=self.config.n_agents or any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<8 for a in actions):raise ValueError('Invalid eight-direction actions')
        before=int(self.explored.sum());old=self.positions.copy();busy=[b is not None for b in self.busy]
        self.positions,reasons,conflicts=joint_motion(old,actions,self._blocked,busy);self.collision_events+=conflicts
        self.distance_m+=float(np.linalg.norm(self.positions-old,axis=1).sum()*self.dx)
        self.actions.append(dict(time_s=self.time_s,actions=[ACTIONS[a] for a in actions]));self.step_count+=1;self.time_s=self.step_count*self.config.step_s
        values=self.source.values(self.time_s);rescued=0
        for i,p_ in enumerate(self.positions):
            p=tuple(p_);b=self.busy[i]
            # Synthetic service feasibility, not a flight restriction or a human mortality model.
            safe=values[0][p]<=self.config.service_depth_limit_m and np.hypot(values[1][p],values[2][p])<=self.config.service_speed_limit_m_s
            if b:
                if safe:
                    b['remaining']=max(0,b['remaining']-self.config.step_s)
                    if b['remaining']==0:self.target_status[p]='rescued';self.busy[i]=None;rescued+=1;self.events.append(dict(type='rescued',cell=list(p),time_s=self.time_s))
                else:self.target_status[p]='discovered';self.busy[i]=None;self.events.append(dict(type='service_interrupted',cell=list(p),time_s=self.time_s))
            if not busy[i] and p in self.target_status and self.target_status[p] not in ('rescued','rescuing') and safe:
                self.target_status[p]='rescuing';self.busy[i]=dict(cell=p,remaining=self.config.service_s)
        new=self._sample([ACTIONS[a] for a in actions],reasons)
        # Both tasks optimize actual sampling; the E2 difference is service reward and unfinished-person cost.
        reward=.1*sum(new)+.15*self.dynamic_gain-.2*sum(r in ('boundary','obstacle','collision') for r in reasons)+5*rescued
        done=self.time_s>=self.config.duration_s
        if done and self.config.task=='E2':reward-=2*sum(s!='rescued' for s in self.target_status.values())
        return self.view(),float(reward),done
    def metrics(self):
        scores=[s['metrics']['correct_coverage'] for s in self.snapshots]
        return dict(current_correct_coverage_auc=float(np.mean(scores[:-1])),first_coverage=float(self.explored.mean()),rescued=sum(s=='rescued' for s in self.target_status.values()),targets=len(self._targets),distance_m=self.distance_m,collision_events=self.collision_events,first_90_s=next((s['time_s'] for s in self.snapshots if np.mean(s['explored'])>=.9),None))

def config(task,method,seed,n_agents=2):return ScaleConfig(task=task,n_targets=0 if task=='E1' else 5,representation='raw' if method in ('B0','B2') else 'gmrf',seed=seed,n_agents=n_agents)
