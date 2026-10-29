"""Task adapters for the unified eight-direction workbench.

The registered sur_rescue package and its checkpoints are left unchanged.
Exploration has no personnel, signal source, rescue dwell or rescue penalty.
"""
from dataclasses import dataclass,asdict,replace
import math
import numpy as np
from sur_rescue.environment import RescueEnv,RescueConfig,ACTIONS,joint_motion
from sur_rescue.gmrf import DualGMRF

@dataclass(frozen=True)
class ExplorationConfig(RescueConfig):
    n_targets:int=0
    def validate(self,source):
        if type(self.n_targets) is not int or self.n_targets!=0:raise ValueError('自主探索不设置待救人员')
        if not math.isfinite(self.service_s):raise ValueError('无效时间参数')
        # Only reuse the common validation contract; no placeholder target is generated.
        RescueConfig(**dict(asdict(self),n_targets=1)).validate(source)

class ExplorationEnv(RescueEnv):
    def __init__(self,source,config=ExplorationConfig(),scenario=None):
        config.validate(source);self.source,self.config=source,config;self.shape=source.shape;self.dx=source.dx
        self.initial_positions=np.array([[0,0],[self.shape[0]-1,0]],dtype=int)
        if scenario is None:
            rng=np.random.default_rng(config.seed);available=np.ones(self.shape,dtype=bool)
            for r,c in self.initial_positions:available[r,c]=False
            ids=np.flatnonzero(available);blocked=np.zeros(self.shape,dtype=bool)
            blocked.flat[rng.choice(ids,int(available.sum()*config.obstacle_fraction),replace=False)]=True
        else:
            if scenario.get('targets'):raise ValueError('自主探索场景不得包含人员目标')
            blocked=np.asarray(scenario['obstacles'],dtype=bool)
        if blocked.shape!=self.shape or any(blocked[r,c] for r,c in self.initial_positions):raise ValueError('障碍图或起点无效')
        self._blocked=blocked.copy();self._targets=[];self.fields=DualGMRF(self.shape,self._blocked,[],config.seed);self._done=False
    def reset(self):
        view=super().reset();self._done=bool(self.explored.all());return view
    def step(self,actions):
        if self._done or self.time_s>=self.config.duration_s-1e-8:raise ValueError('探索任务已经结束')
        if len(actions)!=2:raise ValueError('每架无人机需要一个八方向动作')
        actions=[ACTIONS.index(a) if isinstance(a,str) and a in ACTIONS else a for a in actions]
        if any(isinstance(a,(bool,np.bool_)) or not isinstance(a,(int,np.integer)) or not 0<=a<8 for a in actions):raise ValueError('无效的八方向动作')
        before=int(self.explored.sum());old=self.positions.copy()
        self.positions,reasons,conflicts=joint_motion(old,actions,self._blocked,[False,False])
        self.distance_m+=float(np.linalg.norm(self.positions-old,axis=1).sum()*self.dx);self.collision_events+=conflicts
        self.actions.append(dict(time_s=self.time_s,actions=[ACTIONS[a] for a in actions]));self.step_count+=1;self.time_s=round(self.step_count*self.config.step_s,9)
        new=self._sample([ACTIONS[a] for a in actions],reasons)
        exploration=sum(.1*(1+before/math.prod(self.shape))*n if reason=='move' else -1. if reason in ('boundary','obstacle','collision') else 0. for n,reason in zip(new,reasons))
        delay=min(10.,.1*math.log(max(self.step_count,1))*math.log(math.prod(self.shape)-int(self.explored.sum())+1))
        self._done=bool(self.explored.all() or self.time_s>=self.config.duration_s-1e-8)
        return self.view(),float(exploration-delay),self._done
    def result(self):
        snapshots=self.snapshots;times=np.array([s['time_s'] for s in snapshots]);scores=np.array([s['metrics']['correct_coverage'] for s in snapshots])
        coverage=np.array([np.mean(s['explored']) for s in snapshots]);duration=float(times[-1])
        quality=float(np.dot(scores[:-1],np.diff(times))/duration) if duration else float(scores[0])
        average=float(np.dot(coverage[:-1],np.diff(times))/duration) if duration else float(coverage[0])
        return dict(schema='hyra-exploration-run-v1',n_agents=2,grid=self.source.process['grid'],source_sha256=self.source.digest,
            config=dict(duration_s=self.config.duration_s,range_m=(self.config.sense_width//2)*self.dx,fov_deg=360.,action_s=self.config.step_s,sample_s=self.config.step_s),rescue_config=asdict(self.config),
            summary=dict(time_average_correct_coverage=quality,time_average_first_coverage=average,first_90_percent_coverage_s=next((s['time_s'] for s in snapshots if np.mean(s['explored'])>=.9),None),undiscovered_change_fraction=None,distance_m=self.distance_m,turn_s=0.,collision_events=self.collision_events,
                final=snapshots[-1]['metrics'],rescued=0,total_targets=0,rescue_fraction=None,explored_fraction=float(self.explored.mean())),
            snapshots=snapshots,trajectory=self.trajectory,actions=self.actions,observations=self.observations,changes=[],events=[],
            truth=[self.source.labels(s['time_s']).astype(int).ravel().tolist() for s in snapshots],
            scenario_truth=dict(obstacles=self._blocked.ravel().astype(int).tolist(),targets=[],scope='offline evaluator only; synthetic obstacles'),
            audit=dict(gmrf_environment=self.environment_diagnostics,gmrf_signal=self.fields.signal_diagnostics,hydraulic_cache_max_frames=2,actual_frames_only=True,
                exploration_only=True,rescue_reward_enabled=False,personnel_signal_sources=0,
                execution_permissions='same legal arrived exploration records and known poses; no personnel or rescue state',
                scope='engineering exploration; hydraulic prediction and physics auxiliary not yet enabled'))

def make_env(source,task,config):
    if task=='exploration':return ExplorationEnv(source,ExplorationConfig(**config))
    if task=='exploration_rescue':return RescueEnv(source,RescueConfig(**config))
    raise ValueError('未知任务类型')
