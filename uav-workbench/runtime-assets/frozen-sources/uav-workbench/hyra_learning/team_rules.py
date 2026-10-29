"""Legal shared-map rule competition with fixed row partitions and safety."""
import itertools
import math
import numpy as np

from hyra_uav.core import ACTIONS, footprint, move
from hyra_uav.policies import ScanPolicy
from .team_env import path_separation


class TeamRules:
    def __init__(self,name,view,config,separation_m=2.):
        if name not in ('R0','R1','R2') or len(view.agents)!=2:
            raise ValueError('Registered rule competition uses R0/R1/R2 with two agents')
        self.name,self.config,self.separation = name,config,separation_m
        self.scans=[]
        rows=view.bed.shape[0]
        for i,agent in enumerate(view.agents):
            low,high=(0,rows//2-1) if i==0 else (rows//2,rows-1)
            step=config.speed_m_s*config.action_s
            xs=list(np.arange(0,(agent.shape[1]-1)*agent.dx,step))+[(agent.shape[1]-1)*agent.dx]
            zs=list(np.arange(low*agent.dx,high*agent.dx,step))+[high*agent.dx]
            scan=ScanPolicy(agent,config)
            scan.waypoints=[(float(x),float(z)) for j,z in enumerate(zs) for x in (xs if j%2==0 else xs[::-1])]
            scan.waypoints+=scan.waypoints[-2:0:-1]
            scan.index=min(range(len(scan.waypoints)),key=lambda k:math.hypot(scan.waypoints[k][0]-agent.pose.x_m,scan.waypoints[k][1]-agent.pose.z_m))
            self.scans.append(scan)

    def safe(self,views,actions):
        poses=[v.pose for v in views]
        for _ in range(round(self.config.action_s/self.config.sample_s)):
            next_=[move(p,ACTIONS[a],self.config.sample_s,self.config,views[0].shape,views[0].dx)[0] for p,a in zip(poses,actions)]
            if path_separation(*poses,*next_)<self.separation-1e-9:return False
            poses=next_
        return True

    def choose(self,view):
        scores=[]
        for i,agent in enumerate(view.agents):
            if self.name=='R0':
                desired=self.scans[i].choose(agent,self.config)
                scores.append([1000. if a==desired else 0. for a in ACTIONS]);continue
            known=agent.labels>=0
            age=np.maximum(0,agent.time_s-agent.observed_s)/self.config.duration_s
            utility=np.where(known,age*self.config.age_weight if self.name=='R2' else 0.,1.)
            region=np.indices(agent.shape)[0]<(agent.shape[0]//2)
            if i==1:region=~region
            utility=utility*region
            targets=np.argwhere((~known)&region)
            by_first=[]
            for first in ACTIONS:
                best=-float('inf')
                for second in ACTIONS:
                    pose,seen,score=agent.pose,np.zeros(agent.shape,dtype=bool),0.
                    for depth,action in enumerate((first,second)):
                        for _ in range(round(self.config.action_s/self.config.sample_s)):
                            pose,_=move(pose,action,self.config.sample_s,self.config,agent.shape,agent.dx)
                            visible=footprint(pose,self.config,agent.shape,agent.dx)
                            score+=float(utility[visible&~seen].sum())*(.8**depth);seen|=visible
                    if len(targets):score-=.025*np.hypot(targets[:,1]*agent.dx-pose.x_m,targets[:,0]*agent.dx-pose.z_m).min()/agent.dx
                    best=max(best,score)
                by_first.append(best)
            scores.append(by_first)
        candidates=[a for a in itertools.product(range(4),repeat=2) if self.safe(view.agents,a)]
        if not candidates:return (3,3)
        return max(candidates,key=lambda a:scores[0][a[0]]+scores[1][a[1]])
