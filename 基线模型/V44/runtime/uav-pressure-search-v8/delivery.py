"""Person-centred radius study, centred on the current recipient in each simulated second."""
import numpy as np
from mission_env import MissionSupply
from population import MOVES, EPS, segment_circle
from supply import intentions
from judge import within_radius, DELIVERY_RADIUS_M

class DeliveryMission(MissionSupply):
    def __init__(self, water, population, mission):
        super().__init__(water, population, mission)
        self.radius = DELIVERY_RADIUS_M
        self.in_range = np.zeros(self.count,bool)
        self.followed = np.zeros(self.count,bool)
        self.outside_during_follow = np.zeros(self.count,bool)
        self.first_five = np.zeros(self.count,bool)

    def step(self,actions,assigned):
        if self.t>=self.duration:raise ValueError("Mission ended")
        actions=np.asarray(actions)
        if actions.shape!=(4,) or not np.issubdtype(actions.dtype,np.integer) or np.any((actions<0)|(actions>8)):
            raise ValueError("Invalid joint action")
        pos=self.pos
        ids=np.flatnonzero(self.visible)
        if len(ids):
            gap=self.seen_xy[ids,None,:]-pos[None,:,:]
            self.in_range[ids] |= ((gap*gap).sum(-1) <= (self.radius+1e-9)**2).any(axis=1)
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
        next_owner=np.full(self.count,-1,int)
        next_progress=np.zeros(self.count,int)
        for i,j in enumerate(assigned):
            if j<0 or actions[i]!=8 or stopped[i] or not self.visible[j]:continue
            end=self.t+5
            if not self.truth['available'][self.t:end+1,j].all() or self.deadline[j]<end:continue
            path=self.truth['points'][self.t:end+1,j]
            if np.linalg.norm(track[0,i]-path[0])<=self.radius+1e-9:
                self.followed[j]=True
                if not within_radius(track[:,i],path):self.outside_during_follow[j]=True
            if not within_radius(track[:,i],path):continue
            self.first_five[j]=True
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

