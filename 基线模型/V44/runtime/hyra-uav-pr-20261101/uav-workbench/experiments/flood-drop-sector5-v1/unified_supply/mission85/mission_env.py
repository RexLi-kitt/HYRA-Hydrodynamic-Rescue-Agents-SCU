"""Versioned instance-parameterised mission. The frozen v1 files remain hash-verifiable."""
import numpy as np
from population import START, MOVES, EPS, segment_circle
from supply import Supply as FrozenSupply, intentions, align_service

class MissionSupply(FrozenSupply):
    def __init__(self,water,population,mission):
        self.count,self.window,self.duration=mission
        if not (0 < self.count <= 80 and 0 < self.window < self.duration <= 2400 and self.duration % 5 == 0):
            raise ValueError("Invalid mission")
        if population["points"].shape[1]!=self.count or len(water)*5-5<self.duration:
            raise ValueError("Population or map does not cover the mission")
        self.water=water
        self.truth=population
        self.radius=10
        self.t=0
        self.pos=START.copy()
        self.heading=500-self.pos
        self.heading/=np.linalg.norm(self.heading,axis=1)[:,None]
        self.last=np.full((100,100),-1,int)
        self.measured=np.zeros((3,100,100))
        self.seen_xy=np.zeros((self.count,2))
        self.seen_v=np.zeros((self.count,2))
        self.seen_at=np.full(self.count,-1,int)
        self.visible=np.zeros(self.count,bool)
        self.active=np.zeros(self.count,bool)
        self.owner=np.full(self.count,-1,int)
        self.progress=np.zeros(self.count,int)
        self.supplied=np.full(self.count,-1,int)
        self.deadline=np.minimum(self.duration,population['fall']+self.window)
        self.conflicts=0
        self.boundary=0
        self.sample()

    def step(self,actions,assigned):
        if self.t>=self.duration:raise ValueError("Mission ended")
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
        next_owner=np.full(self.count,-1,int)
        next_progress=np.zeros(self.count,int)
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
        return dict(delivered=int(ok.sum()),total=self.count,all=bool(ok.all()),
            ST=self.t//5,ACC=self.conflicts/max(1,self.t//5),RCR=float(ok.mean()),
            FRT=int(self.supplied[ok].min()//5) if ok.any() else None,
            CRT=int(self.supplied.max()//5) if ok.all() else None,coverage=float((self.last>=0).mean()),
            discovered=int((self.seen_at>=0).sum()),conflicts=self.conflicts,boundary=self.boundary,
            actual_falls=int(fell.sum()),delivered_standing=int((ok&~fell).sum()),delivered_after_fall=int((ok&fell).sum()),
            censored_window=int((fell&(fall+self.window>self.duration)).sum()),
            timeout=not bool(ok.all() and np.all(self.last>=0)),delivery_s=self.supplied.tolist())

