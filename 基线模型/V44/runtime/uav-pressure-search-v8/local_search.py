"""Sensor-trained spatial search, observed-target dispatch and local traffic coordination."""
import numpy as np
from judge import DELIVERY_RADIUS_M, INNER_SQ
from local_route import Lookahead,endpoints
from approach_policy import DIRECTIONS,approach,candidate_footprints,CENTERS,pool,smooth
from supply import PAIRS,raw_conflicts
from planner import bins,intensity

class Cooperative:
    def __init__(self,theta,prior,mode):
        self.legacy=Lookahead(theta,[1,4,35],[.5,0,2])
        self.mode=mode
        self.prior=prior/prior.mean()
        self.lookup=intensity(theta)
        self.invalid={}
        self.observed=set()
        self.contacts=np.zeros((20,20))
        self.wet=None
        self.attempt={}
        self.grid=np.zeros(10000)
        self.index=np.indices((20,20))

    def assign(self,o):
        actions=np.full(4,-1,int)
        goals=o.pos.copy()
        claims=[]
        for j in np.flatnonzero(o.active & (o.seen_at>=0)):
            age=o.t-o.seen_at[j]
            vel=o.seen_v[j]
            if age>(200 if np.linalg.norm(vel)<.03 else 35):continue
            prediction=np.clip(o.seen_xy[j]+age*vel,0,1000)
            if not o.visible[j]:
                rel=prediction-o.pos
                if ((np.linalg.norm(rel,axis=1)<=20)&((rel*o.heading).sum(-1)>=0)).any():
                    self.invalid[j]=o.seen_at[j]
                if self.invalid.get(j,-1)==o.seen_at[j]:continue
            dist=np.linalg.norm(o.pos-prediction,axis=1)
            for i in np.flatnonzero(dist<240):
                if self.attempt.get((i,j),(-1,0))[1]>=3 and self.attempt[i,j][0]+30>=o.t and o.owner[j]!=i:continue
                cost=dist[i]+(.3*age)+(min(np.linalg.norm(vel),3)*13 if not self.wet else 0)
                claims.append((o.owner[j]!=i,not(o.visible[j] and dist[i]<=DELIVERY_RADIUS_M+1e-7),cost,i,j,prediction))
        claimed=set()
        for _,_,_,i,j,xy in sorted(claims,key=lambda x:x[:5]):
            if actions[i]<0 and j not in claimed:
                actions[i]=j;goals[i]=xy;claimed.add(j)
        return actions,goals

    def search(self,o,ends,dist,assigned,targets):
        for j in np.flatnonzero(o.visible):
            if j not in self.observed:
                self.observed.add(j)
                x,y=(o.seen_xy[j]//50).astype(int).clip(0,19)
                self.contacts[y,x]+=1
        ids=o.observed_cells
        h,u,v=o.measured.reshape(3,-1)[:,ids]
        self.grid[ids]=self.lookup[bins(h,np.hypot(u,v))]
        known=o.last>=0
        cov=pool(known)
        q=pool(self.grid.reshape(100,100))
        rate=(q+.8*smooth(q)+.2*self.lookup.mean())/(cov+.8*smooth(cov)+.2)
        signal=np.minimum(5.,rate/max(rate.mean(),1e-12))
        local=smooth(smooth(self.contacts))
        # A newly sighted cluster redirects remaining search to its surroundings.
        power,mult=(1.0,4.) if self.mode=="spatial" else (1.5,10.)
        density=.12+np.power(self.prior,power)*(1+.45*signal)+mult*local
        age=np.where(known,np.minimum(.35,(o.t-o.last)/1300.),1.)
        mass=np.repeat(np.repeat(density,5,0),5,1)*age
        cells,mask=candidate_footprints(ends,DIRECTIONS)
        raw=(mass.ravel()[cells]*mask).sum(-1)
        score=raw/(np.maximum(1,raw.max(axis=1,keepdims=True)))
        utility=pool(mass);reservation=np.ones((20,20))
        for i in np.flatnonzero(assigned>=0):
            reservation*=.45+.55*np.clip(np.linalg.norm(CENTERS-targets[i],axis=-1)/85,0,1)
        for i in np.flatnonzero(assigned<0):
            dc=np.linalg.norm(CENTERS-o.pos[i],axis=-1)
            attraction=utility*reservation/(1+dc/270.)
            # Once near the most valuable block, use physical view gain inside it.
            aim=CENTERS[np.unravel_index(attraction.argmax(),attraction.shape)]
            delta=np.linalg.norm(aim-o.pos[i])-np.linalg.norm(ends[i]-aim,axis=-1)
            score[i]+=np.clip(delta/50.,-1,1)*(1.65 if np.linalg.norm(o.pos[i]-aim)>65 else .42)
            reservation*=.16+.84*np.clip(np.linalg.norm(CENTERS-aim,axis=-1)/105,0,1)
        score[dist<1]=-99
        return score

    def approach_target(self,o,i,j,target,ends):
        return approach(o.pos[i],target,o.seen_v[j],o.visible[j],ends)

    @staticmethod
    def safe(o,actions,assigned,ends):
        # Same observable ray, approach and boundary rules as the physical simulator.
        speed=np.zeros((4,2));stop=np.full(4,5.)
        for i,(a,j) in enumerate(zip(actions,assigned)):
            if a==8:
                if j>=0 and o.visible[j]:
                    v=o.seen_v[j];speed[i]=v*min(1.,10/max(1.e-12,np.linalg.norm(v)))
            else:
                speed[i]=10*DIRECTIONS[a]
                stop[i]=np.linalg.norm(ends[i,a]-o.pos[i])/10.
                if j>=0 and o.visible[j]:
                    delta=o.seen_xy[j]-o.pos[i]
                    along=delta@DIRECTIONS[a]
                    r2=delta@delta-along*along
                    if delta@delta>INNER_SQ+1.e-6 and r2<=INNER_SQ and along>=0:
                        stop[i]=min(stop[i],max(0.,(along-np.sqrt(max(0.,INNER_SQ-r2)))/10))
        bound=np.full((4,2),np.inf)
        np.divide(1000-o.pos,speed,out=bound,where=speed>0)
        np.divide(-o.pos,speed,out=bound,where=speed<0)
        stop=np.maximum(0,np.minimum(stop,bound.min(axis=1)))
        knots=np.unique(np.r_[0.,5.,stop])
        tracks=o.pos+np.minimum(knots[:,None],stop)[...,None]*speed
        return raw_conflicts(tracks)

    def act(self,o):
        if self.wet is None and (len(np.flatnonzero(o.last>=0))>=40 or o.t>=60):
            self.wet=float(np.mean(o.measured[0][o.last>=0]>=.01))>=.8
        ends,length=endpoints(o.pos)
        use_original=self.wet or self.mode=="dispatch"
        if use_original:
            actions,_=self.legacy.act(o)
        assigned,targets=self.assign(o)
        if not use_original:
            score=self.search(o,ends,length,assigned,targets)
            actions=score.argmax(axis=1)
        for i,j in enumerate(assigned):
            if j>=0:
                actions[i]=self.approach_target(o,i,j,targets[i],ends[i])
                if actions[i]==8 and o.visible[j]:
                    prev,k=self.attempt.get((i,j),(-100,0))
                    self.attempt[i,j]=(o.t, k+1 if prev+5==o.t and o.owner[j]!=i else 1)
        conflicts=self.safe(o,actions,assigned,ends)
        for _ in range(4):
            if not len(conflicts):break
            left,right=PAIRS[:,conflicts[0]]
            def priority(i):
                j=assigned[i]
                return (j>=0 and o.owner[j]==i,j>=0 and actions[i]==8,j>=0,-i)
            choices=sorted((left,right),key=priority)
            change=None
            for i in choices:
                anchor=targets[i] if assigned[i]>=0 else ends[i,min(actions[i],7)]
                # A yielding UAV may wait without claiming service or try a nearby heading.
                alternatives=[8]+list(np.argsort(np.linalg.norm(ends[i]-anchor,axis=1)))
                for alt in alternatives:
                    trial=actions.copy();trial[i]=alt
                    claims=assigned.copy()
                    if alt==8:claims[i]=-1
                    test=self.safe(o,trial,claims,ends)
                    if len(test)<len(conflicts):
                        change=(trial,claims,test);break
                if change:break
            if change is None:break
            actions,assigned,conflicts=change
        return actions,assigned

