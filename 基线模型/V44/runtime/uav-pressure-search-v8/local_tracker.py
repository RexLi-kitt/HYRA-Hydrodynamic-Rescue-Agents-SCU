"""Risk-aware search and reassignment driven by observed contacts and failed attempts."""
import numpy as np
from judge import DELIVERY_RADIUS_M
from local_search import Cooperative
from local_route import Lookahead
from planner import CENTERS,pool,smooth,bins
from approach_policy import DIRECTIONS,candidate_footprints

CONFIG={"tracker":(.8,2,40),"urgent":(1.4,4,25),"spread":(-.6,5,40)}
class Tracking(Cooperative):
    def __init__(self,theta,prior,version):
        super().__init__(theta,prior,"concentrated")
        self.bias,self.frontier,self.pursuit=CONFIG[version]
        self.legacy=Lookahead(theta,[1,4,35],[.6,self.bias,self.frontier])
        self.track=np.full(4,-1)
        self.since=np.zeros(4,int)
        self.rest={}
        self.deadreckoning={}
        self.contact_map=np.zeros((20,20))
        self.first_seen=set()
        self.last_actions=np.zeros(4,int)
        self.sector_failed=np.zeros(4,int)
    def assign(self,o):
        age=o.t-o.seen_at
        claims=[]
        for j in np.flatnonzero(o.active & (o.seen_at>=0)):
            v=o.seen_v[j]
            stationary=np.linalg.norm(v)<.002
            if age[j]>(480 if stationary else 30):continue
            predicted=np.clip(o.seen_xy[j]+v*age[j],0,1000)
            if not o.visible[j]:
                ray=predicted-o.pos
                if ((np.linalg.norm(ray,axis=1)<=20)&((ray*o.heading).sum(-1)>=0)).any():
                    self.invalid[j]=o.seen_at[j]
                if self.invalid.get(j,-1)==o.seen_at[j]:continue
            distances=np.linalg.norm(o.pos-predicted,axis=1)
            for i in np.flatnonzero(distances<(320 if stationary else 180)):
                continuing=o.owner[j]==i
                if self.track[i]==j and o.t-self.since[i]>=self.pursuit and not continuing:
                    self.rest[i,j]=o.t+45
                if self.rest.get((i,j),-1)>o.t and not continuing:continue
                bonus=0 if (self.wet or stationary) else 28
                merit=distances[i]+bonus+.24*age[j]
                claims.append((not continuing, not(o.visible[j] and distances[i]<=DELIVERY_RADIUS_M+1e-7 and (stationary or self.wet)),merit,i,j,predicted))
        assigned=np.full(4,-1,int);goals=o.pos.copy();used=set()
        for _,_,_,i,j,xy in sorted(claims,key=lambda v:v[:5]):
            if assigned[i]<0 and j not in used:
                assigned[i]=j;goals[i]=xy;used.add(j)
        self.since[assigned!=self.track]=o.t
        self.track=assigned.copy()
        return assigned,goals
    def search(self,o,ends,dist,assigned,targets):
        ids=o.observed_cells
        h,u,v=o.measured.reshape(3,-1)[:,ids]
        self.grid[ids]=self.lookup[bins(h,np.hypot(u,v))]
        for j in np.flatnonzero(o.visible):
            if j in self.first_seen:continue
            self.first_seen.add(j)
            p=o.seen_xy[j]
            radius=65 if o.t<500 else 90
            self.contact_map+=np.exp(-np.square(CENTERS-p).sum(-1)/(2*radius**2))
        known=o.last>=0
        cov=pool(known)
        q=pool(self.grid.reshape(100,100))
        signal=(q+smooth(q)+.12*self.lookup.mean())/(cov+smooth(cov)+.12)
        signal/=max(signal.mean(),1e-10)
        centre=self.contact_map/max(1.,self.contact_map.max())
        if len(self.first_seen)>=2:
            density=.05+self.prior**.95*(.22+4.*centre)+4.5*centre+np.minimum(signal,10)*(.15+centre)
        else:density=.15+self.prior**1.3+.65*np.minimum(signal,8)
        age=np.where(known,np.minimum(.30,(o.t-o.last)/1000),1.)
        mass=np.repeat(np.repeat(density,5,0),5,1)*age
        cells,mask=candidate_footprints(ends,DIRECTIONS)
        local=(mass.ravel()[cells]*mask).sum(-1)
        future=local/np.maximum(1.,local.max(axis=1,keepdims=True))
        coarse=pool(mass);reserved=np.ones_like(coarse)
        for i in np.flatnonzero(assigned>=0):
            reserved*=.55+.45*np.clip(np.linalg.norm(CENTERS-targets[i],axis=-1)/85,0,1)
        for i in np.flatnonzero(assigned<0):
            dc=np.linalg.norm(CENTERS-o.pos[i],axis=-1)
            utility=coarse*reserved/(1+dc/250)
            goal=CENTERS[np.unravel_index(utility.argmax(),utility.shape)]
            d=np.linalg.norm(o.pos[i]-goal)
            future[i]+=(d-np.linalg.norm(ends[i]-goal,axis=-1))/50*(2 if d>70 else .45)
            reserved*=.1+.9*np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/120,0,1)
        future[dist<1]=-99
        return future

