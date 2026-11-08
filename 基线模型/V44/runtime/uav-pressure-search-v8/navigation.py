"""Plan feasible sequences of the eight legal flight directions from shared past measurements."""
import numpy as np
from judge import DELIVERY_RADIUS_M as RADIUS
from approach_policy import DIRECTIONS, approach
from local_tracker import Tracking

# These parameters only change pursuit and scheduling decisions.
PROFILES={
    "brief":(4,25,35,0.),"steady":(5,45,60,0.),"persist":(5,65,95,4.),
    "opportunity":(4,25,60,18.),"mobile":(5,45,90,22.),"quick":(3,20,40,10.)
}

def flight_edges(positions,visible,target,velocity):
    directions=DIRECTIONS[None,:,:]
    starts=positions[:,None,:]
    limits=np.full((len(positions),8,2),np.inf)
    np.divide(1000-starts,directions,out=limits,where=directions>0)
    np.divide(-starts,directions,out=limits,where=directions<0)
    length=limits.min(-1).clip(0.,50.)
    offset=target-starts
    along=(offset*directions).sum(-1)
    side=((offset-along[:,:,None]*directions)**2).sum(-1)
    disk=(.5*RADIUS)**2
    entry=along-np.sqrt(np.maximum(0.,disk-side))
    hit=visible[:,None] & (side<=disk) & (entry>=0) & (entry<=length)
    length=np.where(hit,entry,length)
    ends=starts+length[:,:,None]*directions
    diff=np.clip(target+5*velocity,0,1000)-ends
    square=(diff*diff).sum(-1)
    seen=(square<=400+1e-9)&((diff*directions).sum(-1)>=-1e-9)
    return ends,seen,np.sqrt(square),length

def intercept(pos,target,velocity,visible,depth):
    # Frontier deduplicates converging paths; keep 22 spatially distinct poses.
    positions=pos.reshape(1,2);vis=np.array([visible]);paths=np.empty((1,0),int)
    for step in range(depth):
        ends,seen,dist,length=flight_edges(positions,vis,target+5*step*velocity,velocity)
        distance=dist.ravel()
        routes=np.column_stack((np.repeat(paths,8,axis=0),np.tile(np.arange(8),len(paths))))
        valid=length.ravel()>.01
        arrivals=np.flatnonzero(seen.ravel() & valid & (distance<=RADIUS+1e-9))
        if len(arrivals):
            idx=arrivals[np.argmin(distance[arrivals])]
            return routes[idx].tolist()
        if step+1==depth or not valid.any():break
        flat=ends.reshape(-1,2)
        predicted=target+(step+2)*5*velocity
        diff=predicted-flat
        lateral=np.abs(diff @ DIRECTIONS.T)
        side=np.min(lateral,axis=1)
        # Retain near positions and useful directions for the next intercept.
        cost=distance+.75*side-5*seen.ravel()+np.where(valid,0,1e5)
        order=np.argsort(cost,kind="stable")
        key=np.round(flat[order]*20).astype(int)
        # Visibility matters if the next move can use the short approach.
        keys=np.column_stack((key,seen.ravel()[order]))
        _,idx=np.unique(keys,axis=0,return_index=True)
        kept=order[np.sort(idx)[:22]]
        positions,vis,paths=flat[kept],seen.ravel()[kept],routes[kept]
    return []

class Navigator(Tracking):
    def __init__(self,theta,prior,profile):
        super().__init__(theta,prior,"tracker")
        self.depth,self.budget,self.memory,self.penalty=PROFILES[profile]
        self.plans={}
        self.blocked={}
        self.target_since={}
    def assign(self,o):
        age=o.t-o.seen_at
        claims=[]
        for j in np.flatnonzero(o.active & (o.seen_at>=0)):
            velocity=o.seen_v[j]
            moving=np.linalg.norm(velocity)>.02
            if age[j]>(self.memory if moving else 480):continue
            loc=np.clip(o.seen_xy[j]+velocity*age[j],0,1000)
            delta=loc-o.pos
            distance=np.linalg.norm(delta,axis=1)
            if not o.visible[j] and ((distance<=20)&((delta*o.heading).sum(-1)>=0)).any():
                self.invalid[j]=o.seen_at[j]
            if not o.visible[j] and self.invalid.get(j,-1)==o.seen_at[j]:continue
            for i in np.flatnonzero(distance<(240 if moving else 360)):
                owner=o.owner[j]==i
                if self.track[i]==j and o.t-self.since[i]>=self.budget and not owner:
                    self.blocked[i,j]=max(self.blocked.get((i,j),0),o.t+25)
                if self.blocked.get((i,j),-1)>o.t and not owner:continue
                ready=o.visible[j] and distance[i]<=RADIUS+1e-9
                # Prefer contacts accessible on a legal flight ray, or an existing approach.
                side=np.min(np.abs(DIRECTIONS @ delta[i]))
                costly=min(1.,max(0.,(side-.5*RADIUS)/5.))
                cost=distance[i]+.30*age[j]+self.penalty*costly-12*(self.track[i]==j)
                claims.append((not owner,not ready,cost,i,j,loc))
        assigned=np.full(4,-1,int);goals=o.pos.copy();taken=set()
        for _,_,_,i,j,loc in sorted(claims,key=lambda x:x[:5]):
            if assigned[i]<0 and j not in taken:
                assigned[i]=j;goals[i]=loc;taken.add(j)
        self.since[assigned!=self.track]=o.t
        self.track=assigned.copy()
        return assigned,goals
    def approach_target(self,o,i,j,target,ends):
        pos=o.pos[i];velocity=o.seen_v[j];visible=o.visible[j]
        gap=np.linalg.norm(target-pos)
        if visible and gap<=RADIUS+1e-9:return 8
        entry=self.plans.get(i)
        if entry and entry[0]==j and entry[1]==o.t and np.linalg.norm(entry[2]-pos)<.02 and np.linalg.norm(entry[3]-target)<max(.25,.5*RADIUS):
            plan=entry[4]
        elif gap<90:
            plan=intercept(pos,target,velocity,visible,self.depth)
        else:plan=[]
        if not plan:
            self.plans.pop(i,None)
            return approach(pos,target,velocity,visible,ends)
        action=plan[0]
        predicted=flight_edges(pos.reshape(1,2),np.array([visible]),target,velocity)[0][0,action]
        self.plans[i]=(j,o.t+5,predicted,np.clip(target+5*velocity,0,1000),plan[1:])
        return action
