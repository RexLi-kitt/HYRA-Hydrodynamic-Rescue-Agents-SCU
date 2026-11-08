"""Evaluate stationary intercepts and coordinated search with only shared current/past observations."""
import numpy as np
from navigation import Navigator,flight_edges,DIRECTIONS
from local_route import endpoints
from local_search import Cooperative,PAIRS
from judge import DELIVERY_RADIUS_M as RADIUS
from approach_policy import candidate_footprints,advect,CENTERS,pool
from planner import smooth

VARIANTS={"wait":(False,0.,1.),"current":(True,.9,1.),"sweep":(True,1.8,2.),"sweep_fast":(True,1.2,.6)}
class WaitingNavigator(Navigator):
    def __init__(self,theta,prior,profile):
        super().__init__(theta,prior,"mobile")
        self.fresh_search,self.urgency,self.spread=VARIANTS[profile]
        self.standing=np.ones((100,100));self.drifting=np.zeros((100,100))
        self.last_flow=None
        self.waiting=np.zeros(4,bool)
    def waiting_plan(self,pos,heading,target,velocity,visible):
        # A waiting UAV holds its last heading and issues no delivery command.
        # At 0.4 m/s a pedestrian drifting over a parked UAV can close 2 m every step.
        poses=pos.reshape(1,2);heads=heading.reshape(1,2)
        vis=np.array([visible]);paths=np.empty((1,0),int)
        for step in range(5):
            loc=target+step*5*velocity
            ends,seen,dist,length=flight_edges(poses,vis,loc,velocity)
            future=np.clip(loc+5*velocity,0,1000)
            ray=future-poses
            gap=np.linalg.norm(ray,axis=1)
            sees=(gap<=20+1e-9)&((ray*heads).sum(1)>=-1e-9)
            valid=np.column_stack((length>.01,np.ones(len(poses),bool))).ravel()
            all_pos=np.concatenate((ends,poses[:,None,:]),axis=1).reshape(-1,2)
            all_head=np.concatenate((np.broadcast_to(DIRECTIONS,(len(poses),8,2)),heads[:,None,:]),axis=1).reshape(-1,2)
            all_vis=np.column_stack((seen,sees)).ravel()
            all_dist=np.column_stack((dist,gap)).ravel()
            routes=np.column_stack((np.repeat(paths,9,axis=0),np.tile(np.arange(9),len(paths))))
            successes=np.flatnonzero(valid&all_vis&(all_dist<=RADIUS+1e-9))
            if len(successes):
                idx=successes[np.argmin(all_dist[successes])]
                return routes[idx].tolist()
            if step==4:break
            delta=target+(step+2)*5*velocity-all_pos
            side=np.min(np.abs(delta@DIRECTIONS.T),axis=1)
            # Prefer useful poses looking at the predicted next location.
            cost=all_dist+.65*side-5*all_vis+np.where(valid,0.,1e5)
            order=np.argsort(cost,kind="stable")
            # Include the sign of heading relative to the drift; it matters while waiting.
            key=np.column_stack((np.round(all_pos[order]*10).astype(int),all_vis[order],np.sign(all_head[order] @ velocity)))
            _,ix=np.unique(key,axis=0,return_index=True)
            kept=order[np.sort(ix)[:24]]
            poses,heads,vis,paths=all_pos[kept],all_head[kept],all_vis[kept],routes[kept]
        return []
    def intercept_action(self,o,i,j,target,ends):
        pos=o.pos[i];vel=o.seen_v[j]
        if np.linalg.norm(vel)<.025:
            return super().approach_target(o,i,j,target,ends),False
        gap=np.linalg.norm(target-pos)
        if o.visible[j] and gap<=RADIUS+1e-9:return 8,False
        key=("wait",i)
        old=self.plans.get(key)
        if old and old[0]==j and old[1]==o.t and np.linalg.norm(old[2]-pos)<.02 and np.linalg.norm(old[3]-target)<.35*RADIUS and old[4]:
            moves=old[4]
        elif gap<85:moves=self.waiting_plan(pos,o.heading[i],target,vel,o.visible[j])
        else:moves=[]
        if not moves:
            self.plans.pop(key,None)
            return super().approach_target(o,i,j,target,ends),False
        a=moves[0]
        next_pos=pos if a==8 else flight_edges(pos[None],np.array([o.visible[j]]),target,vel)[0][0,a]
        self.plans[key]=(j,o.t+5,next_pos,np.clip(target+5*vel,0,1000),moves[1:])
        return a,a==8
    def pressure_search(self,o,ends,length,assigned,targets):
        ids=o.observed_cells
        if self.last_flow is not None:
            h,u,v=self.last_flow
            self.drifting=advect(self.drifting,u*.5,v*.5)
            speed2=u*u+v*v
            wet=(h/(h+.05))*(speed2*(1+.25*h/(1+np.sqrt(speed2)+h)))
            raw=wet/(1+speed2*(1+.25*h/(1+np.sqrt(speed2)+h)))+.02*(h/(h+1))**4
            fall=self.standing*(-np.expm1(-5*.005962180452265221*raw))
            self.standing-=fall;self.drifting+=fall
        if len(ids):
            self.last_flow=np.median(o.measured.reshape(3,-1)[:,ids],axis=1)
        self.standing.ravel()[ids]=0.;self.drifting.ravel()[ids]=0.
        flow=np.asarray(self.last_flow[1:]) if self.last_flow is not None else np.zeros(2)
        if np.linalg.norm(flow)>.04:self.axis=flow/np.linalg.norm(flow)
        axis=getattr(self,"axis",np.zeros(2))
        y,x=np.mgrid[:100,:100]
        along=((x-49.5)*axis[0]+(y-49.5)*axis[1])/50
        mass=(self.standing+self.drifting)*np.exp(self.urgency*np.clip(along,-1,1)*max(0,1-o.t/1700.))
        ids,view=candidate_footprints(ends,DIRECTIONS)
        gain=(mass.ravel()[ids]*view).sum(-1)
        coarse=.5*pool(mass)+.5*smooth(pool(mass))
        reserve=np.ones_like(coarse)
        for i in np.flatnonzero(assigned>=0):
            reserve*=.12+.88*np.clip(np.linalg.norm(CENTERS-targets[i],axis=-1)/180,0,1)
        actions=np.zeros(4,int)
        # A separate unclaimed search action for every free UAV.
        for i in np.roll(np.arange(4),o.t//60%4):
            if assigned[i]>=0:continue
            distances=np.linalg.norm(CENTERS-o.pos[i],axis=-1)
            score=coarse*reserve/(1+distances/180)
            score[distances<40]*=.3
            goal=CENTERS[np.unravel_index(score.argmax(),score.shape)]
            advance=np.linalg.norm(o.pos[i]-goal)-np.linalg.norm(ends[i]-goal,axis=-1)
            step=gain[i]/max(.1,float(gain[i].max()))+self.spread*advance/50.
            step[length[i]<1]=-1000
            actions[i]=step.argmax()
            reserve*=.05+.95*np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/220,0,1)
        return actions
    def act(self,o):
        if self.wet is None and ((o.last>=0).sum()>=40 or o.t>=60):
            self.wet=float(np.mean(o.measured[0][o.last>=0]>=.01))>=.8
        ends,length=endpoints(o.pos)
        assigned,targets=self.assign(o)
        if self.wet and self.fresh_search:
            actions=self.pressure_search(o,ends,length,assigned,targets)
        elif self.wet:
            # Suppress the legacy search engine's second dispatcher.
            from types import SimpleNamespace
            search_obs=SimpleNamespace(**{**vars(o),"active":np.zeros_like(o.active)})
            actions,_=self.legacy.act(search_obs)
        else:
            actions=self.search(o,ends,length,assigned,targets).argmax(axis=1)
        claims=assigned.copy()
        for i,j in enumerate(assigned):
            if j>=0:
                actions[i],wait=self.intercept_action(o,i,j,targets[i],ends[i])
                if wait:claims[i]=-1
        conflicts=self.safe(o,actions,claims,ends)
        for _ in range(4):
            if not len(conflicts):break
            pair=PAIRS[:,conflicts[0]]
            def priority(i):
                j=claims[i]
                return (j>=0 and o.owner[j]==i,j>=0 and actions[i]==8,j>=0,-int(i))
            changed=False
            for i in sorted(pair,key=priority):
                goal=targets[i] if assigned[i]>=0 else ends[i,min(7,actions[i])]
                for a in [8,*np.argsort(np.linalg.norm(ends[i]-goal,axis=1))]:
                    trial=actions.copy();new=claims.copy();trial[i]=a
                    if a==8:new[i]=-1
                    test=self.safe(o,trial,new,ends)
                    if len(test)<len(conflicts):
                        actions,claims,conflicts=trial,new,test;changed=True;break
                if changed:break
            if not changed:break
        return actions,claims
