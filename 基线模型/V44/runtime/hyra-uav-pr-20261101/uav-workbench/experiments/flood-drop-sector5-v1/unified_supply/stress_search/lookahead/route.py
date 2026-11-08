"""Two-step coverage utility, using the same physical action set and sensor cadence."""
import numpy as np
from policy import CENTERS, DIRECTIONS, StressSearch, candidate_footprints, pool, smooth

AXIS = np.arange(100)*10+5
CELL_X, CELL_Y = np.meshgrid(AXIS,AXIS)


def endpoints(pos):
    # A move hitting a wall stops on its ray at the first boundary hit.
    along = pos[...,None,:]
    limits = np.full(along.shape[:-2]+(8,2),np.inf)
    np.divide(1000-along,DIRECTIONS,out=limits,where=DIRECTIONS>0)
    np.divide(-along,DIRECTIONS,out=limits,where=DIRECTIONS<0)
    distance = np.clip(limits.min(axis=-1),0,50)
    return np.clip(along+distance[...,None]*DIRECTIONS,0,1000), distance


class Lookahead(StressSearch):
    def __init__(self,theta,base_params,params):
        super().__init__(theta,base_params)
        self.second, self.downstream, self.frontier = params

    def act(self,obs):
        actions,assigned=super().act(obs)
        free=np.flatnonzero(assigned<0)
        if not self.use_belief or not len(free):
            return actions,assigned
        mass=self.standing+self.drifting
        velocity=self.last_speed
        speed=np.linalg.norm(velocity)
        if self.downstream and speed>.01:
            axis=velocity/speed
            downstream=((CELL_X-500)*axis[0]+(CELL_Y-500)*axis[1])/500
            # Emphasise one end of the flow early; let recent observation dominate later.
            mass=mass*np.exp(self.downstream*max(0,1-obs.t/1320)*downstream)
        flat=mass.ravel()
        ends,dist= endpoints(obs.pos[free])
        ids,seen=candidate_footprints(ends,DIRECTIONS)
        gain=(flat[ids]*seen).sum(-1)
        next_ends,next_dist=endpoints(ends)
        next_ids,next_seen=candidate_footprints(next_ends,DIRECTIONS)
        # Two 20 m views cannot overlap after a full 50 m move. Near a boundary
        # exclude any cell already counted in the first view.
        a,b,c=np.nonzero(next_dist<40)
        if len(a):
            cell=next_ids[a,b,c]
            dx=(cell%100)*10+5-ends[a,b,0,None]
            dy=(cell//100)*10+5-ends[a,b,1,None]
            in_previous=(dx*dx+dy*dy<=400+1e-7) & (dx*DIRECTIONS[b,0,None]+dy*DIRECTIONS[b,1,None]>=-1e-7)
            next_seen[a,b,c] &= ~in_previous
        next_gain=(flat[next_ids]*next_seen).sum(-1)
        scale=max(.03,float(mass.mean()))
        next_gain[next_dist<1]=-1000
        gain[dist<1]=-1000
        coarse=pool(mass)
        priorities=.5*coarse+.5*smooth(coarse)
        reservation=np.ones_like(priorities)
        for i in np.flatnonzero(assigned>=0):
            reservation*=np.clip(np.linalg.norm(CENTERS-obs.seen_xy[assigned[i]],axis=-1)/170,.06,1)
        for a,i in enumerate(free):
            dc=np.linalg.norm(CENTERS-obs.pos[i],axis=-1)
            attraction=priorities*reservation/(1+dc/190)
            attraction[dc<60]*=.25
            goal=CENTERS[np.unravel_index(attraction.argmax(),attraction.shape)]
            separation=np.linalg.norm(ends[a,:,None]-obs.pos[None,:],axis=-1)
            separation[:,i]=np.inf
            progress=(np.linalg.norm(obs.pos[i]-goal)-np.linalg.norm(next_ends[a]-goal,axis=-1))/100
            value=gain[a,:,None]+self.second*next_gain[a]+scale*self.frontier*progress
            value-=4*scale*np.clip(1-separation.min(-1)/90,0,1)[:,None]
            actions[i]=np.unravel_index(value.argmax(),value.shape)[0]
            reservation*=np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/180,.05,1)
        return actions,assigned
