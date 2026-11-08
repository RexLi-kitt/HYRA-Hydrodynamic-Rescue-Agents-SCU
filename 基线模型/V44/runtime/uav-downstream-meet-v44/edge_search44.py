"""Track the mass of unseen drifting cohorts intercepted by the retaining perimeter."""
import numpy as np
from continuous44 import Continuous
from beam16 import Cohorts

CENTRES = np.arange(5.,1000.,10.)
# West, east, south (y=0), north (y=1000). Corners deposit on the first side hit.
EDGE_POINTS = np.concatenate([np.column_stack((np.full(100,x),CENTRES)) for x in (0.,1000.)] +
                             [np.column_stack((CENTRES,np.full(100,y))) for y in (0.,1000.)])
EDGE_CELL = np.clip((EDGE_POINTS/10).astype(int),0,99)
EDGE_INDEX = EDGE_CELL[:,1]*100 + EDGE_CELL[:,0]


class RetainedCohorts(Cohorts):
    def __init__(self,count):
        super().__init__(count)
        self.retained = np.zeros((count,400))

    def deposit(self,k,ix,iy,old,delta):
        if not len(ix):
            return
        initial = np.column_stack((CENTRES[ix],CENTRES[iy])) + old
        wall = np.where(delta>0,1000.,0.)
        times = np.full(initial.shape,np.inf)
        np.divide(wall-initial,delta,out=times,where=delta!=0)
        axis = times.argmin(axis=1)
        t = times[np.arange(len(ix)),axis]
        point = initial+t[:,None]*delta
        side = 2*axis+(delta[axis]>0)
        along = np.clip((point[np.arange(len(ix)),1-axis]/10).astype(int),0,99)
        np.add.at(self.retained[k],100*side+along,self.unseen[k,iy,ix])
        self.unseen[k,iy,ix] = 0.

    def observe(self,o):
        previous = self.shift.copy()
        super().observe(o)
        active = np.flatnonzero(np.isfinite(self.fall)&(self.fall+self.personal_window_seconds>=o.t+10))
        for k in active:
            old,new = previous[k],self.shift[k]
            if np.array_equal(old,new):
                continue
            prior = (CENTRES[:,None]+old>0)&(CENTRES[:,None]+old<1000)
            inside = (CENTRES[:,None]+new>0)&(CENTRES[:,None]+new<1000)
            leaving = prior & ~inside
            if not leaving.any():
                continue
            # Split the departing strip into disjoint columns and rows.
            for xs,ys in ((leaving[:,0],prior[:,1]),(prior[:,0]&inside[:,0],leaving[:,1])):
                ix,iy = np.meshgrid(np.flatnonzero(xs),np.flatnonzero(ys))
                self.deposit(k,ix.ravel(),iy.ravel(),old,new-old)
        delta = EDGE_POINTS[:,None,:]-o.pos
        visible = ((delta*delta).sum(2)<=400+1e-9)&((delta*o.heading).sum(2)>=-1e-9)
        self.retained[:,visible.any(axis=1)] = 0.


class EdgeSearch(Continuous):
    def __init__(self,allocation):
        super().__init__()
        self.belief = RetainedCohorts(12)
        self.belief.personal_window_seconds = 720
        self.allocation = allocation
        self.free = np.array([],dtype=int)
        self.edge_calls = 0

    def pressure_search(self,o,ends,length,assigned,targets):
        self.free = np.flatnonzero(assigned<0)
        return super().pressure_search(o,ends,length,assigned,targets)

    def fields(self,o):
        fields = super().fields(o)
        b = self.belief
        remaining = b.fall+b.personal_window_seconds-o.t
        active = np.isfinite(b.fall)&(remaining>=10)
        weights = (1+.35*np.exp(-remaining[active]/180))/b.count
        edge = weights @ b.retained[active]
        self.edge_fields = None
        if np.any(edge) and len(self.free):
            stationary = fields[0].copy()
            np.add.at(stationary,EDGE_INDEX,edge)
            self.edge_fields = (stationary,fields[1])
            distance = np.linalg.norm(o.pos[self.free,None,:]-EDGE_POINTS,axis=2)
            self.edge_drone = self.free[np.argmax((edge/(1+distance/150)**2).sum(axis=1))]
        return fields

    def path(self,o,i,fields,reserved):
        if self.edge_fields is not None and (self.allocation=="all" or i==self.edge_drone):
            fields = self.edge_fields
            self.edge_calls += 1
        return super().path(o,i,fields,reserved)
