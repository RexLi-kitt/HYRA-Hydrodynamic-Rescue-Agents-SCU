"""Re-rank the same V44 finalists using continuous cohort displacement."""
import ctypes
from pathlib import Path
import numpy as np
from tracker44 import Downstream
from shortlist44 import shortlist,DIRECTIONS,expected_ends

F64=np.ctypeslib.ndpointer(dtype=np.float64,flags="C_CONTIGUOUS")
native=ctypes.CDLL(str(Path(__file__).with_name("fraction.dll"))).mobile
native.argtypes=[ctypes.c_int]*3+[F64]*8+[ctypes.c_double,ctypes.c_int,F64]
native.restype=None
yy,xx=np.mgrid[:100,:100]

def as_float(x):
    return np.ascontiguousarray(x,dtype=float)

class Continuous(Downstream):
    def __init__(self, dry_service=False):
        super().__init__("extended")
        self.search_calls=self.changed_first=self.changed_route=0
        self.checked=0
        self.step_cost=as_float(self.lookup_steps[1])
        self.static_cost=as_float(self.lookup_steps[0])
        self.dry_service=dry_service
        self.use_static_cost=False
        self.dry_search_calls=0
        self.checked_dry=0

    def fields(self,o):
        self.use_static_cost=False
        if self.dry_service and len(o.observed_cells):
            samples=o.measured.reshape(3,-1)[:,o.observed_cells]
            self.use_static_cost=bool(np.all(samples[0]<.01) and np.all(samples[1:]==0.))
        fields=super().fields(o)
        b=self.belief
        ids=np.flatnonzero(np.isfinite(b.fall)&(b.fall+b.personal_window_seconds>=o.t+10))
        weights=(1+.35*np.exp(-(b.fall[ids]+b.personal_window_seconds-o.t)/180))/b.count
        bias=np.exp(self.tilt*((xx-49.5)*b.axis[0]+(yy-49.5)*b.axis[1])/50*max(0,1-o.t/1450))
        self.pack=(as_float(b.shift[ids]),as_float(weights[:,None,None]*b.unseen[ids]),as_float(bias))
        return fields

    def candidates(self,p,h,field,seen,moving,offset,axis):
        # An already displaced cohort keeps its position. Only the cost of
        # reaching a person from the next sighting changes while flow is stopped.
        return super().candidates(p,h,field,seen,moving and not self.use_static_cost,offset,axis)

    def path(self,o,i,fields,reserved):
        shifts,masses,bias=self.pack
        if o.t<60 or not len(shifts):
            return super().path(o,i,fields,reserved)
        self.search_calls+=1
        routes,values=shortlist(self,o,i,fields,reserved)
        old=int(np.argmax(values))
        nr,steps=routes.shape
        poses=np.empty((nr,steps,2));pos=np.broadcast_to(o.pos[i],(nr,2))
        for t in range(steps):
            ends,_=expected_ends(pos)
            pos=ends[np.arange(nr),routes[:,t]]
            poses[:,t]=pos
        heads=as_float(DIRECTIONS[routes])
        cost=self.static_cost if self.use_static_cost else self.step_cost
        cost_axis=(1.,0.) if self.use_static_cost else self.belief.axis
        self.dry_search_calls+=int(self.use_static_cost)
        args=(nr,steps,len(shifts),poses,heads,shifts,masses,bias,cost,as_float(self.belief.field[1:]),as_float(cost_axis),self.easy)
        rounded=np.empty(nr);continuous=np.empty(nr)
        native(*args,0,rounded);native(*args,1,continuous)
        if self.checked<2 or (self.use_static_cost and self.checked_dry<2):
            assert routes[old].tolist()==super().path(o,i,fields,reserved)
            # Cross-check rounded scorer with the original per-step moving field
            # on the actual survivor paths, including repeated views and easy-service weights.
            seen=np.zeros((nr,10000),bool);ref=np.zeros(nr)
            for k in range(steps):
                gain,ix,visible=self.candidates(poses[:,k:k+1],heads[:,k:k+1],fields[1],seen,True,(k+1)*5*self.belief.field[1:],self.belief.axis)
                ref+=.97**k*gain[:,0]
                rows,cols=np.nonzero(visible[:,0]);seen[rows,ix[rows,0,cols]]=True
            assert np.allclose(ref,rounded,atol=1e-7),np.max(abs(ref-rounded))
            if self.use_static_cost:
                self.checked_dry+=1
            self.checked=min(2,self.checked+1)
        adjusted=values+(continuous-rounded)
        best=int(np.argmax(adjusted))
        if adjusted[best]<=adjusted[old]+1e-8:
            best=old
        self.changed_route+=int(best!=old)
        self.changed_first+=int(routes[best,0]!=routes[old,0])
        return self.select_route(o,i,routes,poses,adjusted,best)

    def select_route(self,o,i,routes,poses,scores,best):
        return routes[best].tolist()
