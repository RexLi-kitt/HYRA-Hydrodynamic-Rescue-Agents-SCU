"""Causal search over cohorts of possible yet-unseen survivors; shared V7 pursuit."""
import numpy as np
from extra import WaitingNavigator
from local_route import endpoints
from approach_policy import candidate_footprints,DIRECTIONS,CENTERS,pool
from planner import H,V,Planner,smooth

# directional bias, time-to-loss bias, learned contact effect, assigned-area attraction
PROFILES={
    "cohort":(0.,1.,0.,.75),"upstream":(-1.2,.5,0.,.8),"urgent":(-.45,3.,0.,.8),
    "sensor":(-.6,1.,.8,.7),"sensor_only":(0.,.4,1.4,.95),"sensor_up":(-1.5,1.,.75,1.),
    "cohort_wide":(-.6,1.,0.,1.65),"neutral":(0.,0.,0.,1.),
    "legacy_up":(-1.8,0.,0.,1.),"legacy_flat":(0.,0.,0.,1.),
    "legacy_hard_up":(-3.,0.,0.,1.),"legacy_switch":(-2.,0.,0.,1.)}
GRID=np.arange(100)*10+5
X,Y=np.meshgrid(GRID,GRID)
STENCIL=np.array([(xx,yy) for yy in range(-2,3) for xx in range(-2,3)])
K=.005962180452265221
def rate(h,speed):
    z=speed**2*(1+.25*h/(1+speed+h))
    return K*(h/(h+.05)*z/(1+z)+.02*(h/(h+1))**4)
class Cohorts:
    def __init__(self,count=10):
        self.personal_window_seconds=1140
        self.count=count
        self.clock=-np.log1p(-(np.arange(count)+.5)/count)
        self.fall=np.full(count,np.inf)
        self.shift=np.zeros((count,2))
        self.unseen=np.ones((count,100,100))
        self.dose=0.
        self.time=0
        self.field=np.zeros(3)
        self.axis=np.zeros(2)
    def observe(self,o):
        dt=o.t-self.time
        h,u,v=self.field
        lam=rate(h,np.hypot(u,v))
        next_dose=self.dose+lam*dt
        crossing=(self.clock<=next_dose)&~np.isfinite(self.fall)
        if crossing.any():self.fall[crossing]=self.time+(self.clock[crossing]-self.dose)/lam
        travel=np.clip(o.t-self.fall,0,dt)
        self.shift+=travel[:,None]*np.array([u,v])[None,:]
        self.dose=next_dose;self.time=o.t
        if len(o.observed_cells):
            self.field=np.median(o.measured.reshape(3,-1)[:,o.observed_cells],axis=1)
            direction=self.field[1:];speed=np.linalg.norm(direction)
            if speed>.04:self.axis=direction/speed
        # Remove hypotheses actually in the four observed sectors, in origin coordinates.
        origin=o.pos[None,:,:]-self.shift[:,None,:]
        indices=np.floor(origin/10).astype(int)[:,:,None,:]+STENCIL
        delta=indices*10+5-origin[:,:,None,:]
        seen=(np.square(delta).sum(-1)<=400+1e-9)&((delta*o.heading[None,:,None,:]).sum(-1)>=-1e-9)
        ok=((indices>=0)&(indices<100)).all(-1)
        k,i,j=np.nonzero(ok&seen)
        xx,yy=indices[k,i,j].T
        self.unseen[k,yy,xx]=0.
    def mass(self,time,urgency):
        out=np.zeros((100,100))
        along=(X-500)*self.axis[0]+(Y-500)*self.axis[1]
        for k,(offset,fall) in enumerate(zip(self.shift,self.fall)):
            remaining=self.personal_window_seconds+fall-time
            if remaining<=0:continue
            dx,dy=np.rint(offset/10).astype(int)
            if abs(dx)>=100 or abs(dy)>=100:continue
            x0,x1=max(0,dx),min(100,100+dx)
            y0,y1=max(0,dy),min(100,100+dy)
            new=self.unseen[k,y0-dy:y1-dy,x0-dx:x1-dx]
            w=1.
            if np.isfinite(fall):
                # Weight people still locatable, with deadlines or domain exits approaching.
                w+=urgency*(.55*np.exp(-remaining/240)+.45*np.exp(-np.maximum(0,500-along[y0:y1,x0:x1])/105))
            out[y0:y1,x0:x1]+=w*new
        return out/self.count
class SearchPolicy(WaitingNavigator):
    def __init__(self,name,relative=None):
        super().__init__(np.zeros(49),np.ones((20,20)),"wait")
        self.name=name
        self.bias,self.urgency,self.sensor,self.attract=PROFILES[name]
        self.lookup=np.repeat((H>=.01).astype(float),len(V))
        self.legacy.original=Planner("water",3.)
        self.cohorts=Cohorts()
        self.relative=relative
        self.fresh_search=not name.startswith("legacy")
        if not self.fresh_search:self.legacy.downstream=self.bias
    def act(self,o):
        if self.name=="legacy_switch":self.legacy.downstream=self.bias if o.t<1100 else .8
        return super().act(o)
    def pressure_search(self,o,ends,length,assigned,targets):
        self.cohorts.observe(o)
        axis=self.cohorts.axis
        along=((X-500)*axis[0]+(Y-500)*axis[1])/500
        mass=self.cohorts.mass(o.t,self.urgency)
        mass*=np.exp(self.bias*along*max(0.,1-o.t/1800.))
        if self.sensor:
            band=np.clip(((along+1)*2.5).astype(int),0,4)
            lateral=np.clip((np.abs((Y-500)*axis[0]-(X-500)*axis[1])*3/500).astype(int),0,2)
            mass*=self.relative[min(5,o.t//380),band,lateral]**self.sensor
        # Score legal directions at the current step and at two subsequent observation times.
        ids1,vis1=candidate_footprints(ends,DIRECTIONS)
        second,travel=endpoints(ends)
        ids2,vis2=candidate_footprints(second,DIRECTIONS)
        gain1=(mass.ravel()[ids1]*vis1).sum(-1)
        gain2=(mass.ravel()[ids2]*vis2).sum(-1)
        gain2[travel<39]=0.
        third=np.clip(second+(second-ends[:,:,None,:]),0,1000)
        idx3,v3=candidate_footprints(third,DIRECTIONS)
        gain3=(mass.ravel()[idx3]*v3).sum(-1)
        gain3[np.linalg.norm(third-second,axis=-1)<40]=0.
        free=np.flatnonzero(assigned<0);chosen=[]
        coarse=.5*pool(mass)+.5*smooth(pool(mass))
        reserve=np.ones_like(coarse)
        for i in np.flatnonzero(assigned>=0):
            reserve*=.15+.85*np.clip(np.linalg.norm(CENTERS-targets[i],axis=-1)/165,0,1)
        actions=np.zeros(4,int)
        scale=max(.08,float(mass.mean()))
        for i in np.roll(free,o.t//50%max(1,len(free))):
            p=o.pos[i];dc=np.linalg.norm(CENTERS-p,axis=-1)
            utility=coarse*reserve/(1+dc/245)
            utility[dc<55]*=.28
            goal=CENTERS[np.unravel_index(utility.argmax(),utility.shape)]
            progress=(np.linalg.norm(p-goal)-np.linalg.norm(second[i]-goal,axis=-1))/100.
            score=gain1[i,:,None]+.8*gain2[i]+.42*gain3[i]+4.5*scale*self.attract*progress
            for prev in chosen:
                sep=np.linalg.norm(second[i]-prev,axis=-1)
                score-=4*scale*np.clip(1-sep/145,0,1)
            score[length[i]<1]=-1e7
            score[travel[i]<1]=-1e7
            a,b=np.unravel_index(score.argmax(),score.shape)
            actions[i]=a;chosen.append(second[i,a,b])
            reserve*=.08+.92*np.clip(np.linalg.norm(CENTERS-goal,axis=-1)/185,0,1)
        return actions
