"""Learn next-observation discovery counts; actor uses only sensed history and candidate geometry."""
import numpy as np
from search8 import SearchPolicy,X,Y,CENTERS,pool,smooth
from approach_policy import DIRECTIONS,candidate_footprints
from local_route import endpoints

PARAMS={
    "net_soft":(.35,.3,.65,1.2),"net_balanced":(.7,.5,1.,1.4),
    "net_direct":(1.,0.,1.2,.9),"net_wide":(.7,.65,1.5,2.2),
    "net_steady":(.85,-.25,.55,.8),"net_early":(.55,1.,.85,1.7),
    "horizon":(0.,.5,1.,1.4),"horizon_wide":(0.,.65,1.5,2.2)}
def features(o,positions,headings,belief,axis,offset):
    heads=np.broadcast_to(headings,positions.shape)
    indices,view=candidate_footprints(positions,heads)
    last=o.last.ravel()[indices]
    visits=(last>=0)&view
    fresh=(view & ~visits).sum(-1)
    seen=visits.sum(-1)
    age=np.where(visits,np.minimum(600,np.maximum(0,o.t-last))/600.,0).sum(-1)
    mass=(belief.ravel()[indices]*view).sum(-1)
    along=((positions-500)*axis).sum(-1)/500
    cross=np.abs((positions-500)@np.array([-axis[1],axis[0]]))/500
    edge=np.clip(np.minimum(positions,1000-positions).min(-1)/70.,0,1)
    dot=(heads*axis).sum(-1)
    phase=np.cos(positions*2*np.pi/10)
    t=np.full(mass.shape,(o.t+offset)/2280)
    return np.stack([mass/6,fresh/6,seen/6,age/6,along,cross,edge,dot,t,
                     phase[...,0]+phase[...,1],np.abs(heads[...,0]*heads[...,1])*2,
                     along*t,along*dot],axis=-1)
def expected(feat,weights):
    return np.exp(np.clip(np.tanh(feat@weights["w"]+weights["b"])@weights["out"]+weights["bias"],-8,1))
class PredictSearch(SearchPolicy):
    def __init__(self,name,weights,disable=False):
        super().__init__("urgent")
        self.name=name;self.weights=weights
        self.blend,self.tilt,self.far,self.region=PARAMS[name]
        if disable:self.blend=0.
    def pressure_search(self,o,ends,length,assigned,targets):
        self.cohorts.observe(o)
        axis=self.cohorts.axis
        mass=self.cohorts.mass(o.t,2.)
        along=((X-500)*axis[0]+(Y-500)*axis[1])/500
        second,travel=endpoints(ends)
        third=np.clip(second+second-ends[:,:,None,:],0,1000)
        def values(positions,offset):
            f=features(o,positions,DIRECTIONS,mass,axis,offset)
            mech=f[...,0]*6
            # A calibrated expected count reweights feasible, observable views.
            if self.blend:
                neural=expected(f,self.weights)
                neural=neural*(mech.mean()+.03)/(neural.mean()+1e-9)
                mech=(1-self.blend)*mech+self.blend*neural
            return mech
        g1=values(ends,5);g2=values(second,10);g3=values(third,15)
        g2[travel<38]=0;g3[np.linalg.norm(third-second,axis=-1)<38]=0
        map_value=mass*np.exp(self.tilt*along*max(0.,1-o.t/1500))
        coarse=.5*pool(map_value)+.5*smooth(pool(map_value))
        reserve=np.ones_like(coarse)
        for i in np.flatnonzero(assigned>=0):
            reserve*=.1+.9*np.clip(np.linalg.norm(CENTERS-targets[i],axis=-1)/180,0,1)
        free=np.flatnonzero(assigned<0)
        result=np.zeros(4,int);booked=[]
        scale=max(.05,float(mass.mean()))
        for i in np.roll(free,o.t//60%max(1,len(free))):
            p=o.pos[i];dc=np.linalg.norm(CENTERS-p,axis=-1)
            utility=coarse*reserve/(1+dc/270)
            utility[dc<50]*=.3
            aim=CENTERS[np.unravel_index(utility.argmax(),utility.shape)]
            progress=(np.linalg.norm(p-aim)-np.linalg.norm(second[i]-aim,axis=-1))/100.
            score=g1[i,:,None]+.82*g2[i]+.44*g3[i]+4*scale*self.far*progress
            for endpoint in booked:
                score-=4.5*self.region*scale*np.clip(1-np.linalg.norm(second[i]-endpoint,axis=-1)/165,0,1)
            score[length[i]<1]=-1e7;score[travel[i]<1]=-1e7
            a,b=np.unravel_index(score.argmax(),score.shape);result[i]=a
            booked.append(second[i,a,b])
            reserve*=.1+.9*np.clip(np.linalg.norm(CENTERS-aim,axis=-1)/(170*self.region),0,1)
        return result
