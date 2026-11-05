import numpy as np
from history_model import History,features,pool
from long_search import SearchEnv
from long_environment import sample_water
def transport(demand,u,v,seconds):
    """Forward bilinear transport; mass leaving square is lost, not edge-clipped."""
    rr,cc=np.mgrid[:10,:10];xx=cc+u*seconds/100;yy=rr+v*seconds/100
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);fx=xx-x0;fy=yy-y0
    moved=np.zeros((10,10))
    for dx,dy,w in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        x=x0+dx;y=y0+dy;valid=(x>=0)&(x<10)&(y>=0)&(y<10)
        np.add.at(moved,(y[valid],x[valid]),(demand*w)[valid])
    assert np.isfinite(moved).all() and (moved>=0).all()
    return moved

class FlowFogEnv(SearchEnv):
    def __init__(self,water,scenario,method,predict=None):
        self.method=method;self.predict=predict;self.history=History();self.prediction=None;self.predicted_at=None
        self.patch=bool(scenario.get('patch_fog',False));self.fog_orientation=int(scenario['seed'])%2
        self.sensor_radii=None;self.fog_uav_observations=0;self.observation_calls=0
        super().__init__(water,scenario,'coverage100')

    def radii(self):
        # Synthetic central 400m band, alternating orientation by seed. Current local visibility is public.
        if not self.patch:return np.full(self.cfg.uavs,self.cfg.sensor)
        q=self.pos[:,self.fog_orientation];dense=(q>=300)&(q<=700)
        return np.where(dense,min(20.,self.cfg.sensor),self.cfg.sensor)

    def observe_world(self):
        if not self.patch:return super().observe_world()
        before=int(self.known.sum());found=int(self.discovered.sum());r=self.radii();self.sensor_radii=r
        self.fog_uav_observations+=int((r<self.cfg.sensor).sum());self.observation_calls+=1
        mask=np.zeros((100,100),bool)
        for p,radius in zip(self.pos,r):mask|=np.sum((self.cells-p)**2,axis=-1)<=radius**2
        self.known|=mask;self.last[mask]=self.t
        a=self.t//5;b=min(a+1,len(self.water)-1);alpha=self.t%5/5
        self.measured[:,mask]=(1-alpha)*self.water[a][:,mask]+alpha*self.water[b][:,mask]
        distances=np.linalg.norm(self.person[:,None]-self.pos[None],axis=-1)
        self.visible=(self.status>0)&(self.status!=4)&(distances<=r[None]).any(1)
        ids=np.flatnonzero(self.visible);self.first_seen[ids[self.first_seen[ids]<0]]=self.t
        self.discovered[ids]=True;self.seen_xy[ids]=self.person[ids];self.seen_at[ids]=self.t
        if len(ids):
            self.seen_v[ids]=sample_water(self.water,self.t,self.person[ids])[:,1:]
            self.seen_v[ids[self.stuck[ids]]]=0
        self.belief[ids]=np.where(self.status[ids]==1,1,2)
        return int(self.known.sum())-before,int(self.discovered.sum())-found

    def search_score(self,stale,risk):
        if self.method=='coverage100':return super().search_score(stale,risk)
        base=stale*(1+risk)
        prior=np.where(self.known,np.where(self.measured[0]>=self.cfg.wet_depth,2.,.5),1.)
        cover=pool(np.where(self.known,np.minimum(stale,2.),2.)*prior)
        fresh=self.known&(self.t-self.last<=120)
        u=pool(np.where(fresh,self.measured[1],0));v=pool(np.where(fresh,self.measured[2],0))
        if self.method=='flow_coverage':target=.5*cover+.5*transport(cover,u,v,60)
        else:
            x=self.history.get(self)
            if self.prediction is None or self.predicted_at!=self.history.time:
                self.prediction=np.maximum(0,self.predict(x)).reshape(10,10);self.predicted_at=self.history.time
            target=self.prediction
            if self.method.endswith('_flowblend'):
                target=.5*target+.5*transport(target,u,v,60)
                # Normalize only spatial emphasis; keep coverage half of target, no truth-derived scale.
                target=.5*cover/(cover.mean()+1e-6)+.5*target/(target.mean()+1e-6)
            else:return base*(1+8*np.repeat(np.repeat(target,10,0),10,1))
        return base*(.25+np.repeat(np.repeat(target,10,0),10,1))

