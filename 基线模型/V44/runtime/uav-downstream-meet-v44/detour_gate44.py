"""Compare the committed part of an edge detour with the available interior route."""
import numpy as np
from continuous44 import Continuous, as_float, native, expected_ends
from edge_search44 import EdgeSearch, EDGE_POINTS


class DetourGate(EdgeSearch):
    def __init__(self):
        super().__init__("all")
        self.compared = self.accepted = 0

    def prefix_scores(self,o,i,fields,routes):
        b=self.belief
        pos=np.repeat(o.pos[i,None],2,axis=0)
        head=np.repeat(o.heading[i,None],2,axis=0)
        poses=[]; heads=[]
        interior=np.zeros(2); edge=np.zeros(2)
        seen=np.zeros((2,10000),bool)
        seen_edge=np.zeros((2,len(EDGE_POINTS)),bool)
        for k,actions in enumerate(routes.T):
            ends,lengths=expected_ends(pos)
            nxt=ends[np.arange(2),actions]
            distance=lengths[np.arange(2),actions]
            moving=distance>1e-8
            head[moving]=(nxt-pos)[moving]/distance[moving,None]
            pos=nxt
            poses.append(pos.copy()); heads.append(head.copy())
            gain,ids,mask=self.candidates(pos[:,None],head[:,None],fields[0],seen,False,np.zeros(2),b.axis)
            interior+=.97**k*gain[:,0]
            a,j=np.nonzero(mask[:,0]); seen[a,ids[a,0,j]]=True
            ray=EDGE_POINTS[None]-pos[:,None]
            visible=((ray*ray).sum(2)<=400+1e-9)&((ray*head[:,None]).sum(2)>=-1e-9)
            ix,iy=np.clip(np.rint(ray+20).astype(int),0,40).transpose(2,0,1)
            steps=self.static_cost[iy,ix]
            # Exclude cohorts without enough remaining time to approach and drop.
            sight_time=o.t+5*(k+1)
            for c in np.flatnonzero(np.isfinite(b.fall)):
                time_left=min(2400,b.fall[c]+b.personal_window_seconds)-sight_time
                valid=visible & ~seen_edge & (time_left>=5*steps+10)
                mass=b.retained[c]/b.count
                edge+=.97**k*(mass*valid*(1+self.easy*(2.8/(2+steps)-1))).sum(axis=1)
            seen_edge|=visible
        shifts,mass,bias=self.pack
        if len(shifts):
            value=np.empty(2)
            native(2,routes.shape[1],len(shifts),as_float(np.stack(poses,axis=1)),
                   as_float(np.stack(heads,axis=1)),shifts,mass,bias,self.step_cost,
                   as_float(b.field[1:]),as_float(b.axis),self.easy,1,value)
            interior+=value
        return interior,edge

    def path(self,o,i,fields,reserved):
        interior=Continuous.path(self,o,i,fields,reserved)
        if self.edge_fields is None:
            return interior
        border=Continuous.path(self,o,i,self.edge_fields,reserved)
        n=min(self.commit,len(interior),len(border),(2400-o.t-10)//5)
        if n<=0 or interior[:n]==border[:n]:
            return interior
        self.compared+=1
        inland,edge=self.prefix_scores(o,i,fields,np.array([interior[:n],border[:n]]))
        if edge[1]>edge[0]+1e-8 and edge[1]-edge[0]>inland[0]-inland[1]+1e-8:
            self.accepted+=1
            return border
        return interior
