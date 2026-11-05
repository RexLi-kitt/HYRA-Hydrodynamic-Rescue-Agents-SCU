import numpy as np
import torch
from torch import nn
def pool(a):return np.asarray(a).reshape(10,10,10,10).mean((1,3))

def features(e):
    """Only public observations/configuration; one row per 100m tile."""
    known=e.known
    age=np.where(known,np.clip((e.t-e.last)/240,0,2),2.)
    seen=np.zeros((10,10));pending=seen.copy()
    for j in np.flatnonzero(e.discovered):
        c,r=np.clip((e.seen_xy[j]/100).astype(int),0,9)
        seen[r,c]+=1
        if e.belief[j]==1:pending[r,c]+=1
    rr,cc=np.mgrid[:10,:10]
    values=[pool(known),pool(age),pool(np.where(known,e.measured[0]/2,0)),
            pool(np.where(known,e.measured[1],0)),pool(np.where(known,e.measured[2],0)),
            pool(known&(e.measured[0]>=e.cfg.wet_depth)),seen/5,pending/5,
            (cc+.5)/10,(rr+.5)/10]
    values += [np.full((10,10),v) for v in (e.t/e.cfg.duration,e.cfg.sensor/100,e.cfg.people/80,e.cfg.window/300)]
    x=np.stack(values,-1).reshape(100,14).astype(np.float32)
    assert np.isfinite(x).all()
    return x

class History:
    def __init__(self):self.frames=[];self.time=None
    def get(self,e):
        if self.time!=e.t and (not self.frames or e.t%30==0):
            self.frames.append(features(e));self.frames=self.frames[-4:];self.time=e.t
        # Repeat the first actual observation, never future padding.
        return np.stack([self.frames[0]]*(4-len(self.frames))+self.frames,axis=1)

def statistic(x):
    a=x[:,-1];return (.1+a[:,1])*(.25+2*a[:,5]+1-a[:,0])

class Model(nn.Module):
    def __init__(self,kind,mean):
        super().__init__();self.kind=kind
        if kind=='gru':self.recurrent=nn.GRU(14,32,batch_first=True);self.head=nn.Linear(32,1)
        else:self.head=nn.Linear(56,1)
        nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(mean,1e-5)))))
    def forward(self,x):
        h=self.recurrent(x)[0][:,-1] if self.kind=='gru' else x.flatten(1)
        return torch.nn.functional.softplus(self.head(h).squeeze(-1))+1e-6

