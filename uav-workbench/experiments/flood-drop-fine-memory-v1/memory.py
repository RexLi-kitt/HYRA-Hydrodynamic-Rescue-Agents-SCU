"""Public 20m observation-footprint memory inside each 100m demand tile."""
from pathlib import Path
import sys,copy
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-search-redesign-v1'))
from adapter import ResidualEnv
from history_model import features

def pack_fine(a):
    # [2,50,50] -> [50,10,10], preserving internal 5x5 orientation
    return a.reshape(2,10,5,10,5).transpose(0,2,4,1,3).reshape(50,10,10)
def public_features(e):
    base=features(e).reshape(10,10,14).transpose(2,0,1).copy()
    base[8:10]=0
    age=np.where(e.known,np.clip((e.t-e.last)/240,0,2),2.)
    fine=np.stack([e.known,age]).reshape(2,50,2,50,2).mean((2,4))
    return np.concatenate([base,pack_fine(fine)],0).astype(np.float32)
class FineHistory:
    def __init__(self):self.frames=[];self.time=None
    def get(self,e):
        if self.time!=e.t and (not self.frames or e.t%30==0):
            self.frames.append(public_features(e));self.frames=self.frames[-4:];self.time=e.t
        return np.stack([self.frames[0]]*(4-len(self.frames))+self.frames)
class FineEnv(ResidualEnv):
    def __init__(self,w,s,predict=None,alpha=.25):
        super().__init__(w,s,predict,alpha);self.history=FineHistory()
class MemoryGRU(nn.Module):
    def __init__(self,mean,coarse=False):
        super().__init__();self.coarse=coarse
        self.recurrent=nn.GRU(64,32,batch_first=True);self.head=nn.Linear(32,1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(mean,1e-5)))))
    def forward(self,x):
        if self.coarse:
            x=x.clone();x[:,:,14:]=0
        b,t,c,h,w=x.shape
        z=x.permute(0,3,4,1,2).reshape(-1,t,c)
        return (torch.nn.functional.softplus(self.head(self.recurrent(z)[0][:,-1]).squeeze(-1))+1e-6).reshape(b,h,w)
def augment(x,y,k,flip):
    z=torch.rot90(x,k,(-2,-1)).clone();q=torch.rot90(y,k,(-2,-1)).clone()
    for _ in range(k):
        u=z[:,:,3].clone();v=z[:,:,4].clone();z[:,:,3]=v;z[:,:,4]=-u
    b,t=x.shape[:2]
    full=x[:,:,14:].reshape(b,t,2,5,5,10,10).permute(0,1,2,5,3,6,4).reshape(b,t,2,50,50)
    full=torch.rot90(full,k,(-2,-1))
    if flip:
        z=z.flip(-1);q=q.flip(-1);z[:,:,3]=-z[:,:,3];full=full.flip(-1)
    z[:,:,14:]=full.reshape(b,t,2,10,5,10,5).permute(0,1,2,4,6,3,5).reshape(b,t,50,10,10)
    return z,q
def normalizer(x):
    mu=x.mean((0,1,3,4));sd=x.std((0,1,3,4)).clip(.01)
    mu[3:5]=0;sd[3:5]=max(float(np.sqrt(np.mean(x[:,:,3:5]**2))),.01)
    mu[8:10]=0;sd[8:10]=1
    for sl in (slice(14,39),slice(39,64)):
        mu[sl]=float(x[:,:,sl].mean());sd[sl]=max(float(x[:,:,sl].std()),.01)
    return mu.astype(np.float32),sd.astype(np.float32)
def predictor(model,mu,sd):
    def predict(x):
        with torch.inference_mode():
            z=(x-mu[None,:,None,None])/sd[None,:,None,None]
            return model(torch.from_numpy(z)[None])[0].numpy().reshape(-1)
    return predict

