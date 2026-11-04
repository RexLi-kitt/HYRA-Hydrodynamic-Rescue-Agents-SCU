"""ConvLSTM map demand and matched per-tile GRU. No privileged runtime inputs."""
import numpy as np
import torch
from torch import nn
from history_model import Model

def maps(x):
    a=np.asarray(x,np.float32).reshape(10,10,4,14).transpose(2,3,0,1).copy()
    a[:,8:10]=0 # remove absolute position from both newly trained models
    return a

def augment(x,y,k,flip=False):
    z=torch.rot90(x,k,(-2,-1)).clone();q=torch.rot90(y,k,(-2,-1)).clone()
    for _ in range(k):
        u=z[:,:,3].clone();v=z[:,:,4].clone();z[:,:,3]=v;z[:,:,4]=-u
    if flip:
        z=z.flip(-1);q=q.flip(-1);z[:,:,3]=-z[:,:,3]
    return z,q

class DemandModel(nn.Module):
    def __init__(self,kind,mean):
        super().__init__();self.kind=kind
        if kind=='gru':self.net=Model('gru',mean)
        elif kind=='convlstm':
            self.cell=nn.Conv2d(14+16,4*16,3,padding=1);self.head=nn.Conv2d(16,1,1)
            nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(mean,1e-5)))))
        else:raise ValueError(kind)
    def forward(self,x):
        b,t,c,h,w=x.shape
        if self.kind=='gru':return self.net(x.permute(0,3,4,1,2).reshape(-1,t,c)).reshape(b,h,w)
        hidden=x.new_zeros(b,16,h,w);cell=hidden.clone()
        for frame in x.unbind(1):
            i,f,o,g=self.cell(torch.cat([frame,hidden],1)).chunk(4,1)
            cell=f.sigmoid()*cell+i.sigmoid()*g.tanh();hidden=o.sigmoid()*cell.tanh()
        return torch.nn.functional.softplus(self.head(hidden)[:,0])+1e-6
