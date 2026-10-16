"""Full-source nonlinear controls, using native PyTorch MLP/GRU layers.

FloodGNN-GRU's pinned code motivated an explicit temporal control. This is an
independent upstream-only implementation, not a reproduction of its graph model.
"""
import numpy as np
import torch
from torch import nn


class SourceTargetNetwork(nn.Module):
    def __init__(self,common_size,kind='mlp'):
        super().__init__();self.kind=kind
        if kind=='mlp':
            self.encoder=nn.Sequential(nn.Linear(432,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU());encoded=128
        elif kind=='gru':
            self.encoder=nn.GRU(6,32,batch_first=True);encoded=6*32
        else:raise ValueError('Unknown nonlinear control')
        self.reader=nn.Sequential(nn.Linear(common_size+encoded,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,26))

    def forward(self,common,history):
        b=len(history)
        if self.kind=='mlp':encoded=self.encoder(history.reshape(b,432))
        else:
            # Stored records run recent-to-old; recurrence consumes old-to-recent.
            sequence=history.flip(2).reshape(b*6,12,6)
            _,hidden=self.encoder(sequence);encoded=hidden[0].reshape(b,6*32)
        encoded=encoded[:,None,:].expand(-1,common.shape[1],-1)
        return .05*self.reader(torch.cat([common,encoded],dim=-1))


def normalization(common,history,weights):
    weights=np.array(weights,float,copy=True);weights/=weights.sum()
    mean=np.einsum('n,ntf->f',weights,common)/common.shape[1]
    var=np.einsum('n,ntf->f',weights,(common-mean)**2)/common.shape[1]
    hm=np.einsum('n,nsbf->f',weights,history)/(6*12)
    hv=np.einsum('n,nsbf->f',weights,(history-hm)**2)/(6*12)
    return dict(common_mean=mean,common_scale=np.maximum(np.sqrt(var),1e-5),
        history_mean=hm,history_scale=np.maximum(np.sqrt(hv),1e-5))


def predict_network(model,normalization_,common,history,device='cuda',batch_size=512):
    model.eval();out=[]
    with torch.no_grad():
        for begin in range(0,len(common),batch_size):
            c=(common[begin:begin+batch_size]-normalization_['common_mean'])/normalization_['common_scale']
            h=(history[begin:begin+batch_size]-normalization_['history_mean'])/normalization_['history_scale']
            out.append(model(torch.as_tensor(c,dtype=torch.float32,device=device),torch.as_tensor(h,dtype=torch.float32,device=device)).cpu().numpy())
    return np.concatenate(out)


class PhysicalSourceTargetNetwork(nn.Module):
    """Identical full-raw encoder for P4/P5; each lead sees its own response only."""
    def __init__(self,common_size,kind='mlp'):
        super().__init__();self.kind=kind
        self.encoder=SourceTargetNetwork(common_size,kind).encoder
        encoded=128 if kind=='mlp' else 192
        self.reader=nn.Sequential(nn.Linear(common_size+encoded+4,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,common,history,physical,global_component,probability):
        b=len(history)
        if self.kind=='mlp':encoded=self.encoder(history.reshape(b,432))
        else:
            _,hidden=self.encoder(history.flip(2).reshape(b*6,12,6));encoded=hidden[0].reshape(b,192)
        public=torch.cat([common,encoded[:,None,:].expand(-1,12,-1)],dim=-1)
        public=public[:,:,None,:].expand(-1,-1,26,-1)
        horizon=torch.arange(26,device=common.device,dtype=common.dtype)[None,None,:].expand(b,12,-1)/25.
        probability=probability[:,None,None].expand(-1,12,26)
        extra=torch.stack([horizon,physical/.05,global_component/.05,probability],dim=-1)
        return physical+.05*self.reader(torch.cat([public,extra],dim=-1))[...,0]


def predict_physical_network(model,normalization_,common,history,physical,global_component,probability,device='cuda',batch_size=256):
    model.eval();out=[]
    with torch.no_grad():
        for begin in range(0,len(common),batch_size):
            sl=slice(begin,begin+batch_size)
            c=(common[sl]-normalization_['common_mean'])/normalization_['common_scale']
            h=(history[sl]-normalization_['history_mean'])/normalization_['history_scale']
            tensors=[torch.as_tensor(x,dtype=torch.float32,device=device) for x in (c,h,physical[sl],global_component[sl],probability[sl])]
            out.append(model(*tensors).cpu().numpy())
    return np.concatenate(out)
