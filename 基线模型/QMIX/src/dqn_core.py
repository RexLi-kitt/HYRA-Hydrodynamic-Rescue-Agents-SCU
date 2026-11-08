from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from . import rescue as geometry
from .rnn_agent import RNNAgent as RNN
N_ACTIONS=585
BASE=geometry.BASE
PERSON_DIM=geometry.PERSON_DIM
ACTION_DIM=geometry.ACTION_DIM
FEATURE_DIM=geometry.FEATURE_DIM
LOCAL_DIM=geometry.search.LOCAL_DIM
def encode(o):
    return geometry.encode(o)


def last_action_onehot(actions,device):
    a=torch.as_tensor(actions,dtype=torch.long,device=device)
    if bool(((a < -1)|(a>=N_ACTIONS)).any()):raise ValueError('Previous action must be -1 or 0..584')
    result=torch.zeros((*a.shape,N_ACTIONS),dtype=torch.float32,device=device)
    valid=a>=0
    if bool(valid.any()):result[valid]=nn.functional.one_hot(a[valid],N_ACTIONS).float()
    return result


def mlp(input_size,output_size):
    net=nn.Sequential(nn.Linear(input_size,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,output_size))
    nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
    return net


class RecurrentQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=RNN(998+N_ACTIONS,SimpleNamespace(rnn_hidden_dim=64,n_actions=N_ACTIONS))
        self.search=mlp(LOCAL_DIM+64+998,9)
        self.rescue=mlp(PERSON_DIM+ACTION_DIM+9+64,1)

    def init_hidden(self,n,device=None):
        device=device or next(self.parameters()).device
        return torch.zeros((n,64),dtype=torch.float32,device=device)

    def forward(self,features,previous,hidden):
        if features.ndim!=2 or features.shape[-1]!=geometry.INPUT_DIM:raise ValueError('Bad public feature shape')
        n=features.shape[0]
        if previous.shape!=(n,N_ACTIONS) or hidden.shape!=(n,64):raise ValueError('Bad recurrent state shape')
        base_q,new_hidden=self.base(torch.cat((features[:,:998],previous),dim=-1),hidden)
        patch=features[:,998:BASE]
        split=BASE+64*PERSON_DIM
        people=features[:,BASE:split].reshape(n,64,PERSON_DIM)
        actions=features[:,split:BASE+FEATURE_DIM].reshape(n,64,9,ACTION_DIM)
        q=self.from_context(patch,people,actions,base_q,new_hidden,features[:,:998])
        return q,new_hidden

    def from_context(self,patch,people,actions,base_q,new_hidden,public):
        n=patch.shape[0]
        if (patch.shape!=(n,LOCAL_DIM) or people.shape!=(n,64,PERSON_DIM) or
            actions.shape!=(n,64,9,ACTION_DIM) or base_q.shape!=(n,N_ACTIONS) or new_hidden.shape!=(n,64) or public.shape!=(n,998)):
            raise ValueError('Bad fixed-context feature shape')
        move_q=self.search(torch.cat((patch,new_hidden,public),dim=-1))
        joint=base_q.reshape(n,9,65)+move_q[:,:,None]
        idx=torch.nonzero(people[:,:,0]>.5,as_tuple=False)
        if idx.numel():
            batch,person=idx[:,0],idx[:,1];k=len(batch)
            move=torch.eye(9,dtype=patch.dtype,device=patch.device)[None].expand(k,9,9)
            inputs=torch.cat((people[batch,person,None,:].expand(k,9,PERSON_DIM),actions[batch,person],move,
                              new_hidden[batch,None,:].expand(k,9,64)),dim=-1)
            scores=self.rescue(inputs).squeeze(-1)
            delta=torch.zeros((n,64,9),device=patch.device,dtype=patch.dtype).index_put((batch,person),scores)
            joint=joint+torch.cat((torch.zeros_like(delta[:,:1]),delta),dim=1).transpose(1,2)
        q=joint.reshape(n,N_ACTIONS)
        if not torch.isfinite(q).all():raise FloatingPointError('Nonfinite Q')
        return q


def legal_q(q,features):
    mask=features[:,-N_ACTIONS:]>.5
    if not bool(mask[:,520].all()):raise ValueError('Safe hover absent')
    return q.masked_fill(~mask,-torch.inf)


@torch.no_grad()
def select(q,features,epsilon=0.):
    if not 0.<=epsilon<=1.:raise ValueError('Bad epsilon')
    masked=legal_q(q,features);mask=features[:,-N_ACTIONS:]>.5
    greedy=masked.argmax(-1)
    picked=torch.rand(len(q),device=q.device)<epsilon
    random_action=torch.distributions.Categorical(probs=mask.float()).sample()
    return torch.where(picked,random_action,greedy)
