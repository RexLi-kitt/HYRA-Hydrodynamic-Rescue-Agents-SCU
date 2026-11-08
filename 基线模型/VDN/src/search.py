import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from . import public_model as original_model, boundary as geometry
LOCAL_SIDE=15
LOCAL_DIM=452
ARMS=('original','mask','local_mask')
def local_coverage(o):
    """North/east aligned 10m cells around this UAV, never current hidden water."""
    pos=np.asarray(o.pos,dtype=float);last=np.asarray(o.last)
    if pos.shape!=(4,2) or last.shape!=(100,100):raise ValueError('Public geometry shape')
    offset=np.arange(-7,8);out=[]
    for p in pos:
        col,row=np.floor(p/10).astype(int)
        rr=row+offset[:,None];cc=col+offset[None,:]
        valid=(rr>=0)&(rr<100)&(cc>=0)&(cc<100)
        known=valid&(last[np.clip(rr,0,99),np.clip(cc,0,99)]>=0)
        subcell=p/10-np.floor(p/10)-.5
        out.append(np.r_[known.ravel(),valid.ravel(),subcell])
    return np.asarray(out,np.float32)


def encode(o,arm):
    if arm not in ARMS:raise ValueError(arm)
    x=original_model.encode(o)
    if arm=='local_mask':x=np.concatenate((x,local_coverage(o)),axis=1)
    if arm=='original':
        mask=np.tile(np.r_[True,(np.asarray(o.seen_at)>=0)&np.asarray(o.active,bool)],(4,9))
    else:mask=geometry.action_mask(o)
    # Exact Boolean masks travel with rollout samples; they are not neural features.
    return np.concatenate((x,mask.astype(np.float32)),axis=1)


def residual():
    net=nn.Sequential(nn.Linear(LOCAL_DIM,64),nn.ReLU(),nn.Linear(64,64))
    nn.init.orthogonal_(net[0].weight,gain=2**.5);nn.init.zeros_(net[0].bias)
    nn.init.zeros_(net[2].weight);nn.init.zeros_(net[2].bias)
    return net


class Actor(original_model.Actor):
    def __init__(self,use_local):
        super().__init__();self.use_local=use_local
        if use_local:self.local=residual()
    def distributions(self,x):
        old=x[:,:998]
        if not self.use_local:return super().distributions(old)
        h=self.net(old)+self.local(x[:,998:998+LOCAL_DIM])
        return Categorical(logits=self.move(h)),Categorical(logits=self.target(h).masked_fill(~(old[:,-65:]>.5),-1e9))


class Critic(original_model.Critic):
    def __init__(self,use_local):
        super().__init__();self.use_local=use_local
        if use_local:self.local=residual()
    def forward(self,x):
        old=x[:,:998]
        if not self.use_local:return super().forward(old)
        h=self.net[:-1](old)+self.local(x[:,998:998+LOCAL_DIM])
        return self.net[-1](h)


class Policy:
    def __init__(self,arm,device='cuda:0'):
        self.arm=arm;self.device=device;local=arm=='local_mask'
        self.actor=Actor(local).to(device);self.critic=Critic(local).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=.0005,eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=.0005,eps=1e-5)
    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)
    def joint(self,x):
        m,t=self.actor.distributions(x)
        logits=(m.logits[:,:,None]+t.logits[:,None,:]).reshape(-1,585)
        mask=x[:,-585:]>.5
        if not bool(mask[:,520].all()):raise ValueError('Safe no-target hover absent')
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf))
    def act(self,x,deterministic=False):
        x=self.tensor(x)
        if self.arm=='original':
            m,t=self.actor.distributions(x)
            a=m.probs.argmax(-1) if deterministic else m.sample()
            b=t.probs.argmax(-1) if deterministic else t.sample()
            actions=torch.stack((a,b),dim=-1);logp=m.log_prob(a)+t.log_prob(b)
        else:
            d=self.joint(x);packed=d.probs.argmax(-1) if deterministic else d.sample()
            actions=torch.stack((packed//65,packed%65),dim=-1);logp=d.log_prob(packed)
        return actions,logp.unsqueeze(-1),self.critic(x)
    def evaluate_actions(self,shared,obs,rnn,rnn_critic,actions,masks,available,active):
        x=self.tensor(obs);a=torch.as_tensor(actions,dtype=torch.long,device=self.device)
        if self.arm=='original':
            m,t=self.actor.distributions(x);logp=m.log_prob(a[:,0])+t.log_prob(a[:,1]);entropy=m.entropy()+t.entropy()
        else:
            d=self.joint(x);packed=a[:,0]*65+a[:,1]
            if not bool((x[:,-585:]>.5).gather(1,packed[:,None]).all()):raise ValueError('PPO action illegal in its rollout mask')
            logp=d.log_prob(packed);entropy=d.entropy()
        active=self.tensor(active);entropy=(entropy[:,None]*active).sum()/active.sum()
        return self.critic(self.tensor(shared)),logp[:,None],entropy

