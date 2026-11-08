"""Public-information encoder and factored joint-action MAPPO policy.

No target planner: both movement and target are sampled from policy heads.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

DIM=998

def encode(o):
    """Encode only attributes of the referee's public observation object."""
    t=float(o.t); last=np.asarray(o.last); known=last>=0
    measured=np.asarray(o.measured)
    maps=np.stack((known.astype(float),np.where(known,np.clip(measured[0]/2.,0,1),0),
                   np.where(known,np.clip(measured[1]/3.,-1,1),0),
                   np.where(known,np.clip(measured[2]/3.,-1,1),0)))
    pooled=maps.reshape(4,10,10,10,10).mean(axis=(2,4)).reshape(-1)
    team=np.r_[np.asarray(o.pos).reshape(-1)/1000.,np.asarray(o.heading).reshape(-1),t/2400.]
    observed=np.asarray(o.seen_at)>=0;active=np.asarray(o.active,dtype=bool)
    possible=observed&active
    xy=np.where(observed[:,None],np.asarray(o.seen_xy)/1000.,0.)
    v=np.where(observed[:,None],np.clip(np.asarray(o.seen_v)/3.,-1.,1.),0.)
    age=np.where(observed,np.minimum(1.,np.maximum(0.,t-np.asarray(o.seen_at))/2400.),0.)
    rows=[]
    for i in range(4):
        person=np.column_stack((xy,v,age,np.asarray(o.visible,dtype=float),active.astype(float),
                                (np.asarray(o.owner)==i).astype(float))).reshape(-1)
        own=np.eye(4,dtype=float)[i];target_mask=np.r_[1.,possible.astype(float)]
        rows.append(np.r_[pooled,team,person,own,target_mask])
    out=np.asarray(rows,dtype=np.float32)
    if out.shape!=(4,DIM) or not np.isfinite(out).all():raise ValueError('invalid public features')
    return out

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,64),nn.ReLU(),nn.LayerNorm(64),
                               nn.Linear(64,64),nn.ReLU(),nn.LayerNorm(64))
        self.move=nn.Linear(64,9);self.target=nn.Linear(64,65)
        for layer in self.net:
            if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,gain=2**.5);nn.init.zeros_(layer.bias)
        for layer in (self.move,self.target):nn.init.orthogonal_(layer.weight,gain=.01);nn.init.zeros_(layer.bias)
    def distributions(self,x):
        h=self.net(x);mask=x[:,-65:]>.5
        if not bool(mask[:,0].all()):raise ValueError('no-target must be available')
        return Categorical(logits=self.move(h)),Categorical(logits=self.target(h).masked_fill(~mask,-1e9))

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,64),nn.ReLU(),nn.LayerNorm(64),
                               nn.Linear(64,64),nn.ReLU(),nn.LayerNorm(64),nn.Linear(64,1))
        for layer in self.net:
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,gain=1. if layer.out_features==1 else 2**.5);nn.init.zeros_(layer.bias)
    def forward(self,x):return self.net(x)

class Policy:
    def __init__(self,device,lr=.0005):
        self.device=device;self.actor=Actor().to(device);self.critic=Critic().to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=lr,eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=lr,eps=1e-5)
    def tensor(self,x):return torch.as_tensor(x,dtype=torch.float32,device=self.device)
    def act(self,obs,deterministic=False):
        x=self.tensor(obs);m,t=self.actor.distributions(x)
        a=m.probs.argmax(-1) if deterministic else m.sample();b=t.probs.argmax(-1) if deterministic else t.sample()
        return torch.stack((a,b),dim=-1), (m.log_prob(a)+t.log_prob(b)).unsqueeze(-1),self.critic(x)
    def evaluate_actions(self,shared,obs,rnn,rnn_critic,actions,masks,available,active):
        x=self.tensor(obs);a=torch.as_tensor(actions,dtype=torch.long,device=self.device)
        m,t=self.actor.distributions(x)
        logp=(m.log_prob(a[:,0])+t.log_prob(a[:,1])).unsqueeze(-1)
        entropy=(m.entropy()+t.entropy()).unsqueeze(-1)
        active=self.tensor(active);entropy=(entropy*active).sum()/active.sum()
        return self.critic(self.tensor(shared)),logp,entropy
