"""Six explicit research implementations. Untested draft until the readiness suite runs.

Sources and deviations are registered in protocol.json. This is not a claim to
reproduce published benchmark scores. No prohibited algorithm is a fallback.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, kl_divergence
from torch.nn import functional as F

ALGORITHMS=('happo','hatrpo','hasac','maac','mat','coma')

def mlp(inputs,outputs,width=64):
    return nn.Sequential(nn.Linear(inputs,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh(),nn.Linear(width,outputs))

def distribution(logits,mask):
    if not mask.any(-1).all(): raise ValueError('No available action')
    return Categorical(logits=logits.masked_fill(~mask,-1e9))

def optimize(loss,optimizer,parameters):
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
    optimizer.zero_grad(set_to_none=True); loss.backward()
    parameters=list(parameters); norm=nn.utils.clip_grad_norm_(parameters,10.,error_if_nonfinite=True)
    optimizer.step(); return float(norm)

def gae(rewards,values,dones,gamma=.995,lam=.95):
    result=torch.zeros_like(rewards); future=0.; next_value=0.
    for t in reversed(range(len(rewards))):
        live=1-dones[t]
        delta=rewards[t]+gamma*next_value*live-values[t]
        future=delta+gamma*lam*live*future; result[t]=future; next_value=values[t]
    return result,result+values

def td_lambda(rewards,next_q,dones,gamma=.995,lam=.8):
    result=torch.zeros_like(next_q); future=torch.zeros_like(next_q[-1])
    for t in reversed(range(len(rewards))):
        future=rewards[t]+gamma*(1-dones[t])*((1-lam)*next_q[t]+lam*future)
        result[t]=future
    return result

def flat_params(module): return torch.cat([p.detach().reshape(-1) for p in module.parameters()])

def set_params(module,vector):
    cursor=0
    with torch.no_grad():
        for p in module.parameters():
            size=p.numel();p.copy_(vector[cursor:cursor+size].view_as(p));cursor+=size

def gradient(value,module,create_graph=False):
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(value,tuple(module.parameters()),create_graph=create_graph)])

def conjugate_gradient(product,b,steps=10):
    x=torch.zeros_like(b); r=b.clone(); p=r.clone(); rr=r@r
    for _ in range(steps):
        if rr<1e-12: break
        z=product(p); alpha=rr/(p@z+1e-12);x+=alpha*p;r-=alpha*z
        new=r@r;p=r+(new/(rr+1e-12))*p;rr=new
    return x

def trust_update(actor,obs,mask,actions,old_logp,weighted_adv,delta=.01):
    """HATRPO: KL-constrained natural-gradient step with backtracking."""
    old=distribution(actor(obs).detach(),mask)
    def objective():
        dist=distribution(actor(obs),mask)
        return (torch.exp(dist.log_prob(actions)-old_logp)*weighted_adv).mean(),kl_divergence(old,dist).mean()
    value,_=objective();g=gradient(value,actor).detach();before=flat_params(actor)
    def product(v):
        _,kl=objective();first=gradient(kl,actor,True)
        return gradient((first*v).sum(),actor).detach()+.1*v
    direction=conjugate_gradient(product,g)
    curvature=direction@product(direction)
    if not torch.isfinite(curvature) or curvature<=0: return dict(accepted=False,kl=0.)
    full=direction*torch.sqrt(2*delta/(curvature+1e-12)); expected=g@full
    for k in range(10):
        fraction=.5**k;set_params(actor,before+fraction*full)
        with torch.no_grad(): new,kl=objective()
        if torch.isfinite(new) and torch.isfinite(kl) and kl<=delta and new-value.detach()>.1*fraction*expected:
            return dict(accepted=True,kl=float(kl))
    set_params(actor,before);return dict(accepted=False,kl=0.)

class CounterfactualCritic(nn.Module):
    """Q_i(a_i), conditioned on allowed joint observation and OTHER actions."""
    def __init__(self,dim):
        super().__init__();self.net=mlp(4*dim+36+4,9)
    def forward(self,obs,actions):
        batch=len(obs);one=F.one_hot(actions,9).float(); result=[]
        for i in range(4):
            other=one.clone();other[:,i]=0
            ids=torch.zeros(batch,4,device=obs.device);ids[:,i]=1
            result.append(self.net(torch.cat((obs.flatten(1),other.flatten(1),ids),-1)))
        return torch.stack(result,1)

class JointQ(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=mlp(4*dim+36,1)
    def forward(self,obs,actions):
        return self.net(torch.cat((obs.flatten(1),F.one_hot(actions,9).float().flatten(1)),-1)).squeeze(-1)

class AttentionCritic(nn.Module):
    """MAAC adaptation: query own observation; attend to other obs-action tokens."""
    def __init__(self,dim):
        super().__init__();self.state=mlp(dim,64);self.token=mlp(dim+9,64)
        self.attention=nn.MultiheadAttention(64,4,batch_first=True,dropout=0.)
        self.output=mlp(128,9)
    def forward(self,obs,actions):
        state=self.state(obs);token=self.token(torch.cat((obs,F.one_hot(actions,9).float()),-1))
        context,_=self.attention(state,token,token,attn_mask=torch.eye(4,dtype=torch.bool,device=obs.device),need_weights=False)
        return self.output(torch.cat((state,context),-1))

class TransformerPolicy(nn.Module):
    """MAT adaptation: observation encoder and causal joint-action decoder."""
    def __init__(self,dim):
        super().__init__();self.input=nn.Linear(dim,64);self.positions=nn.Parameter(torch.zeros(4,64))
        self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,dropout=0.,batch_first=True),1,enable_nested_tensor=False)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(64,4,128,dropout=0.,batch_first=True),1)
        self.action_embed=nn.Embedding(10,64);self.head=nn.Linear(64,9);self.value_head=nn.Linear(64,1)
    def encode(self,obs): return self.encoder(self.input(obs)+self.positions)
    def decode(self,encoded,actions):
        # Start token 9, followed by prior agents' actions. Agent i never sees a_i.
        shifted=torch.cat((torch.full_like(actions[:,:1],9),actions[:,:-1]),1)
        tokens=self.action_embed(shifted)+self.positions
        causal=torch.triu(torch.ones(4,4,device=encoded.device,dtype=torch.bool),diagonal=1)
        return self.head(self.decoder(tokens,encoded,tgt_mask=causal))
    def forward(self,obs,actions):
        encoded=self.encode(obs)
        return self.decode(encoded,actions),self.value_head(encoded).mean(1).squeeze(-1)
    def sample(self,obs,mask,deterministic=False):
        encoded=self.encode(obs);actions=torch.zeros((len(obs),4),device=obs.device,dtype=torch.long);logs=[]
        for i in range(4):
            d=distribution(self.decode(encoded,actions)[:,i],mask[:,i])
            actions[:,i]=d.probs.argmax(-1) if deterministic else d.sample()
            logs.append(d.log_prob(actions[:,i]))
        return actions,torch.stack(logs,1),self.value_head(encoded).mean(1).squeeze(-1)

class Learner(nn.Module):
    def __init__(self,algorithm,dim,device='cpu',gamma=.995):
        super().__init__()
        if algorithm not in ALGORITHMS: raise ValueError('Unregistered algorithm')
        self.algorithm=algorithm;self.dim=dim;self.device_name=device;self.gamma=gamma;self.alpha=.05
        self.updates=0;self.optimizers=[]
        if algorithm=='mat':
            self.policy=TransformerPolicy(dim);self.to(device)
            self.optimizers=[torch.optim.Adam(self.policy.parameters(),lr=3e-4)];return
        self.actors=nn.ModuleList([mlp(dim,9) for _ in range(4)])
        if algorithm in ('happo','hatrpo'): self.critic=mlp(dim*4,1)
        elif algorithm=='coma': self.critic=CounterfactualCritic(dim)
        elif algorithm=='maac': self.critic=AttentionCritic(dim)
        else: self.critic=nn.ModuleList([JointQ(dim),JointQ(dim)])
        if algorithm in ('coma','hasac','maac'):
            self.target=copy.deepcopy(self.critic);self.target.requires_grad_(False)
        self.to(device)
        self.optimizers=[torch.optim.Adam(a.parameters(),lr=3e-4) for a in self.actors]
        self.optimizers.append(torch.optim.Adam(self.critic.parameters(),lr=3e-4))

    def tensor(self,x,dtype=torch.float32): return torch.as_tensor(x,dtype=dtype,device=self.device_name)
    def actor_distributions(self,obs,mask): return [distribution(a(obs[:,i]),mask[:,i]) for i,a in enumerate(self.actors)]
    def value(self,obs):
        if self.algorithm=='mat': return self.policy.value_head(self.policy.encode(obs)).mean(1).squeeze(-1)
        if self.algorithm in ('happo','hatrpo'): return self.critic(obs.flatten(1)).squeeze(-1)
        return torch.zeros(len(obs),device=obs.device)

    @torch.no_grad()
    def act(self,obs,mask,deterministic=False):
        o=self.tensor(obs)[None];m=self.tensor(mask,torch.bool)[None]
        if self.algorithm=='mat': a,l,v=self.policy.sample(o,m,deterministic)
        else:
            ds=self.actor_distributions(o,m)
            a=torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in ds],1)
            l=torch.stack([d.log_prob(a[:,i]) for i,d in enumerate(ds)],1);v=self.value(o)
        return a[0].cpu().numpy(),l[0].cpu().numpy(),float(v[0])

    def batch(self,rows):
        return {k:self.tensor(np.stack([r[k] for r in rows]),torch.long if k in ('action','next_action') else torch.bool if 'mask' in k else torch.float32)
                for k in rows[0]}

    def update_on_policy(self,rows):
        b=self.batch(rows);o=b['obs'];m=b['mask'];a=b['action'];r=b['reward'];done=b['done']
        metrics={};entropy_weight=.01
        if self.algorithm=='coma':
            with torch.no_grad():
                next_values=self.target(b['next_obs'],b['next_action']).gather(-1,b['next_action'][...,None]).squeeze(-1)
                targets=td_lambda(r,next_values,done,self.gamma)
            for _ in range(4):
                q=self.critic(o,a).gather(-1,a[...,None]).squeeze(-1)
                loss=F.mse_loss(q,targets);optimize(loss,self.optimizers[4],self.critic.parameters())
            with torch.no_grad(): all_q=self.critic(o,a)
            for i,d in enumerate(self.actor_distributions(o,m)):
                baseline=(d.probs.detach()*all_q[:,i]).sum(-1)
                adv=(all_q[:,i].gather(1,a[:,i,None]).squeeze(-1)-baseline).detach()
                loss=-(d.log_prob(a[:,i])*adv+entropy_weight*d.entropy()).mean()
                optimize(loss,self.optimizers[i],self.actors[i].parameters())
            self.soft_target(.05);metrics['critic_loss']=float(F.mse_loss(q.detach(),targets))
        else:
            advantage,returns=gae(r,b['value'],done,self.gamma)
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
            if self.algorithm=='mat':
                for _ in range(4):
                    for ids in torch.randperm(len(o),device=o.device).split(128):
                        logits,value=self.policy(o[ids],a[ids]);d=distribution(logits,m[ids])
                        ratio=(d.log_prob(a[ids])-b['logp'][ids]).exp();adv=advantage[ids,None]
                        loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*F.mse_loss(value,returns[ids])-entropy_weight*d.entropy().mean()
                        optimize(loss,self.optimizers[0],self.policy.parameters())
            else:
                for _ in range(4):
                    value=self.value(o);loss=F.mse_loss(value,returns)
                    optimize(loss,self.optimizers[4],self.critic.parameters())
                factor=torch.ones_like(advantage);order=torch.randperm(4).tolist();metrics['update_order']=order
                for i in order:
                    if self.algorithm=='hatrpo':
                        result=trust_update(self.actors[i],o[:,i],m[:,i],a[:,i],b['logp'][:,i],factor*advantage)
                        metrics[f'actor_{i}']=result
                    else:
                        for _ in range(4):
                            for ids in torch.randperm(len(o),device=o.device).split(128):
                                d=distribution(self.actors[i](o[ids,i]),m[ids,i]);ratio=(d.log_prob(a[ids,i])-b['logp'][ids,i]).exp()
                                adv=advantage[ids];loss=-(factor[ids]*torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)).mean()-entropy_weight*d.entropy().mean()
                                optimize(loss,self.optimizers[i],self.actors[i].parameters())
                    with torch.no_grad():
                        new=distribution(self.actors[i](o[:,i]),m[:,i]).log_prob(a[:,i])
                        factor*=torch.exp(new-b['logp'][:,i])
                if not torch.isfinite(factor).all(): raise FloatingPointError('Nonfinite HAPPO/HATRPO correction')
                metrics['max_factor']=float(factor.max())
        self.updates+=1;self.check_finite();return metrics

    @torch.no_grad()
    def sample_joint(self,obs,mask):
        ds=self.actor_distributions(obs,mask);a=torch.stack([d.sample() for d in ds],1)
        logs=torch.stack([d.log_prob(a[:,i]) for i,d in enumerate(ds)],1)
        return a,logs

    def soft_target(self,tau=.01):
        with torch.no_grad():
            for p,q in zip(self.critic.parameters(),self.target.parameters()): q.lerp_(p,tau)

    def update_off_policy(self,rows):
        b=self.batch(rows);o=b['obs'];a=b['action'];m=b['mask']
        with torch.no_grad():
            na,nlog=self.sample_joint(b['next_obs'],b['next_mask'])
            if self.algorithm=='hasac':
                next_q=torch.minimum(*[q(b['next_obs'],na) for q in self.target])-self.alpha*nlog.sum(-1)
                target=b['reward']+self.gamma*(1-b['done'])*next_q
            else:
                next_q=self.target(b['next_obs'],na).gather(-1,na[...,None]).squeeze(-1)-self.alpha*nlog
                target=b['reward'][:,None]+self.gamma*(1-b['done'][:,None])*next_q
        if self.algorithm=='hasac': loss=sum(F.mse_loss(q(o,a),target) for q in self.critic)
        else: loss=F.mse_loss(self.critic(o,a).gather(-1,a[...,None]).squeeze(-1),target)
        optimize(loss,self.optimizers[4],self.critic.parameters())
        order=torch.randperm(4).tolist() if self.algorithm=='hasac' else list(range(4))
        # HASAC uses sequential actor updates; MAAC computes the common pre-update
        # action context once. Exact discrete expectations avoid action rounding.
        context,_=self.sample_joint(o,m)
        for i in order:
            if self.algorithm=='hasac': context,_=self.sample_joint(o,m)
            with torch.no_grad():
                if self.algorithm=='hasac':
                    candidates=[]
                    for action in range(9):
                        joint=context.clone();joint[:,i]=action
                        candidates.append(torch.minimum(*[q(o,joint) for q in self.critic]))
                    q_values=torch.stack(candidates,-1)
                else: q_values=self.critic(o,context)[:,i]
            d=distribution(self.actors[i](o[:,i]),m[:,i])
            actor_loss=(d.probs*(self.alpha*d.logits-q_values)).sum(-1).mean()
            optimize(actor_loss,self.optimizers[i],self.actors[i].parameters())
        self.soft_target();self.updates+=1;self.check_finite()
        return dict(critic_loss=float(loss.detach()),update_order=order)

    def check_finite(self):
        if any(not torch.isfinite(p).all() for p in self.parameters()): raise FloatingPointError('Nonfinite model parameter')

    def checkpoint(self):
        return dict(algorithm=self.algorithm,dim=self.dim,gamma=self.gamma,alpha=self.alpha,updates=self.updates,
                    model=self.state_dict(),optimizers=[x.state_dict() for x in self.optimizers])

    def restore(self,state):
        if state['algorithm']!=self.algorithm or state['dim']!=self.dim: raise ValueError('Checkpoint identity mismatch')
        self.load_state_dict(state['model']);self.updates=state['updates'];self.alpha=state['alpha']
        for optimizer,values in zip(self.optimizers,state['optimizers']): optimizer.load_state_dict(values)
