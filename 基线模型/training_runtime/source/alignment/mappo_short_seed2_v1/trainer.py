"""Use the checked-in MAPPO PPO update, with public-obs factored action policy."""
import sys
from pathlib import Path
import numpy as np
import torch
from model import Policy

UPSTREAM=Path(r'D:\HYRA-MAPPO-20261031\upstream\on-policy')
sys.path.insert(0,str(UPSTREAM))
from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO

NORM_KEYS=('running_mean','running_mean_sq','debiasing_term')
def norm_state(trainer):
    # Upstream ValueNorm attributes can be unregistered Tensors on CUDA, so
    # its ordinary state_dict() may be empty even after updates.
    return {key:getattr(trainer.value_normalizer,key).detach().clone() for key in NORM_KEYS}

def load_norm_state(trainer,state):
    if set(state)!=set(NORM_KEYS):raise ValueError('incomplete ValueNorm state')
    with torch.no_grad():
        for key in NORM_KEYS:
            target=getattr(trainer.value_normalizer,key);target.copy_(state[key].to(target.device))

def make(device):
    args=get_config().parse_args([]);args.use_recurrent_policy=False;args.use_naive_recurrent_policy=False
    args.ppo_epoch=5;args.num_mini_batch=1;args.lr=.0005;args.critic_lr=.0005
    args.gamma=.99;args.gae_lambda=.95;args.clip_param=.2;args.entropy_coef=.01
    p=Policy(device,lr=args.lr);return args,p,R_MAPPO(args,p,device=device)

def update(trainer,x,actions,old_logp,values,returns,advantages,epochs=5):
    x=np.asarray(x,np.float32);a=np.asarray(actions,np.int64);N=len(x)
    old=np.asarray(old_logp,np.float32).reshape(N,1);values=np.asarray(values,np.float32).reshape(N,1)
    returns=np.asarray(returns,np.float32).reshape(N,1);adv=np.asarray(advantages,np.float32).reshape(N,1)
    adv=(adv-adv.mean())/(adv.std()+1e-5)
    ones=np.ones((N,1),np.float32);dummy=np.zeros((N,1,64),np.float32)
    sample=(x,x,dummy,dummy,a,values,returns,ones,ones,old,adv,None)
    stats=[]
    for _ in range(epochs):
        value_loss,cgrad,policy_loss,entropy,agrad,ratio=trainer.ppo_update(sample)
        tensors=[value_loss,cgrad,policy_loss,entropy,agrad,ratio]
        if not all(bool(torch.isfinite(torch.as_tensor(v)).all()) for v in tensors):raise FloatingPointError('nonfinite PPO output')
        grads=[p.grad for p in list(trainer.policy.actor.parameters())+list(trainer.policy.critic.parameters()) if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):raise FloatingPointError('nonfinite or absent gradients')
        # PPO provides the pre-update ratio from this epoch. This diagnostic
        # adds no gradients and does not affect the sampled policy or update.
        r=ratio.detach(); safe=r.clamp_min(1e-12)
        kl=((r-1.)-safe.log()).mean()
        clip_fraction=((r-1.).abs()>trainer.clip_param).float().mean()
        if not bool(torch.isfinite(kl)) or not bool(torch.isfinite(clip_fraction)):raise FloatingPointError('nonfinite KL diagnostic')
        def norm(module):
            parts=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None]
            return float(torch.stack(parts).sum().sqrt().item()) if parts else 0.
        stats.append(dict(value_loss=float(value_loss.item()),policy_loss=float(policy_loss.item()),entropy=float(entropy.item()),
            actor_grad_norm=float(torch.as_tensor(agrad).item()),critic_grad_norm=float(torch.as_tensor(cgrad).item()),
            ratio_mean=float(ratio.mean().item()),approx_kl=float(kl.item()),clip_fraction=float(clip_fraction.item()),
            move_head_grad_norm_after_clip=norm(trainer.policy.actor.move),target_head_grad_norm_after_clip=norm(trainer.policy.actor.target),
            gradient_devices=sorted({str(g.device) for g in grads})))
    return stats
