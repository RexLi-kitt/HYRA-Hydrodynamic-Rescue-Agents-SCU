"""Compact recurrent contexts and a fresh replay for a frozen original RNN.

Context values are computed from public observations and real previous actions.
The new replay is not a migration of the old replay, which lacks local patches.
"""
import copy
import numpy as np
import torch
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('dqn_frontier_v2_replay_core',Path(__file__).resolve().parent/'policy_core.py')
p=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(p)


@torch.no_grad()
def capture(model,o,previous,hidden):
    device=next(model.parameters()).device
    x=p.encode(o);xt=torch.as_tensor(x,dtype=torch.float32,device=device)
    last=p.last_action_onehot(previous,device)
    old_q,next_hidden=model.base(torch.cat((xt[:,:998],last),dim=-1),hidden)
    people=x[:,p.BASE:p.BASE+64*p.PERSON_DIM].reshape(4,64,p.PERSON_DIM)
    actions=x[:,p.BASE+64*p.PERSON_DIM:p.BASE+p.FEATURE_DIM].reshape(4,64,9,p.ACTION_DIM)
    idx=np.argwhere(people[:,:,0]>.5)
    patch=x[:,998:p.BASE]
    if not np.all((patch[:,:450]==0)|(patch[:,:450]==1)):raise ValueError('Patch must be Boolean')
    state=dict(public=x[:,:998].copy(),base_q=old_q.cpu().numpy().copy(),hidden=next_hidden.cpu().numpy().copy(),
        patch_bits=np.packbits(patch[:,:450].astype(np.uint8),axis=1),subcell=patch[:,450:].copy(),
        mask_bits=np.packbits(x[:,-585:].astype(np.uint8),axis=1),
        indices=idx.astype(np.uint8),people=people[idx[:,0],idx[:,1]].copy(),
        actions=actions[idx[:,0],idx[:,1]].copy())
    return state,next_hidden.detach()


def tensor_batch(states,device):
    n=len(states)*4
    if not states:raise ValueError('Empty state batch')
    patches=[];base=[];hidden=[];masks=[];public=[]
    people=np.zeros((n,64,p.PERSON_DIM),np.float32)
    actions=np.zeros((n,64,9,p.ACTION_DIM),np.float32)
    for k,state in enumerate(states):
        patches.append(np.concatenate((np.unpackbits(state['patch_bits'],axis=1,count=450).astype(np.float32),state['subcell']),axis=1))
        base.append(state['base_q']);hidden.append(state['hidden']);public.append(state['public'])
        masks.append(np.unpackbits(state['mask_bits'],axis=1,count=585).astype(bool))
        ids=state['indices'].astype(np.int64)
        if len(ids):
            if (ids[:,0]>=4).any() or (ids[:,1]>=64).any():raise ValueError('Bad sparse context index')
            people[4*k+ids[:,0],ids[:,1]]=state['people']
            actions[4*k+ids[:,0],ids[:,1]]=state['actions']
    def t(x,dtype=torch.float32):return torch.as_tensor(x,dtype=dtype,device=device)
    return dict(patch=t(np.concatenate(patches)),people=t(people),actions=t(actions),
        base_q=t(np.concatenate(base)),new_hidden=t(np.concatenate(hidden)),public=t(np.concatenate(public)),
        mask=t(np.concatenate(masks),torch.bool))


def qvalues(model,states,device):
    batch=tensor_batch(states,device);mask=batch.pop('mask')
    q=model.from_context(**batch)
    if not bool(mask[:,520].all()):raise ValueError('Safe action absent')
    return q,mask


def choose(model,state,device,epsilon):
    with torch.no_grad():
        q,mask=qvalues(model,[state],device)
        greedy=q.masked_fill(~mask,-torch.inf).argmax(-1)
        # Match PyMARL's draw order, including RNG consumption at epsilon=0.
        picked=torch.rand(len(q),device=device)<epsilon
        random_action=torch.distributions.Categorical(probs=mask.float()).sample()
        return torch.where(picked,random_action,greedy).cpu().numpy()


class Replay:
    def __init__(self,capacity=8192):
        self.capacity=capacity;self.items=[];self.index=0;self.draws=0
    def insert(self,state,actions,reward,next_state,terminated,truncated=False):
        a=np.asarray(actions,dtype=np.int64)
        if a.shape!=(4,) or ((a<0)|(a>=585)).any():raise ValueError('Invalid replay action')
        mask=np.unpackbits(state['mask_bits'],axis=1,count=585).astype(bool)
        if not mask[np.arange(4),a].all():raise ValueError('Replay action illegal')
        if terminated and truncated:raise ValueError('Ambiguous end')
        item=dict(state=state,action=a.copy(),reward=float(reward),next_state=next_state,
                  terminated=bool(terminated),truncated=bool(truncated))
        if len(self.items)<self.capacity:self.items.append(item)
        else:self.items[self.index]=item
        self.index=(self.index+1)%self.capacity
    def sample(self,count):
        if len(self.items)<count:raise ValueError('Replay warmup incomplete')
        ids=np.random.choice(len(self.items),count,replace=False);self.draws+=count
        return [self.items[int(i)] for i in ids],ids.tolist()
    def state(self):return copy.deepcopy(vars(self))
    def restore(self,state):self.__dict__.update(copy.deepcopy(state))


def double_dqn_update(online,target,optimizer,items,device,gamma=.99):
    current=[r['state'] for r in items];following=[r['next_state'] for r in items]
    q,mask=qvalues(online,current,device)
    acts=torch.as_tensor(np.concatenate([x['action'] for x in items]),device=device,dtype=torch.long)
    if not bool(mask.gather(1,acts[:,None]).all()):raise ValueError('Stored action outside stored mask')
    selected=q.gather(1,acts[:,None]).squeeze(1)
    with torch.no_grad():
        q_next,next_mask=qvalues(online,following,device)
        best=q_next.masked_fill(~next_mask,-torch.inf).argmax(-1)
        target_next,target_mask=qvalues(target,following,device)
        if not torch.equal(next_mask,target_mask):raise AssertionError('Online/target masks differ')
        bootstrap=target_next.gather(1,best[:,None]).squeeze(1)
        reward=torch.as_tensor(np.repeat([r['reward'] for r in items],4),device=device,dtype=torch.float32)
        terminated=torch.as_tensor(np.repeat([r['terminated'] for r in items],4),device=device,dtype=torch.float32)
        # Artificial truncation preserves bootstrap; the true task end does not.
        expected=reward+gamma*(1.-terminated)*bootstrap
    loss=(selected-expected).square().mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TD loss')
    optimizer.zero_grad(set_to_none=True)
    if not loss.requires_grad:
        return dict(loss=float(loss.detach()),grad_norm=0.,td_abs=float((selected-expected).detach().abs().mean()),
                    agent_transitions=len(acts),updated=False,skip_reason='no_trainable_path')
    loss.backward()
    gradients=[p.grad for p in online.parameters() if p.requires_grad and p.grad is not None]
    if not all(torch.isfinite(g).all() for g in gradients):raise FloatingPointError('Nonfinite TD gradient')
    if not gradients or not any(bool((g!=0).any()) for g in gradients):
        optimizer.zero_grad(set_to_none=True)
        return dict(loss=float(loss.detach()),grad_norm=0.,td_abs=float((selected-expected).detach().abs().mean()),
                    agent_transitions=len(acts),updated=False,skip_reason='zero_gradient')
    grad=torch.nn.utils.clip_grad_norm_([p for p in online.parameters() if p.requires_grad],10.)
    optimizer.step()
    return dict(loss=float(loss.detach()),grad_norm=float(grad),td_abs=float((selected-expected).detach().abs().mean()),
                agent_transitions=len(acts),updated=True,skip_reason=None)
