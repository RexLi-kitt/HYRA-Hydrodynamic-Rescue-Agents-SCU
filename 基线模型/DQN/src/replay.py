import numpy as np
import torch
from . import dqn_core as p
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

