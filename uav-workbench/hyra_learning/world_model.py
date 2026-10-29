"""Training-only ten-second forecast and local SWE consistency auxiliary.

Inputs are current clean recurrent actor features and static bed/coordinates.
True current/future fields, local rates and current sources are label/loss data,
never encoder inputs. No imagined policy samples or prediction-memory updates.
"""
from pathlib import Path

from .runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .physics import SWEOperator
from .training import Learner


class WorldHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(hidden+3,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,9))
        self.register_buffer('rate_scale',torch.tensor([.1,.05,.05])[None,:,None,None])

    def forward(self, recurrent_context, bed):
        # Same known static prior as the actor; retain datum precision until
        # normalization before converting to the neural network dtype.
        b, nz, nx = bed.shape
        local_bed = bed-bed.mean(dim=(-2,-1),keepdim=True)
        scale = bed.std(dim=(-2,-1),keepdim=True,correction=0).clamp_min(.05)
        local_bed = (local_bed/scale).to(recurrent_context.dtype)
        z,x = torch.meshgrid(torch.linspace(-1,1,nz,device=bed.device,dtype=recurrent_context.dtype),
                             torch.linspace(-1,1,nx,device=bed.device,dtype=recurrent_context.dtype),indexing='ij')
        coordinates = torch.stack((x,z),dim=-1)[None].expand(b,-1,-1,-1)
        context = recurrent_context[:,None,None].expand(-1,nz,nx,-1)
        raw = self.decoder(torch.cat((context,local_bed[...,None],coordinates),dim=-1)).permute(0,3,1,2)
        def state(value):
            depth = 5*torch.sigmoid(value[:,0:1]-5.1)
            momentum = depth*10*torch.tanh(value[:,1:3])
            return torch.cat((depth,momentum),dim=1)
        return dict(current=state(raw[:,:3]),rate=raw[:,3:6]*self.rate_scale,future=state(raw[:,6:9]))


class AuxiliaryLabels:
    def __init__(self, entries):
        self.entries = entries
        self.arrays = []
        for entry in entries:
            with np.load(entry['auxiliary_path'],allow_pickle=False) as packed:
                self.arrays.append({key:packed[key] for key in packed.files})

    def batch(self, case_indices, time_indices, device):
        rows = []
        for case, t in zip(case_indices,time_indices):
            item = self.arrays[int(case)]
            t = int(t)
            rows.append(dict(current=item['U'][2*t],future=item['U'][2*t+2],rate=item['rate'][t],
                             dt=item['dt'][t],source=item['source'][t],bed=item['bed'],
                             drains=item['drains'],manning=item['manning'],solid=item['solid'],
                             wall_x=item['wall_x'],wall_y=item['wall_y']))
        result = {}
        for key in rows[0]:
            array = np.stack([r[key] for r in rows])
            dtype = torch.bool if array.dtype == bool else torch.float64 if key in ('bed','dt') else torch.float32
            result[key] = torch.as_tensor(array,dtype=dtype,device=device)
        return result


def balanced_weights(labels):
    wet = (labels['current'][:,0] >= .05) | (labels['future'][:,0] >= .05)
    nw = wet.sum(dim=(-2,-1),keepdim=True)
    nd = (~wet).sum(dim=(-2,-1),keepdim=True)
    classes = ((nw > 0).float()+(nd > 0).float()).clamp_min(1)
    return (wet.float()/nw.clamp_min(1)+(~wet).float()/nd.clamp_min(1))/classes


def field_loss(prediction,target,weights):
    scale = prediction.new_tensor([.1,.05,.05])[None,:,None,None]
    errors = F.smooth_l1_loss(prediction/scale,target/scale,reduction='none').mean(dim=1)
    return (errors*weights).sum(dim=(-2,-1)).mean()


def supervised_loss(predictions,labels):
    weights = balanced_weights(labels)
    return (field_loss(predictions['current'],labels['current'],weights)
            +field_loss(predictions['future'],labels['future'],weights)
            +.25*field_loss(predictions['rate'],labels['rate'],weights))


def conservation_loss(predictions,labels):
    # dt is a training label time coordinate, not an input to the head/actor.
    # A short, accepted internal step avoids a false 5s/10s PDE difference.
    current, rate = predictions['current'],predictions['rate']
    evolved = torch.zeros_like(current)
    for dt in torch.unique(labels['dt'].detach().cpu()).tolist():
        indices = torch.nonzero(labels['dt'] == dt).flatten()
        result = SWEOperator().step(current[indices],labels['bed'][indices],dt,
                  source=labels['source'][indices],drains=labels['drains'][indices],manning=labels['manning'][indices],
                  solid=labels['solid'][indices],wall_x=labels['wall_x'][indices],wall_y=labels['wall_y'][indices])
        evolved = evolved.index_copy(0,indices,result)
    physical_rate = (evolved-current)/labels['dt'][:,None,None,None].to(current.dtype)
    return field_loss(rate,physical_rate,balanced_weights(labels))


def clean_recurrent_context(actor,batch):
    b,times,n = batch['images'].shape[:3]
    hidden = actor.initial_hidden(b*n,batch['images'].device)
    states = []
    for t in range(times):
        previous = torch.zeros(b*n,4,device=hidden.device) if t == 0 else F.one_hot(batch['actions'][:,t-1].reshape(-1),4).float()
        _,hidden,_,_ = actor(batch['images'][:,t].reshape(b*n,*actor.image_shape),
                              batch['poses'][:,t].reshape(b*n,6),previous,t,hidden,enhance=False)
        states.append(hidden.reshape(b,n,-1).mean(dim=1))
    return torch.stack(states,dim=1)


class WorldLearner(Learner):
    def __init__(self,n_agents,state_dim,plan,device,method,labels,seed):
        super().__init__(n_agents,state_dim,plan,device,method != 'QMIX')
        self.method,self.labels = method,labels
        self.auxiliary_rng = np.random.default_rng(seed+419)
        self.world = WorldHead().to(device) if method in ('Sur-QMIX+Data','Sur-QMIX+Physics') else None
        if self.world is not None:
            extra = list(self.world.parameters())
            self.parameters += extra
            self.optimizer.add_param_group(dict(params=extra))

    def additional_loss(self,batch):
        if self.world is None:
            return None,{}
        contexts = clean_recurrent_context(self.actor,batch)
        b_indices,t_indices = [],[]
        anchors = self.plan['auxiliary']['training_anchors_per_episode']
        for b,length in enumerate(batch['lengths'].detach().cpu().tolist()):
            chosen = self.auxiliary_rng.choice(length,min(anchors,length),replace=False)
            b_indices.extend([b]*len(chosen));t_indices.extend(chosen.tolist())
        selected_context = contexts[b_indices,t_indices]
        selected_cases = batch['case_index'][b_indices,t_indices].detach().cpu().tolist()
        labels = self.labels.batch(selected_cases,t_indices,self.device)
        prediction = self.world(selected_context,labels['bed'])
        data_loss = supervised_loss(prediction,labels)
        physics_loss = conservation_loss(prediction,labels) if self.method == 'Sur-QMIX+Physics' else data_loss.new_zeros(())
        weight = self.plan['auxiliary']['auxiliary_weight']
        relative = self.plan['auxiliary']['physics_weight_relative_to_auxiliary']
        result = weight*(data_loss+relative*physics_loss)
        return result,dict(auxiliary_data_loss=float(data_loss.detach().cpu()),
                           auxiliary_physics_loss=float(physics_loss.detach().cpu()),auxiliary_anchors=len(t_indices))

    def checkpoint(self,replay,rng,episode,history):
        checkpoint = super().checkpoint(replay,rng,episode,history)
        checkpoint.update(world=self.world.state_dict() if self.world is not None else None,
                          auxiliary_rng=self.auxiliary_rng.bit_generator.state,method=self.method)
        return checkpoint

    def restore(self,checkpoint,replay,rng):
        if checkpoint['method'] != self.method:
            raise ValueError('Checkpoint method does not match')
        if self.world is not None:
            self.world.load_state_dict(checkpoint['world'])
        self.auxiliary_rng.bit_generator.state = checkpoint['auxiliary_rng']
        super().restore(checkpoint,replay,rng)
