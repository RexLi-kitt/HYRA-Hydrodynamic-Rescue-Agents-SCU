"""Nine-channel, eight-action QMIX/Sur-QMIX engineering pilot.

This is a new model family, never a relabelled four-action checkpoint.
Training budget here checks execution/update/recovery, not paper performance.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,json,math,sys,time
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hyra_learning.vendor.pymarl_qmix import QMixer
from .hydraulics import HydraulicFrames,file_digest
from .environment import RescueEnv,RescueConfig,ACTIONS

class RescueAgent(nn.Module):
    def __init__(self):
        super().__init__();self.n_actions=8;self.hidden_dim=64
        self.conv=nn.Sequential(nn.Conv2d(9,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.Flatten())
        self.image=nn.Linear(400,32);self.action=nn.Linear(8,32);self.gru=nn.GRUCell(96,64);self.output=nn.Linear(64,8)
        self.register_buffer('direction',torch.ones(32)/math.sqrt(32))
    def hidden(self,n):return torch.zeros(n,64,device=self.direction.device)
    def decode(self,z,previous,t,hidden):
        frequency=torch.exp(torch.arange(0,32,2,device=z.device)*(-math.log(10000.)/32))
        time_=torch.stack(((t*frequency).sin(),(t*frequency).cos()),dim=-1).flatten()[None].expand(len(z),-1)
        h=self.gru(torch.cat((z,self.action(previous),time_),dim=-1),hidden);return self.output(h),h
    def forward(self,images,previous,t,hidden,enhance=False,valid=None):
        if images.shape[1:]!=(9,5,5) or previous.shape!=(len(images),8):raise ValueError('Expected nine-channel observations and eight previous actions')
        z=F.relu(self.image(self.conv(images)));surprise=z.new_zeros(len(z))
        if enhance:
            if not self.training or not torch.is_grad_enabled():raise RuntimeError('Sur enhancement is training-only')
            virtual,_=self.decode(z,previous,t,hidden)
            gradient=torch.autograd.grad(virtual.max(dim=-1).values.sum(),z,retain_graph=True,create_graph=False)[0]
            norm=gradient.detach().norm(dim=-1);mask=torch.ones(len(z),dtype=torch.bool,device=z.device) if valid is None else valid.bool()
            if mask.any():
                low,high=norm[mask].min(),norm[mask].max();surprise=torch.where(mask,(norm-low)/(high-low).clamp_min(1e-12),0.)
                z=z+surprise[:,None]*self.direction
        q,h=self.decode(z,previous,t,hidden);return q,h,surprise

def masked_q(q,available):
    available=available.bool();safe=available.clone();safe[~safe.any(dim=-1)]=True
    return q.masked_fill(~safe,-1e9)

def execute(actor,env,rng=None,epsilon=0.,collect=False):
    view=env.reset();hidden=actor.hidden(2);previous=torch.zeros(2,8);actor.eval();total=0.;done=False;t=0
    rows={k:[] for k in ('images','states','available','actions','rewards','terminal')}
    while not done:
        if collect:
            rows['images'].append(view.images.copy());rows['states'].append(env.training_state());rows['available'].append(view.available.copy())
        with torch.no_grad():q,hidden,_=actor(torch.from_numpy(view.images.copy()),previous,t,hidden,False)
        actions=masked_q(q,torch.from_numpy(view.available.copy())).argmax(-1).numpy()
        if epsilon:
            for i in range(2):
                if rng.random()<epsilon:
                    legal=np.flatnonzero(view.available[i]);actions[i]=rng.choice(legal) if len(legal) else 0
        view,reward,done=env.step(actions);total+=reward
        if collect:rows['actions'].append(actions.copy());rows['rewards'].append(reward);rows['terminal'].append(float(done))
        previous=F.one_hot(torch.as_tensor(actions),8).float();t+=1
    if collect:
        rows['images'].append(view.images.copy());rows['states'].append(env.training_state());rows['available'].append(view.available.copy())
        rows={k:torch.as_tensor(np.asarray(v),dtype=torch.long if k=='actions' else torch.bool if k=='available' else torch.float32) for k,v in rows.items()}
    return total,rows

def batch_episodes(episodes):
    length=max(len(e['actions']) for e in episodes);out={}
    for k in episodes[0]:
        out[k]=torch.zeros((len(episodes),length+(k not in ('actions','rewards','terminal')),*episodes[0][k].shape[1:]),dtype=episodes[0][k].dtype)
        for i,e in enumerate(episodes):out[k][i,:len(e[k])]=e[k]
    out['lengths']=torch.tensor([len(e['actions']) for e in episodes]);out['valid']=torch.arange(length)[None]<out['lengths'][:,None]
    return out

def recurrent(actor,batch,enhance):
    b,times,agents=batch['images'].shape[:3];h=actor.hidden(b*agents);out=[]
    for t in range(times):
        previous=torch.zeros(b*agents,8) if t==0 else F.one_hot(batch['actions'][:,t-1].reshape(-1),8).float()
        valid=(batch['lengths']>=t)[:,None].expand(-1,agents).reshape(-1)
        q,h,_=actor(batch['images'][:,t].reshape(-1,9,5,5),previous,t,h,enhance,valid);out.append(q.reshape(b,agents,8))
    return torch.stack(out,dim=1)

class Learner:
    def __init__(self,state_dim,surprise):
        self.actor=RescueAgent();self.mixer=QMixer(SimpleNamespace(n_agents=2,state_shape=state_dim,mixing_embed_dim=32,hypernet_layers=2,hypernet_embed=64))
        self.target_actor=deepcopy(self.actor).eval().requires_grad_(False);self.target_mixer=deepcopy(self.mixer).eval().requires_grad_(False)
        self.parameters=list(self.actor.parameters())+list(self.mixer.parameters());self.optimizer=torch.optim.RMSprop(self.parameters,lr=1e-4,alpha=.99,eps=1e-5)
        self.surprise=surprise;self.updates=0
    def train(self,batch):
        self.actor.train();q=recurrent(self.actor,batch,self.surprise)
        chosen=q[:,:-1].gather(-1,batch['actions'][...,None]).squeeze(-1);value=self.mixer(chosen,batch['states'][:,:-1]).squeeze(-1)
        with torch.no_grad():
            clean=recurrent(self.actor,batch,False);target=recurrent(self.target_actor,batch,False)
            selected=masked_q(clean[:,1:],batch['available'][:,1:]).argmax(-1,keepdim=True)
            next_=self.target_mixer(target[:,1:].gather(-1,selected).squeeze(-1),batch['states'][:,1:]).squeeze(-1)
            expected=batch['rewards']+.99*(1-batch['terminal'])*next_
        mask=batch['valid'].float();loss=((value-expected).square()*mask).sum()/mask.sum()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite eight-action TD loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.parameters,10.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite eight-action gradient')
        self.optimizer.step();self.updates+=1
        if self.updates%5==0:self.target_actor.load_state_dict(self.actor.state_dict());self.target_mixer.load_state_dict(self.mixer.state_dict())
        return dict(loss=float(loss.detach()),gradient_norm=float(norm),updates=self.updates)

def sources():return {p.name:file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def train_pilot(source,output,method='Sur-QMIX8',seed=11,episodes=12,stop_after=None,resume=False):
    if method not in ('QMIX8','Sur-QMIX8'):raise ValueError('Unknown eight-action model')
    output=Path(output);hydro=HydraulicFrames(source);config=RescueConfig(seed=seed)
    config.validate(hydro);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed)
    env=RescueEnv(hydro,config);env.reset();learner=Learner(len(env.training_state()),method=='Sur-QMIX8');rng=np.random.default_rng(seed)
    spec=dict(schema='hyra-dual-gmrf-eight-pilot-v1',method=method,seed=seed,episodes=episodes,config=asdict_config(config),source_sha256=hydro.digest,sources=sources(),scope='engineering-only; not converged, not paper performance reproduction')
    replay=[];history=[];first=0
    if resume:
        if json.loads((output/'plan.json').read_text(encoding='utf-8'))!=spec:raise ValueError('Pilot plan/source changed')
        pointer=json.loads((output/'latest.json').read_text(encoding='utf-8'));p=output/pointer['path']
        if file_digest(p)!=pointer['sha256']:raise ValueError('Recovery digest mismatch')
        state=torch.load(p,weights_only=True)
        for name in ('actor','mixer','target_actor','target_mixer'):getattr(learner,name).load_state_dict(state[name])
        learner.optimizer.load_state_dict(state['optimizer']);learner.updates=state['updates'];replay=state['replay'];history=state['history'];first=state['episode'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',spec)
        torch.save(learner.actor.state_dict(),output/'initial-actor.pt')
    end=min(stop_after or episodes,episodes)
    if end<first:raise ValueError('Cannot rewind pilot')
    start=time.perf_counter()
    for ep in range(first,end):
        # Vary only the fit scenario seeds; no holdout/model selection here.
        cfg=RescueConfig(seed=seed+ep);env=RescueEnv(hydro,cfg);total,data=execute(learner.actor,env,rng,epsilon=1-.9*ep/max(episodes-1,1),collect=True)
        replay.append(data);replay=replay[-32:];update=None
        if len(replay)>=4:update=learner.train(batch_episodes([replay[i] for i in rng.choice(len(replay),4,replace=False)]))
        history.append(dict(episode=ep+1,return_=total,steps=len(data['actions']),scenario_seed=cfg.seed,update=update))
        recovery=output/f'recovery-{ep+1:03d}.pt'
        if recovery.exists():raise ValueError('Refuse to overwrite pilot recovery')
        state={name:getattr(learner,name).state_dict() for name in ('actor','mixer','target_actor','target_mixer')}
        state.update(optimizer=learner.optimizer.state_dict(),updates=learner.updates,replay=replay,history=history,episode=ep+1,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        torch.save(state,recovery);write(output/'latest.json',dict(path=recovery.name,sha256=file_digest(recovery),episode=ep+1))
        write(output/'progress.json',dict(episode=ep+1,updates=learner.updates,segment_elapsed_s=time.perf_counter()-start))
        print(json.dumps(dict(method=method,episode=ep+1,updates=learner.updates)),flush=True)
    if end==episodes:
        torch.save(learner.actor.state_dict(),output/'actor.pt')
        metadata=dict(**spec,n_actions=8,action_names=list(ACTIONS),image_shape=[9,5,5],trained_episodes=episodes,updates=learner.updates,actor_sha256=file_digest(output/'actor.pt'))
        write(output/'actor.json',metadata);write(output/'history.json',history)
    return history

def asdict_config(config):
    from dataclasses import asdict
    return asdict(config)

def load_actor(folder):
    folder=Path(folder);meta=json.loads((folder/'actor.json').read_text(encoding='utf-8'))
    if meta['n_actions']!=8 or meta['action_names']!=list(ACTIONS) or meta['image_shape']!=[9,5,5]:raise ValueError('Not an eight-action dual-GMRF actor')
    if meta['sources']!=sources() or file_digest(folder/'actor.pt')!=meta['actor_sha256']:raise ValueError('Eight-action model/source digest mismatch')
    actor=RescueAgent();actor.load_state_dict(torch.load(folder/'actor.pt',weights_only=True,map_location='cpu'),strict=True);actor.eval();return actor,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['QMIX8','Sur-QMIX8'],default='Sur-QMIX8');p.add_argument('--seed',type=int,default=11);p.add_argument('--episodes',type=int,default=12);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    train_pilot(a.source,a.output,a.method,a.seed,a.episodes,a.stop_after,a.resume)
if __name__=='__main__':main()
