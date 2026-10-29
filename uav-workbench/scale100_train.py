"""Six matched eight-action learners on the registered 100m two-task contract."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('PYTHONUTF8','1')
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from types import SimpleNamespace
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sur_rescue.learning import RescueAgent,masked_q,batch_episodes
from hyra_learning.vendor.pymarl_qmix import QMixer
from hyra_learning.physics import SWEOperator
from scale100_env import FloodCase,ScaleEnv,config

METHODS=('B0','B1','B2','B3','D','P')
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)
def sources():
    root=Path(__file__).parent
    paths=[root/'scale100_env.py',root/'scale100_train.py',root/'hyra_learning/physics.py',root/'hyra_learning/vendor/pymarl_qmix.py',*sorted((root/'sur_rescue').glob('*.py'))]
    return {str(p.relative_to(root)):digest(p) for p in paths}

class Agent(RescueAgent):
    def __init__(self):
        super().__init__();self.conv[0]=nn.Conv2d(21,16,3,padding=1)
        # Allocated in ALL groups. Only D/P optimize it; identical labels/capacity.
        self.aux=nn.Sequential(nn.Linear(64+400,128),nn.ReLU(),nn.Linear(128,2*3*400))
    def forward(self,images,previous,t,hidden,enhance=False,valid=None):
        if images.shape[1:]!=(21,5,5):raise ValueError('Wrong model contract: requires 21-channel legal observations')
        z=F.relu(self.image(self.conv(images)));surprise=z.new_zeros(len(z))
        if enhance:
            if not self.training or not torch.is_grad_enabled():raise RuntimeError('Sur is training only')
            virtual,_=self.decode(z,previous,t,hidden);gradient=torch.autograd.grad(virtual.max(-1).values.sum(),z,retain_graph=True,create_graph=False)[0]
            norm=gradient.detach().norm(dim=-1);mask=torch.ones(len(z),dtype=torch.bool,device=z.device) if valid is None else valid.bool()
            if mask.any():
                low,high=norm[mask].min(),norm[mask].max();surprise=torch.where(mask,(norm-low)/(high-low).clamp_min(1e-12),0.)
                z=z+surprise[:,None]*self.direction
        q,h=self.decode(z,previous,t,hidden);return q,h,surprise
    def predict(self,hidden,bed):
        out=self.aux(torch.cat((hidden,bed.flatten(1)),dim=-1)).reshape(-1,2,3,20,20)
        h=.5*torch.sigmoid(out[:,:,0]);mx=h*torch.tanh(out[:,:,1]);mz=h*torch.tanh(out[:,:,2])
        return torch.stack((h,mx,mz),dim=2)

def rollout(actor,env,rng,epsilon=0.,collect=True):
    view=env.reset();n=env.config.n_agents;hidden=actor.hidden(n);previous=torch.zeros(n,8);actor.eval();total=0.;done=False;t=0
    rows={k:[] for k in ('images','states','available','actions','rewards','terminal')}
    while not done:
        if collect:
            for k,v in [('images',view.images),('states',env.training_state()),('available',view.available)]:rows[k].append(v.copy())
        with torch.no_grad():q,hidden,_=actor(torch.from_numpy(view.images.copy()),previous,t,hidden)
        actions=masked_q(q,torch.from_numpy(view.available.copy())).argmax(-1).numpy()
        for i in range(n):
            if epsilon and rng.random()<epsilon:
                legal=np.flatnonzero(view.available[i]);actions[i]=rng.choice(legal) if len(legal) else 0
        view,reward,done=env.step(actions);total+=reward
        if collect:
            rows['actions'].append(actions.copy());rows['rewards'].append(reward);rows['terminal'].append(float(done))
        previous=F.one_hot(torch.as_tensor(actions),8).float();t+=1
    if collect:
        for k,v in [('images',view.images),('states',env.training_state()),('available',view.available)]:rows[k].append(v.copy())
        rows={k:torch.as_tensor(np.asarray(v),dtype=torch.long if k=='actions' else torch.bool if k=='available' else torch.float32) for k,v in rows.items()}
    return total,rows,env.metrics()

def recurrent(actor,batch,enhance):
    b,times,n=batch['images'].shape[:3];h=actor.hidden(b*n);out=[];hs=[]
    for t in range(times):
        previous=torch.zeros(b*n,8) if t==0 else F.one_hot(batch['actions'][:,t-1].reshape(-1),8).float()
        valid=(batch['lengths']>=t)[:,None].expand(-1,n).reshape(-1)
        q,h,_=actor(batch['images'][:,t].reshape(-1,21,5,5),previous,t,h,enhance,valid);out.append(q.reshape(b,n,8));hs.append(h.reshape(b,n,64))
    return torch.stack(out,dim=1),torch.stack(hs,dim=1)

class Learner:
    def __init__(self,state_dim,method,n_agents=2):
        self.actor=Agent();self.mixer=QMixer(SimpleNamespace(n_agents=n_agents,state_shape=state_dim,mixing_embed_dim=32,hypernet_layers=2,hypernet_embed=64))
        self.target_actor=deepcopy(self.actor).eval().requires_grad_(False);self.target_mixer=deepcopy(self.mixer).eval().requires_grad_(False)
        self.parameters=list(self.actor.parameters())+list(self.mixer.parameters());self.optimizer=torch.optim.RMSprop(self.parameters,lr=1e-4,alpha=.99,eps=1e-5)
        self.method=method;self.updates=0;self.operator=SWEOperator(dx=5.)
    def train(self,batch,cases):
        self.actor.train();q,hidden=recurrent(self.actor,batch,self.method in ('B2','B3','D','P'))
        chosen=q[:,:-1].gather(-1,batch['actions'][...,None]).squeeze(-1);value=self.mixer(chosen,batch['states'][:,:-1]).squeeze(-1)
        with torch.no_grad():
            clean,_=recurrent(self.actor,batch,False);target,_=recurrent(self.target_actor,batch,False)
            selected=masked_q(clean[:,1:],batch['available'][:,1:]).argmax(-1,keepdim=True)
            next_=self.target_mixer(target[:,1:].gather(-1,selected).squeeze(-1),batch['states'][:,1:]).squeeze(-1)
            expected=batch['rewards']+.99*(1-batch['terminal'])*next_
        mask=batch['valid'].float();td=((value-expected).square()*mask).sum()/mask.sum();data=td.new_zeros(());physics=td.new_zeros(())
        if self.method in ('D','P'):
            # Clean shared recurrent representation. No future field is an encoder input.
            _,aux_hidden=recurrent(self.actor,batch,False)
            times=torch.linspace(0,int(batch['lengths'].min())-1,4).long().tolist();n=hidden.shape[2];dl=[];pl=[]
            scale=torch.tensor([.5,.25,.25])[None,:,None,None]
            for b,case in enumerate(cases):
                for t in times:
                    bed=torch.tensor(case.bed,dtype=torch.float32)[None].expand(n,-1,-1)
                    pred=self.actor.predict(aux_hidden[b,t],bed)
                    before=torch.tensor(case.probes['before'][t],dtype=torch.float32)[None].expand(n,-1,-1,-1)
                    after=torch.tensor(case.probes['after'][t],dtype=torch.float32)[None].expand(n,-1,-1,-1)
                    dl.append((((pred[:,0]-before)/scale).square().mean()+((pred[:,1]-after)/scale).square().mean())*.5)
                    if self.method=='P':
                        stepped=self.operator.step(pred[:,0],bed,float(case.probes['dt'][t]),source=torch.tensor(case.probes['source'][t],dtype=torch.float32))
                        pl.append(((stepped-pred[:,1])/scale).square().mean())
            data=torch.stack(dl).mean()
            if pl:physics=torch.stack(pl).mean()
        loss=td+.05*data+.05*physics
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.parameters,10.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step();self.updates+=1
        if self.updates%50==0:self.target_actor.load_state_dict(self.actor.state_dict());self.target_mixer.load_state_dict(self.mixer.state_dict())
        return dict(td=float(td.detach()),data=float(data.detach()),physics=float(physics.detach()),gradient_norm=float(norm),updates=self.updates)

def train(dataset,output,task,method,seed,episodes=24,stop_after=None,resume=False,phase='development-diagnostic'):
    dataset=Path(dataset);output=Path(output);manifest=json.loads(dataset.read_text(encoding='utf-8'))
    entries=[r for r in manifest['cases'] if r['split']=='fit'];cases=[FloodCase(r['path']) for r in entries]
    for row,case in zip(entries,cases):
        if row['sha256']!=case.digest:raise ValueError('Dataset changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    cfg=config(task,method,seed);env=ScaleEnv(cases[0],cfg);env.reset();learner=Learner(len(env.training_state()),method,cfg.n_agents)
    spec=dict(schema='hyra-scale100-training-v1',phase=phase,task=task,method=method,seed=seed,episodes=episodes,config=asdict(cfg),sources=sources(),dataset_sha256=digest(dataset),fit_cases=[dict(path=r['path'],sha256=r['sha256'],parent=r['parent_terrain']) for r in entries],observation_channels=21,actions=8,discount=.99,lr=1e-4,batch_episodes=4,replay_episodes=32,target_update_steps=50,data_weight=.05,physics_weight=.05,formal=phase=='formal')
    replay=[];history=[];first=0
    if resume:
        if json.loads((output/'plan.json').read_text(encoding='utf-8'))!=spec:raise ValueError('Refuse resume with changed training contract')
        pointer=json.loads((output/'latest.json').read_text());checkpoint=output/pointer['path']
        if digest(checkpoint)!=pointer['sha256']:raise ValueError('Recovery digest mismatch')
        state=torch.load(checkpoint,weights_only=False,map_location='cpu') # locally written, verified digest only
        for name in ('actor','mixer','target_actor','target_mixer'):getattr(learner,name).load_state_dict(state[name])
        learner.optimizer.load_state_dict(state['optimizer']);learner.updates=state['updates'];replay=state['replay'];history=state['history'];first=state['episode'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',spec);torch.save(learner.actor.state_dict(),output/'initial-actor.pt')
    start=time.perf_counter();end=min(stop_after or episodes,episodes)
    if end<first:raise ValueError('Cannot rewind training')
    for ep in range(first,end):
        case_index=ep%len(cases);cfg=config(task,method,seed+1009*ep);env=ScaleEnv(cases[case_index],cfg)
        epsilon=1-.95*min(ep/max(1,int(episodes*.8)),1)
        total,data,metrics=rollout(learner.actor,env,rng,epsilon,True);replay.append((case_index,data));replay=replay[-32:];update=None
        if len(replay)>=4:
            selected=[replay[i] for i in rng.choice(len(replay),4,replace=False)];update=learner.train(batch_episodes([d for _,d in selected]),[cases[i] for i,_ in selected])
        history.append(dict(episode=ep+1,return_=total,metrics=metrics,epsilon=epsilon,case_index=case_index,scenario_seed=cfg.seed,update=update))
        write(output/'progress.json',dict(status='running',episode=ep+1,total_episodes=episodes,updates=learner.updates,elapsed_segment_s=time.perf_counter()-start,last=history[-1]))
        if (ep+1)%4==0 or ep+1==end:
            # Two rotating recovery slots preserve the last verified generation, bounding disk use.
            slot=f'recovery-{ep+1:06d}.pt';temp=output/(slot+'.tmp')
            state={name:getattr(learner,name).state_dict() for name in ('actor','mixer','target_actor','target_mixer')}
            state.update(optimizer=learner.optimizer.state_dict(),updates=learner.updates,replay=replay,history=history,episode=ep+1,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
            torch.save(state,temp);temp.replace(output/slot);write(output/'latest.json',dict(path=slot,sha256=digest(output/slot),episode=ep+1));write(output/'history.json',history)
            for obsolete in sorted(output.glob('recovery-??????.pt'))[:-2]:obsolete.unlink()
            print(json.dumps(dict(task=task,method=method,episode=ep+1,updates=learner.updates,elapsed_s=time.perf_counter()-start)),flush=True)
    if end==episodes:
        torch.save(learner.actor.state_dict(),output/'actor.pt');write(output/'actor.json',dict(**spec,trained_episodes=episodes,updates=learner.updates,actor_sha256=digest(output/'actor.pt')))
        write(output/'progress.json',dict(status='completed',episode=end,total_episodes=episodes,updates=learner.updates,elapsed_segment_s=time.perf_counter()-start,last=history[-1]))
    return history

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=['E1','E2'],required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--episodes',type=int,default=24);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--phase',choices=['development-diagnostic','formal'],default='development-diagnostic');a=p.parse_args()
    train(a.dataset,a.output,a.task,a.method,a.seed,a.episodes,a.stop_after,a.resume,a.phase)
if __name__=='__main__':main()
