"""Controlled auxiliary-head activity diagnostic; previous versions stay frozen.

Same21-channel actor, labels, TD update and shared-backbone LR. Give all methods
the same fit-only constant initialization of the unused auxiliary head; D/P
share all v3 settings, replacing only auxiliary ReLU with LeakyReLU(.05). P alone adds
the equally weighted physical constraint. This is development, not formal.
"""
import argparse,json,time,math
from pathlib import Path
from scale100_train import *
import scale100_train as base

def resilient_sources():return dict(base.sources(),**{'scale100_aux_resilient.py':digest(__file__)})

def fit_prior(cases):return np.mean([c.probes['before'].mean(axis=(0,2,3)) for c in cases],axis=0)

class ResilientLearner(base.Learner):
    def __init__(self,state_dim,method,n_agents=2,prior=(.05,0.,0.)):
        super().__init__(state_dim,method,n_agents)
        self.actor.aux[1]=nn.LeakyReLU(.05);self.target_actor.aux[1]=nn.LeakyReLU(.05)
        h=float(prior[0]);require_prior=0<h<.5 and abs(prior[1]/h)<1 and abs(prior[2]/h)<1
        if not require_prior:raise ValueError('Fit prior is outside the shared auxiliary output range')
        with torch.no_grad():
            bias=self.actor.aux[-1].bias.reshape(2,3,20,20)
            bias[:,0].fill_(math.log(h/(.5-h)));bias[:,1].fill_(math.atanh(float(prior[1])/h));bias[:,2].fill_(math.atanh(float(prior[2])/h))
        self.target_actor.load_state_dict(self.actor.state_dict())
        shared=[p for name,p in self.actor.named_parameters() if not name.startswith('aux.')]+list(self.mixer.parameters())
        self.optimizer=torch.optim.RMSprop([dict(params=shared,lr=1e-4),dict(params=list(self.actor.aux.parameters()),lr=1e-3,eps=1e-8)],alpha=.99,eps=1e-5)
        self.data_weight=1.;self.physics_weight=1.
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
            _,aux_hidden=recurrent(self.actor,batch,False);times=torch.linspace(0,int(batch['lengths'].min())-1,4).long().tolist();n=hidden.shape[2];dl=[];pl=[];scale=torch.tensor([.5,.25,.25])[None,:,None,None]
            for b,case in enumerate(cases):
                for t in times:
                    bed=torch.tensor(case.bed,dtype=torch.float32)[None].expand(n,-1,-1);pred=self.actor.predict(aux_hidden[b,t],bed)
                    before=torch.tensor(case.probes['before'][t],dtype=torch.float32)[None].expand(n,-1,-1,-1);after=torch.tensor(case.probes['after'][t],dtype=torch.float32)[None].expand(n,-1,-1,-1)
                    dl.append((((pred[:,0]-before)/scale).square().mean()+((pred[:,1]-after)/scale).square().mean())*.5)
                    if self.method=='P':
                        stepped=self.operator.step(pred[:,0],bed,float(case.probes['dt'][t]),source=torch.tensor(case.probes['source'][t],dtype=torch.float32));pl.append(((stepped-pred[:,1])/scale).square().mean())
            data=torch.stack(dl).mean()
            if pl:physics=torch.stack(pl).mean()
        gradient_audit=None
        if self.method in ('D','P') and self.updates in (0,31,127,252):
            shared=[p for name,p in self.actor.named_parameters() if name.startswith(('conv.','image.','gru.'))]
            def gradient_norm(loss):
                if not loss.requires_grad:return 0.
                grads=torch.autograd.grad(loss,shared,retain_graph=True,allow_unused=True)
                return float(torch.stack([g.detach().square().sum() for g in grads if g is not None]).sum().sqrt())
            gradient_audit=dict(td_shared_norm=gradient_norm(td),data_shared_norm=gradient_norm(data),physics_shared_norm=gradient_norm(physics))
        loss=td+self.data_weight*data+self.physics_weight*physics
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite auxiliary-tuning loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(self.parameters,10.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite auxiliary-tuning gradient')
        self.optimizer.step();self.updates+=1
        if self.updates%50==0:self.target_actor.load_state_dict(self.actor.state_dict());self.target_mixer.load_state_dict(self.mixer.state_dict())
        return dict(td=float(td.detach()),data=float(data.detach()),physics=float(physics.detach()),gradient_norm=float(norm),updates=self.updates,shared_gradient_audit=gradient_audit)

def train_resilient(dataset,output,task,method,seed=11,episodes=256,stop_after=None,resume=False,phase='development-aux-resilient'):
    dataset=Path(dataset);output=Path(output);manifest=json.loads(dataset.read_text());entries=[r for r in manifest['cases'] if r['split']=='fit'];cases=[FloodCase(r['path']) for r in entries]
    for row,case in zip(entries,cases):
        if row['sha256']!=case.digest:raise ValueError('Fit source changed')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(seed);rng=np.random.default_rng(seed);prior=fit_prior(cases)
    cfg=config(task,method,seed);env=ScaleEnv(cases[0],cfg);env.reset();learner=ResilientLearner(len(env.training_state()),method,cfg.n_agents,prior)
    spec=dict(schema='hyra-scale100-aux-resilient-v4',model_family='same_actor_aux_resilient_v4',phase=phase,task=task,method=method,seed=seed,episodes=episodes,config=asdict(cfg),sources=resilient_sources(),dataset_sha256=digest(dataset),fit_cases=[dict(path=r['path'],sha256=r['sha256'],parent=r['parent_terrain']) for r in entries],fit_only_prior_U=prior.tolist(),observation_channels=21,actions=8,discount=.99,lr=1e-4,auxiliary_head_lr=1e-3,auxiliary_head_eps=1e-8,auxiliary_activation='leaky_relu_0.05',batch_episodes=4,replay_episodes=32,target_update_steps=50,data_weight=1.,physics_weight=1.,formal=phase=='formal')
    first=0;history=[];replay=[]
    if resume:
        if json.loads((output/'plan.json').read_text())!=spec:raise ValueError('Tuned training contract changed')
        pointer=json.loads((output/'latest.json').read_text());checkpoint=output/pointer['path']
        if digest(checkpoint)!=pointer['sha256']:raise ValueError('Tuned recovery digest mismatch')
        state=torch.load(checkpoint,weights_only=False,map_location='cpu')
        for name in ('actor','mixer','target_actor','target_mixer'):getattr(learner,name).load_state_dict(state[name])
        learner.optimizer.load_state_dict(state['optimizer']);learner.updates=state['updates'];history=state['history'];replay=state['replay'];first=state['episode'];rng.bit_generator.state=state['numpy_rng'];torch.set_rng_state(state['torch_rng'])
    else:
        output.mkdir(parents=True,exist_ok=False);write(output/'plan.json',spec);torch.save(learner.actor.state_dict(),output/'initial-actor.pt')
    end=min(stop_after or episodes,episodes);start=time.perf_counter()
    if end<first:raise ValueError('Cannot rewind tuned training')
    for ep in range(first,end):
        i=ep%len(cases);cfg=config(task,method,seed+1009*ep);env=ScaleEnv(cases[i],cfg);epsilon=1-.95*min(ep/max(1,int(episodes*.8)),1)
        total,data,metrics=rollout(learner.actor,env,rng,epsilon,True);replay.append((i,data));replay=replay[-32:];update=None
        if len(replay)>=4:
            selected=[replay[j] for j in rng.choice(len(replay),4,replace=False)];update=learner.train(batch_episodes([d for _,d in selected]),[cases[j] for j,_ in selected])
        history.append(dict(episode=ep+1,return_=total,metrics=metrics,epsilon=epsilon,case_index=i,scenario_seed=cfg.seed,update=update))
        write(output/'progress.json',dict(status='running',episode=ep+1,total_episodes=episodes,updates=learner.updates,elapsed_segment_s=time.perf_counter()-start,last=history[-1]))
        if (ep+1)%4==0 or ep+1==end:
            name=f'recovery-{ep+1:06d}.pt';temp=output/(name+'.tmp');state={key:getattr(learner,key).state_dict() for key in ('actor','mixer','target_actor','target_mixer')}
            state.update(optimizer=learner.optimizer.state_dict(),updates=learner.updates,history=history,replay=replay,episode=ep+1,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state());torch.save(state,temp);temp.replace(output/name)
            write(output/'latest.json',dict(path=name,sha256=digest(output/name),episode=ep+1));write(output/'history.json',history)
            for old in sorted(output.glob('recovery-??????.pt'))[:-2]:old.unlink()
            print(json.dumps(dict(task=task,method=method,episode=ep+1,updates=learner.updates)),flush=True)
    if end==episodes:
        torch.save(learner.actor.state_dict(),output/'actor.pt');write(output/'actor.json',dict(**spec,trained_episodes=episodes,updates=learner.updates,actor_sha256=digest(output/'actor.pt')))
        write(output/'progress.json',dict(status='completed',episode=end,total_episodes=episodes,updates=learner.updates,elapsed_segment_s=time.perf_counter()-start,last=history[-1]))
    return history

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',choices=['E1','E2'],required=True);p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=11);p.add_argument('--episodes',type=int,default=256);p.add_argument('--stop-after',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();train_resilient(a.dataset,a.output,a.task,a.method,a.seed,a.episodes,a.stop_after,a.resume)
if __name__=='__main__':main()


