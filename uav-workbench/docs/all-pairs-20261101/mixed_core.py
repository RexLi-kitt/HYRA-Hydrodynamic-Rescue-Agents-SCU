"""Research adapters for all six allowed learners; not an algorithm substitution."""
import numpy as np
import torch
from torch.nn import functional as F

def install(e):
    from flood1km.learners import Learner,distribution,optimize,gae,td_lambda,trust_update
    legacy_adapt=e.adapted;legacy_update=e.update_block;OriginalController=e.Controller
    legacy_pairs={('hatrpo','hasac'),('hatrpo','maac'),('hasac','maac')}
    on_policy={'happo','hatrpo','mat','coma'}

    def adapted(algorithm):
        if algorithm in ('hatrpo','hasac','maac'):return legacy_adapt(algorithm)
        ck=torch.load(e.PARENTS[algorithm],map_location='cpu',weights_only=False)['learner']
        model=Learner(algorithm,e.DIM);state=model.state_dict()
        for name,old in ck['model'].items():
            new=state[name]
            if new.shape==old.shape:new.copy_(old);continue
            assert old.ndim==2 and new.shape[0]==old.shape[0],name
            new.zero_();n=old.shape[1]
            if n==e.OLD_DIM:
                assert new.shape[1]==e.DIM;new[:,:e.OLD_DIM].copy_(old)
            else:
                tail=n-4*e.OLD_DIM;assert tail in (0,36,40) and new.shape[1]==4*e.DIM+tail,(name,old.shape,new.shape)
                for i in range(4):new[:,i*e.DIM:i*e.DIM+e.OLD_DIM].copy_(old[:,i*e.OLD_DIM:(i+1)*e.OLD_DIM])
                if tail:new[:,4*e.DIM:].copy_(old[:,4*e.OLD_DIM:])
        model.load_state_dict(state);model.alpha=ck['alpha'];model.eval();model.check_finite()
        return model,ck

    @torch.no_grad()
    def mixed_sample(models,obs,mask,deterministic=False,force=None):
        roles=obs[:,0,-4:].long();actions=torch.zeros((len(obs),4),dtype=torch.long);logs=[]
        encoded={j:m.policy.encode(obs) for j,m in enumerate(models) if m.algorithm=='mat'}
        for i in range(4):
            logits=[m.policy.decode(encoded[j],actions)[:,i] if m.algorithm=='mat' else m.actors[i](obs[:,i]) for j,m in enumerate(models)]
            d=distribution(torch.where((roles[:,i]==1)[:,None],logits[1],logits[0]),mask[:,i])
            actions[:,i]=force[1] if force is not None and force[0]==i else (d.probs.argmax(-1) if deterministic else d.sample())
            logs.append(d.log_prob(actions[:,i]))
        return actions,torch.stack(logs,1)

    class MixedController(OriginalController):
        def decide(self,models,obs,boundary,roles,deterministic=False):
            if models is None or tuple(m.algorithm for m in models) in legacy_pairs:
                return super().decide(models,obs,boundary,roles,deterministic)
            forced,detail=self.service.select(obs,boundary,np.full(4,8,dtype=np.int64))
            mask=e.service_mask(boundary,forced,detail['phases']);mask,gdetail=self.guard.mask(obs,mask,detail['phases'])
            proposals=forced.copy()
            for i,p in enumerate(detail['phases']):
                if p=='learned':proposals[i]=8 if mask[i,8] else int(np.flatnonzero(mask[i])[0])
            if self.coverage:
                planned,pdetail=self.planner.select(obs,mask,proposals,detail['phases'])
                for i in pdetail['planned_ids']:mask[i]=False;mask[i,planned[i]]=True
            else:pdetail=dict(planned_ids=[])
            aug=e.augment(obs,roles);o=torch.from_numpy(aug)[None]
            a,l=mixed_sample(models,o,torch.from_numpy(mask)[None],deterministic);a=a[0].numpy();logs=l[0].numpy()
            with torch.no_grad():values=np.asarray([float(m.value(o)[0]) for m in models],np.float32)
            assert mask[np.arange(4),a].all()
            controlled=mask.sum(-1)==1;assert np.abs(logs[controlled]).max(initial=0)<1e-6
            service=np.array([p!='learned' for p in detail['phases']]);assert np.array_equal(a[service],forced[service])
            self.guard.remember(a)
            return a,mask,logs,values,aug,dict(planned=pdetail['planned_ids'],service=service.tolist(),free=(mask.sum(-1)>1).tolist(),guard=gdetail['constrained_agents'])

    def owned_rows(rows,owner):
        result=e.owned_rows(rows,owner)
        for r in result:r['value']=float(np.asarray(r['value'])[owner])
        return result

    def update_on(model,rows,owner):
        b=model.batch(owned_rows(rows,owner));o=b['obs'];mask=b['mask'];a=b['action'];r=b['reward'];done=b['done']
        own=o[:,0,-4:]==owner;free=(mask.sum(-1)>1)&own;metrics={}
        if model.algorithm=='coma':
            with torch.no_grad():
                nq=model.target(b['next_obs'],b['next_action']).gather(-1,b['next_action'][...,None]).squeeze(-1)
                targets=td_lambda(r,nq,done,model.gamma)
            for _ in range(4):
                q=model.critic(o,a).gather(-1,a[...,None]).squeeze(-1)
                loss=((q-targets).square()*own).sum()/own.sum()
                optimize(loss,model.optimizers[4],model.critic.parameters())
            with torch.no_grad():all_q=model.critic(o,a)
            for i in range(4):
                ids=torch.where(free[:,i])[0]
                if not len(ids):continue
                d=distribution(model.actors[i](o[ids,i]),mask[ids,i]);qi=all_q[ids,i]
                baseline=(d.probs.detach()*qi).sum(-1)
                advantage=(qi.gather(1,a[ids,i,None]).squeeze(-1)-baseline).detach()
                loss_i=-(d.log_prob(a[ids,i])*advantage+.01*d.entropy()).mean()
                optimize(loss_i,model.optimizers[i],model.actors[i].parameters())
            model.soft_target(.05);metrics['critic_loss']=float(loss.detach())
        else:
            advantage,returns=gae(r,b['value'],done,model.gamma)
            advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-6)
            if model.algorithm=='mat':
                for _ in range(4):
                    for ids in torch.randperm(len(o)).split(128):
                        logits,value=model.policy(o[ids],a[ids]);d=distribution(logits,mask[ids]);f=free[ids]
                        ratio=(d.log_prob(a[ids])-b['logp'][ids]).exp();adv=advantage[ids,None]
                        policy=-(torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)*f).sum()/f.sum().clamp_min(1)
                        loss=policy+.5*F.mse_loss(value,returns[ids])-.01*(d.entropy()*f).sum()/f.sum().clamp_min(1)
                        optimize(loss,model.optimizers[0],model.policy.parameters())
                metrics['last_loss']=float(loss.detach())
            else:
                for _ in range(4):
                    loss=F.mse_loss(model.value(o),returns);optimize(loss,model.optimizers[4],model.critic.parameters())
                factor=torch.ones_like(advantage);order=torch.randperm(4).tolist();metrics['update_order']=order
                for i in order:
                    active=torch.where(free[:,i])[0]
                    if not len(active):continue  # also prevents Adam momentum moving inactive actors
                    if model.algorithm=='hatrpo':
                        metrics[f'actor_{i}']=trust_update(model.actors[i],o[active,i],mask[active,i],a[active,i],b['logp'][active,i],factor[active]*advantage[active])
                    else:
                        for _ in range(4):
                            for ids in active[torch.randperm(len(active))].split(128):
                                d=distribution(model.actors[i](o[ids,i]),mask[ids,i]);ratio=(d.log_prob(a[ids,i])-b['logp'][ids,i]).exp()
                                adv=advantage[ids];loss_i=-(factor[ids]*torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)).mean()-.01*d.entropy().mean()
                                optimize(loss_i,model.optimizers[i],model.actors[i].parameters())
                    with torch.no_grad():factor*=torch.exp(distribution(model.actors[i](o[:,i]),mask[:,i]).log_prob(a[:,i])-b['logp'][:,i])
                if not torch.isfinite(factor).all():raise FloatingPointError('Nonfinite ownership correction')
                metrics['critic_loss']=float(loss.detach())
        model.updates+=1;model.check_finite();return metrics

    def update_off(models,owner,rows):
        model=models[owner];b=model.batch(rows);o=b['obs'];a=b['action'];mask=b['mask'];own=o[:,0,-4:]==owner
        has_mat=any(m.algorithm=='mat' for m in models)
        with torch.no_grad():
            na,nlog=mixed_sample(models,b['next_obs'],b['next_mask']);next_own=b['next_obs'][:,0,-4:]==owner
            if model.algorithm=='hasac':
                nq=torch.minimum(*[q(b['next_obs'],na) for q in model.target])-model.alpha*(nlog*next_own).sum(-1)
                target=b['reward']+model.gamma*(1-b['done'])*nq
            else:
                nq=model.target(b['next_obs'],na).gather(-1,na[...,None]).squeeze(-1)-model.alpha*nlog
                target=b['reward'][:,None]+model.gamma*(1-b['done'][:,None])*nq
        if model.algorithm=='hasac':loss=sum(F.mse_loss(q(o,a),target) for q in model.critic)
        else:loss=((model.critic(o,a).gather(-1,a[...,None]).squeeze(-1)-target).square()*own).sum()/own.sum()
        norm=optimize(loss,model.optimizers[4],model.critic.parameters());context,_=mixed_sample(models,o,mask);records=[]
        order=torch.randperm(4).tolist() if model.algorithm=='hasac' else list(range(4))
        for i in order:
            active=own[:,i]&(mask[:,i].sum(-1)>1)
            if not active.any():continue
            oi=o[active];mi=mask[active];context_i=mixed_sample(models,oi,mi)[0] if model.algorithm=='hasac' else context[active]
            with torch.no_grad():
                if model.algorithm=='hasac' or has_mat:
                    values=[]
                    for act in range(9):
                        if has_mat:joint=mixed_sample(models,oi,mi,force=(i,act))[0]
                        else:joint=context_i.clone();joint[:,i]=act
                        values.append(torch.minimum(*[q(oi,joint) for q in model.critic]) if model.algorithm=='hasac' else model.critic(oi,joint)[:,i,act])
                    q_values=torch.stack(values,-1)
                else:q_values=model.critic(oi,context_i)[:,i]
            d=distribution(model.actors[i](oi[:,i]),mi[:,i]);aloss=(d.probs*(model.alpha*d.logits-q_values)).sum(-1).mean()
            optimize(aloss,model.optimizers[i],model.actors[i].parameters());records.append(dict(agent=i,loss=float(aloss.detach())))
        model.soft_target(.01);model.updates+=1;model.check_finite()
        return dict(critic_loss=float(loss.detach()),gradient_norm=norm,actors=records)

    def update_block(models,rows,replay):
        pair=tuple(m.algorithm for m in models)
        if pair in legacy_pairs:return legacy_update(models,rows,replay)
        owners=[j for j,m in enumerate(models) if m.algorithm in on_policy];losses={}
        # With two on-policy learners, update one on a fresh frozen-team batch,
        # alternating each block. Never reuse that on-policy batch for its partner.
        if owners:
            j=owners[sum(models[k].updates for k in owners)%len(owners)]
            losses[models[j].algorithm]=update_on(models[j],rows,j)
        replay.extend(rows);del replay[:-e.REPLAY_SIZE]
        for j,m in enumerate(models):
            if m.algorithm in ('hasac','maac'):
                losses[m.algorithm]=[update_off(models,j,e.random.sample(replay,min(64,len(replay)))) for _ in range(4)]
        return losses

    e.adapted=adapted;e.Controller=MixedController;e.update_block=update_block
    return dict(legacy_adapt=legacy_adapt,legacy_update=legacy_update,original_controller=OriginalController,
                mixed_sample=mixed_sample,legacy_pairs=legacy_pairs)
