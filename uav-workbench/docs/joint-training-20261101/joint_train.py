"""Research hybrid fine-tuning of HASAC with MAT/HATRPO; no convergence claim."""
import argparse,copy,gc,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
EXP=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
ROOT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
sys.path[:0]=[str(EXP/'priority-20261101'),str(EXP)]
from flood1km.learners import Learner,distribution,optimize
from flood1km.train import seed_all,rng_state,restore_rng,aggregate
from flood1km.data import sha
from flood1km.env import Env
from service_commitment import ServiceCommitment
from coverage_controller import service_mask
from retry_guard import RetryGuard
from priority_train import write
OLD_DIM=1468;DIM=1472
ROLES=((0,1,1,0),(1,0,0,1));LAYOUTS=(9001,9002,9003,9004);SEEDS=(3101,3102)
PARENTS={'hasac':ROOT/'uav6-full1024-20261101/hasac/checkpoint-1024.pt','mat':ROOT/'uav6-total256-20261101/mat/latest.pt','hatrpo':ROOT/'uav6-full1024-20261101/hatrpo/checkpoint-1024.pt'}

def augment(obs,roles):return np.concatenate((obs,np.broadcast_to(np.array(roles,dtype=np.float32),(4,4))),axis=1)

def adapted(algorithm):
    ck=torch.load(PARENTS[algorithm],map_location='cpu',weights_only=False)['learner']
    model=Learner(algorithm,DIM);state=model.state_dict()
    for name,old in ck['model'].items():
        new=state[name]
        if new.shape==old.shape:new.copy_(old);continue
        assert old.ndim==2 and new.shape[0]==old.shape[0],name
        new.zero_();n=old.shape[1]
        if n==OLD_DIM:assert new.shape[1]==DIM;new[:,:OLD_DIM].copy_(old)
        else:
            tail=n-4*OLD_DIM;assert tail in (0,36) and new.shape[1]==4*DIM+tail,(name,old.shape,new.shape)
            for i in range(4):new[:,i*DIM:i*DIM+OLD_DIM].copy_(old[:,i*OLD_DIM:(i+1)*OLD_DIM])
            if tail:new[:,4*DIM:].copy_(old[:,4*OLD_DIM:])
    model.load_state_dict(state);model.alpha=ck['alpha'];model.eval();model.check_finite()
    return model,ck

@torch.no_grad()
def sample(models,obs,mask,deterministic=False,force=None):
    roles=obs[:,0,-4:].long();assert ((roles==0)|(roles==1)).all()
    actions=torch.zeros((len(obs),4),dtype=torch.long);logs=[]
    encoded={j:m.policy.encode(obs) for j,m in enumerate(models) if m.algorithm=='mat'}
    for i in range(4):
        logits=[]
        for j,m in enumerate(models):logits.append(m.policy.decode(encoded[j],actions)[:,i] if m.algorithm=='mat' else m.actors[i](obs[:,i]))
        selected=torch.where((roles[:,i]==1)[:,None],logits[1],logits[0]);d=distribution(selected,mask[:,i])
        actions[:,i]=force[1] if force is not None and force[0]==i else (d.probs.argmax(-1) if deterministic else d.sample())
        logs.append(d.log_prob(actions[:,i]))
    return actions,torch.stack(logs,1)

def decide(models,service,guard,obs,boundary,roles,deterministic=False):
    forced,detail=service.select(obs,boundary,np.full(4,8,dtype=np.int64))
    mask=service_mask(boundary,forced,detail['phases']);mask,_=guard.mask(obs,mask,detail['phases'])
    aug=augment(obs,roles);a,l=sample(models,torch.from_numpy(aug)[None],torch.from_numpy(mask)[None],deterministic)
    a=a[0].numpy();l=l[0].numpy();guard.remember(a)
    assert mask[np.arange(4),a].all()
    controlled=np.array([p!='learned' for p in detail['phases']]);assert np.array_equal(a[controlled],forced[controlled])
    assert np.abs(l[controlled]).max(initial=0)<1e-6
    with torch.no_grad():value=float(models[1].value(torch.from_numpy(aug)[None])[0])
    return a,mask,l,value,aug

def rollout(models,water,layout,roles,collect=False,deterministic=False):
    env=Env(water,layout);obs=env.observe();service=ServiceCommitment('commit');guard=RetryGuard();previous=None;rows=[];done=False;ret=0.
    while not done:
        a,m,l,v,aug=decide(models,service,guard,obs,env.available(),roles,deterministic)
        if previous is not None:previous.update(next_mask=m.copy(),next_action=a.copy());rows.append(previous)
        nxt,reward,done,info=env.step(a);ret+=reward
        previous=dict(obs=aug,mask=m.copy(),action=a.copy(),logp=l.copy(),value=v,reward=reward,done=float(done),next_obs=augment(nxt,roles),next_mask=env.available().copy(),next_action=np.full(4,8,dtype=np.int64)) if collect else None
        obs=nxt
    if collect:rows.append(previous)
    assert len({x['target'] for x in env.events if x['type']=='rescued'})==env.metrics()['rescued']
    return dict(**env.metrics(),return_sum=ret,guard_agent_steps=guard.constrained_steps),rows

def partner_rows(rows):
    result=[]
    for r in rows:
        x=dict(r);mask=r['mask'].copy();logp=r['logp'].copy();inactive=r['obs'][0,-4:]==0
        mask[inactive]=False;mask[np.flatnonzero(inactive),r['action'][inactive]]=True;logp[inactive]=0
        x.update(mask=mask,logp=logp);result.append(x)
    return result

def update_hasac(models,rows):
    model=models[0];b=model.batch(rows);o=b['obs'];a=b['action'];mask=b['mask'];own=o[:,0,-4:]==0
    with torch.no_grad():
        na,nlog=sample(models,b['next_obs'],b['next_mask']);next_own=b['next_obs'][:,0,-4:]==0
        nq=torch.minimum(*[q(b['next_obs'],na) for q in model.target])-model.alpha*(nlog*next_own).sum(-1)
        target=b['reward']+model.gamma*(1-b['done'])*nq
    loss=sum(F.mse_loss(q(o,a),target) for q in model.critic);norm=optimize(loss,model.optimizers[4],model.critic.parameters())
    records=[]
    for i in torch.randperm(4).tolist():
        active=own[:,i]&(mask[:,i].sum(-1)>1)
        if not active.any():continue
        oi=o[active];mi=mask[active]
        with torch.no_grad():
            values=[]
            for candidate in range(9):
                # Re-evaluate later causal MAT decisions for this counterfactual action.
                joint,_=sample(models,oi,mi,force=(i,candidate))
                values.append(torch.minimum(*[q(oi,joint) for q in model.critic]))
            q_values=torch.stack(values,-1)
        dist=distribution(model.actors[i](oi[:,i]),mi[:,i])
        actor_loss=(dist.probs*(model.alpha*dist.logits-q_values)).sum(-1).mean()
        anorm=optimize(actor_loss,model.optimizers[i],model.actors[i].parameters());records.append(dict(agent=i,loss=float(actor_loss.detach()),norm=anorm))
    model.soft_target(.01);model.updates+=1;model.check_finite()
    return dict(critic_loss=float(loss.detach()),critic_gradient_norm=norm,actors=records)

def update_block(models,rows,replay):
    # Rollouts were collected with both models fixed; partner is updated on-policy first.
    partner=models[1].update_on_policy(partner_rows(rows))
    replay.extend(rows);del replay[:-4096]
    losses=[update_hasac(models,random.sample(replay,min(64,len(replay)))) for _ in range(4)]
    return dict(partner=partner,hasac=losses)

def model_hash(models):
    h=hashlib.sha256()
    for model in models:
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def checkpoint(models,replay,episode):return dict(models=[m.checkpoint() for m in models],replay=replay,joint_episode=episode,rng=rng_state())
def restore(ck):
    models=[Learner(x['algorithm'],x['dim']) for x in ck['models']]
    for m,s in zip(models,ck['models']):m.restore(s);m.eval()
    restore_rng(ck['rng']);return models,ck['replay']

def evaluate(models,cases,out,label):
    saved=rng_state();rows=[]
    try:
        for assignment,roles in enumerate(ROLES):
            for seed in SEEDS:
                for name,water in cases:
                    for layout in LAYOUTS:
                        seed_all(seed+layout);metric,_=rollout(models,water,layout,roles)
                        rows.append(dict(assignment=assignment,action_seed=seed,case=name,layout=layout,**metric))
        result=dict(episodes=rows,summary=aggregate(rows),n=len(rows),whole_task_completed=sum(r['task_complete'] for r in rows),
          by_assignment={str(i):aggregate([r for r in rows if r['assignment']==i]) for i in range(2)},
          by_action_seed={str(s):aggregate([r for r in rows if r['action_seed']==s]) for s in SEEDS})
        write(out/(label+'.json'),result);return result
    finally:restore_rng(saved)

def admissible(c,b):
    s,t=c['summary'],b['summary'];worst=lambda x:min(v['RCR'] for v in x['by_assignment'].values())
    return bool(s['RCR']>=t['RCR']-1e-12 and s['coverage']>=t['coverage']-.01-1e-12 and s['ACC']<=.02+1e-12 and worst(c)>=worst(b)-1e-12 and (s['RCR']>=t['RCR']+.00625-1e-12 or s['coverage']>=t['coverage']+.005-1e-12))

def preflight(cases,out):
    checks=[]
    for algorithm in PARENTS:
        seed_all(41);m,ck=adapted(algorithm);old=Learner(algorithm,OLD_DIM);old.load_state_dict(ck['model']);obs=Env(cases[0][1],9001).observe();mask=np.ones((4,9),bool)
        for roles in ROLES:
            seed_all(73);a,l,v=old.act(obs,mask);seed_all(73);b,k,w=m.act(augment(obs,roles),mask)
            assert np.array_equal(a,b) and np.allclose(l,k,atol=2e-6) and abs(v-w)<2e-6
        checks.append(dict(test='zero_initialized_role_input_preserves_parent',algorithm=algorithm,passed=True));del ck,old,m;gc.collect()
    for partner in ['mat','hatrpo']:
        seed_all(91);a,_=adapted('hasac');b,_=adapted(partner);models=[a,b]
        _,rows=rollout(models,cases[0][1],990001,ROLES[0],True)
        own=partner_rows(rows)
        for r in own:
            inactive=r['obs'][0,-4:]==0;assert (r['mask'][inactive].sum(1)==1).all() and (r['logp'][inactive]==0).all()
        before=[model_hash([m]) for m in models];saved=copy.deepcopy(checkpoint(models,[],0));replay=[]
        losses=update_block(models,rows,replay);after=model_hash(models)
        assert all(model_hash([m])!=h for m,h in zip(models,before))
        restored,rp=restore(saved);losses2=update_block(restored,rows,rp)
        assert model_hash(restored)==after and losses==losses2
        if partner=='hatrpo':
            for i in [0,3]:
                for k,v in models[1].actors[i].state_dict().items():assert torch.equal(v,saved['models'][1]['model'][f'actors.{i}.{k}'])
        # Real serialized checkpoint includes optimizer/replay/RNG; next update must match.
        file=out/f'preflight-{partner}.pt';torch.save(checkpoint(models,replay,2),file)
        nextloss=update_block(models,rows,replay);expected=model_hash(models)
        resumed,rp2=restore(torch.load(file,map_location='cpu',weights_only=False));actual=update_block(resumed,rows,rp2)
        assert expected==model_hash(resumed) and nextloss==actual
        checks.append(dict(test='ownership_gradients_both_models_learn_and_exact_serialized_resume',partner=partner,passed=True))
        del models,restored,resumed,saved,replay,rp,rp2;gc.collect()
    write(out/'preflight.json',dict(status='passed',checks=checks))

def train(partner,cases,out):
    out.mkdir();seed_all(1001);models=[adapted('hasac')[0],adapted(partner)[0]];replay=[];history=[];begin=time.perf_counter()
    original=evaluate(models,cases['validation'],out,'validation-0000');best=original;best_ep=0;pending=[]
    for ep in range(1,257):
        if (out.parent/'STOP_AFTER_CHECKPOINT').exists():break
        case,water=cases['train'][(ep-1)%4];roles=ROLES[((ep-1)//4)%2] # balanced roles for every water parent
        metric,rows=rollout(models,water,930000+ep,roles,True);pending.extend(rows)
        loss=None
        if ep%2==0:loss=update_block(models,pending,replay);pending=[]
        with (out/'episodes.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(joint_episode=ep,case=case,layout=930000+ep,roles=roles,loss=loss,elapsed_s=time.perf_counter()-begin,**metric))+'\n')
        write(out/'status.json',dict(status='training',joint_episode=ep,target=256,best_joint_episode=best_ep,elapsed_s=time.perf_counter()-begin))
        if ep%64==0:
            val=evaluate(models,cases['validation'],out,f'validation-{ep:04d}');accepted=admissible(val,best)
            if accepted:best,best_ep=val,ep
            path=out/f'checkpoint-{ep:04d}.pt';temp=path.with_suffix('.partial');torch.save(checkpoint(models,replay,ep),temp);os.replace(temp,path)
            write(out/'latest.json',dict(joint_episode=ep,path=str(path),sha256=sha(path)));write(out/'best.json',dict(joint_episode=best_ep,validation=best))
            history.append(dict(joint_episode=ep,adopted=accepted,summary=val['summary'],by_assignment=val['by_assignment'],whole_task_completed=val['whole_task_completed']))
            write(out/'validation-history.json',history);print(partner,'joint',ep,json.dumps(val['summary']),flush=True)
    result=dict(status='completed' if ep==256 else 'stopped',joint_episodes=ep,best_joint_episode=best_ep,best_validation=best,history=history,elapsed_s=time.perf_counter()-begin)
    write(out/'status.json',result);return result

def main(out):
    out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest=ROOT/'uav6-pilot-20261101/data/dataset.json';items=json.loads(manifest.read_text(encoding='utf8'))['cases'];cases={}
    for item in items:
        p=Path(item['path']);assert sha(p)==item['sha256'];cases.setdefault(item['split'],[]).append((p.name,np.load(p,mmap_mode='r',allow_pickle=False)))
    protocol=dict(date='2026-11-01',kind='research mixed-learner joint fine-tuning',pairs=['HASAC1024+MAT256','HASAC1024+HATRPO1024'],joint_episodes_per_pair=256,
      parents={a:dict(path=str(p),sha256=sha(p)) for a,p in PARENTS.items()},source_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest),
      observation='append four public ownership bits; zero-initialized added input columns exactly preserve initial parent policy; no hidden/future water',
      training='two complete episodes with both policies frozen; then MAT/HATRPO on-policy update of owned actions only; four HASAC batch64 updates with mixed-policy bootstrap; replay capacity4096, reset old replay and optimizer state',
      hasac='bootstrap entropy only for owned HASAC agents; nine-action counterfactual expectation re-evaluates causal MAT responses; stochastic conditional-context approximation',
      role_schedule='each four training water parents use same diagonal allocation, next four opposite; both equally represented',
      updates='original per-algorithm update mechanisms, adapted ownership masks and HASAC targets; not an official algorithm reproduction and no HARL monotonicity guarantee',
      controls='same auto-service and RetryGuard for all; no coverage watchdog; fixed reward/task; 4 training synthetic parent fields',
      validation='reuse 2 parents x4 layouts x2 action seeds x2 allocations=32; every64 joint episodes; sealed50 untouched',
      best_rule='RCR nondecrease, coverage drop<=.01, ACC<=.02, worst-allocation RCR nondecrease, then RCR gain>=.00625 or coverage gain>=.005',
      stop='numerical/runtime failure or manual STOP_AFTER_CHECKPOINT only; no performance early stop',
      sources=['https://www.jmlr.org/papers/v25/23-0488.html','https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html','https://arxiv.org/abs/2205.14953'])
    write(out/'protocol.json',protocol);write(out/'queue-status.json',dict(status='preflight',pid=os.getpid()))
    preflight(cases['train'],out);results={}
    for partner in ['mat','hatrpo']:
        write(out/'queue-status.json',dict(status='training',current='hasac_'+partner,completed=list(results),pid=os.getpid(),elapsed_s=time.perf_counter()-begin))
        results[partner]=train(partner,cases,out/('hasac_'+partner));gc.collect()
        if results[partner]['status']!='completed':break
    assert {a:sha(p) for a,p in PARENTS.items()}=={a:v['sha256'] for a,v in protocol['parents'].items()}
    write(out/'results.json',results);write(out/'queue-status.json',dict(status='completed' if len(results)==2 and all(r['status']=='completed' for r in results.values()) else 'stopped',completed={a:r['joint_episodes'] for a,r in results.items()},elapsed_s=time.perf_counter()-begin))

if __name__=='__main__':
    out=Path(sys.argv[1])
    try:main(out)
    except Exception:
        if out.exists():write(out/'failure.json',dict(traceback=traceback.format_exc()));write(out/'queue-status.json',dict(status='failed'))
        raise
