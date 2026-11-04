"""Isolated, bounded joint warm-start research. No prohibited learner or test leakage."""
import copy,gc,hashlib,json,os,random,shutil,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

BASE=Path(__file__).resolve().parent
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
EXP=REPO/'uav-workbench/experiments/flood1km-six-20261101'
ARCHIVE=OUT/'uav-linked-training-20261101'
DOC=REPO/'uav-workbench/docs/linked-training-20261101'
sys.path[:0]=[str(EXP),str(EXP/'priority-20261101')]
from flood1km.learners import Learner,distribution,optimize
from flood1km.train import seed_all,rng_state,restore_rng
from flood1km.data import generate,sha
from flood1km.env import Env
from service_commitment import ServiceCommitment
from coverage_controller import service_mask
from retry_guard import RetryGuard
from observed_planner import CoverageWatchdog

OLD_DIM=1468;DIM=1472
ROLES=((0,1,1,0),(1,0,0,1));ACTION_SEEDS=(3101,3102)
PARENTS={'hatrpo':OUT/'uav6-priority-20261101/hatrpo/checkpoint-0512.pt',
         'hasac':OUT/'uav6-total256-20261101/hasac/latest.pt',
         'maac':OUT/'uav6-total256-20261101/maac/latest.pt'}
PAIRS={'hatrpo_hasac':('hatrpo','hasac'),'hatrpo_maac':('hatrpo','maac'),'hasac_maac':('hasac','maac')}
TRAIN_SEED=1001;REPLAY_SIZE=1024

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,path)

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def augment(obs,roles):return np.concatenate((obs,np.broadcast_to(np.asarray(roles,np.float32),(4,4))),axis=1)
def model_hash(models):
    h=hashlib.sha256()
    for model in models:
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def adapted(algorithm):
    ck=torch.load(PARENTS[algorithm],map_location='cpu',weights_only=False)['learner']
    model=Learner(algorithm,DIM);state=model.state_dict()
    for name,old in ck['model'].items():
        new=state[name]
        if new.shape==old.shape:new.copy_(old);continue
        assert old.ndim==2 and new.shape[0]==old.shape[0],name
        new.zero_();n=old.shape[1]
        if n in (OLD_DIM,OLD_DIM+9):
            tail=n-OLD_DIM;assert new.shape[1]==DIM+tail
            new[:,:OLD_DIM].copy_(old[:,:OLD_DIM])
            if tail:new[:,DIM:].copy_(old[:,OLD_DIM:])
        else:
            tail=n-4*OLD_DIM;assert tail in (0,36) and new.shape[1]==4*DIM+tail,(name,old.shape,new.shape)
            for i in range(4):new[:,i*DIM:i*DIM+OLD_DIM].copy_(old[:,i*OLD_DIM:(i+1)*OLD_DIM])
            if tail:new[:,4*DIM:].copy_(old[:,4*OLD_DIM:])
    model.load_state_dict(state);model.alpha=ck['alpha'];model.eval();model.check_finite()
    return model,ck

@torch.no_grad()
def sample(models,obs,mask,deterministic=False):
    roles=obs[:,0,-4:].long();actions=[];logs=[]
    for i in range(4):
        logits=torch.where((roles[:,i]==1)[:,None],models[1].actors[i](obs[:,i]),models[0].actors[i](obs[:,i]))
        d=distribution(logits,mask[:,i]);a=d.probs.argmax(-1) if deterministic else d.sample()
        actions.append(a);logs.append(d.log_prob(a))
    return torch.stack(actions,1),torch.stack(logs,1)

class Controller:
    def __init__(self,pure=False,coverage=True):
        self.service=ServiceCommitment('commit');self.guard=RetryGuard();self.planner=CoverageWatchdog()
        self.pure=pure;self.coverage=coverage
        if pure:self.planner.active=True

    def decide(self,models,obs,boundary,roles,deterministic=False):
        forced,detail=self.service.select(obs,boundary,np.full(4,8,dtype=np.int64))
        mask=service_mask(boundary,forced,detail['phases']);mask,gdetail=self.guard.mask(obs,mask,detail['phases'])
        # Plan BEFORE policy sampling. No action-dependent post-hoc log-probability.
        # Learned agents start from a legal deterministic public placeholder.
        proposals=forced.copy()
        for i,p in enumerate(detail['phases']):
            if p=='learned':proposals[i]=8 if mask[i,8] else int(np.flatnonzero(mask[i])[0])
        if self.coverage:
            planned,pdetail=self.planner.select(obs,mask,proposals,detail['phases'])
            for i in pdetail['planned_ids']:
                mask[i]=False;mask[i,planned[i]]=True
        else:pdetail=dict(active=False,planned_ids=[],planned_agents=0)
        aug=augment(obs,roles)
        if self.pure:
            a=proposals.copy()
            for i in pdetail['planned_ids']:a[i]=planned[i]
            logs=np.zeros(4,np.float32);value=0.
        else:
            a,l=sample(models,torch.from_numpy(aug)[None],torch.from_numpy(mask)[None],deterministic)
            a=a[0].numpy();logs=l[0].numpy();value=0.
            for m in models:
                if m.algorithm=='hatrpo':
                    with torch.no_grad():value=float(m.value(torch.from_numpy(aug)[None])[0])
        assert mask[np.arange(4),a].all()
        controlled=mask.sum(-1)==1
        assert np.abs(logs[controlled]).max(initial=0)<1e-6
        service=np.array([p!='learned' for p in detail['phases']]);assert np.array_equal(a[service],forced[service])
        self.guard.remember(a)
        return a,mask,logs,value,aug,dict(planned=pdetail['planned_ids'],service=service.tolist(),free=(mask.sum(-1)>1).tolist(),guard=gdetail['constrained_agents'])

def rollout(models,water,layout,roles,collect=False,deterministic=False,pure=False,coverage=True,trace_path=None):
    env=Env(water,layout);obs=env.observe();controller=Controller(pure,coverage);previous=None;rows=[];done=False;ret=0.
    planner_steps=free_steps=service_steps=blocked_steps=0;traces=[]
    while not done:
        a,mask,logs,value,aug,detail=controller.decide(models,obs,env.available(),roles,deterministic)
        if previous is not None:previous.update(next_mask=mask.copy(),next_action=a.copy());rows.append(previous)
        before=env.pos.copy();nxt,reward,done,info=env.step(a);ret+=reward
        assert not (controller.planner.known & ~env.known).any()
        blocked=int(((np.linalg.norm(env.pos-before,axis=1)<1e-8)&(a!=8)).sum());blocked_steps+=blocked
        planner_steps+=len(detail['planned']);free_steps+=sum(detail['free']);service_steps+=sum(detail['service'])
        if trace_path:
            traces.append(dict(position=env.pos.copy(),action=a.copy(),planned=[i in detail['planned'] for i in range(4)],
              service=detail['service'],free=detail['free'],info=[info[k] for k in ('new_cells','found','rescued','conflicts','boundary')]))
        if collect:previous=dict(obs=aug,mask=mask.copy(),action=a.copy(),logp=logs.copy(),value=value,reward=reward,done=float(done),
          next_obs=augment(nxt,roles),next_mask=env.available().copy(),next_action=np.full(4,8,dtype=np.int64))
        obs=nxt
    if collect:rows.append(previous)
    metric=env.metrics();assert len({x['target'] for x in env.events if x['type']=='rescued'})==metric['rescued']
    metric.update(return_sum=ret,planner_fraction=planner_steps/(4*env.t),neural_free_fraction=free_steps/(4*env.t),
      service_fraction=service_steps/(4*env.t),blocked_agent_steps=blocked_steps,unserved_ids=np.flatnonzero(env.status!=2).tolist())
    if trace_path:
        np.savez_compressed(trace_path,**{k:np.asarray([t[k] for t in traces]) for k in traces[0]},
          known=env.known,people=env.people,target_status=env.status,final_position=env.pos)
    return metric,rows

def owned_rows(rows,owner):
    result=[]
    for r in rows:
        x=dict(r);mask=r['mask'].copy();logs=r['logp'].copy();inactive=r['obs'][0,-4:]!=owner
        mask[inactive]=False;mask[np.flatnonzero(inactive),r['action'][inactive]]=True;logs[inactive]=0
        x.update(mask=mask,logp=logs);result.append(x)
    return result

def update_off(models,owner,rows):
    model=models[owner];b=model.batch(rows);o=b['obs'];a=b['action'];mask=b['mask'];own=o[:,0,-4:]==owner
    with torch.no_grad():
        na,nlog=sample(models,b['next_obs'],b['next_mask']);next_own=b['next_obs'][:,0,-4:]==owner
        if model.algorithm=='hasac':
            nq=torch.minimum(*[q(b['next_obs'],na) for q in model.target])-model.alpha*(nlog*next_own).sum(-1)
            target=b['reward']+model.gamma*(1-b['done'])*nq
        else:
            nq=model.target(b['next_obs'],na).gather(-1,na[...,None]).squeeze(-1)-model.alpha*nlog
            target=b['reward'][:,None]+model.gamma*(1-b['done'][:,None])*nq
    if model.algorithm=='hasac':loss=sum(F.mse_loss(q(o,a),target) for q in model.critic)
    else:
        prediction=model.critic(o,a).gather(-1,a[...,None]).squeeze(-1)
        loss=((prediction-target).square()*own).sum()/own.sum()
    norm=optimize(loss,model.optimizers[4],model.critic.parameters());records=[]
    context,_=sample(models,o,mask)
    order=torch.randperm(4).tolist() if model.algorithm=='hasac' else list(range(4))
    for i in order:
        active=own[:,i]&(mask[:,i].sum(-1)>1)
        if not active.any():continue
        oi=o[active];mi=mask[active]
        if model.algorithm=='hasac':context_i,_=sample(models,oi,mi)
        else:context_i=context[active]
        with torch.no_grad():
            if model.algorithm=='hasac':
                values=[]
                for act in range(9):
                    joint=context_i.clone();joint[:,i]=act
                    values.append(torch.minimum(*[q(oi,joint) for q in model.critic]))
                q_values=torch.stack(values,-1)
            else:q_values=model.critic(oi,context_i)[:,i]
        dist=distribution(model.actors[i](oi[:,i]),mi[:,i])
        aloss=(dist.probs*(model.alpha*dist.logits-q_values)).sum(-1).mean()
        anorm=optimize(aloss,model.optimizers[i],model.actors[i].parameters());records.append(dict(agent=i,loss=float(aloss.detach()),gradient_norm=anorm))
    model.soft_target(.01);model.updates+=1;model.check_finite()
    return dict(critic_loss=float(loss.detach()),gradient_norm=norm,actors=records)

def update_block(models,rows,replay):
    losses={}
    # On-policy data belongs to the fixed team that collected this two-episode block.
    for owner,m in enumerate(models):
        if m.algorithm=='hatrpo':losses[m.algorithm]=m.update_on_policy(owned_rows(rows,owner))
    replay.extend(rows);del replay[:-REPLAY_SIZE]
    for owner,m in enumerate(models):
        if m.algorithm in ('hasac','maac'):
            losses[m.algorithm]=[update_off(models,owner,random.sample(replay,min(64,len(replay)))) for _ in range(4)]
    return losses

def checkpoint(models,replay,episode,identity):
    return dict(models=[m.checkpoint() for m in models],replay=replay,joint_episode=episode,rng=rng_state(),identity=identity)
def restore(ck):
    models=[Learner(x['algorithm'],x['dim']) for x in ck['models']]
    for m,s in zip(models,ck['models']):m.restore(s);m.eval()
    restore_rng(ck['rng']);return models,ck['replay']
def save(path,ck):
    temp=path.with_suffix('.partial');torch.save(ck,temp);os.replace(temp,path)

def summary(rows):
    return dict(n=len(rows),completed=sum(r['task_complete'] for r in rows),completion=float(np.mean([r['task_complete'] for r in rows])),
      **{k:float(np.mean([r[k] for r in rows])) for k in ('RCR','coverage','ACC','ST','planner_fraction','neural_free_fraction')},
      max_conflicts=max(r['conflicts'] for r in rows),unserved=sum(20-r['rescued'] for r in rows))

def evaluate(models,cases,out,episode,pure=False,coverage=True):
    saved=rng_state();rows=[];label=f'eval-{episode:04d}'+('' if coverage else '-no-cover');tracedir=out/label;tracedir.mkdir(exist_ok=False)
    try:
        for stratum,name,water in cases:
            layouts=(9001,9401) if stratum=='legacy' else (9501,9502)
            for assignment,roles in enumerate(ROLES[:1] if pure else ROLES):
                for mode,seeds in ([('fixed',(3101,))] if pure else [('sample',ACTION_SEEDS),('fixed',(3101,))]):
                    for aseed in seeds:
                        for layout in layouts:
                            seed_all(aseed+layout)
                            key=f'{name}-{layout}-{assignment}-{mode}-{aseed}'
                            metric,_=rollout(models,water,layout,roles,deterministic=mode=='fixed',pure=pure,coverage=coverage,trace_path=tracedir/(key+'.npz'))
                            rows.append(dict(case=name,stratum=stratum,layout=layout,assignment=assignment,mode=mode,action_seed=aseed,trace=f'{label}/{key}.npz',**metric))
        grouped={f'{s}/{m}':summary([r for r in rows if r['stratum']==s and r['mode']==m]) for s in ('legacy','new') for m in ('sample','fixed') if any(r['stratum']==s and r['mode']==m for r in rows)}
        result=dict(joint_episode=episode,episodes=rows,groups=grouped,summary=summary(rows),
          worst_assignment_RCR=min(summary([r for r in rows if r['assignment']==a])['RCR'] for a in {r['assignment'] for r in rows}))
        write(out/(label+'.json'),result);return result
    finally:restore_rng(saved)

def quality(val):
    s=val['summary'];violations=sum(g['ACC']>.02+1e-12 for g in val['groups'].values())
    return (-violations,s['completion'],s['RCR'],val['worst_assignment_RCR'],s['coverage'],-s['ACC'],-s['ST'])

def gain(a,b):
    x,y=a['summary'],b['summary']
    return x['completion']>=y['completion']+1/x['n']-1e-12 or x['RCR']>=y['RCR']+.005-1e-12 or x['coverage']>=y['coverage']+.005-1e-12 or (x['completion']>=y['completion'] and x['RCR']>=y['RCR'] and x['coverage']>=y['coverage']-.001 and x['ST']<=y['ST']-5)

def improving(a,b):
    x,y=a['summary'],b['summary']
    return (x['completion']>=y['completion']-1e-12 and x['RCR']>=y['RCR']-1e-12 and x['coverage']>=y['coverage']-.001-1e-12
      and all(g['ACC']<=.02+1e-12 for g in a['groups'].values()) and x['max_conflicts']<=10 and gain(a,b))

def trend2048(folder):
    vals={e:read(folder/f'eval-{e:04d}.json') for e in (256,768,896,1024)}
    checks=dict(improves_768_to_896=improving(vals[896],vals[768]),improves_896_to_1024=improving(vals[1024],vals[896]),
       material_gain_over_256=improving(vals[1024],vals[256]),best_after_256=read(folder/'best.json')['joint_episode']>256)
    return dict(eligible=all(checks.values()),checks=checks)

def validate_records(folder):
    checks=[]
    for p in sorted(folder.glob('eval-*.json')):
        val=read(p)
        for r in val['episodes']:
            t=np.load(folder/r['trace'],allow_pickle=False)
            assert len(t['action'])==r['ST']
            assert int(t['info'][:,2].sum())==r['rescued'] and int(t['info'][:,3].sum())==r['conflicts']
            assert int((t['target_status']==2).sum())==r['rescued']
            assert abs(float(t['known'].mean())-r['coverage'])<1e-12
        checks.append(dict(file=p.name,n=len(val['episodes']),passed=True))
    return checks

def stop_requested(run):return (run/'STOP_AFTER_CHECKPOINT').exists() or (ARCHIVE/'STOP_AFTER_CHECKPOINT').exists()

def train_stage(cid,cases,run,target,identity):
    folder=run/cid;folder.mkdir(exist_ok=True);start=time.perf_counter()
    if (folder/'latest.json').exists():
        latest=read(folder/'latest.json');p=folder/latest['file'];assert sha(p)==latest['sha256']
        ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['identity']==identity
        models,replay=restore(ck);begin=ck['joint_episode'];best_info=read(folder/'best.json');best=read(folder/best_info['evaluation']);best_ep=best_info['joint_episode']
    else:
        seed_all(TRAIN_SEED);models=[adapted(a)[0] for a in PAIRS[cid]];replay=[];begin=0
        best=evaluate(models,cases['validation'],folder,0);best_ep=0
        save(folder/'checkpoint-0000.pt',checkpoint(models,replay,0,identity));write(folder/'best.json',dict(joint_episode=0,evaluation='eval-0000.json',file='checkpoint-0000.pt',quality=quality(best)))
        evaluate(models,cases['validation'],folder,0,coverage=False)
    pending=[];ep=begin
    for ep in range(begin+1,target+1):
        name,water=cases['train'][(ep-1)%len(cases['train'])];roles=ROLES[((ep-1)//len(cases['train']))%2]
        metric,rows=rollout(models,water,8000000+TRAIN_SEED*10000+ep,roles,True);pending.extend(rows)
        loss=None
        if ep%2==0:loss=update_block(models,pending,replay);pending=[]
        record=dict(joint_episode=ep,case=name,layout=8000000+TRAIN_SEED*10000+ep,roles=roles,loss=loss,elapsed_stage_s=time.perf_counter()-start,**metric)
        with (folder/'episodes.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        write(folder/'status.json',dict(status='training',joint_episode=ep,target=target,best_joint_episode=best_ep,elapsed_stage_s=time.perf_counter()-start))
        if ep%32==0:print(json.dumps(dict(candidate=cid,episode=ep,target=target,elapsed_stage_s=time.perf_counter()-start)),flush=True)
        check=ep%64==0
        evaluate_now=(ep<=256 and check) or (ep>256 and ep%128==0)
        if check or (ep%2==0 and stop_requested(run)):
            p=folder/f'checkpoint-{ep:04d}.pt';save(p,checkpoint(models,replay,ep,identity))
            write(folder/'latest.json',dict(joint_episode=ep,file=p.name,sha256=sha(p)))
        if evaluate_now:
            val=evaluate(models,cases['validation'],folder,ep)
            if quality(val)>quality(best):best,best_ep=val,ep
            write(folder/'best.json',dict(joint_episode=best_ep,evaluation=f'eval-{best_ep:04d}.json',file=f'checkpoint-{best_ep:04d}.pt',quality=quality(best)))
            print(json.dumps(dict(candidate=cid,evaluation=ep,best=best_ep,summary=val['summary'])),flush=True)
        if ep%2==0 and stop_requested(run):break
    result=dict(status='stage_completed' if ep==target else 'manual_stopped',joint_episode=ep,target=target,best_joint_episode=best_ep,best_summary=best['summary'],elapsed_stage_s=time.perf_counter()-start)
    write(folder/'status.json',result)
    if ep==target:evaluate(models,cases['validation'],folder,ep,coverage=False)
    del models,replay;gc.collect();return result

def preflight(cases,run):
    checks=[];water=cases['train'][0][1]
    for algorithm in PARENTS:
        seed_all(41);m,ck=adapted(algorithm);old=Learner(algorithm,OLD_DIM);old.load_state_dict(ck['model']);obs=Env(water,9001).observe();mask=np.ones((4,9),bool)
        for roles in ROLES:
            seed_all(73);a,l,v=old.act(obs,mask);seed_all(73);b,k,w=m.act(augment(obs,roles),mask)
            assert np.array_equal(a,b) and np.allclose(l,k,atol=2e-6) and abs(v-w)<2e-6
            acts=torch.tensor([[1,2,3,4]])
            if algorithm=='maac':
                with torch.no_grad():assert torch.allclose(old.critic(torch.from_numpy(obs)[None],acts),m.critic(torch.from_numpy(augment(obs,roles))[None],acts),atol=2e-6)
        checks.append(dict(test='parent_policy_and_role_padding_preserved',algorithm=algorithm,passed=True))
    for cid,pair in PAIRS.items():
        seed_all(91);models=[adapted(a)[0] for a in pair];_,rows=rollout(models,water,990001,ROLES[0],True)
        assert any(r['mask'].sum()>4 for r in rows),'no free decisions available for learning'
        saved=copy.deepcopy(checkpoint(models,[],0,{}));before=[model_hash([m]) for m in models];replay=[]
        loss=update_block(models,rows,replay)
        assert all(model_hash([m])!=h for m,h in zip(models,before))
        for j,m in enumerate(models):
            for i in range(4):
                if ROLES[0][i]!=j:
                    for k,v in m.actors[i].state_dict().items():assert torch.equal(v,saved['models'][j]['model'][f'actors.{i}.{k}'])
        p=run/f'preflight-{cid}.pt';save(p,checkpoint(models,replay,2,{}))
        loss1=update_block(models,rows,replay);expected=model_hash(models)
        restored,rp=restore(torch.load(p,map_location='cpu',weights_only=False));loss2=update_block(restored,rows,rp)
        assert model_hash(restored)==expected and loss1==loss2
        checks.append(dict(test='both_models_update_ownership_and_exact_resume',pair=cid,passed=True))
        del models,restored,saved,replay,rp;gc.collect()
    models=[adapted(a)[0] for a in PAIRS['hatrpo_hasac']]
    e=Env(water,9001);obs=e.observe();boundary=e.available();seed_all(51)
    x=Controller().decide(models,obs,boundary,ROLES[0]);e.water=np.zeros_like(water);seed_all(51)
    y=Controller().decide(models,obs,boundary,ROLES[0]);assert np.array_equal(x[0],y[0])
    c=Controller(pure=True);a,mask,logs,_,_,detail=c.decide(None,obs,boundary,ROLES[0]);assert (mask[np.array(detail['planned'])].sum(1)==1).all()
    checks.append(dict(test='observed_only_controller_and_presampling_rule_masks',passed=True))
    write(run/'preflight.json',dict(status='passed',checks=checks))

def prepare(run):
    manifest=read(OUT/'uav6-pilot-20261101/data/dataset.json');rows=manifest['cases']
    for r in rows:assert sha(Path(r['path']))==r['sha256']
    terrain=REPO/'terrain-workbench/terrain-lab'
    for split,seeds in [('train',list(range(31001,31013))),('validation',[41001,41002,41003,41004])]:
        for s in seeds:rows.append(generate(run/'data',terrain,s,split))
    assert len({r['seed'] for r in rows})==22
    write(run/'dataset.json',dict(cases=rows,train_parents=16,validation_parents=6,test_generated=False))
    cases={'train':[],'validation':[]}
    for r in rows:
        p=Path(r['path']);assert sha(p)==r['sha256'];a=np.load(p,mmap_mode='r',allow_pickle=False)
        if r['split']=='train':cases['train'].append((p.stem,a))
        else:cases['validation'].append(('legacy' if r['seed'] in (2101,2201) else 'new',p.stem,a))
    return cases

def protocol(run):
    sources=[Path(__file__),BASE/'observed_planner.py',EXP/'coverage_controller.py',EXP/'service_commitment.py',EXP/'priority-20261101/retry_guard.py']+list((EXP/'flood1km').glob('*.py'))+list((REPO/'terrain-workbench/terrain-lab').glob('flood_*.py'))
    return dict(date='2026-11-01',authorization='User explicitly authorized new old-model linkage training: all 256, choose better pairs to 1024; conditionally 2048. Supersedes prior prescreen stop only for this new experiment.',
      pairs=PAIRS,parents={k:dict(path=str(v),sha256=sha(v),pretrained_episodes={'hatrpo':512,'hasac':256,'maac':256}[k]) for k,v in PARENTS.items()},
      source_hashes={str(p):sha(p) for p in sources},seed=TRAIN_SEED,threads=2,device='cpu',no_subagents=True,
      dataset=dict(train_existing=[1101,1201,1301,1401],train_new=list(range(31001,31013)),validation_existing=[2101,2201],validation_new=[41001,41002,41003,41004],sealed50_used=False),
      semantics='2+2 diagonal role assignment, swapped every 16 training episodes; all 16 parents see both roles. Four public role bits added with zero input weights. Two complete episodes per joint update.',
      controls='Same service/RetryGuard/coverage watchdog in training and evaluation. Coverage planning runs BEFORE network sampling on deterministic legal placeholders; planned actions become singleton masks. No hindsight relabeling of forced actions as policy samples.',
      learning='HATRPO on-policy owned actions first; then owned HASAC/MAAC actors with mixed-team bootstrap and joint observed critic; MAAC loss uses owned agents. Replay capacity1024, four batch64 updates per off-policy learner per two episodes. Fresh replay/optimizers, preserve parent parameters/alpha. Research adaptation; no monotonicity guarantee.',
      validation='6 parents x2 layouts x2 roles x(2 stochastic seeds+1 fixed)=72 per checkpoint; legacy/new and fixed/stochastic separated; two old parents use9001/9401, four new use9501/9502. A single training seed screening, not replicated significance.',
      schedule='3 pairs each256 new joint episodes, checkpoints every64, evaluate0/64/128/192/256. Rank best checkpoints, choose top2 to total1024, continue latest state not restart best; evaluate every128 after256. Latest and best kept separately. No performance early stop before each authorized stage.',
      ranking='lexicographic: fewer legacy/new x sample/fixed ACC>.02 groups, then higher whole-task completion, RCR, worst assignment RCR, coverage, lower ACC, shorter capped steps. Expansion selection is not acceptance. Tie broken by pair id.',
      extend2048='For at most2 finalists: last768->896 and896->1024 both improve with nondecreasing completion/RCR, coverage loss<=.001, every groupACC<=.02 and maxconflicts<=10; improvement means completion+1/n OR RCR+.005 OR coverage+.005 OR nondecreasing task metrics and>=5 fewer capped steps. Also improve1024 over256 and best checkpoint after256. Otherwise stop1024.',
      hard_stop='manual STOP_AFTER_CHECKPOINT, numerical/runtime/integrity failure, disk<8GiB at stage boundary, wall>4h at stage boundary. No automatic failure retry. Up to4864 new joint episodes (3*256+2*768+2*1024), not multiplied by number of member models.',
      caveats='Synthetic small-scale research, no buildings/weather/comms/battery/real delivery verification; no real rescue accuracy claim; no new hierarchical task head in this experiment; selected weights need future multi-training-seed confirmation.',
      references=['https://www.jmlr.org/papers/v25/23-0488.html','https://proceedings.mlr.press/v97/iqbal19a.html'])

def guard_budget(run,start):
    if stop_requested(run):raise RuntimeError('Manual stop requested at stage boundary')
    if shutil.disk_usage(run).free<8*1024**3:raise RuntimeError('Less than8GiB free; no next stage')
    if time.perf_counter()-start>4*3600:raise RuntimeError('Four-hour boundary reached; no next stage')

def report(run):
    queue=read(run/'queue-status.json');lines=['# 旧模型联动分阶段训练执行记录','',f"队列状态：{queue['status']}。旧权重保留，本轮新计数为共同经历的任务回合，不能按两种模型翻倍。",'',
      '|组合|本轮实际联合回合|保留最佳回合|完整任务比例|人员服务比例|覆盖比例|潜在冲突/步|','|---|---:|---:|---:|---:|---:|---:|']
    for cid in PAIRS:
        folder=run/cid
        if not (folder/'best.json').exists():continue
        best=read(folder/'best.json');s=read(folder/best['evaluation'])['summary'];status=read(folder/'status.json') if (folder/'status.json').exists() else {'joint_episode':0}
        lines.append(f"|{cid}|{status['joint_episode']}|{best['joint_episode']}|{s['completion']:.2%}|{s['RCR']:.2%}|{s['coverage']:.4%}|{s['ACC']:.5f}|")
    lines+=['','每次评价共72回放，6个合成父水场；含两种分工、固定动作和两种随机动作种子。全量逐场与legacy/new分层保留，不是72个独立洪水事件。此表仅汇总；结论要同时查看最差分工及冲突尾部。','',
      '学习收益必须对照本组合第0回合；补漏控制收益另与no-cover评价对照。原奖励、服务阈值、运动及冲突判定保持。训练0权重经过相同新增控制器评价，不能拿旧口径分数当基线。','',
      '256后选前两组扩训是按用户要求进行研究探索，不表示通过工程或现场验收。2048须满足事前趋势条件。旧模型训练轮次与本轮新增共同回合分别记录于protocol.json。','',
      '更改：只新增隔离的双模型混合训练入口、预采样覆盖动作掩码和归档，不修改原源码。重放缓存及优化器重新初始化，角色输入零初始化；无高层任务分配接口。回退仍为原项目和父权重。','',
      '检查：前置父策略等价、动作所有权、两成员参数更新、真实序列化后精确续训、因果输入与掩码检查；每阶段核对逐场轨迹和末态，代码/权重哈希。失败则停止，不循环重试。','',
      '局限及未做：封存50测试、三个训练种子重复、真实水情与人员、建筑通信续航验证均未执行。仅扩大至16训练/6验证合成父水场。','',
      '论文依据：[HARL](https://www.jmlr.org/papers/v25/23-0488.html)、[MAAC](https://proceedings.mlr.press/v97/iqbal19a.html)。混合训练为本项目适配，不继承论文的单调改进保证。']
    if (run/'selection-0256.json').exists():lines+=['','256筛选：'+json.dumps(read(run/'selection-0256.json'),ensure_ascii=False)]
    if (run/'extension-1024.json').exists():lines+=['','2048条件：'+json.dumps(read(run/'extension-1024.json'),ensure_ascii=False)]
    (run/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

def publish(run,final=False):
    report(run)
    for target in (ARCHIVE,DOC):
        target.mkdir(parents=True,exist_ok=True)
        for name in ('linked_train.py','observed_planner.py','protocol-addendum.md'):shutil.copy2(BASE/name,target/name)
        for p in run.rglob('*'):
            if not p.is_file() or p.suffix in ('.pt','.partial','.npy'):continue
            dest=target/p.relative_to(run);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        write(target/'live-run.json',dict(status=read(run/'queue-status.json')['status'],live_path=str(run),checkpoint_location=str(run),large_water_fields='live data path in dataset.json; not duplicated'))
    weights=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in run.rglob('checkpoint-*.pt')]
    for target in (ARCHIVE,DOC):write(target/'checkpoint-index.json',weights)
    if final:
        checks={str(p):sha(Path(p)) for p in read(run/'protocol.json')['source_hashes']}
        assert checks==read(run/'protocol.json')['source_hashes']
        for k,p in PARENTS.items():assert sha(p)==read(run/'protocol.json')['parents'][k]['sha256']
        audits={cid:validate_records(run/cid) for cid in PAIRS if (run/cid).exists()}
        write(run/'integrity-audit.json',dict(status='passed',source_and_parents_unchanged=True,records=audits))
        for target in (ARCHIVE,DOC):shutil.copy2(run/'integrity-audit.json',target/'integrity-audit.json')
        note='## 2026-11-01 旧模型联动阶段队列结束\n\n状态'+read(run/'queue-status.json')['status']+'。3组256→较优2组1024→符合趋势者2048，实际轮次、最佳版本及全部退化以outputs/uav-linked-training-20261101/report.md和queue-status.json为准。16训练/6验证合成父水场；封存50未用；原权重及源码未修改，结果双份保存。不要重复启动同配置。\n\n'
        update_logs(note)

def update_logs(note):
    for root in (OUT,REPO/'terrain-workbench/docs'):
        for name in ('HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md'):
            p=root/name
            if name=='HYRA自动续作进度.md':p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
            else:
                with p.open('a',encoding='utf8') as f:f.write('\n'+note)

def main():
    run=BASE/'run01';run.mkdir(exist_ok=False);start=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    proto=protocol(run);write(run/'protocol.json',proto);write(run/'queue-status.json',dict(status='preparing',pid=os.getpid(),started=time.strftime('%Y-%m-%d %H:%M:%S')))
    cases=prepare(run);preflight(cases,run)
    identity=dict(protocol_sha256=sha(run/'protocol.json'),dataset_sha256=sha(run/'dataset.json'))
    (run/'pure_planning').mkdir();evaluate(None,cases['validation'],run/'pure_planning',0,pure=True)
    write(run/'queue-status.json',dict(status='training',stage=256,current=list(PAIRS)[0],pid=os.getpid()))
    update_logs('## 2026-11-01 旧模型联动新队列已启动\n\n用户新增授权三组旧模型共同训练：HATRPO512+HASAC256、HATRPO512+MAAC256、HASAC256+MAAC256。先各新增256共同回合，选前两组到1024，符合事前持续改善条件者到2048。已扩充16训练/6验证合成父水场并完成启动检查；服务和补漏预采样掩码，权重所有权分开。旧预选停训结论保留，此为新数据/控制协议，不是重跑旧配置。outputs/uav-linked-training-20261101/live-run.json为实时入口，勿重复启动。单CPU两线程串行，封存50未用，未外发。\n\n')
    publish(run)
    for cid in PAIRS:
        guard_budget(run,start);write(run/'queue-status.json',dict(status='training',stage=256,current=cid,pid=os.getpid()))
        state=train_stage(cid,cases,run,256,identity);publish(run)
        if state['status']=='manual_stopped':write(run/'queue-status.json',dict(status='manual_stopped',current=cid));publish(run,True);return
    ranked=sorted(PAIRS,key=lambda cid:(quality(read(run/cid/read(run/cid/'best.json')['evaluation'])),cid),reverse=True);selected=ranked[:2]
    write(run/'selection-0256.json',dict(ranking=ranked,selected=selected,kind='research expansion, not acceptance',scores={c:read(run/c/'best.json') for c in PAIRS}))
    for cid in selected:
        guard_budget(run,start);write(run/'queue-status.json',dict(status='training',stage=1024,current=cid,pid=os.getpid()))
        state=train_stage(cid,cases,run,1024,identity);publish(run)
        if state['status']=='manual_stopped':write(run/'queue-status.json',dict(status='manual_stopped',current=cid));publish(run,True);return
    decisions={cid:trend2048(run/cid) for cid in selected};write(run/'extension-1024.json',decisions)
    for cid in selected:
        if not decisions[cid]['eligible']:continue
        guard_budget(run,start);write(run/'queue-status.json',dict(status='training',stage=2048,current=cid,pid=os.getpid()))
        state=train_stage(cid,cases,run,2048,identity);publish(run)
        if state['status']=='manual_stopped':write(run/'queue-status.json',dict(status='manual_stopped',current=cid));publish(run,True);return
    write(run/'queue-status.json',dict(status='completed',completed={c:read(run/c/'status.json')['joint_episode'] for c in PAIRS},elapsed_s=time.perf_counter()-start,selected1024=selected,extension=decisions,pid=os.getpid()))
    publish(run,True)

if __name__=='__main__':
    try:main()
    except Exception:
        run=BASE/'run01'
        if run.exists():
            write(run/'failure.json',dict(traceback=traceback.format_exc()))
            write(run/'queue-status.json',dict(status='failed',pid=os.getpid()))
            try:publish(run)
            except Exception:pass
        raise
