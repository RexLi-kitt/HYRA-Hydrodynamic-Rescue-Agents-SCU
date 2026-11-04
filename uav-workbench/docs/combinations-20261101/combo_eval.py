"""Frozen-weight heterogeneous fleet experiments; no learner updates."""
import gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

EXP=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
ROOT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
sys.path[:0]=[str(EXP/'priority-20261101'),str(EXP)]
from flood1km.learners import Learner, distribution
from flood1km.train import aggregate,seed_all
from flood1km.data import sha
from flood1km.env import Env
from priority_train import rollout,write

MEMBERS={
 'HASAC1024':('hasac',1024,ROOT/'uav6-full1024-20261101/hasac/checkpoint-1024.pt'),
 'HATRPO1024':('hatrpo',1024,ROOT/'uav6-full1024-20261101/hatrpo/checkpoint-1024.pt'),
 'HASAC256':('hasac',256,ROOT/'uav6-total256-20261101/hasac/latest.pt'),
 'HATRPO512':('hatrpo',512,ROOT/'uav6-priority-20261101/hatrpo/checkpoint-0512.pt'),
 'MAT256':('mat',256,ROOT/'uav6-total256-20261101/mat/latest.pt'),
}
PAIRS=[('P1','HASAC1024','HATRPO1024'),('P2','HASAC256','HATRPO512'),('P3','HASAC1024','MAT256'),('P4','HATRPO1024','MAT256')]
ASSIGNMENTS=((0,1,1,0),(1,0,0,1))
SEEDS=(3101,3102)
LAYOUTS=(9001,9002,9003,9004)

class MixedFleet:
    """Each UAV uses its assigned expert. MAT sees the actual final prefix."""
    def __init__(self,models,owners): self.models=models;self.owners=tuple(owners)
    @torch.no_grad()
    def act(self,obs,mask,deterministic=False):
        o=torch.as_tensor(obs,dtype=torch.float32)[None]
        m=torch.as_tensor(mask,dtype=torch.bool)[None]
        actions=torch.zeros((1,4),dtype=torch.long);logs=[]
        encoded={k:model.policy.encode(o) for k,model in enumerate(self.models) if model.algorithm=='mat' and k in self.owners}
        for i,owner in enumerate(self.owners):
            model=self.models[owner]
            logits=model.policy.decode(encoded[owner],actions)[:,i] if model.algorithm=='mat' else model.actors[i](o[:,i])
            d=distribution(logits,m[:,i])
            actions[:,i]=d.probs.argmax(-1) if deterministic else d.sample()
            logs.append(d.log_prob(actions[:,i]))
        a=actions[0].numpy();l=torch.stack(logs,1)[0].numpy()
        assert mask[np.arange(4),a].all()
        return a,l,0. # Critic is unused during these evaluation-only rollouts.

def state_hash(model):
    h=hashlib.sha256()
    for name,t in model.state_dict().items():h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def preflight(models,water):
    e=Env(water,9001);o=e.observe();mask=e.available();checks=[]
    for name,model in models.items():
        wrapper=MixedFleet([model],(0,0,0,0))
        for deterministic in [False,True]:
            seed_all(777);a,l,_=model.act(o,mask,deterministic)
            seed_all(777);b,k,_=wrapper.act(o,mask,deterministic)
            assert np.array_equal(a,b) and np.allclose(l,k,rtol=0,atol=1e-6),(name,deterministic)
            checks.append(dict(check='same_expert_action_equivalence',model=name,deterministic=deterministic,passed=True))
    forced=np.zeros_like(mask);forced[np.arange(4),[8,0,1,8]]=True
    for pair,a,b in PAIRS:
        for owners in ASSIGNMENTS:
            act,logs,_=MixedFleet([models[a],models[b]],owners).act(o,forced)
            assert np.array_equal(act,[8,0,1,8]) and np.allclose(logs,0,atol=1e-6)
    checks.append(dict(check='all_pairs_service_singleton_masks',passed=True))
    mat=models['MAT256'];original=mat.policy.decode;prefixes=[]
    def capture(encoded,actions):prefixes.append(actions.detach().clone());return original(encoded,actions)
    mat.policy.decode=capture
    try:
        a,_,_=MixedFleet([models['HASAC1024'],mat],(0,1,0,1)).act(o,forced)
    finally: mat.policy.decode=original
    assert len(prefixes)==2 and int(prefixes[0][0,0])==a[0] and np.array_equal(prefixes[1][0,:3].numpy(),a[:3])
    checks.append(dict(check='MAT_conditions_on_final_mixed_fleet_prefix',passed=True))
    return dict(status='passed',checks=checks)

def summarize(rows):
    return dict(summary=aggregate(rows),whole_task_completed=sum(r['task_complete'] for r in rows),n=len(rows),
                by_action_seed={str(s):aggregate([r for r in rows if r['action_seed']==s]) for s in SEEDS})

def run(out):
    out.mkdir(parents=True,exist_ok=False);begin=time.perf_counter()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(1001)
    manifest=ROOT/'uav6-pilot-20261101/data/dataset.json'
    items=json.loads(manifest.read_text(encoding='utf8'))['cases']
    cases=[]
    for item in items:
        if item['split']!='validation':continue
        p=Path(item['path']);assert sha(p)==item['sha256'];cases.append((p.name,np.load(p,mmap_mode='r',allow_pickle=False)))
    protocol=dict(date='2026-11-01',request='实际组合，不限制原ABC配对',kind='frozen-weight heterogeneous fleet evaluation',
      training_episodes_added=0,learning_updates=0,parents={k:dict(algorithm=a,episode=ep,path=str(p),sha256=sha(p)) for k,(a,ep,p) in MEMBERS.items()},
      pairs=[dict(id=k,first=a,second=b) for k,a,b in PAIRS],assignments=[list(x) for x in ASSIGNMENTS],
      assignment_meaning='0/1 selects first/second model for UAV ids 0..3; two opposite diagonal allocations; both retained and equally weighted',
      single_controls=list(MEMBERS),common_controller='ServiceCommitment commit and RetryGuard enabled for all members and all combinations; no coverage watchdog',
      action_selection='stochastic; each expert emits only assigned UAV actions, MAT uses actual preceding mixed actions',
      validation=dict(parents=[x[0] for x in cases],layouts=list(LAYOUTS),action_seeds=list(SEEDS),per_single=16,per_pair=32,total=208,
                      scope='reused development validation; 2 water parents, not 208 independent flood events; sealed50 untouched'),
      success_rule=['pooled RCR >= max of both singles','pooled coverage >= max of both singles minus .01','ACC <= .02',
                    'worst action-seed RCR >= max of both singles worst action-seed RCR',
                    'each assignment RCR >= max single RCR minus .025',
                    'RCR gain >= .00625 OR coverage gain >= .005 relative to maximum respective single score'],
      limits='No claim of joint training or synergy from merely averaging scores. No hidden people or future water supplied to policy. Original scores and weights untouched.',
      cost='208 serial CPU2 rollouts, each max300 steps; expected minutes; stop on error or STOP file between rollouts',
      script_sha256=sha(Path(__file__)),manifest_sha256=sha(manifest))
    write(out/'protocol.json',protocol) # Freeze before evaluating any candidate.
    models={}
    for name,(algorithm,ep,p) in MEMBERS.items():
        ck=torch.load(p,map_location='cpu',weights_only=False);model=Learner(algorithm,ck['learner']['dim'])
        model.load_state_dict(ck['learner']['model']);model.eval();model.check_finite();models[name]=model
        del ck;gc.collect()
    originals={k:state_hash(m) for k,m in models.items()}
    write(out/'preflight.json',preflight(models,cases[0][1]))
    jobs=[(k,models[k],None) for k in MEMBERS]
    for name,a,b in PAIRS:
        for allocation,owners in enumerate(ASSIGNMENTS):jobs.append((f'{name}-allocation{allocation}',MixedFleet([models[a],models[b]],owners),allocation))
    completed=0;results={}
    for label,policy,allocation in jobs:
        rows=[]
        for seed in SEEDS:
            for case,water in cases:
                for layout in LAYOUTS:
                    if (out/'STOP').exists():raise RuntimeError('Manual STOP present')
                    seed_all(seed+layout)
                    metrics=rollout(policy,water,layout,True,False)
                    rows.append(dict(case=case,layout=layout,action_seed=seed,allocation=allocation,**metrics));completed+=1
                    write(out/'queue-status.json',dict(status='running',job=label,completed_rollouts=completed,total_rollouts=208,elapsed_s=time.perf_counter()-begin,pid=os.getpid()))
        result=dict(episodes=rows,**summarize(rows));write(out/(label+'.json'),result);results[label]=result
        print(label,json.dumps(result['summary']),flush=True)
    decisions=[]
    for name,a,b in PAIRS:
        rows=results[name+'-allocation0']['episodes']+results[name+'-allocation1']['episodes'];combined=dict(episodes=rows,**summarize(rows))
        c=combined['summary'];sa=results[a]['summary'];sb=results[b]['summary'];rmax=max(sa['RCR'],sb['RCR']);covmax=max(sa['coverage'],sb['coverage'])
        sworst=max(min(v['RCR'] for v in results[k]['by_action_seed'].values()) for k in [a,b])
        rules=dict(rescue_nonlower=c['RCR']>=rmax-1e-12,coverage_tolerance=c['coverage']>=covmax-.01-1e-12,conflict=c['ACC']<=.02+1e-12,
          action_seed_robustness=min(x['RCR'] for x in combined['by_action_seed'].values())>=sworst-1e-12,
          allocation_robustness=min(results[f'{name}-allocation{i}']['summary']['RCR'] for i in [0,1])>=rmax-.025-1e-12,
          material_gain=c['RCR']>=rmax+.00625-1e-12 or c['coverage']>=covmax+.005-1e-12)
        combined.update(first=a,second=b,rules=rules,qualified_for_next_training=all(rules.values()))
        write(out/(name+'-pooled.json'),combined);decisions.append(dict(pair=name,first=a,second=b,**summarize(rows),rules=rules,qualified_for_next_training=all(rules.values())))
    assert originals=={k:state_hash(m) for k,m in models.items()}
    write(out/'audit.json',dict(weights_unchanged=True,parameter_hashes=originals,training_episodes_added=0,rollouts=completed,source_protocol_hash=sha(out/'protocol.json')))
    write(out/'results.json',dict(single_controls={k:summarize(results[k]['episodes']) for k in MEMBERS},pairs=decisions,elapsed_s=time.perf_counter()-begin))
    write(out/'queue-status.json',dict(status='completed',completed_rollouts=completed,total_rollouts=208,elapsed_s=time.perf_counter()-begin,training_episodes_added=0))

if __name__=='__main__':
    out=Path(sys.argv[1])
    try: run(out)
    except Exception:
        if out.exists():write(out/'failure.json',dict(traceback=traceback.format_exc()))
        raise
