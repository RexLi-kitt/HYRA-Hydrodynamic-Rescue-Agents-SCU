import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031\uav-workbench\experiments\flood1km-six-20261101')
sys.path.insert(0,str(repo))
from flood1km.env import Env
from flood1km.learners import Learner,distribution
from flood1km.train import aggregate,seed_all
from train_with_service import decision
from service_commitment import ServiceCommitment
base=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs\uav6-pilot-20261101')
new=base.parent/'uav6-total256-20261101'
out=Path(__file__).with_suffix('');out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
write('protocol.json',dict(algorithms=['mat','coma'],no_training=True,no_weight_change=True,
    greedy_checkpoints=[64,256],stochastic_checkpoint=256,stochastic_seeds=[3101,3102],validation_layouts=[9001,9002,9003,9004],
    scope='Diagnostic sampling only; retain original deterministic score and do not replace official evaluation.',
    script_hash=sha(Path(__file__)),checks={a:{'64':sha(base/a/'latest.pt'),'256':sha(new/a/'latest.pt')} for a in ['mat','coma']}))
items=json.loads((base/'data/dataset.json').read_text(encoding='utf-8'))['cases']
cases=[(Path(x['path']).name,np.load(x['path'],mmap_mode='r')) for x in items if x['split']=='validation']
results={}
for a in ['mat','coma']:
    results[a]={}
    for total in [64,256]:
        ck=torch.load((base if total==64 else new)/a/'latest.pt',map_location='cpu',weights_only=False)['learner']
        model=Learner(a,ck['dim']);model.restore(ck)
        modes=[('greedy',0)]+([('sample',3101),('sample',3102)] if total==256 else [])
        for mode,seed in modes:
            rows=[]
            for name,water in cases:
                for layout in [9001,9002,9003,9004]:
                    seed_all(seed+layout);e=Env(water,layout);obs=e.observe();done=False;c=ServiceCommitment('commit')
                    count=0;probhover=0.;entropy=0.;topprob=0.;freehover=0;free_top_hover=0;forced=0;rewardtotal=0.;qgap=0.;gapcount=0
                    hist=[0]*9;free_hist=[0]*9;new_late=0;conflict_late=0
                    while not done:
                        actions,mask,logs,_,detail,nforced=decision(model,c,obs,e.available(),mode=='greedy')
                        with torch.no_grad():
                            o=model.tensor(obs)[None];mt=model.tensor(mask,torch.bool)[None]
                            if a=='mat':
                                logits,_=model.policy(o,model.tensor(actions,torch.long)[None]);ds=distribution(logits,mt);probs=ds.probs[0].numpy();ent=ds.entropy()[0].numpy()
                            else:
                                ds=model.actor_distributions(o,mt);probs=np.stack([d.probs[0].numpy() for d in ds]);ent=np.array([float(d.entropy()[0]) for d in ds])
                                q=model.critic(o,model.tensor(actions,torch.long)[None])[0].numpy()
                        free=mask.sum(1)>1
                        count+=int(free.sum());probhover+=float(probs[free,8].sum());entropy+=float(ent[free].sum());topprob+=float(probs[free].max(1).sum())
                        freehover+=int(((actions==8)&free).sum());free_top_hover+=int(((probs.argmax(1)==8)&free).sum());forced+=nforced
                        if a=='coma':qgap+=float((q[free,8]-(q[free]*probs[free]).sum(1)).sum());gapcount+=int(free.sum())
                        for act in actions:hist[int(act)]+=1
                        for act in actions[free]:free_hist[int(act)]+=1
                        obs,reward,done,info=e.step(actions);rewardtotal+=reward
                        if e.t>200:new_late+=info['new_cells'];conflict_late+=info['conflicts']
                    rows.append(dict(case=name,layout=layout,seed=seed,**e.metrics(),free_count=count,free_hover_probability_sum=probhover,
                        free_entropy_sum=entropy,free_top_probability_sum=topprob,free_executed_hover=freehover,free_argmax_hover=free_top_hover,
                        forced_count=forced,action_histogram=hist,free_action_histogram=free_hist,return_sum=rewardtotal,new_cells_last100=new_late,
                        conflicts_last100=conflict_late,hover_q_advantage_sum=qgap,hover_q_advantage_count=gapcount))
            n=sum(x['free_count'] for x in rows)
            summary=dict(metrics=aggregate(rows),free_decisions=n,
                mean_hover_probability=sum(x['free_hover_probability_sum'] for x in rows)/max(1,n),
                mean_entropy=sum(x['free_entropy_sum'] for x in rows)/max(1,n),
                mean_top_probability=sum(x['free_top_probability_sum'] for x in rows)/max(1,n),
                executed_hover_fraction=sum(x['free_executed_hover'] for x in rows)/max(1,n),
                argmax_hover_fraction=sum(x['free_argmax_hover'] for x in rows)/max(1,n),
                estimated_hover_q_advantage=sum(x['hover_q_advantage_sum'] for x in rows)/max(1,sum(x['hover_q_advantage_count'] for x in rows)),
                mean_return=float(np.mean([x['return_sum'] for x in rows])))
            if mode=='greedy':
                old=json.loads((new/a/f'validation-{total:04d}-service.json').read_text(encoding='utf-8'))['episodes']
                assert all(all(row[k]==prev[k] for k in ['coverage','rescued','conflicts','boundaries']) for row,prev in zip(rows,old))
            key=f'{total}-{mode}-{seed}';results[a][key]=summary;write(a+'-'+key+'.json',dict(rows=rows,summary=summary))
            print(a,key,json.dumps(summary),flush=True)
    train=[json.loads(x) for x in (new/a/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    blocks=[]
    for start in [65,97,129,161,193,225]:
        batch=[x for x in train if start<=x['total_episode']<start+32]
        blocks.append(dict(start=start,end=start+31,**{k:float(np.mean([x[k] for x in batch])) for k in ['coverage','RCR','return_sum','ACC','boundaries']}))
    write(a+'-training-blocks.json',blocks)
write('summary.json',results)
write('hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
