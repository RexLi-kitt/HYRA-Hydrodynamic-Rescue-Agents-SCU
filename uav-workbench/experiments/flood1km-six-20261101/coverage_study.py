"""Fixed coverage planning ablation and COMA 64/256 extra-episode learning check."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from flood1km.env import Env
from flood1km.learners import Learner,ALGORITHMS
from flood1km.train import aggregate,seed_all,rng_state,restore_rng,sources
from flood1km.data import sha
from service_commitment import ServiceCommitment
from coverage_controller import CoverageWatchdog,service_mask

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);seed_all(1001)
    manifest=args.baseline/'data/dataset.json';items=json.loads(manifest.read_text(encoding='utf-8'))['cases'];cases={}
    for x in items:
        p=Path(x['path']);assert sha(p)==x['sha256'];cases.setdefault(x['split'],[]).append((p.name,np.load(p,mmap_mode='r')))
    identity=json.loads((args.baseline/'coma/identity.json').read_text(encoding='utf-8'))
    for k,v in identity['source_hashes'].items():assert sources()[k]==v,k
    assert sha(manifest)==identity['data_hash']
    prereg=dict(version='coverage-watchdog-and-coma-budget-v1',
        diagnostic='Original commit late steps 201..300: HAPPO/HASAC/MAAC zero new coverage in all 800 steps, MAT 794/800, HATRPO770/800, COMA706/800.',
        frozen='Original six 64-episode checkpoints, service_commitment.py unchanged, same physics/rewards/data/validation',
        planner='After 5 consecutive steps with no newly inferred observed cells, sticky takeover of free agents; quadrants, novel sector gain, nearest unseen goal, two-step hole approach, predicted collision avoidance. Service retains priority.',
        planner_information='Own reconstructed conservative sensed history, local observed mask, fully observed pooled blocks, public positions/headings only; no hidden people, no water truth.',
        planner_acceptance='For each method coverage > original commit, RCR >= original commit; conflicts and all-task completion still reported. Never erase failed variants.',
        learning='COMA chosen because prior commit had highest rescue, with its conflict increase retained. Continue original parameters/optimizers; same reward and network; service forced actions represented by singleton legal masks BEFORE sampling.',
        training_extra_episodes=256,checkpoints_evaluated=[64,256],training_seed=1001,training_layout_start=910000,
        matched_budget_note='Neural+service and neural+service+watchdog use identical weights at a given budget. 0/64/256 are deliberately different learning budgets; not an equal-compute six-algorithm comparison.',
        learning_record='Actual masked-sampled action and log probability; next actual mask/action recorded causally; no off-policy overridden action mislabeled as policy sample.',
        validation_layouts=[9001,9002,9003,9004],test50_used=False,maximum_seconds=1800,parallel_jobs=1,
        files={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('coverage_controller.py'),Path(__file__).with_name('service_commitment.py')]},
        sources=sources(),data_hash=sha(manifest),checkpoints={a:sha(args.baseline/a/'latest.pt') for a in ALGORITHMS})
    write(out/'protocol.json',prereg)
    def stage(name,**kw):
        elapsed=time.perf_counter()-start;write(out/'progress.json',dict(stage=name,elapsed_s=elapsed,**kw))
        if elapsed>1800:raise TimeoutError('Fixed 1800s budget exceeded')
    def evaluate(model,label,planner):
        state=rng_state();rows=[];events=[]
        try:
            for name,water in cases['validation']:
                for layout in [9001,9002,9003,9004]:
                    stage('validation',label=label,layout=layout)
                    e=Env(water,layout);obs=e.observe();done=False;c=ServiceCommitment('commit');p=CoverageWatchdog();zero=0;late_new=0
                    with (out/(label+'-trace.jsonl')).open('a',encoding='utf-8') as log:
                        while not done:
                            proposed=model.act(obs,e.available(),True)[0]
                            actions,decision=c.select(obs,e.available(),proposed)
                            planning={}
                            if planner:
                                actions,planning=p.select(obs,e.available(),actions,decision['phases'])
                                assert not (p.known & ~e.known).any(),'Coverage memory inferred unseen cells'
                            obs,_,done,info=e.step(actions);zero+=int(info['new_cells']==0)
                            if e.t>200:late_new+=info['new_cells']
                            info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()}
                            log.write(json.dumps(dict(case=name,layout=layout,step=e.t,pos=e.pos.tolist(),proposed=proposed.tolist(),executed=actions.tolist(),service=decision,planning=planning,info=info))+'\n')
                    rows.append(dict(case=name,layout=layout,zero_gain_steps=zero,late_new_cells=late_new,planner_agent_steps=p.steps,planner_activation=p.activation_step,**e.metrics()))
                    events.extend(dict(case=name,layout=layout,**x) for x in e.events)
            result=dict(episodes=rows,events=events,summary=aggregate(rows),planner=planner)
            write(out/(label+'.json'),result);print(label,json.dumps(result['summary']),flush=True);return result
        finally:restore_rng(state)
    results={'frozen_planner':{},'learning':{}}
    for algo in ALGORITHMS:
        ck=torch.load(args.baseline/algo/'latest.pt',map_location='cpu',weights_only=False)['learner']
        m=Learner(algo,ck['dim']);m.restore(ck)
        results['frozen_planner'][algo]=evaluate(m,algo+'-frozen-watchdog',True)['summary']
        write(out/'partial-results.json',results)
    # No planner during learning: isolate extra training with the service controller.
    seed_all(1001);ck=torch.load(args.baseline/'coma/latest.pt',map_location='cpu',weights_only=False)['learner']
    m=Learner('coma',ck['dim']);m.restore(ck);pending=[];steps=0;finite_forced_logps=True
    for ep in range(256):
        stage('training',extra_episode=ep+1)
        name,water=cases['train'][ep%len(cases['train'])];e=Env(water,910000+ep);obs=e.observe();done=False;c=ServiceCommitment('commit');trajectory=[];ret=0.;forced_count=0
        while not done:
            boundary=e.available();forced,decision=c.select(obs,boundary,np.full(4,8,dtype=np.int64))
            mask=service_mask(boundary,forced,decision['phases']);actions,logs,value=m.act(obs,mask)
            controlled=np.array([x!='learned' for x in decision['phases']])
            assert np.array_equal(actions[controlled],forced[controlled]);assert np.all(np.abs(logs[controlled])<1e-6)
            if trajectory:trajectory[-1]['next_mask']=mask.copy();trajectory[-1]['next_action']=actions.copy()
            nxt,reward,done,_=e.step(actions);steps+=1;ret+=reward;forced_count+=int(controlled.sum())
            trajectory.append(dict(obs=obs.copy(),mask=mask.copy(),action=actions.copy(),reward=reward,done=float(done),next_obs=nxt.copy(),
                next_mask=e.available().copy(),next_action=np.full(4,8,dtype=np.int64),logp=logs.copy(),value=value))
            obs=nxt
        pending.extend(trajectory);loss={}
        if (ep+1)%2==0:loss=m.update_on_policy(pending);pending=[]
        with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(extra_episode=ep+1,case=name,layout=910000+ep,steps=steps,return_sum=ret,forced_agent_steps=forced_count,loss=loss,**e.metrics()))+'\n')
        if (ep+1)%32==0:
            torch.save(dict(learner=m.checkpoint(),extra_episode=ep+1,rng=rng_state(),protocol=prereg),out/f'coma-extra-{ep+1:03d}.pt')
            print('training',ep+1,e.metrics()['coverage'],flush=True)
        if ep+1 in [64,256]:
            result={}
            for planner in [False,True]:
                label=f'coma-extra-{ep+1:03d}-'+('watchdog' if planner else 'service')
                result['watchdog' if planner else 'service']=evaluate(m,label,planner)['summary']
            results['learning'][str(ep+1)]=result;write(out/'partial-results.json',results)
    for a,h in prereg['checkpoints'].items():assert sha(args.baseline/a/'latest.pt')==h
    write(out/'results.json',results);stage('completed',training_steps=steps,forced_action_consistency_passed=True)
    write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='hashes.json'})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
