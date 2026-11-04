"""Frozen-weight, equal-rule evaluation of two explicit hybrid controllers."""
import argparse,json,time,traceback
from pathlib import Path
import numpy as np
import torch
from flood1km.env import Env
from flood1km.learners import Learner,ALGORITHMS
from flood1km.train import aggregate,sources
from flood1km.data import sha
from service_commitment import ServiceCommitment

def write(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def run(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    torch.set_num_threads(2);torch.manual_seed(901);torch.use_deterministic_algorithms(True)
    original=json.loads((args.baseline/'maac/identity.json').read_text(encoding='utf-8'))
    current=sources()
    for k,v in original['source_hashes'].items():assert current[k]==v,k
    manifest=args.baseline/'data/dataset.json';assert sha(manifest)==original['data_hash']
    paths=[Path(x['path']) for x in json.loads(manifest.read_text(encoding='utf-8'))['cases'] if x['split']=='validation']
    checkpoints={a:sha(args.baseline/a/'latest.pt') for a in ALGORITHMS}
    write(out/'protocol.json',dict(version='service-commitment-v1',modes=['hold','commit'],algorithms=list(ALGORITHMS),
        fixed_validation_layouts=[9001,9002,9003,9004],parents=[p.name for p in paths],
        aim='Increase completed service, preserving coverage/conflict/timeout reporting; no success assumption.',
        baseline='Original 64-episode checkpoints. Failed BC candidate not substituted.',
        unchanged='Water, initial states, personnel, service radius/time/water checks, collision resolution, 300-step limit, reward, weights, metrics.',
        hold='Force hover only when already inside a discovered unserved target service radius.',
        commit='Maintain unique assignment to discovered target; approach and hover, otherwise proposed neural action.',
        commitment='Existing ownership first, keep assignment, then shortest agent-target distance; deterministic ties.',
        release='Completed/no longer observed target, last observed local water unsafe, 40-step assignment timeout or 3 consecutive stalls; agent-target cooldown 10 steps.',
        unknown_water='Not certified safe; approach/hover may be attempted but only unchanged Env determines actual service completion over full interval.',
        info='Controller receives exactly actor observation and action mask, no Env/truth/future access.',
        attribution='Learning policy plus explicit controller. Not neural learning gain or original-algorithm ranking. No retraining.',
        budgets='CPU2, serial, 6 algorithms x 2 wrappers x 8 validations; final 50-scene test unused.',
        controller_hash=sha(Path(__file__).with_name('service_commitment.py')),script_hash=sha(Path(__file__)),
        source_hashes=current,data_hash=sha(manifest),checkpoint_hashes=checkpoints))
    cases=[(p.name,np.load(p,mmap_mode='r')) for p in paths]
    summary={}
    for algo in ALGORITHMS:
        ck=torch.load(args.baseline/algo/'latest.pt',map_location='cpu',weights_only=False)['learner']
        model=Learner(algo,ck['dim']);model.restore(ck)
        baseline=json.loads((args.baseline/algo/'final-validation.json').read_text(encoding='utf-8'))
        summary[algo]={'baseline':baseline['summary']};write(out/(algo+'-baseline.json'),baseline)
        for mode in ['hold','commit']:
            rows=[];events=[]
            for name,water in cases:
                for layout in [9001,9002,9003,9004]:
                    write(out/'progress.json',dict(algorithm=algo,mode=mode,case=name,layout=layout,elapsed_s=time.perf_counter()-start))
                    env=Env(water,layout);obs=env.observe();done=False;c=ServiceCommitment(mode);controlled=0;prior_events=0
                    with (out/(algo+'-'+mode+'-trace.jsonl')).open('a',encoding='utf-8') as log:
                        while not done:
                            proposed=model.act(obs,env.available(),True)[0];action,decision=c.select(obs,env.available(),proposed)
                            controlled+=sum(x!='learned' for x in decision['phases'])
                            obs,_,done,info=env.step(action)
                            info={k:v.item() if isinstance(v,np.generic) else v for k,v in info.items()}
                            log.write(json.dumps(dict(case=name,layout=layout,step=env.t,proposed=proposed.tolist(),executed=action.tolist(),decision=decision,info=info),ensure_ascii=False)+'\n')
                            for event in env.events[prior_events:]:events.append(dict(case=name,layout=layout,**event,phase=decision['phases'][event['agent']]))
                            prior_events=len(env.events)
                    rows.append(dict(case=name,layout=layout,changed_actions=c.overrides,controller_agent_steps=controlled,
                        assignments=c.assignments,releases=c.releases,**env.metrics()))
            result=dict(episodes=rows,events=events,summary=aggregate(rows),
                changed_action_fraction=sum(r['changed_actions'] for r in rows)/sum(r['ST']*4 for r in rows),
                controlled_agent_step_fraction=sum(r['controller_agent_steps'] for r in rows)/sum(r['ST']*4 for r in rows))
            write(out/(algo+'-'+mode+'.json'),result);summary[algo][mode]=result['summary']
            print(algo,mode,json.dumps(result['summary']),flush=True)
        assert sha(args.baseline/algo/'latest.pt')==checkpoints[algo]
    write(out/'summary.json',summary)
    write(out/'progress.json',dict(status='completed',elapsed_s=time.perf_counter()-start,training_executed=False,final_test_executed=False))
    write(out/'hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='hashes.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:run(args)
    except Exception:
        if args.output.exists():write(args.output/'failure.json',dict(traceback=traceback.format_exc()))
        raise
