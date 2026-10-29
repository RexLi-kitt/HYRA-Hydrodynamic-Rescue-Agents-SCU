"""Development-only policy outcomes. Never opens holdout hydraulic arrays."""
from pathlib import Path
import argparse,json
from scale100_train import *
from scale100_development import DATA,OUT

def evaluate(folder):
    torch.set_num_threads(1);manifest=json.loads(DATA.read_text());row=next(r for r in manifest['cases'] if r['split']=='development');case=FloodCase(row['path']);results=[]
    for task in ('E1','E2'):
        for method in METHODS:
            path=folder/f'{task}-{method}-s11'
            if not (path/'actor.json').exists():continue
            meta=json.loads((path/'actor.json').read_text())
            if meta['sources']!=sources() or digest(path/'actor.pt')!=meta['actor_sha256']:raise ValueError('Model/source mismatch')
            actor=Agent();actor.load_state_dict(torch.load(path/'actor.pt',weights_only=True));actor.eval()
            for seed in (11,29,47):
                env=ScaleEnv(case,config(task,method,seed));total,_,metrics=rollout(actor,env,np.random.default_rng(seed),0.,False)
                results.append(dict(task=task,method=method,scenario_seed=seed,training_seed=11,training_episodes=meta['trained_episodes'],return_=total,metrics=metrics,actor_sha256=meta['actor_sha256'],split='development',parent=row['parent_terrain']))
    report=dict(schema='hyra-scale100-development-evaluation-v1',rows=results,independent_parent_count=1,selection='final checkpoint, no best-seed selection',holdout_policy_evaluation=False,scope='engineering/learning diagnostic only; not formal confirmation')
    write(folder/'development-evaluation.json',report)
    for task in ('E1','E2'):
        for method in METHODS:
            rows=[r for r in results if r['task']==task and r['method']==method]
            if rows:print(json.dumps(dict(task=task,method=method,coverage=float(np.mean([r['metrics']['first_coverage'] for r in rows])),rescued=float(np.mean([r['metrics']['rescued'] for r in rows])))),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=OUT);a=p.parse_args();evaluate(a.folder)
