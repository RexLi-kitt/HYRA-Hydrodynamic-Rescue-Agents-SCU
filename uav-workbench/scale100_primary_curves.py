"""Recover the prespecified primary AUC curves from all saved pilot checkpoints.

Re-executes each saved actor and checks the original development outcomes before
adding independent AUC metrics; never chooses a checkpoint using holdout data.
"""
import json,argparse
from pathlib import Path
from scale100_train import *
from scale100_run_audit import audit,require
import scale100_run_audit as audit_module

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
PILOT=ROOT/'generated/scale100-budget-pilot-20261026'

def run(folder=PILOT):
    torch.set_num_threads(1);dataset=json.loads(DATA.read_text());entry=next(r for r in dataset['cases'] if r['split']=='development');case=FloodCase(entry['path']);require(case.digest==entry['sha256'],'Development case changed')
    rows=[];missing=[]
    for task in ('E1','E2'):
        for method in METHODS:
            model=folder/f'{task}-{method}-s11'
            if not (model/'actor.json').exists() or any(not (model/f'development-ep{ep:04d}.json').exists() for ep in (64,128,256)):missing.append(model.name);continue
            meta=json.loads((model/'actor.json').read_text());require(meta['sources']==sources() and meta['dataset_sha256']==digest(DATA),'Training source/data mismatch')
            for episode in (64,128,256):
                actor_path=model/f'actor-ep{episode:04d}.pt';original_path=model/f'development-ep{episode:04d}.json';original=json.loads(original_path.read_text());require([r['seed'] for r in original['rows']]==[11,29,47],'Changed development scenarios')
                target=model/f'primary-ep{episode:04d}.json';actor_hash=digest(actor_path);audit_hash=digest(audit_module.__file__)
                if target.exists():
                    record=json.loads(target.read_text());require(record['actor_sha256']==actor_hash and record['audit_source_sha256']==audit_hash and record['original_development_sha256']==digest(original_path),'Primary-curve provenance changed')
                else:
                    actor=Agent();actor.load_state_dict(torch.load(actor_path,weights_only=True));actor.eval();scenarios=[]
                    for expected in original['rows']:
                        seed=expected['seed'];env=ScaleEnv(case,config(task,method,seed));total,data,metrics=rollout(actor,env,np.random.default_rng(seed),0.,True)
                        require(abs(total-expected['return_'])<1e-9,'Saved actor does not reproduce its original return')
                        for key,value in expected['metrics'].items():
                            current=metrics[key]
                            require(current is None if value is None else current is not None and abs(float(current)-float(value))<1e-9,'Saved actor does not reproduce original metric '+key)
                        scenarios.append(dict(seed=seed,original_results_reproduced=True,**audit(env,data['images'].numpy())))
                    record=dict(task=task,method=method,training_seed=11,episode=episode,actor_sha256=actor_hash,audit_source_sha256=audit_hash,curve_source_sha256=digest(__file__),original_development_sha256=digest(original_path),parent=entry['parent_terrain'],split='development',rows=scenarios,means=dict(first_coverage_auc=float(np.mean([r['first_coverage_auc'] for r in scenarios])),balanced_current_correct_coverage_auc=float(np.mean([r['balanced_current_correct_coverage_auc'] for r in scenarios])),final_first_coverage=float(np.mean([r['final_first_coverage'] for r in scenarios])),rescue_fraction=float(np.mean([r['rescue_fraction'] for r in scenarios])) if task=='E2' else None))
                    write(target,record)
                rows.append({k:record[k] for k in ('task','method','training_seed','episode','actor_sha256','means')})
    report=dict(schema='hyra-scale100-primary-learning-curves-v1',curve_source_sha256=digest(__file__),training_sources=sources(),dataset_sha256=digest(DATA),completed_models=12-len(missing),missing_models=missing,all_predeclared_points_preserved=True,independent_development_parents=1,training_seeds=[11],holdout_policy_evaluated=False,rows=rows,scope='reproduced original greedy checkpoint outcomes before measuring primary AUC; no convergence or ranking claim')
    write(folder/'primary-curves.json',report);print(json.dumps(dict(models=report['completed_models'],points=len(rows),holdout_evaluated=False)),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=PILOT);a=p.parse_args();run(a.folder)
