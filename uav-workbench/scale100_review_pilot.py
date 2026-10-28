"""Audit completed pilot artifacts, retaining every prespecified development point."""
import argparse,json,math,time
from pathlib import Path
from scale100_train import *
from scale100_run_audit import audit,require
import scale100_run_audit as audit_module

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
PILOT=ROOT/'generated/scale100-budget-pilot-20261026'

def review(folder=PILOT):
    torch.set_num_threads(1);plan=json.loads((folder/'plan.json').read_text())
    require(plan['dataset_sha256']==digest(DATA),'Pilot dataset changed');require(plan['source_sha256']==sources(),'Pilot training source changed')
    dataset=json.loads(DATA.read_text());entry=next(r for r in dataset['cases'] if r['split']=='development');case=FloodCase(entry['path']);require(case.digest==entry['sha256'],'Development case changed')
    rows=[];missing=[];reference=None
    for task in ('E1','E2'):
        for method in METHODS:
            job=f'{task}-{method}-s11';model=folder/job
            if not (model/'actor.json').exists() or any(not (model/f'development-ep{ep:04d}.json').exists() for ep in (64,128,256)):missing.append(job);continue
            meta=json.loads((model/'actor.json').read_text());history=json.loads((model/'history.json').read_text())
            require((meta['task'],meta['method'],meta['seed'],meta['episodes'],meta['phase'],meta['formal'])==(task,method,11,256,'development-diagnostic',False),'Wrong task/method/seed/phase attribution: '+job)
            require(meta['dataset_sha256']==digest(DATA) and meta['config']==asdict(config(task,method,11)),'Training data or environment contract mismatch: '+job)
            require(meta['sources']==sources() and meta['actor_sha256']==digest(model/'actor.pt'),'Model/source integrity failed: '+job)
            require(meta['trained_episodes']==256 and meta['updates']==253 and len(history)==256,'Incomplete training budget: '+job)
            require([r['episode'] for r in history]==list(range(1,257)),'Duplicated/missing episode: '+job)
            for r in history:
                require(math.isfinite(r['return_']),'Nonfinite return')
                if r['update']:
                    require(all(math.isfinite(float(v)) for v in r['update'].values()),'Nonfinite optimizer diagnostic')
                    if method not in ('D','P'):require(r['update']['data']==0 and r['update']['physics']==0,'Unexpected baseline auxiliary loss')
                    if method=='D':require(r['update']['physics']==0,'Data method received physics loss')
            initial=torch.load(model/'initial-actor.pt',weights_only=True)
            final=torch.load(model/'actor.pt',weights_only=True)
            for prefix in ('conv.','gru.'):
                require(any(not torch.equal(initial[k],final[k]) for k in initial if k.startswith(prefix)),'Shared encoder/GRU did not update: '+job)
            if reference is None:reference=initial
            else:require(all(torch.equal(reference[k],initial[k]) for k in reference),'Unmatched initial parameters for seed11: '+job)
            points=[]
            for episode in (64,128,256):
                ev=json.loads((model/f'development-ep{episode:04d}.json').read_text())
                require(ev['episode']==episode and ev['checkpoint_episode']==episode,'Wrong evaluated checkpoint')
                require(not ev['holdout_evaluated'] and len(ev['rows'])==3,'Changed development/holdout opportunities')
                require([r['seed'] for r in ev['rows']]==[11,29,47] and all(r['parent']==entry['parent_terrain'] for r in ev['rows']),'Mismatched development scenarios')
                points.append(dict(episode=episode,coverage=float(np.mean([r['metrics']['first_coverage'] for r in ev['rows']])),balanced_current_quality=float(np.mean([r['metrics']['current_correct_coverage_auc'] for r in ev['rows']])),rescued=float(np.mean([r['metrics']['rescued'] for r in ev['rows']])),collision_attempts=float(np.mean([r['metrics']['collision_events'] for r in ev['rows']]))))
            target=model/'independent-audit.json';existing=json.loads(target.read_text()) if target.exists() else None
            if existing and existing.get('audit_source_sha256')==digest(audit_module.__file__) and existing['actor_sha256']==meta['actor_sha256']:audited=existing['rows']
            else:
                actor=Agent();actor.load_state_dict(torch.load(model/'actor.pt',weights_only=True));actor.eval();audited=[]
                for seed in (11,29,47):
                    env=ScaleEnv(case,config(task,method,seed));total,data,_=rollout(actor,env,np.random.default_rng(seed),0.,True)
                    audited.append(dict(seed=seed,return_=total,**audit(env,data['images'].numpy())))
                write(target,dict(schema='hyra-scale100-independent-run-audit-v1',audit_source_sha256=digest(audit_module.__file__),actor_sha256=meta['actor_sha256'],source_sha256=sources(),task=task,method=method,training_episodes=256,parent=entry['parent_terrain'],split='development',holdout_evaluated=False,rows=audited))
            rows.append(dict(job=job,task=task,method=method,actor_sha256=meta['actor_sha256'],episodes=256,updates=253,all_development_points=points,full_trajectory_audits_passed=all(r['audit_passed'] for r in audited),audited_decisions=3*180,maximum_stall_s=max(r['maximum_stationary_streak_s'] for r in audited)))
    report=dict(schema='hyra-scale100-budget-pilot-review-v1',created_at=time.time(),review_source_sha256=digest(__file__),audit_source_sha256=digest(audit_module.__file__),training_sources=sources(),dataset_sha256=digest(DATA),completed=len(rows),expected=12,missing_jobs=missing,complete=len(rows)==12,training_seed=11,matched_initial_tensors=True if rows else None,initial_parameter_check_scope='completed models listed in rows only',independent_development_parent_count=1,development_scenario_seeds=[11,29,47],formal_budget_frozen=False,formal_training_started=False,holdout_policy_evaluated=False,rows=rows,limitations=['one training seed and one development terrain do not prove method ranking or convergence','initial, intermediate and final returns have different exploration epsilon; use greedy development checkpoints for progress','physical forecast accuracy and task benefit must be evaluated separately'])
    write(folder/'review.json',report);print(json.dumps(dict(completed=len(rows),expected=12,all_completed_audits_passed=all(r['full_trajectory_audits_passed'] for r in rows),formal_budget_frozen=False)),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=PILOT);a=p.parse_args();review(a.folder)
