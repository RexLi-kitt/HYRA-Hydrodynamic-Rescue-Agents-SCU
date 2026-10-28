"""Review every auxiliary-tuning task/checkpoint, including negative outcomes."""
import json,math,argparse
from pathlib import Path
from scale100_aux_tuning import *
from scale100_run_audit import audit,require
import scale100_run_audit as audit_module

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'
OUT=ROOT/'generated/scale100-aux-tuning-20261026'

def review(folder=OUT):
    torch.set_num_threads(1);manifest=json.loads(DATA.read_text());entry=next(r for r in manifest['cases'] if r['split']=='development');case=FloodCase(entry['path']);rows=[];missing=[];reference=None
    for task in ('E1','E2'):
        for method in ('D','P'):
            job=f'{task}-{method}-s11';model=folder/job
            if not (model/'actor.json').exists() or any(not (model/f'actor-ep{ep:04d}.json').exists() for ep in (64,128,256)):missing.append(job);continue
            meta=json.loads((model/'actor.json').read_text());history=json.loads((model/'history.json').read_text())
            require(meta['sources']==tuned_sources() and meta['actor_sha256']==digest(model/'actor.pt') and meta['dataset_sha256']==digest(DATA),'Tuned artifact integrity failed')
            require((meta['task'],meta['method'],meta['seed'],meta['trained_episodes'],meta['updates'],meta['formal'])==(task,method,11,256,253,False),'Tuned task/budget attribution mismatch')
            require(meta['config']==asdict(config(task,method,11)) and meta['data_weight']==meta['physics_weight']==1. and meta['auxiliary_head_lr']==1e-3,'Tuned control setting mismatch')
            require(len(history)==256 and [r['episode'] for r in history]==list(range(1,257)),'Tuned history missing episodes')
            gradients=[]
            for row in history:
                require(math.isfinite(row['return_']),'Nonfinite tuned return')
                if row['update']:
                    update=row['update'];require(all(math.isfinite(float(v)) for k,v in update.items() if k!='shared_gradient_audit'),'Nonfinite tuned loss/gradient')
                    if method=='D':require(update['physics']==0,'Data received a physics loss')
                    if update['shared_gradient_audit']:gradients.append(dict(update=update['updates'],**update['shared_gradient_audit']))
            require(len(gradients)==4 and any(g['data_shared_norm']>0 for g in gradients),'Missing shared-encoder auxiliary gradients')
            if method=='P':require(any(g['physics_shared_norm']>0 for g in gradients),'Physics gradients never reached the representation')
            initial=torch.load(model/'initial-actor.pt',weights_only=True)
            if reference is None:reference=initial
            else:require(all(torch.equal(initial[k],reference[k]) for k in reference),'Tuned initial parameters differ across task/method')
            points=[]
            for episode in (64,128,256):
                actor_file=model/f'actor-ep{episode:04d}.pt';ev_file=model/f'development-ep{episode:04d}.json';expected=json.loads(ev_file.read_text());header=json.loads((model/f'actor-ep{episode:04d}.json').read_text());require(header['actor_sha256']==digest(actor_file) and header['trained_episodes']==episode,'Tuned checkpoint checksum/episode mismatch')
                require([r['seed'] for r in expected['rows']]==[11,29,47],'Tuned development seeds differ')
                target=model/f'primary-ep{episode:04d}.json'
                if target.exists():
                    record=json.loads(target.read_text());require(record['actor_sha256']==header['actor_sha256'] and record['audit_source_sha256']==digest(audit_module.__file__) and record['original_development_sha256']==digest(ev_file),'Cached tuned audit changed')
                else:
                    actor=Agent();actor.load_state_dict(torch.load(actor_file,weights_only=True));actor.eval();scenarios=[]
                    for original in expected['rows']:
                        seed=original['seed'];env=ScaleEnv(case,config(task,method,seed));total,data,metrics=rollout(actor,env,np.random.default_rng(seed),0.,True)
                        require(abs(total-original['return_'])<1e-9,'Tuned actor no longer reproduces original development return')
                        for key,value in original['metrics'].items():require(metrics[key] is None if value is None else metrics[key] is not None and abs(float(metrics[key])-float(value))<1e-9,'Tuned development metric mismatch')
                        scenarios.append(dict(seed=seed,**audit(env,data['images'].numpy())))
                    record=dict(episode=episode,actor_sha256=header['actor_sha256'],audit_source_sha256=digest(audit_module.__file__),original_development_sha256=digest(ev_file),split='development',parent=entry['parent_terrain'],rows=scenarios);write(target,record)
                points.append(dict(episode=episode,first_coverage_auc=float(np.mean([r['first_coverage_auc'] for r in record['rows']])),balanced_current_quality=float(np.mean([r['balanced_current_correct_coverage_auc'] for r in record['rows']])),rescue_fraction=float(np.mean([r['rescue_fraction'] for r in record['rows']])) if task=='E2' else None,final_first_coverage=float(np.mean([r['final_first_coverage'] for r in record['rows']]))))
            last=history[-1]['update'];last_gradient=last['shared_gradient_audit'];live=bool(last_gradient and (last['data']<1e-12 or last_gradient['data_shared_norm']>0) and (method!='P' or last['physics']<1e-12 or last_gradient['physics_shared_norm']>0))
            rows.append(dict(job=job,task=task,method=method,episodes=256,updates=253,actor_sha256=meta['actor_sha256'],all_development_points=points,shared_gradient_audits=gradients,shared_auxiliary_mechanism_live_at_end=live,full_trajectory_audits_passed=True,fit_only_prior_U=meta['fit_only_prior_U']))
    forecasts={task:str(folder/f'{task}-matched-forecast.json') for task in ('E1','E2') if (folder/f'{task}-matched-forecast.json').exists()}
    report=dict(schema='hyra-scale100-aux-tuning-review-v2',review_source_sha256=digest(__file__),training_sources=tuned_sources(),dataset_sha256=digest(DATA),completed=len(rows),expected=4,missing_jobs=missing,complete=len(rows)==4,matched_initial_tensors=True if rows else None,independent_development_parents=1,holdout_policy_evaluated=False,rows=rows,matched_forecast_paths=forecasts,formal_backend_ready=False,scope='engineering/development evidence; no positive outcome is required or asserted')
    write(folder/'review.json',report);print(json.dumps(dict(completed=len(rows),expected=4,matched_forecast_tasks=list(forecasts),formal_backend_ready=False)),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=OUT);a=p.parse_args();review(a.folder)
