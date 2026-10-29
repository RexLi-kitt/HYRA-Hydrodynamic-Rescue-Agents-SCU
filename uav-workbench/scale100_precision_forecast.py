"""Matched-input D/P reconstruction and internal-short-step diagnostic.

Both actors receive the SAME legal rule-generated observation/action stream.
The forecast horizon is the actual accepted solver microstep, not 5s or 60s.
All truth/source access below belongs to offline scoring, never actor input.
"""
import argparse,json,math
from pathlib import Path
from scale100_train import *
from scale100_run_audit import require
from scale100_rules import water_aware
import scale100_rules as rule_module

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'

def score(pred,before,after,bed,dt,source,known):
    # pred: [agents, current/microstep, h/hu/hv, z,x].
    n=pred.shape[0];scale=torch.tensor([.5,.25,.25],dtype=pred.dtype)[None,:,None,None]
    before=torch.as_tensor(before,dtype=pred.dtype)[None].expand(n,-1,-1,-1);after=torch.as_tensor(after,dtype=pred.dtype)[None].expand(n,-1,-1,-1)
    stepped=SWEOperator().step(pred[:,0],torch.as_tensor(bed,dtype=pred.dtype),float(dt),source=torch.as_tensor(source,dtype=pred.dtype))
    residual=pred[:,1]-stepped;difference=pred[:,1]-after;current=pred[:,0]-before
    volume=(pred[:,1,0]-pred[:,0,0]).sum(dim=(-2,-1))*25.-float(np.sum(source)*dt*25.)
    mask=torch.tensor(np.asarray(known).copy(),dtype=torch.bool)
    result=dict(current_h_mae_m=float(current[:,0].abs().mean()),microstep_h_mae_m=float(difference[:,0].abs().mean()),microstep_h_rmse_m=float(difference[:,0].square().mean().sqrt()),microstep_hu_mae_m2_s=float(difference[:,1].abs().mean()),microstep_hv_mae_m2_s=float(difference[:,2].abs().mean()),data_normalized_mse=float((current/scale).square().mean()/2+(difference/scale).square().mean()/2),local_physics_normalized_mse=float((residual/scale).square().mean()),volume_residual_abs_mean_m3=float(volume.abs().mean()),known_fraction=float(mask.float().mean()),known_h_mae_m=float(difference[:,0][:,mask].abs().mean()) if mask.any() else None,unknown_h_mae_m=float(difference[:,0][:,~mask].abs().mean()) if (~mask).any() else None)
    require(all(v is None or math.isfinite(v) for v in result.values()),'Nonfinite forecast diagnostic');return result

def load_pair(folder,task,episode=256):
    actors={};metas={}
    for method in ('D','P'):
        model=folder/f'{task}-{method}-s11';stem='actor' if episode==256 else f'actor-ep{episode:04d}';meta=json.loads((model/(stem+'.json')).read_text())
        require((meta['task'],meta['method'],meta['seed'],meta['trained_episodes'])==(task,method,11,episode),'Matched diagnostic needs the same task/seed/episode budget')
        if meta.get('model_family')=='same_actor_aux_precision_v3':
            from scale100_aux_precision import precision_sources
            expected_sources=precision_sources()
        elif meta.get('model_family')=='same_actor_aux_optimization_v2':
            from scale100_aux_tuning import tuned_sources
            expected_sources=tuned_sources()
        else:expected_sources=sources()
        require(meta['sources']==expected_sources and digest(model/(stem+'.pt'))==meta['actor_sha256'],'Model/source mismatch')
        actor=Agent();actor.load_state_dict(torch.load(model/(stem+'.pt'),weights_only=True));actor.eval();actors[method]=actor;metas[method]=meta
    require(metas['D']['config']==metas['P']['config'] and metas['D']['dataset_sha256']==metas['P']['dataset_sha256']==digest(DATA),'D/P environment or dataset differs')
    return actors,metas

def run(folder,task,output,episode=256):
    torch.set_num_threads(1);actors,metas=load_pair(folder,task,episode);manifest=json.loads(DATA.read_text());row=next(r for r in manifest['cases'] if r['split']=='development');case=FloodCase(row['path']);require(case.digest==row['sha256'],'Development source changed')
    fit_cases=[FloodCase(r['path']) for r in manifest['cases'] if r['split']=='fit'];prior=np.mean([c.probes['before'].mean(axis=(0,2,3)) for c in fit_cases],axis=0)
    points=[];sample_times=list(range(0,181,10))
    for seed in (11,29,47):
        env=ScaleEnv(case,config(task,'D',seed));view=env.reset();hidden={m:a.hidden(2) for m,a in actors.items()};previous=torch.zeros(2,8)
        for t in range(181):
            with torch.no_grad():
                for method,actor in actors.items():
                    _,hidden[method],_=actor(torch.from_numpy(view.images.copy()),previous,t,hidden[method],False)
                if t in sample_times:
                    before=case.probes['before'][t];after=case.probes['after'][t];dt=float(case.probes['dt'][t]);source=case.probes['source'][t]
                    for method,actor in actors.items():
                        bed=torch.as_tensor(case.bed,dtype=torch.float32)[None].expand(2,-1,-1);pred=actor.predict(hidden[method],bed)
                        points.append(dict(seed=seed,time_s=t*5.,method=method,actual_microstep_s=dt,**score(pred,before,after,case.bed,dt,source,view.explored)))
                    # Available-information persistence uses stale arrived memory and zero unknowns.
                    h,u,v=env.water;memory=np.stack((h,h*u,h*v));legal=torch.tensor(np.stack((memory,memory)),dtype=torch.float32)[None].expand(2,-1,-1,-1,-1)
                    points.append(dict(seed=seed,time_s=t*5.,method='legal_memory_persistence',actual_microstep_s=dt,**score(legal,before,after,case.bed,dt,source,view.explored)))
                    constant=np.broadcast_to(prior[:,None,None],(3,20,20)).copy();constant=torch.tensor(np.stack((constant,constant)),dtype=torch.float32)[None].expand(2,-1,-1,-1,-1)
                    points.append(dict(seed=seed,time_s=t*5.,method='fit_mean_constant',actual_microstep_s=dt,**score(constant,before,after,case.bed,dt,source,view.explored)))
                    # This oracle is an offline temporal yardstick, never an executable baseline.
                    oracle=torch.tensor(np.stack((before,before)),dtype=torch.float32)[None].expand(2,-1,-1,-1,-1)
                    points.append(dict(seed=seed,time_s=t*5.,method='truth_current_persistence_oracle',actual_microstep_s=dt,**score(oracle,before,after,case.bed,dt,source,view.explored)))
            if t<180:
                actions=water_aware(env);view,_,_=env.step(actions);previous=F.one_hot(torch.tensor(actions),8).float()
        require(all(p.grad is None for actor in actors.values() for p in actor.parameters()),'Diagnostic created training gradients')
    keys=[k for k in points[0] if k not in ('seed','time_s','method','actual_microstep_s')];summary={}
    for method in ('D','P','legal_memory_persistence','fit_mean_constant','truth_current_persistence_oracle'):
        rows=[r for r in points if r['method']==method];summary[method]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
    report=dict(schema='hyra-scale100-matched-forecast-diagnostic-v1',task=task,training_seed=11,trained_episodes=256,actor_sha256={m:meta['actor_sha256'] for m,meta in metas.items()},diagnostic_source_sha256=digest(__file__),rule_source_sha256=digest(rule_module.__file__),dataset_sha256=digest(DATA),parent=row['parent_terrain'],independent_parent_count=1,split='development',same_legal_inputs_and_previous_actions=True,model_policy_controls_reference_trajectory=False,evaluated_decision_indices=sample_times,scenario_seeds=[11,29,47],microstep_s_min=min(r['actual_microstep_s'] for r in points),microstep_s_max=max(r['actual_microstep_s'] for r in points),post_terminal_label_scope='terminal900s reconstruction is also scored against its actual900s+microstep label; this label never enters any actor',holdout_policy_evaluated=False,summary=summary,rows=points,limitations=['microstep reconstruction accuracy is not a long-horizon flood forecast','same-parent points, agents and seeds are correlated; they are not independent flood events','lower physics residual can coexist with wrong fields; report data error and volume/local residual together','these are auxiliary heads on a common legal observation stream, not model-based planning','truth-current persistence uses privileged current truth and is only an offline temporal yardstick'])
    report['trained_episodes']=episode;report['fit_mean_constant_U']=prior.tolist();report['model_family']={m:meta.get('model_family','absolute_head_v1') for m,meta in metas.items()};report['model_sources']={m:meta['sources'] for m,meta in metas.items()}
    report['memory_baseline_scope']='Raw arrived measurement ledger before lossy local/global pooling; retains more direct pixel detail than actor tensors. It is a legal sensing comparator, not a policy-performance claim.'
    write(output,report);print(json.dumps(dict(task=task,episodes=episode,horizon_s=[report['microstep_s_min'],report['microstep_s_max']],summary=summary)),flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=ROOT/'generated/scale100-budget-pilot-20261026');p.add_argument('--task',choices=['E1','E2'],required=True);p.add_argument('--episode',type=int,default=256);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.folder,a.task,a.output,a.episode)

