"""Fit on training records only; development diagnostics never open holdouts."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scale100_env import ScaleEnv, config
from sur_rescue.environment import rule_actions
from .data import cases
from .belief import SignalBeliefEstimator, Parameters
from .common import read, write, digest
from . import METHODS


def collect(dataset, split, output):
    if split not in ('fit','development'): raise ValueError('Offline fitting cannot read confirmation events')
    records=[]
    for row,case in cases(dataset,split):
        for seed in (11,29,47):
            env=ScaleEnv(case,config('E2','B3',seed)); env.reset()
            for step in range(25):
                if step in (0,4,12,24):
                    records.append(dict(signal=env.signal.copy(),water=env.water.copy(),valid=env.explored.copy(),
                        stamps=env.observed_s.copy(),now=env.time_s,obstacles=env.obstacles.copy(),
                        truth=env.fields._physical_signal.copy(),water_truth=case.values(env.time_s)[:3].copy(),
                        bed=case.bed.copy(),dx_m=case.dx,initial_depth=case.values(0)[0].copy(),
                        parent=row['parent_terrain'],mechanism=row['mechanism'],seed=seed,step=step))
                if step<24: env.step(rule_actions(env.view()))
    # All truth is explicitly offline labels. No estimator receives this record object.
    np.savez_compressed(output,records=np.array(records,dtype=object))
    return records


def load_records(path):
    # Written by this study on this machine; object pickle is never an external input.
    return np.load(path,allow_pickle=True)['records'].tolist()


def estimator(method, parameters, record, direction='forward'):
    p=Parameters(**{**parameters,'beta':tuple(parameters.get('beta',(0,0,0,0)))})
    return SignalBeliefEstimator(method,p,direction).reset(record['valid'].shape,
        dict(bed=record['bed'],dx_m=record['dx_m'])).update(
        record['signal'],record['water'],record['valid'],record['stamps'],record['now'],record['obstacles'])


def decode(method, vector, base):
    p=dict(base)
    if method in ('R1','C1'): p.update(alpha=float(vector[0]),kappa=float(vector[1]))
    elif method in ('R2','C2'): p['beta']=list(map(float,vector))
    else: p.update(strength=float(vector[0]),cross=float(vector[1]),kappa=float(vector[2]))
    return p


def synthetic_record(record, condition):
    r=dict(record)
    if condition=='original': return r
    rng=np.random.default_rng(20261027+record['seed']+int(record['now']))
    water=record['water_truth'].reshape(3,-1).copy()
    water=water[:,rng.permutation(water.shape[1])].reshape(record['water'].shape)
    # Joint spatial scrambling separates local water features from source location.
    truth=record['truth'].copy()
    if condition in ('attenuation','directional'):
        truth*=np.exp(-np.clip(water[0]/.08,0,5))
    if condition=='directional':
        probe=SignalBeliefEstimator('R1',Parameters(kappa=2)).reset(truth.shape)
        probe.update(truth,water,np.ones(truth.shape,bool),np.zeros(truth.shape),0.)
        truth=(.8*truth.ravel()+.2*(probe.weights()@truth.ravel())).reshape(truth.shape)
    if condition not in ('null','attenuation','directional'): raise ValueError('Unknown synthetic diagnostic')
    r.update(water=water,truth=truth,signal=np.where(record['valid'],truth,0.))
    return r


def fit(records_path, output, condition='original'):
    records=load_records(records_path)
    records=[synthetic_record(r,condition) for r in records]
    truth=np.concatenate([r['truth'].ravel() for r in records])
    base=asdict(Parameters(mean=float(truth.mean()),scale=max(float(truth.std()),1e-3)))
    all_features=[]; all_availability=[]
    for record in records:
        features,availability=estimator('Bpost',base,record).raw_features()
        all_features.append(features); all_availability.append(availability)
    all_features=np.concatenate(all_features); all_availability=np.concatenate(all_availability)
    means=[]; scales=[]
    for column in range(4):
        available=all_features[all_availability[:,column],column]
        means.append(float(available.mean()) if len(available) else 0.)
        scales.append(max(float(available.std()),1e-6) if len(available) else 1.)
    base.update(feature_mean=means,feature_scale=scales)
    params={'Braw':base,'Bpost':base}; audits=[]
    # Deterministic parent-balanced subset, not a selection based on model scores.
    subset=[r for r in records if r['seed']==11 and r['step']==12]
    for method in METHODS[2:]:
        if method in ('R1','C1'): x0=[.25,1.]; bounds=[(0,1),(0,4)]
        elif method in ('R2','C2'): x0=[0.]*4; bounds=[(-2,2)]*4
        else: x0=[.1,.25,1.]; bounds=[(0,10),(-2,2),(0,4)]
        def objective(vector):
            p=decode(method,vector,base); scores=[]
            for record in subset:
                scores.append(estimator(method,p,record).score(record['truth'])['nll'])
            return float(np.mean(scores)+.001*np.dot(vector,vector))
        result=minimize(objective,x0,bounds=bounds,method='L-BFGS-B',options=dict(maxiter=30,ftol=1e-7))
        # Zero effect is an eligible training-only candidate, not a concealed fallback.
        null=np.zeros(len(x0)); candidates=[(objective(null),null),(objective(result.x),result.x)]
        value,chosen=min(candidates,key=lambda item:item[0])
        params[method]=decode(method,chosen,base)
        audits.append(dict(method=method,objective=float(value),iterations=int(result.nit),
                           optimizer_success=bool(result.success),message=str(result.message),
                           null_selected=bool(np.array_equal(chosen,null))))
        print(f'fitted {method}: {value:.6f}',flush=True)
    write(output,dict(schema='hyra-signal-parameters-v1',parameters=params,fit_records_sha256=digest(records_path),
                      fit_only=True,condition=condition,objective='conditional marginal signal NLL incl logdet, plus .001 L2',
                      optimization=audits))
    return params


def diagnostics(records_path, params_path, output):
    records=load_records(records_path); params=read(params_path)['parameters']; rows=[]
    synthetic_params={name:read(Path(params_path).parent/f'parameters-{name}.json')['parameters']
                      for name in ('null','attenuation','directional')}
    # All dev parents, both forcing types and three layouts; one fixed acquisition stage.
    for record in records:
        if record['step']!=12: continue
        for mechanism in ('original','null','attenuation','directional'):
            r=synthetic_record(record,mechanism); truth=r['truth']
            condition_params=params if mechanism=='original' else synthetic_params[mechanism]
            for method in METHODS:
                if method=='Braw':
                    missing=~r['valid']; score=dict(rmse=float(np.sqrt(np.mean(truth[missing]**2))),nll=None,coverage95=None,n=int(missing.sum()))
                    residual=0.
                else:
                    model=estimator(method,condition_params[method],r); score=model.score(truth)
                    residual=model.posterior()['diagnostics']['residual_inf']
                rows.append(dict(method=method,condition=mechanism,parent=record['parent'],
                                 forcing=record['mechanism'],seed=record['seed'],**score,residual=residual))
                # Stratification separates the known initial-depth target-placement association.
                if mechanism=='original' and method!='Braw':
                    pred=model.posterior()['mean']; missing=~record['valid']
                    for name,mask in [('initial_h_below_0.2',record['initial_depth']<.2),('initial_h_at_least_0.2',record['initial_depth']>=.2)]:
                        use=missing&mask
                        if use.any(): rows.append(dict(method=method,condition=name,parent=record['parent'],
                            forcing=record['mechanism'],seed=record['seed'],rmse=float(np.sqrt(np.mean((truth[use]-pred[use])**2))),nll=None,coverage95=None,n=int(use.sum()),residual=residual))
        for method in ('R1','R3'):
            for direction in ('reverse','isotropic'):
                model=estimator(method,params[method],record,direction)
                rows.append(dict(method=method,condition=direction,parent=record['parent'],forcing=record['mechanism'],
                                 seed=record['seed'],**model.score(record['truth']),residual=model.posterior()['diagnostics']['residual_inf']))
    write(output,dict(rows=rows,split='development',synthetic_conditions='artificial frozen-snapshot response, not real signal physics',
                      parameters_frozen=True,synthetic_parameters_fit_on_corresponding_training_records=True,
                      confirmation_opened=False))
