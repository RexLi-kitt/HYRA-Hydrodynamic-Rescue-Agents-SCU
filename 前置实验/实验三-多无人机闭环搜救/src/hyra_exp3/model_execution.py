"""Shared trained/untrained network episode execution with published-curve audit."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from .artifacts import sha256,write_json,peak_memory_bytes
from .learned_predictor import LearnedPredictor
from .planner_replay import audit_episode
from .planning import RollingPlanner
from .records import save_observations
from .simulation import run_episode
from .outcomes import analyze_outcomes


def save_published_forecasts(directory,predictor):
    path=Path(directory)/'published_forecasts.npz'
    keys=sorted(predictor.published_windows)
    length=max((len(predictor.published_windows[k].future_times) for k in keys),default=0)
    times=np.full((len(keys),length),np.nan)
    depths=np.full((len(keys),length),np.nan,np.float32)
    counts=np.zeros(len(keys),np.int16)
    for i,key in enumerate(keys):
        estimate=predictor.published_windows[key]
        counts[i]=len(estimate.future_times)
        times[i,:counts[i]]=estimate.future_times
        depths[i,:counts[i]]=estimate.future_depths
    q=np.asarray(predictor.queries,float).reshape(-1,5)
    np.savez_compressed(path,version=np.asarray([k[0] for k in keys],np.int32),
        row=np.asarray([k[1][0] for k in keys],np.int16),col=np.asarray([k[1][1] for k in keys],np.int16),
        future_times_s=times,predicted_depth_m=depths,length=counts,
        evidence_supported=np.asarray([predictor.published_windows[k].evidence_supported for k in keys],bool),
        queries=q,publications=np.asarray(predictor.publications),issue_latency_ms=predictor.issue_latencies_ms)
    return dict(path=str(path),sha256=sha256(path),unique_windows=len(keys),queries=len(q),
        roles={'0':'known_task','1':'public_frontier'},warmup_wall_s=predictor.warmup_wall_s)


def execute_network_episode(directory,truth,nominal,config,model,bank,method,*,device='cpu',
                            trained=False,provenance=None,planner_settings=None,formal_test=False,experiment_role='development'):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'episode.json').exists():
        raise FileExistsError(directory/'episode.json')
    provenance=provenance or {}
    if formal_test and (not trained or 'formal_registration_path' not in provenance):
        raise ValueError('Independent test requires trained weights and a prior registration')
    if trained:
        path=provenance.get('checkpoint_path')
        if not path or sha256(path)!=provenance.get('checkpoint_sha256'):
            raise ValueError('A verified frozen checkpoint is required to label a model trained')
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    predictor=LearnedPredictor(nominal['times_s'],nominal['depths_m'].reshape(-1,*config.map_shape),
        nominal['positions_m'],nominal['elevation_m'],model,bank,method,device,
        horizon_s=config.forecast_horizon_s,service_duration_s=config.service_duration_s,threshold_m=config.max_service_depth_m)
    result=run_episode(truth['depths_m'].reshape(-1,*config.map_shape),config,times=truth['times_s'],
                       predictor=predictor,planner=RollingPlanner(**(planner_settings or {})))
    audit=audit_episode(result,predictor,truth,config)
    result['outcome_analysis']=analyze_outcomes(truth,result,predictor,config)
    result['summary']['method']=method if trained else 'untrained_network_interface_preflight'
    result['summary']['trained_model']=bool(trained)
    result['summary']['formal_test']=bool(formal_test)
    result['limitations']=['Bounded candidate-route rolling optimization, not a globally optimal planner',
        'Failure categories are descriptive; causal reassignment attribution requires the separately paired policy intervention',
        'Saved 5m/1s hydraulic fields are numerical references, not exact physical solutions']
    result['execution_record']=dict(config=asdict(config),trained_model=bool(trained),method=method,
        model_parameter_count=model.parameter_count(),device=device,provenance=provenance,
        formal_test=bool(formal_test),experiment_role=experiment_role,
        untrained_interface_test_only=not trained,predictor_constructor_received_truth=False,
        same_full_raw_history_contract=True,peak_process_memory_bytes=peak_memory_bytes(),
        gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device)) if str(device).startswith('cuda') else None)
    result['published_forecast_log']=save_published_forecasts(directory,predictor)
    result['observation_log']=save_observations(directory/'observations.npz',result.pop('raw_observations'))
    result['audit']=audit
    write_json(directory/'episode.json',result)
    write_json(directory/'audit.json',audit)
    return result
