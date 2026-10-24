"""Frozen checkpoint loading, common-record diagnostics and validation episodes."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from dataclasses import asdict
import json
import hashlib
import math
import uuid
import numpy as np
import torch

from .artifacts import new_run,finish_run,write_json,sha256
from .diagnostics import load_field,safe_start_grid
from .execution_plan import register
from .learned_predictor import load_frozen_model
from .links import build_graph
from .model_features import full_raw_frame
from .model_execution import execute_network_episode
from .models import METHODS,regions_for_positions
from .protocol import load_protocol
from .simulation import EpisodeConfig
from .statistics import select_bstar


def configure_device():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def layout_seed(study_id,family_id,layout):
    return int.from_bytes(hashlib.sha256(f'{study_id}:{family_id}:layout_noise:{layout}'.encode()).digest()[:4],'big')


def checkpoint_registry(training_run):
    training_run=Path(training_run)
    summary=load_protocol(training_run/'training_summary.json')
    if summary['status']!='completed' or len(summary['models'])!=15 or summary['test_accessed']:
        raise ValueError('All 15 trained models without test access are required')
    from .training_audit import audit_training
    audit_training(training_run.parents[2],training_run)
    registry={}
    for row in summary['models']:
        if not row['trained'] or row['parameter_count']!=summary['parameter_count']:
            raise ValueError('Untrained or unmatched model')
        selected=row['selected']
        if sha256(selected['path'])!=selected['sha256']:
            raise AssertionError('Selected checkpoint changed')
        key=row['method'],row['seed']
        if key in registry:
            raise ValueError('Duplicate model/seed')
        registry[key]=dict(method=row['method'],seed=row['seed'],checkpoint_path=selected['path'],
            checkpoint_sha256=selected['sha256'],selected_step=selected['step'])
    if set(registry)!={(m,s) for m in METHODS for s in (101,202,303)}:
        raise ValueError('Incomplete model/seed matrix')
    bank=load_protocol(training_run/'frozen_training_links.json')
    if bank['validation_or_test_used']:
        raise AssertionError('Link fitting used non-training families')
    return registry,bank


def load_models(registry,device):
    models={}
    for key,info in registry.items():
        model,record=load_frozen_model(info['checkpoint_path'],device)
        if (record['method'],record['seed'])!=key:
            raise ValueError('Checkpoint identity differs')
        models[key]=model
    return models


def config_for(plan,family_id,side,layout,delay_s=0.):
    n={32:4,50:10,70:20}[side]
    return EpisodeConfig(map_shape=(side,side),n_drones=n,n_targets=3*n,
        seed=layout_seed(plan['study_id'],family_id,layout),observation_delay_s=delay_s,
        noise_mode='counter_keyed',reassignment_policy='full')


def public_frame(panel,observations,now,delay,side,device):
    history=full_raw_frame(panel,observations,now,delay)
    tick=int(round(now/5));ticks=tick+np.arange(13)
    valid=ticks<len(panel['times_s']);clipped=ticks.clip(0,len(panel['times_s'])-1)
    nominal=panel['nominal_sample_m'][clipped].T.copy();nominal[:,~valid]=0.
    static=np.column_stack((panel['positions_m']/(side*10.),panel['elevation_m'],panel['nominal_sample_m'][tick]))
    inputs=(torch.as_tensor(history,device=device),torch.as_tensor(static,dtype=torch.float32,device=device),
        torch.as_tensor(regions_for_positions(panel['positions_m'],side),dtype=torch.long,device=device),
        torch.arange(side*side,device=device),torch.as_tensor(nominal,device=device),
        torch.as_tensor(np.broadcast_to(valid,nominal.shape).copy(),device=device))
    return inputs,valid,clipped


def predicted_safe_windows(depths):
    """Complete 12s max of a 5s-knot piecewise-linear forecast, starts 0/10/..40."""
    d=np.asarray(depths,float)
    return np.stack([np.maximum.reduce((d[:,i],d[:,i+1],d[:,i+2],.6*d[:,i+2]+.4*d[:,i+3]))<=.8
                     for i in (0,2,4,6,8)],axis=1)


def _evaluation_run(root,kind,arguments,resume):
    path=new_run(root,kind,arguments) if resume is None else Path(resume).resolve()
    if resume is not None and (not path.is_relative_to(Path(root).resolve()/'results/runs') or load_protocol(path/'arguments.json')!=arguments):
        raise ValueError('Evaluation resume must use the original run and identical data/checkpoints')
    if os.environ.get('HYRA3_STAGE_ANNOUNCEMENT'):
        write_json(os.environ['HYRA3_STAGE_ANNOUNCEMENT'],dict(run=str(path)))
    return path


def evaluate_fixed(root,training_run,data_run,resume=None):
    root,data_run,training_run=Path(root),Path(data_run),Path(training_run)
    registry,bank=checkpoint_registry(training_run);device=configure_device();models=load_models(registry,device)
    run=_evaluation_run(root,'fixed_route_validation',dict(training_run=str(training_run),data_run=str(data_run)),resume)
    write_json(run/'evaluation_spec.json',dict(split='validation',side=50,delays_s=[0.,10.],times_s=list(range(0,1201,10)),
        methods=list(METHODS),seeds=[101,202,303],expected_records=720,all_grid_cells=True,
        service_starts_offsets_s=[0,10,20,30,40],point_and_window_counts_not_independent_sample_units=True))
    print(f'Fixed-record validation directory: {run}',flush=True)
    records=[]
    try:
        paths=sorted((data_run/'jobs').glob('*side50/case.json'))
        cases=[load_protocol(p) for p in paths]
        cases=[c for c in cases if c['family']['split']=='validation']
        if len(cases)!=12:
            raise ValueError('Exactly twelve validation families required')
        for case in cases:
            truth=load_field(case['truth']['manifest']['field_path'])
            safe=safe_start_grid(truth['times_s'],truth['depths_m'])
            for info in case['panels']:
                if sha256(info['path'])!=info['sha256'] or sha256(info['raw_observations']['path'])!=info['raw_observations']['sha256']:
                    raise AssertionError('Common validation records changed')
                panel=load_field(info['path']);obs=load_field(info['raw_observations']['path']);graph=build_graph(bank,panel,50)
                for delay in (0.,10.):
                    directory=run/case['family']['family_id']/f"layout{info['layout']}_delay{delay:g}"
                    saved=[directory/f'{key[0]}_seed{key[1]}.json' for key in registry]
                    if all(p.exists() for p in saved):
                        for key,path in zip(registry,saved):
                            record=load_protocol(path)
                            if (record['checkpoint'],record['source_panel'],record['family_id'],record['delay_s'])!=(registry[key],info,case['family']['family_id'],delay):
                                raise AssertionError('Fixed-record checkpoint or input identity changed')
                            records.append({k:v for k,v in record.items() if k!='frames'})
                        continue
                    frames={key:[] for key in registry}
                    for now in range(0,1201,10):
                        inputs,valid,indices=public_frame(panel,obs,float(now),delay,50,device)
                        label=panel['label_truth_sample_m'][indices].T.astype(float)
                        starts=now+np.arange(0,41,10);window_valid=starts+12<=min(1200,now+60)
                        true_windows=np.zeros((2500,5),bool)
                        for j,t in enumerate(starts):
                            if window_valid[j]:
                                true_windows[:,j]=safe[int(t)]
                        with torch.inference_mode():
                            for key,model in models.items():
                                predicted=model(*inputs,graph,key[0]).cpu().numpy().astype(float)
                                errors=(predicted-label)[:,valid]
                                guessed=predicted_safe_windows(predicted)
                                false_safe=int((guessed&~true_windows)[:,window_valid].sum())
                                false_unsafe=int((~guessed&true_windows)[:,window_valid].sum())
                                frames[key].append(dict(time_s=now,squared_error_sum_m2=float((errors**2).sum()),
                                    absolute_error_sum_m=float(np.abs(errors).sum()),points=errors.size,
                                    false_safe=false_safe,false_unsafe=false_unsafe,windows=2500*int(window_valid.sum())))
                    for key,values in frames.items():
                        total_points=sum(v['points'] for v in values);total_windows=sum(v['windows'] for v in values)
                        directory=run/case['family']['family_id']/f"layout{info['layout']}_delay{delay:g}"
                        path=directory/f'{key[0]}_seed{key[1]}.json'
                        record=dict(family_id=case['family']['family_id'],layout=info['layout'],delay_s=delay,method=key[0],training_seed=key[1],
                            mse_m2=sum(v['squared_error_sum_m2'] for v in values)/total_points,
                            mae_m=sum(v['absolute_error_sum_m'] for v in values)/total_points,
                            window_error_rate=sum(v['false_safe']+v['false_unsafe'] for v in values)/total_windows,
                            false_safe_rate=sum(v['false_safe'] for v in values)/total_windows,
                            false_unsafe_rate=sum(v['false_unsafe'] for v in values)/total_windows,
                            points=total_points,windows=total_windows,frames=values,checkpoint=registry[key],
                            source_panel=info,labels_used_only_after_forward=True,split='validation')
                        write_json(path,record);records.append({k:v for k,v in record.items() if k!='frames'})
                    print(f"Fixed records: {len(records)}/720 model/seed/layout/delay records",flush=True)
                    write_json(run/'progress.json',dict(records=len(records),planned_records=720))
        if len(records)!=720:
            raise AssertionError('Fixed-record evaluation incomplete')
        write_json(run/'fixed_validation_summary.json',dict(status='completed',records=records,validation_families=12,
            expected_records=720,test_accessed=False,model_count=15,method_inputs_are_identical=True))
        finish_run(root,run);write_json(root/'results/latest_fixed_validation.json',dict(run=str(run),status='completed'))
        return run
    except BaseException as error:
        finish_run(root,run,status='failed',error=repr(error));raise


def load_or_execute(directory,truth,nominal,config,model,bank,method,provenance,device,*,formal_test=False,experiment_role='development'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'record.json'
    if marker.exists():
        record=load_protocol(marker)
        if sha256(record['path'])!=record['sha256']:
            raise AssertionError('Completed network episode changed')
        result=load_protocol(record['path'])
        expected=json.loads(json.dumps(asdict(config)))
        if result['execution_record']['config']!=expected or result['execution_record']['method']!=method:
            raise AssertionError('Completed episode has another controller/configuration identity')
        if result['execution_record']['provenance']!=provenance:
            raise AssertionError('Completed episode checkpoint/field provenance differs')
        for name in ('observation_log','published_forecast_log'):
            if sha256(result[name]['path'])!=result[name]['sha256']:
                raise AssertionError('Episode raw log changed')
        return result,record
    target=directory if not any(directory.iterdir()) else directory/'recovery_attempts'/uuid.uuid4().hex
    result=execute_network_episode(target,truth,nominal,config,model,bank,method,device=device,trained=True,provenance=provenance,
        formal_test=formal_test,experiment_role=experiment_role)
    record=dict(path=str(target/'episode.json'),sha256=sha256(target/'episode.json'))
    write_json(marker,record)
    return result,record


def validate_closed_loop(root,training_run,data_run,fixed_run,resume=None):
    root,data_run,training_run=Path(root),Path(data_run),Path(training_run)
    plan=register(root);registry,bank=checkpoint_registry(training_run)
    fixed=load_protocol(Path(fixed_run)/'fixed_validation_summary.json')
    if fixed['status']!='completed' or fixed['test_accessed']:
        raise ValueError('Completed validation-only fixed replay required')
    device=configure_device();models=load_models(registry,device)
    run=_evaluation_run(root,'closed_loop_validation',dict(training_run=str(training_run),data_run=str(data_run),fixed_run=str(fixed_run)),resume)
    write_json(run/'selection_spec.json',dict(side=50,delay_s=0.,families=12,layouts=2,seeds=[101,202,303],methods=list(METHODS),
        planned_episodes=360,B_star_candidates=['H','E'],rule='service rate, fixed window error, fixed depth MSE, H tie-break'))
    print(f'Closed-loop validation directory: {run}',flush=True)
    rows=[]
    try:
        cases=[load_protocol(p) for p in sorted((data_run/'jobs').glob('*side50/case.json'))]
        cases=[c for c in cases if c['family']['split']=='validation']
        if len(cases)!=12:
            raise ValueError('Missing validation families')
        for case in cases:
            truth=load_field(case['truth']['manifest']['field_path']);nominal=load_field(case['nominal']['manifest']['field_path'])
            family=case['family']['family_id']
            for layout in (0,1):
                config=config_for(plan,family,50,layout)
                for key,model in models.items():
                    result,record=load_or_execute(run/family/f'layout{layout}'/f'{key[0]}_seed{key[1]}',
                        truth,nominal,config,model,bank,key[0],registry[key],device)
                    rows.append(dict(family_id=family,side=50,layout=layout,delay_s=0.,method=key[0],training_seed=key[1],
                        summary=result['summary'],outcome_analysis=result['outcome_analysis'],episode=record))
                    write_json(run/'progress.json',dict(episodes=len(rows),planned_episodes=360))
                print(f'Closed-loop validation: {len(rows)}/360 episodes',flush=True)
        chosen=select_bstar(rows,fixed['records'])
        write_json(run/'B_star_selection.json',chosen)
        write_json(run/'validation_summary.json',dict(status='completed',rows=rows,B_star=chosen,
            trained_models=15,families=12,test_accessed=False,episodes=len(rows)))
        finish_run(root,run);write_json(root/'results/latest_closed_loop_validation.json',dict(run=str(run),status='completed'))
        return run
    except BaseException as error:
        finish_run(root,run,status='failed',error=repr(error));raise
