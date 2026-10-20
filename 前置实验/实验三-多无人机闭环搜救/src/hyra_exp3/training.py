"""Equal-budget three-seed training with validation-only checkpoint selection."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import hashlib
import time
import numpy as np
import torch

from .artifacts import finish_run, new_run, sha256, write_json
from .diagnostics import load_field
from .execution_plan import register
from .links import fit_links, build_graph
from .model_features import full_raw_frame
from .models import SharedForecastBackbone, regions_for_positions, METHODS
from .protocol import load_protocol

RECIPE=dict(hidden=32,steps=1500,checkpoint_steps=[500,1000,1500],frames_per_step=2,
    query_cells_per_frame=128,learning_rate=.001,weight_decay=.0001,gradient_clip=1.,
    future_offsets_s=list(range(0,61,5)),history_s=60,raw_drone_slots=20,
    training_seeds=[101,202,303],validation_times_s=[0.,120.,300.,600.,900.,1140.,1200.],
    checkpoint_selection="minimum_validation_future_depth_MSE_then_earlier_checkpoint",
    B_star_selection="separate_validation_closed_loop_H_vs_E_once_after_training",
    same_initialization_sampling_order_and_training_budget=True)


class Panels:
    def __init__(self,data_run,bank):
        self.rows=[]
        for path in sorted((Path(data_run)/'jobs').glob('*/case.json')):
            case=load_protocol(path)
            for info in case['panels']:
                for p,h in ((info['path'],info['sha256']),
                    (info['raw_observations']['path'],info['raw_observations']['sha256'])):
                    if sha256(p)!=h:
                        raise AssertionError('Training/validation record changed')
                panel=load_field(info['path'])
                obs=load_field(info['raw_observations']['path'])
                graph=build_graph(bank,panel,case['side'])
                self.rows.append(dict(case=case,info=info,panel=panel,observations=obs,graph=graph))
        self.train=[r for r in self.rows if r['case']['family']['split']=='train']
        self.validation=[r for r in self.rows if r['case']['family']['split']=='validation']
        if len(self.train)!=144 or len(self.validation)!=72:
            raise ValueError('Complete common records for 24 train and 12 validation families required')
        if any(r['case']['family']['split']=='test' for r in self.rows):
            raise AssertionError('Test records entered model development')


def tensors(row,now,delay,query_ids,device):
    panel=row['panel'];side=row['case']['side']
    history=full_raw_frame(panel,row['observations'],now,delay)
    tick=int(now/5)
    offsets=np.arange(13)
    future_ticks=tick+offsets
    valid=future_ticks<len(panel['times_s'])
    future_ticks=future_ticks.clip(0,len(panel['times_s'])-1)
    nominal=panel['nominal_sample_m'][future_ticks][:,query_ids].T.copy()
    labels=panel['label_truth_sample_m'][future_ticks][:,query_ids].T.copy()
    nominal[:,~valid]=0.;labels[:,~valid]=0.
    positions=panel['positions_m']/(side*10.)
    static=np.column_stack((positions,panel['elevation_m'],panel['nominal_sample_m'][tick]))
    def tensor(value,dtype=torch.float32):
        return torch.as_tensor(value,dtype=dtype,device=device)
    return (tensor(history),tensor(static),tensor(regions_for_positions(panel['positions_m'],side),torch.long),
        tensor(query_ids,torch.long),tensor(nominal),tensor(np.broadcast_to(valid,nominal.shape).copy(),torch.bool),
        row['graph'],tensor(labels))


def prediction_loss(prediction,labels,valid):
    difference=(prediction-labels).square()
    mse=(difference*valid).sum()/valid.sum().clamp_min(1)
    derivative_valid=valid[:,1:]&valid[:,:-1]
    derivative=((prediction.diff(dim=1)-labels.diff(dim=1)).square()*derivative_valid).sum()/derivative_valid.sum().clamp_min(1)
    return mse+.01*derivative,mse


def validation_loss(model,panels,method,device):
    model.eval();total,count=0.,0
    with torch.no_grad():
        for row in panels.validation:
            for now in RECIPE['validation_times_s']:
                seed=int.from_bytes(hashlib.sha256(f"validation-query:{row['case']['family']['family_id']}:{row['case']['side']}:{row['info']['layout']}:{now}".encode()).digest()[:4],'big')
                ids=np.random.default_rng(seed).choice(row['case']['side']**2,128,replace=False)
                args=tensors(row,now,0.,ids,device)
                prediction=model(*args[:6],args[6],method)
                valid=args[5]
                total+=float(((prediction-args[7]).square()*valid).sum())
                count+=int(valid.sum())
    model.train()
    return total/max(1,count)


def train_one(run,panels,method,seed,device,recipe=RECIPE):
    directory=Path(run)/'models'/f'{method}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model=SharedForecastBackbone(hidden=recipe['hidden']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    rng=np.random.default_rng(seed)
    records=[];checkpoints=[];started=time.perf_counter()
    for step in range(1,recipe['steps']+1):
        optimizer.zero_grad(set_to_none=True)
        running=0.
        for _ in range(recipe['frames_per_step']):
            row=panels.train[int(rng.integers(len(panels.train)))]
            now=float(int(rng.integers(121))*10)
            delay=10. if row['case']['side']==50 and rng.random()<.5 else 0.
            ids=rng.choice(row['case']['side']**2,recipe['query_cells_per_frame'],replace=False)
            args=tensors(row,now,delay,ids,device)
            prediction=model(*args[:6],args[6],method)
            loss,mse=prediction_loss(prediction,args[7],args[5])
            (loss/recipe['frames_per_step']).backward()
            running+=float(mse.detach())/recipe['frames_per_step']
        torch.nn.utils.clip_grad_norm_(model.parameters(),recipe['gradient_clip'])
        optimizer.step()
        if not np.isfinite(running):
            raise FloatingPointError('Nonfinite training loss')
        if step%100==0:
            records.append(dict(step=step,training_mse_m2=running,elapsed_wall_s=time.perf_counter()-started))
            write_json(directory/'progress.json',dict(method=method,seed=seed,step=step,training_mse_m2=running))
            print(f'{method} seed{seed} step {step}/{recipe["steps"]}, MSE {running:.6g}',flush=True)
        if step in recipe['checkpoint_steps']:
            val=validation_loss(model,panels,method,device)
            path=directory/f'step{step:04d}.pt'
            torch.save(dict(state_dict=model.state_dict(),hidden=recipe['hidden'],method=method,seed=seed,step=step),path)
            checkpoints.append(dict(step=step,path=str(path),sha256=sha256(path),validation_mse_m2=val))
            write_json(directory/'checkpoints.json',checkpoints)
    best=min(checkpoints,key=lambda c:(c['validation_mse_m2'],c['step']))
    result=dict(method=method,seed=seed,parameter_count=model.parameter_count(),checkpoints=checkpoints,selected=best,
        training_history=records,elapsed_wall_s=time.perf_counter()-started,test_accessed=False,
        distinct_training_families=24,validation_families=12,trained=True)
    write_json(directory/'training_result.json',result)
    return result


def train_all(root,data_run,wait_for_data=False):
    root,data_run=Path(root).resolve(),Path(data_run).resolve()
    register(root)
    run=new_run(root,'five_method_training',dict(data_run=str(data_run),wait_for_data=wait_for_data))
    write_json(run/'training_recipe.json',RECIPE)
    print(f'Training directory: {run}',flush=True)
    started=time.perf_counter()
    try:
        previous=-1
        while not (data_run/'common_data_summary.json').exists():
            if not wait_for_data:
                raise ValueError('Common records not yet completed')
            state=load_protocol(data_run/'state.json')
            if state['status']=='failed':
                raise RuntimeError('Common record collection failed')
            count=len(list((data_run/'jobs').glob('*/case.json')))
            if count!=previous:
                print(f'Waiting for common records: {count}/108 cases',flush=True)
                write_json(run/'progress.json',dict(stage='waiting_for_common_records',common_cases=count))
                previous=count
            time.sleep(10)
        data=load_protocol(data_run/'common_data_summary.json')
        if data['status']!='completed' or data['cases']!=108:
            raise ValueError('Common data incomplete')
        write_json(run/'data_provenance.json',dict(summary_path=str(data_run/'common_data_summary.json'),
            sha256=sha256(data_run/'common_data_summary.json'),test_truth_accessed=False))
        bank=fit_links(root,data_run,run/'frozen_training_links.json')
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        write_json(run/'torch_environment.json',dict(torch=torch.__version__,device=device,
            gpu=torch.cuda.get_device_name() if device=='cuda' else None,
            deterministic_algorithms=True))
        panels=Panels(data_run,bank)
        results=[]
        for seed in RECIPE['training_seeds']:
            for method in METHODS:
                results.append(train_one(run,panels,method,seed,device))
                write_json(run/'progress.json',dict(stage='training',completed_models=len(results),planned_models=15))
        counts={r['parameter_count'] for r in results}
        if len(results)!=15 or len(counts)!=1:
            raise AssertionError('Capacity/seed/method training matrix incomplete')
        write_json(run/'training_summary.json',dict(status='completed',models=results,parameter_count=next(iter(counts)),
            models_trained=15,test_accessed=False,B_star_selected=False,configuration_frozen=False))
        finish_run(root,run,elapsed_wall_s=time.perf_counter()-started)
        write_json(root/'results/latest_model_training.json',dict(run=str(run),status='completed'))
        return run
    except BaseException as error:
        import traceback
        (run/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(root,run,status='failed',error=repr(error))
        raise
