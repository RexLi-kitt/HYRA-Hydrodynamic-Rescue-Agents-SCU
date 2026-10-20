"""Actual train-record GPU forward/backward checks; not a trained model result."""
from pathlib import Path
import argparse
import os
import sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import numpy as np
import torch
from hyra_exp3.artifacts import new_run,finish_run,write_json
from hyra_exp3.protocol import load_protocol
from hyra_exp3.diagnostics import load_field
from hyra_exp3.models import SharedForecastBackbone,METHODS
from hyra_exp3.training import tensors,prediction_loss
from hyra_exp3.evaluation import configure_device


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-run',type=Path,required=True)
    args=parser.parse_args()
    directory=new_run(ROOT,'gpu_training_preflight',dict(data_run=str(args.data_run.resolve()),trained_model=False))
    print(f'GPU training preflight: {directory}',flush=True)
    try:
        device=configure_device()
        if device!='cuda':raise RuntimeError('A CUDA device is required for this GPU preflight')
        torch.cuda.reset_peak_memory_stats();records=[]
        rows,cols=np.indices((16,16));mask=((cols-rows)%16==1)|((cols-rows)%16==2)
        lag=(rows+cols)%13
        graph=dict(E_mask=mask,P_mask=mask,E_lag_steps=lag,P_lag_steps=lag,
                   E_score=np.full((16,16),.2,np.float32),P_score=np.full((16,16),.2,np.float32))
        for side in (32,50,70):
            cases=[load_protocol(p) for p in sorted((args.data_run/'jobs').glob(f'*side{side}/case.json'))]
            case=next(c for c in cases if c['family']['split']=='train');info=case['panels'][0]
            row=dict(case=case,info=info,panel=load_field(info['path']),observations=load_field(info['raw_observations']['path']),graph=graph)
            for delay in ((0.,10.) if side==50 else (0.,)):
                for method in METHODS:
                    torch.manual_seed(101);model=SharedForecastBackbone().to(device)
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
                    inputs=tensors(row,600.,delay,np.arange(128),device)
                    losses=[]
                    for step in range(2):
                        optimizer.zero_grad(set_to_none=True)
                        prediction=model(*inputs[:6],inputs[6],method)
                        loss,mse=prediction_loss(prediction,inputs[7],inputs[5]);loss.backward()
                        finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
                        if not torch.isfinite(loss).item() or not finite:raise FloatingPointError('Nonfinite CUDA training loss/gradient')
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                        if not all(torch.isfinite(p).all().item() for p in model.parameters()):raise FloatingPointError('Nonfinite CUDA optimizer state')
                        losses.append(dict(step=step+1,mse_m2=float(mse.detach()),gradient_norm=float(norm)))
                    records.append(dict(side=side,delay_s=delay,method=method,parameter_count=model.parameter_count(),losses=losses,
                        actual_training_family=case['family']['family_id'],artificial_sparse_graph_fixture=True))
                    print(f'GPU backward checked: side{side}, delay{delay:g}, {method}',flush=True)
                    del model,optimizer,inputs,prediction,loss,mse
        summary=dict(status='passed',records=records,checks=len(records),optimizer_steps=2*len(records),
            trained_model=False,test_accessed=False,models_saved=0,not_model_effect_evidence=True,
            graph_is_gradient_exercise_fixture_not_fitted_links=True,device=device,torch=torch.__version__,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(directory/'gpu_training_preflight_summary.json',summary);finish_run(ROOT,directory)
        print(dict(status='passed',checks=len(records),optimizer_steps=2*len(records)),flush=True)
    except BaseException as error:
        import traceback
        (directory/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,directory,status='failed',error=repr(error));raise
