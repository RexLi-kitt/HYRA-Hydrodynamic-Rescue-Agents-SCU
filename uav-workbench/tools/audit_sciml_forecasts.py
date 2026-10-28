"""Reproduce all common-route forecasts from independently legal views.

Target extraction, masks and scoring do not call AuxiliaryLabels, forecast_rows,
field_loss or supervised_loss. SWEOperator has its own raw-solver validation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F
from hyra_learning.sciml_data import read,write,digest
from hyra_learning.sciml_study import verify
from hyra_learning.networks import SearchAgent
from hyra_learning.world_model import WorldHead
from hyra_learning.physics import SWEOperator
from hyra_learning.observations import encode
from hyra_uav.core import FloodProcess,ACTIONS
from audit_team_replays import rebuild


def score(prediction,arrays,t,view):
    device=prediction['current'].device
    current=torch.as_tensor(arrays['U'][2*t:2*t+1],device=device)
    future=torch.as_tensor(arrays['U'][2*t+2:2*t+3],device=device)
    rate=torch.as_tensor(arrays['rate'][t:t+1],device=device)
    wet=(current[0,0]>=.05)|(future[0,0]>=.05)
    nw=int(wet.sum());nd=wet.numel()-nw;classes=int(nw>0)+int(nd>0)
    weights=(wet.float()/max(nw,1)+(~wet).float()/max(nd,1))/classes
    scale=prediction['current'].new_tensor([.1,.05,.05])[None,:,None,None]
    def loss(a,b):
        diff=(a-b)/scale;absolute=diff.abs()
        huber=torch.where(absolute<1,.5*diff.square(),absolute-.5)
        return float((huber.mean(dim=1)[0]*weights).sum())
    supervised=loss(prediction['current'],current)+loss(prediction['future'],future)+.25*loss(prediction['rate'],rate)
    dt=float(arrays['dt'][t])
    evolved=SWEOperator().step(prediction['current'],torch.as_tensor(arrays['bed'],device=device),dt,
        source=torch.as_tensor(arrays['source'][t],device=device),drains=torch.as_tensor(arrays['drains'],device=device),
        manning=torch.as_tensor(arrays['manning'],device=device),solid=torch.as_tensor(arrays['solid'],device=device),
        wall_x=torch.as_tensor(arrays['wall_x'],device=device),wall_y=torch.as_tensor(arrays['wall_y'],device=device))
    physics_rate=(evolved-prediction['current'])/dt
    residual=prediction['rate']-physics_rate
    physics=loss(prediction['rate'],physics_rate)
    # Preserve the frozen evaluator's contiguous depth-channel reduction order.
    # A strided channels-last slice changes float32 summation roundoff.
    volume=float(residual[:,0].contiguous().sum().abs()*25)
    pred=prediction['future'][0].cpu().numpy();truth=arrays['U'][2*t+2]
    error=np.abs(pred[0]-truth[0]);known=view.agents[0].labels>=0
    stale=known&((view.agents[0].time_s-view.agents[0].observed_s)>=60)
    wet_truth=truth[0]>=.05;wet_pred=pred[0]>=.05
    balanced=[np.count_nonzero(wet_pred[wet_truth==c]!=(wet_truth[wet_truth==c]))/np.count_nonzero(wet_truth==c)
              for c in (False,True) if np.any(wet_truth==c)]
    return dict(unknown_cells=int((~known).sum()),stale_cells=int(stale.sum()),future_h_mae_m=float(error.mean()),
        future_h_unknown_mae_m=float(error[~known].mean()) if (~known).any() else None,
        future_h_stale_mae_m=float(error[stale].mean()) if stale.any() else None,
        future_wet_classbalanced_error=float(np.mean(balanced)),future_momentum_mae_m2_s=float(np.abs(pred[1:]-truth[1:]).mean()),
        current_h_mae_m=float(np.abs(prediction['current'][0,0].cpu().numpy()-arrays['U'][2*t,0]).mean()),
        supervised_loss=supervised,physical_rate_loss=physics,absolute_volume_rate_residual_m3_s=volume)


def audit_route(path,actor,world,views,actions,arrays):
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==60 and [r['time_s'] for r in rows]==list(range(0,600,10))
    hidden=actor.initial_hidden(2);previous=torch.zeros(2,4,device=hidden.device);maximum={}
    with torch.no_grad():
        for t,(row,view) in enumerate(zip(rows,views)):
            encoded=encode(view)
            _,hidden,_,_=actor(torch.as_tensor(encoded['images'],device=hidden.device),
                torch.as_tensor(encoded['poses'],device=hidden.device),previous,t,hidden,enhance=False)
            prediction=world(hidden.mean(dim=0)[None],torch.as_tensor(view.bed.copy(),dtype=torch.float64,device=hidden.device)[None])
            scores=score(prediction,arrays,t,view)
            for field,value in scores.items():
                if value is None:assert row[field] is None
                else:
                    np.testing.assert_allclose(row[field],value,rtol=2e-5,atol=2e-7,err_msg=f'{path.name} time{t*10} {field}')
                    maximum[field]=max(maximum.get(field,0),abs(row[field]-value))
            actual=[ACTIONS.index(a) for a in actions[t]['actions']]
            previous=F.one_hot(torch.tensor(actual,device=hidden.device),4).float()
    return maximum


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    args=parser.parse_args();root=args.output.resolve();plan=read(root/'formal-plan.json');verify(plan)
    freeze=read(root/'model-freeze.json');held=[e for e in read(root/'index.json') if e['split']=='holdout'];assert len(held)==9
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    cache={}
    for entry in held:
        process=FloodProcess(entry['path']);assert process.digest==entry['source_sha256']
        assert digest(Path(entry['auxiliary_path']))==entry['auxiliary_sha256']
        with np.load(entry['auxiliary_path'],allow_pickle=False) as packed:arrays={k:packed[k] for k in packed.files}
        routes={}
        for route in ('R0','R1','R2'):
            replay=read(root/'evaluation/rules'/f"{route}-{entry['id']}.json")['replay']
            rebuilt=rebuild(process,plan['config'],plan['team'],replay['actions'],replay['initial_poses'])
            routes[route]=(rebuilt['views'],replay['actions'])
        cache[entry['id']]=(arrays,routes)
    out=root/'audits/forecasts-v2';out.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):digest(p) for p in (Path(__file__),ROOT/'tools/audit_team_replays.py')}
    reports=[]
    for model in freeze['selected_models']:
        if model['method'] not in ('Sur-QMIX+Data','Sur-QMIX+Physics'):continue
        assert digest(root/model['path'])==model['sha256'];payload=torch.load(root/model['path'],map_location='cpu',weights_only=True)
        actor=SearchAgent().to(plan['runtime']['device']);actor.load_state_dict(payload['actor']);actor.eval()
        world=WorldHead().to(plan['runtime']['device']);world.load_state_dict(payload['world']);world.eval()
        for entry in held:
            arrays,routes=cache[entry['id']]
            for route,(views,actions) in routes.items():
                path=root/'evaluation/forecasts'/f"{model['method'].replace('+','-')}-s{model['seed']}-{entry['id']}-{route}.jsonl"
                target=out/(path.stem+'.json')
                provenance=dict(input_sha256=digest(path),audit_sources=sources,model_sha256=model['sha256'],source_sha256=entry['source_sha256'])
                if target.exists():
                    report=read(target)
                    for k,v in provenance.items():assert report[k]==v
                else:
                    report=dict(status='passed',rows=60,case=entry['id'],method=model['method'],seed=model['seed'],route=route,
                        maximum_absolute_score_disagreement=audit_route(path,actor,world,views,actions,arrays),**provenance)
                    write(target,report)
                reports.append(report)
        print('forecast audit',model['method'],model['seed'],flush=True)
    assert len(reports)==162 and sum(r['rows'] for r in reports)==9720
    write(ROOT/'docs/evidence/20261025-sciml-development/forecast-audit-final.json',dict(status='passed',rows=9720,reports=reports,
        audit_sources=sources,note='absolute_volume_rate_residual is the absolute spatially summed depth-rate discrepancy times cell area; opposite local discrepancies can cancel. The separate physical_rate_loss scores all three local equations.'))


if __name__=='__main__':main()
