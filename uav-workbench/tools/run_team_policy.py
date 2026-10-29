"""Run an exported clean two-UAV actor against a local hydraulic ZIP.

Execution inputs are legal TeamView encodings only. Hydraulic truth belongs
to the simulator and is added to the saved replay for offline evaluation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import torch
from torch.nn import functional as F
from hyra_learning.networks import SearchAgent
from hyra_learning.observations import encode
from hyra_learning.team_env import TeamConfig,TeamEnv
from hyra_uav.core import Config,FloodProcess
from hyra_learning.sciml_data import read,digest,write


def execute(actor,process,config,team):
    env=TeamEnv(process,config,team);view,_=env.reset()
    hidden=actor.initial_hidden(2);previous=torch.zeros(2,4,device=hidden.device)
    actor.eval();total=0.;done=False;step=0
    start=time.perf_counter()
    while not done:
        encoded=encode(view)
        with torch.no_grad():
            q,hidden,_,_=actor(torch.as_tensor(encoded['images'],device=hidden.device),
                torch.as_tensor(encoded['poses'],device=hidden.device),previous,step,hidden,enhance=False)
        actions=q.argmax(dim=-1)
        view,reward,terminated,truncated,_=env.step(actions.cpu().numpy())
        total+=reward;step+=1;done=terminated or truncated
        previous=F.one_hot(actions,4).float()
    if hidden.device.type=='cuda':torch.cuda.synchronize()
    return env,total,time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True,help='Exported actor.pt with adjacent actor.json')
    parser.add_argument('--source',type=Path,required=True,help='Actual hydrodynamic map-result.zip')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'generated/sciml-round1/formal-plan.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve existing replay; choose a new output filename')
    metadata=read(args.model.with_suffix('.json'));plan=read(args.protocol)
    if digest(args.model)!=metadata['actor_sha256']:raise ValueError('Exported actor hash mismatch')
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    weights=torch.load(args.model,map_location='cpu',weights_only=True)
    if not weights or not all(isinstance(value,torch.Tensor) for value in weights.values()):raise ValueError('Expected actor-only state dictionary, not a training checkpoint')
    actor=SearchAgent().to(args.device);actor.load_state_dict(weights,strict=True)
    process=FloodProcess(args.source)
    env,return_,elapsed=execute(actor,process,Config(**plan['config']),TeamConfig(**plan['team']))
    replay=env.result()
    replay['execution']=dict(actor_sha256=metadata['actor_sha256'],selected_checkpoint_sha256=metadata.get('selected_checkpoint_sha256'),
        method=metadata['method'],seed=metadata['seed'],device=args.device,return_=return_,execution_loop_elapsed_s=elapsed,
        scope='Actual greedy two-agent sensing; clean actor only; time excludes loading and offline replay evaluation',
        protocol_sha256=digest(args.protocol))
    write(args.output,replay)
    print(json.dumps(dict(output=str(args.output.resolve()),method=metadata['method'],seed=metadata['seed'],
        time_average_correct_coverage=replay['summary']['time_average_correct_coverage'],return_=return_,
        elapsed_s=elapsed),ensure_ascii=False))


if __name__=='__main__':main()
