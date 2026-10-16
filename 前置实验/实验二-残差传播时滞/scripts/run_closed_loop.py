"""Static learner checks then independent GPU training for all comparison groups."""
import sys
import json
import argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.pipeline import load_config,data_integrity
from hyra_validation.data import load_events
from hyra_validation.diagnostics import freeze_graph
from hyra_validation.training import train_one,closed_loop


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default=str(ROOT/"configs/pilot.v2.json"))
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU training; CUDA unavailable")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    config=load_config(args.config)
    config["training"]["device"]=args.device
    run=new_run("closed_loop_smoke" if args.smoke_only else "closed_loop",config)
    events=load_events(ROOT/config["manifest"])
    data_integrity(events,config,run)
    graphs=freeze_graph(events,config,run)
    write_json(run/"gpu.json",dict(torch=torch.__version__,runtime=torch.version.cuda,
        device=torch.cuda.get_device_name() if args.device=="cuda" else "cpu",capability=torch.cuda.get_device_capability() if args.device=="cuda" else None))
    for enhanced,name in [(False,"static_qmix"),(True,"static_sur_qmix")]:
        _,summary=train_one(events,config,graphs,"B0_raw",5,run/name,static=True,enhancement=enhanced,episodes=8)
        write_json(run/f"{name}_summary.json",summary)
    if not args.smoke_only:
        closed_loop(events,config,graphs,run)
    write_json(run/"completion.json",dict(completed=True,scope="smoke" if args.smoke_only else "pilot",statistical_claim="pilot only"))
    print(run,flush=True)
