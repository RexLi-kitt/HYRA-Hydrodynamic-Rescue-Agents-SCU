"""Frozen checkpoints, single-factor stress cases; never refit on test."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.pipeline import load_config
from hyra_validation.data import load_events
from hyra_validation.agents import SearchMAC
from hyra_validation.training import save_test_rollouts
from hyra_validation.envs import RescueEnv
from hyra_validation.evaluation import compare_paired_events


def boundary_events():
    manifest=json.loads((ROOT/"data/manifests/boundary.json").read_text(encoding="utf-8"))
    events=[]
    for filename in manifest["test"]:
        path=(ROOT/"data/manifests"/filename).resolve()
        with np.load(path,allow_pickle=False) as archive:
            event={k:archive[k] for k in archive.files}
        event["metadata"]=json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if event["metadata"]["split"]!="test":raise ValueError("Boundary split")
        events.append(event)
    return events


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("training_run")
    args=parser.parse_args()
    training=Path(args.training_run).resolve()
    if not (training/"completion.json").exists():
        raise RuntimeError("Training must be complete before frozen boundary evaluation")
    config=load_config(training/"config.json")
    events=load_events(ROOT/config["manifest"])
    graph=json.loads((training/"frozen_structure.json").read_text(encoding="utf-8"))
    distance=json.loads((training/"distance_structure.json").read_text(encoding="utf-8"))
    changes={"base":{},"delay_10s":{"observations":{"delay_s":10}},
        "delay_30s":{"observations":{"delay_s":30}},"noise_005m":{"observations":{"noise_std_m":.05}},
        "service_24s":{"environment":{"service_s":24}},"wide_depth_12m":{"environment":{"depth_limit_m":1.2}},
        "fast_rise":{},"ridge_separated":{}}
    run=new_run("boundaries",dict(training_run=str(training),changes=changes,config=config,
        boundary_manifest=json.loads((ROOT/"data/manifests/boundary.json").read_text(encoding="utf-8"))))
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic=True
    special=boundary_events()
    scores=[];models=[]
    for method in config["methods"]:
        structure=distance if method=="B1_distance" else graph
        for seed in config["training"]["seeds"]:
            source=training/f"{method}_seed{seed}"
            selection=json.loads((source/"validation_selection.json").read_text(encoding="utf-8"))
            checkpoint=source/selection["selected_checkpoint"]/"agent.th"
            shape=json.loads((source/"capacity.json").read_text(encoding="utf-8"))["observation_shape"]
            mac=SearchMAC(shape,config["environment"]["n_agents"],config["training"],"cuda")
            mac.agent.load_state_dict(torch.load(checkpoint,map_location="cuda",weights_only=True))
            mac.agent.eval()
            models.append(dict(method=method,seed=seed,checkpoint=str(checkpoint),model_selection="main validation only"))
            for condition,change in changes.items():
                cfg=copy.deepcopy(config)
                for section,values in change.items():cfg[section].update(values)
                test_events=[e for e in special if e["metadata"]["condition"]==condition] if condition in ("fast_rise","ridge_separated") else events["test"]
                directory=run/condition;directory.mkdir(exist_ok=True)
                rows=save_test_rollouts(mac,test_events,cfg,structure,method,seed,directory)
                scores.extend([dict(row,condition=condition) for row in rows])
            write_json(run/"scores.json",scores)
            print(method,seed,"boundary evaluation finished",flush=True)
    comparisons={condition:compare_paired_events([r for r in scores if r["condition"]==condition],
        dict(metric="weighted_service_yield",seed=config["evaluation"]["bootstrap_seed"])) for condition in changes}
    write_json(run/"comparisons.json",comparisons)
    write_json(run/"models.json",models)
    write_json(run/"completion.json",dict(completed=True,refit_on_test=False,rollouts=len(scores),
        independent_families_per_condition=3))
    print(run,flush=True)
