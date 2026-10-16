"""Paired predictor-input shift; truth is unchanged, no extra online module."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.pipeline import load_config,data_integrity,run_replay
from hyra_validation.data import load_events
from hyra_validation.adapters.anuga import domain_for,simulate,export_event
from hyra_validation.diagnostics import freeze_graph
from hyra_validation.training import closed_loop


def generate(config,run):
    source_manifest=ROOT/config["manifest"]
    manifest=json.loads(source_manifest.read_text(encoding="utf-8"))
    output={s:[] for s in manifest}
    switch=120.
    for split,names in manifest.items():
        for index,name in enumerate(names):
            source=(source_manifest.parent/name).resolve()
            with np.load(source) as z:base={k:z[k] for k in z.files}
            meta=json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
            meta["paired_layout_index"]=index
            # Independent predictor state before the change is literally shared.
            with np.load(source.with_suffix(".solver.npz")) as z:
                raw={k:z[k] for k in z.files}
            initial_index=int(np.flatnonzero(base["nominal_time_s"]==switch)[0])
            initial={k:raw[f"nominal_{k}"][initial_index] for k in ("stage","xmomentum","ymomentum")}
            nominal=domain_for(meta["params"],.75,initial,switch,rate_switch_s=switch,switched_factor=.4)
            nt,nd,states,_=simulate(nominal,300,base["xy_m"])
            pre=base["nominal_time_s"]<switch
            shifted=copy.deepcopy(base)
            shifted["nominal_time_s"]=np.r_[base["nominal_time_s"][pre],nt]
            shifted["nominal_depth_m"]=np.concatenate([base["nominal_depth_m"][pre],nd])
            shifted_raw={k:v.copy() for k,v in raw.items() if not k.startswith("nominal_")}
            shifted_raw["nominal_time_s"]=shifted["nominal_time_s"]
            for k in initial:
                shifted_raw[f"nominal_{k}"]=np.concatenate([raw[f"nominal_{k}"][pre],np.array([s[k] for s in states])])
            for f,release in enumerate(base["release_s"]):
                if release<switch:continue  # Old publications never anticipate the change.
                ti=int(np.flatnonzero(nt==release)[0])
                domain=domain_for(meta["params"],.75,states[ti],release,rate_switch_s=switch,switched_factor=.4)
                ft,fd,_,_=simulate(domain,release+60,base["xy_m"])
                np.testing.assert_array_equal(ft,base["forecast_valid_s"][f])
                shifted["forecast_depth_m"][f]=fd
            np.testing.assert_array_equal(shifted["truth_depth_m"],base["truth_depth_m"])
            np.testing.assert_array_equal(shifted["forecast_depth_m"][base["release_s"]<switch],base["forecast_depth_m"][base["release_s"]<switch])
            for condition,event,solver in [("constant",base,raw),("switch",shifted,shifted_raw)]:
                eid=f"{meta['event_id']}_{condition}"
                path=run/f"{eid}.npz"
                info=dict(meta,event_id=eid,condition=condition,source_event=str(source),
                    switch_time_s=switch,pre_factor=.75,post_factor=.4 if condition=="switch" else .75,
                    forecasting="Independent predictor; continuous boundary slope changes at publication120s only in switch; earlier issues immutable; no observation/truth reset",
                    raw_solver_file=path.with_suffix(".solver.npz").name)
                np.savez_compressed(path.with_suffix(".solver.npz"),**solver)
                export_event({k:event[k] for k in ("time_s","truth_depth_m","xy_m")},
                    {k:v for k,v in event.items() if k not in ("time_s","truth_depth_m","xy_m")},path,info)
                output[split].append(str(path.relative_to(ROOT)))
            print(meta["event_id"],"paired generation complete",flush=True)
    # Manifest lives at project root, so paths above are relative to that root.
    manifest_path=ROOT/f"switch_manifest_{run.name}.json"
    write_json(manifest_path,output)
    cfg=copy.deepcopy(config)
    cfg.update(manifest=manifest_path.name,protocol_version="v3_paired_published_predictor_shift")
    config_path=ROOT/"configs"/f"pilot.switch.{run.name}.json"
    write_json(config_path,cfg)
    write_json(run/"completion.json",dict(completed=True,config=str(config_path),manifest=str(manifest_path),
        unchanged_truth=True,pre_switch_forecasts_immutable=True,fresh_blind_test=False))
    return config_path


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--generate-only",action="store_true")
    parser.add_argument("--config")
    args=parser.parse_args()
    if args.config:
        config_path=Path(args.config)
    else:
        config=load_config(ROOT/"configs/pilot.v2.json")
        protocol=dict(config=config,switch_s=120,pre_factor=.75,post_factor=.4,
            rate_change="continuous boundary, known only on/after publication120",seeds=config["training"]["seeds"],
            conditions=["constant","switch"],same_truth=True,same_layouts=True,
            new_online_modules=False,added_anomaly_signal="offline diagnostic only",test_validity="reused truth families; paired diagnostic, not blind confirmation")
        run=new_run("switch_events",protocol)
        config_path=generate(config,run)
        print("CONFIG",config_path,flush=True)
    if not args.generate_only:
        import torch
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        config=load_config(config_path)
        events=load_events(ROOT/config["manifest"])
        run=new_run("switch_closed_loop",config)
        data_integrity(events,config,run)
        graphs=freeze_graph(events,config,run)
        write_json(run/"gpu.json",dict(device=torch.cuda.get_device_name(),torch=torch.__version__))
        closed_loop(events,config,graphs,run)
        write_json(run/"completion.json",dict(completed=True,scope="paired diagnostic",fresh_blind_test=False))
        print("TRAINING",run,flush=True)
