"""Frozen protocol, data integrity, replay and GPU policy comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .artifacts import ROOT,new_run,write_json
from .data import load_events
from .diagnostics import freeze_graph,replay_evaluate


def load_config(config_path):
    config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("status")!="PILOT_FROZEN_BEFORE_TEST":
        raise ValueError("Unfilled/template/unfrozen configuration")
    def check(obj):
        if obj is None:
            raise ValueError("Null template value")
        if isinstance(obj,dict):
            for value in obj.values():check(value)
        elif isinstance(obj,list):
            for value in obj:check(value)
    check(config)
    return config


def data_integrity(events,config,path):
    rows=[]
    for split,sequence in events.items():
        for event in sequence:
            meta=event["metadata"]
            # Every issue's initial grid state must equal its independently run nominal state.
            restart_error=[]
            if "nominal_time_s" not in event:
                raise ValueError("Missing raw nominal trajectory provenance")
            for f,release in enumerate(event["release_s"]):
                ti=int(np.flatnonzero(event["nominal_time_s"]==release)[0])
                restart_error.append(np.max(np.abs(event["forecast_depth_m"][f,0]-event["nominal_depth_m"][ti])))
            if max(restart_error)>1e-8:
                raise ValueError("Forecast restart differs from independent nominal state")
            rows.append(dict(event_id=meta["event_id"],split=split,source=meta["source"],version=meta["solver_version"],
                forecast_initial_nominal_max_error_m=float(max(restart_error)),mapping_max_distance_m=meta["mapping"]["max_distance_m"],
                raw_solver_file=meta.get("raw_solver_file"),families_isolated=True))
    write_json(path/"data_integrity.json",rows)
    manifest=ROOT/config["manifest"]
    frozen_manifest=json.loads(manifest.read_text(encoding="utf-8"))
    write_json(path/"event_manifest.json",frozen_manifest)
    fingerprints={}
    for names in frozen_manifest.values():
        for name in names:
            p=(manifest.parent/name).resolve()
            for f in (p,p.with_suffix(".json"),p.with_suffix(".solver.npz")):
                fingerprints[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    write_json(path/"data_fingerprints.json",fingerprints)


def run_replay(config_path):
    config=load_config(config_path)
    run=new_run("replay",config)
    events=load_events(ROOT/config["manifest"])
    data_integrity(events,config,run)
    graphs=freeze_graph(events,config,run)
    replay_evaluate(events,graphs,config,run)
    print(run,flush=True)
    return run
