"""Independent held-out fast-rise / ridge-separated flood stress events."""
import sys
import os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.adapters.anuga import generate_event


if __name__=="__main__":
    os.environ.setdefault("ANUGADATA",str(ROOT/"results/setup"))
    settings=dict(seed=20261015,n_per_condition=3,
        fast_rise=dict(rate_range=[.0065,.009],ridge_range=[.02,.12]),
        ridge_separated=dict(rate_range=[.0025,.005],ridge_range=[.65,.85]))
    run=new_run("boundary_events",settings)
    rng=np.random.default_rng(settings["seed"])
    manifest=[]
    for condition in ("fast_rise","ridge_separated"):
        for index in range(3):
            c=settings[condition]
            params=dict(slope=float(rng.uniform(.0008,.002)),ridge=float(rng.uniform(*c["ridge_range"])),
                rate=float(rng.uniform(*c["rate_range"])),onset=float(rng.uniform(5,30)))
            event_id=f"{condition}_{index:03}"
            output=run/f"{event_id}.npz"
            generate_event(params,output,dict(event_id=event_id,event_family_id=event_id,split="test",condition=condition))
            manifest.append(str(output.relative_to(ROOT)).replace("\\","/"))
            print(event_id,flush=True)
    write_json(run/"manifest.json",dict(test=[Path(name).name for name in manifest]))
    write_json(ROOT/"data/manifests/boundary.json",dict(test=["../../"+name for name in manifest]))
    print(run,flush=True)
