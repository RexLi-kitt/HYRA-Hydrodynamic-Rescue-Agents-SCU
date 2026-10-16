"""Generate genuine shallow-water events with frozen family splits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.adapters.anuga import generate_event
import numpy as np

if __name__ == "__main__":
    rng = np.random.default_rng(20261014)
    run = new_run("events",dict(seed=20261014,n_events=12,splits=[6,3,3]))
    manifest = dict(train=[],validation=[],test=[])
    for i in range(12):
        split = "train" if i<6 else "validation" if i<9 else "test"
        params = dict(slope=float(rng.uniform(.0008,.002)),ridge=float(rng.uniform(.02,.12)),
                      rate=float(rng.uniform(.0025,.005)),onset=float(rng.uniform(5,30)))
        event_id = f"flood_{i:03}"
        path = run/f"{event_id}.npz"
        generate_event(params,path,dict(event_id=event_id,event_family_id=event_id,split=split))
        manifest[split].append(str(path.relative_to(ROOT)).replace("\\","/"))
        print(event_id,split,flush=True)
    write_json(run/"manifest.json",{s:[Path(f).name for f in names] for s,names in manifest.items()})
    write_json(ROOT/"data/manifests/pilot.json",{s:["../../"+f for f in names] for s,names in manifest.items()})
    print(run,flush=True)
