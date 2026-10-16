"""Training-family range check; never selects parameters from diagnostic test."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import write_json
from hyra_validation.data import load_events
from hyra_validation.lead import TARGETS
from run_lead_stage1 import response,arrival


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run");args=p.parse_args();run=Path(args.run).resolve()
    events=load_events(run/"manifest.json");rows=[]
    for event in events["train"]:
        if event["metadata"]["condition"]!="C0_stable":continue
        for center in (25,75):
            t,a=response(event["metadata"]["params"],10,center)
            tt,b=response(event["metadata"]["params"],20,center)
            selected=[r for r in TARGETS if np.max(abs(b[:,r]))>=.005]
            peaks=[abs(np.max(abs(a[:,r]))-np.max(abs(b[:,r])))/np.max(abs(b[:,r])) for r in selected]
            timing=[abs(arrival(t,a[:,r])-arrival(tt,b[:,r])) for r in selected]
            rows.append(dict(family=event["metadata"]["event_family_id"],center=center,cells=len(selected),
                median_peak_relative_error=float(np.median(peaks)),median_arrival_difference_s=float(np.median(timing)),
                passed=bool(np.median(peaks)<=.2 and np.median(timing)<=4)))
        print(event["metadata"]["event_family_id"],"mesh range checked",flush=True)
    write_json(run/"training_range_mesh_check.json",dict(rows=rows,all_passed=all(r["passed"] for r in rows),
        purpose="Training-only range audit at pulse90, not full PDE convergence proof or formal test calibration"))
