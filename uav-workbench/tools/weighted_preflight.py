"""Check weighted ablation feasibility on old FIT terrain only, no prediction labels."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.core import FloodProcess
from hyra_uav.relations import weighted_relations
from hyra_uav.experiments import write_json,source_version


def run(source,output,scales=(5,)):
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(stage="fit-only-weighted-relation-feasibility",version=source_version(),pool_size=64,cost_scales_m=list(scales),
              acceptance=dict(min_changed_positive_weights=.5,min_operator_tv=.05),
              matching="identical support and all node degrees; exact per-target weighted mass at each squared lattice distance; incoming weighted mass difference reported",
              reason="binary nearest-neighbour graph had no sufficiently strong degree/distance-matched null; use weighted topographic evidence organisation",
              restrictions="old fit terrains only; no water state, label or predictive performance used")
    write_json(output/"plan.json",plan);reports=[]
    for scale in scales:
        for mechanism in ("slow","rapid","async"):
            process=FloodProcess(source/"scenes"/f"fit-{mechanism}"/"map-result.zip")
            graph=weighted_relations(process,terrain_cost_scale_m=scale)
            audit=dict(mechanism=mechanism,cost_scale_m=scale,source_sha256=process.digest,**graph["audit"])
            audit["passed"]=(audit["changed_positive_weight_fraction"]>=.5 and audit["mean_operator_total_variation"]>=.05 and
                              audit["support_exact"] and audit["out_strength_exact"] and audit["distance_weighted_mass_exact"])
            reports.append(audit);write_json(output/"index.json",reports);print(audit,flush=True)
    valid=[s for s in scales if all(r["passed"] for r in reports if r["cost_scale_m"]==s)]
    write_json(output/"decision.json",dict(ready=bool(valid),selected_cost_scale_m=max(valid) if valid else None,
                selection="weakest topographic weighting that passes the fixed operator-disruption checks",reports=reports))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--scales",nargs="+",type=float,default=[5])
    args=parser.parse_args();run(args.source,args.output,tuple(args.scales))
