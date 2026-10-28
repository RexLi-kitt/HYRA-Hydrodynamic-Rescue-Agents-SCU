"""Fit-only graph-density feasibility audit with the original matching tolerances."""
from functools import partial
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.core import FloodProcess
from hyra_uav.diagnostic import static_graph
from hyra_uav.nonlinear import local_rewire,graph_audit
from hyra_uav.experiments import write_json,source_version


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    plan=dict(stage="fit-only-graph-density-feasibility",version=source_version(),degrees=[16,24,32],pool_size=64,bin_width_cells=2,
              attempts=120000,thresholds=dict(replaced_min=.30,distance_relative_max=.15,fine_histogram_tv_max=.20),
              rationale="degree12 topographic graph almost equals nearest-neighbour graph, constraining meaningful distance-matched corruption",
              selection="smallest density passing all three fit terrains; no holdout or water labels used")
    write_json(output/"plan.json",plan);reports=[]
    for degree in plan["degrees"]:
        for mechanism in ("slow","rapid","async"):
            process=FloodProcess(source/"scenes"/f"fit-{mechanism}"/"map-result.zip")
            graph=static_graph(process,pool_size=64,degree=degree,rewire_fn=partial(local_rewire,bin_width=2))
            audit=graph_audit(graph,2)
            audit.update(degree=degree,mechanism=mechanism,source_sha256=process.digest)
            audit["passed"]=(audit["replaced_edge_fraction"]>=.30 and audit["distance_mean_relative_difference"]<=.15 and
                             audit["fine_histogram_total_variation"]<=.20 and audit["in_degree_equal"] and audit["out_degree_equal"] and audit["coarse_distance_bins_equal"])
            reports.append(audit);write_json(output/"index.json",reports)
            print(degree,mechanism,audit["replaced_edge_fraction"],audit["passed"],flush=True)
    valid=[degree for degree in plan["degrees"] if all(r["passed"] for r in reports if r["degree"]==degree)]
    write_json(output/"decision.json",dict(ready=bool(valid),selected_degree=min(valid) if valid else None,reports=reports))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();run(args.source,args.output)
