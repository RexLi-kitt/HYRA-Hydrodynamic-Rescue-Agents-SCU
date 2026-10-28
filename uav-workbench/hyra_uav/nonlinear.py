"""Fit-only engineering checks for the bounded nonlinear mechanism experiments."""
from functools import partial
from pathlib import Path
import json
import numpy as np
from .core import FloodProcess
from .diagnostic import static_graph
from .experiments import write_json, source_version


def local_rewire(pool, graph, distances, attempts=120000, bin_width=2, seed=20261025):
    """Preserve in/out degrees and each target's coarse distance-bin histogram.

    Nearby row proposals remove the global random-pair bottleneck. No labels,
    observations, water states or performance metrics participate in rewiring.
    """
    wrong=graph.copy();n=len(pool);rng=np.random.default_rng(seed);swaps=0
    lookup=[{int(j):k for k,j in enumerate(row)} for row in pool]
    for _ in range(attempts):
        a=int(rng.integers(n));b=int(rng.choice(pool[a]))
        aa,bb=np.flatnonzero(wrong[a]),np.flatnonzero(wrong[b])
        if not len(aa) or not len(bb):continue
        ka=int(rng.choice(aa));kb=int(rng.choice(bb));u,v=int(pool[a,ka]),int(pool[b,kb])
        av,bu=lookup[a].get(v),lookup[b].get(u)
        if u==v or av is None or bu is None or wrong[a,av] or wrong[b,bu]:continue
        if int(distances[a,u]//bin_width)!=int(distances[a,v]//bin_width) or int(distances[b,v]//bin_width)!=int(distances[b,u]//bin_width):continue
        wrong[a,ka]=wrong[b,kb]=False;wrong[a,av]=wrong[b,bu]=True;swaps+=1
    return wrong,swaps


def graph_audit(graph, bin_width):
    pool,a,b,d=graph["pool"],graph["B3"],graph["A2"],graph["distance"]
    removed=float((a&~b).sum()/max(a.sum(),1))
    before=np.bincount(pool[a],minlength=len(pool));after=np.bincount(pool[b],minlength=len(pool))
    distance_mean_a=float(d[a].mean());distance_mean_b=float(d[b].mean())
    hist_a=np.histogram(d[a],bins=np.arange(0,20.5,.5))[0]/max(a.sum(),1)
    hist_b=np.histogram(d[b],bins=np.arange(0,20.5,.5))[0]/max(b.sum(),1)
    return dict(swaps=graph["swaps"],replaced_edge_fraction=removed,
                in_degree_equal=bool(np.array_equal(before,after)),out_degree_equal=bool(np.array_equal(a.sum(1),b.sum(1))),
                coarse_distance_bins_equal=all(np.array_equal(np.sort(d[i,a[i]]//bin_width),np.sort(d[i,b[i]]//bin_width)) for i in range(len(pool))),
                original_mean_distance_cells=distance_mean_a,wrong_mean_distance_cells=distance_mean_b,
                distance_mean_relative_difference=abs(distance_mean_a-distance_mean_b)/max(distance_mean_a,1e-9),
                fine_histogram_total_variation=float(abs(hist_a-hist_b).sum()/2))


def preflight(old_fit_folder, output):
    """Use ONLY the historical fit terrains, never the already-scored holdouts."""
    old_fit_folder,output=Path(old_fit_folder),Path(output);output.mkdir(parents=True,exist_ok=False)
    plan=dict(stage="P3-fit-only-preflight",version=source_version(),pool_size=64,degree=12,
              candidate_bin_widths_cells=[2,4],attempts_per_graph=120000,
              acceptance=dict(min_replaced_edges=.30,max_relative_mean_distance_difference=.15,max_fine_histogram_tv=.20),
              selection="smallest width passing every old-fit terrain; no water labels or performance inspected")
    write_json(output/"plan.json",plan);reports=[]
    for width in plan["candidate_bin_widths_cells"]:
        for mechanism in ("slow","rapid","async"):
            process=FloodProcess(old_fit_folder/"scenes"/f"fit-{mechanism}"/"map-result.zip")
            graph=static_graph(process,pool_size=64,degree=12,rewire_fn=partial(local_rewire,bin_width=width))
            report=graph_audit(graph,width)
            report.update(mechanism=mechanism,bin_width_cells=width,source_sha256=process.digest)
            report["passed"]=(report["replaced_edge_fraction"]>=.30 and report["distance_mean_relative_difference"]<=.15 and
                             report["fine_histogram_total_variation"]<=.20 and report["in_degree_equal"] and
                             report["out_degree_equal"] and report["coarse_distance_bins_equal"])
            reports.append(report);write_json(output/"index.json",reports)
            print(mechanism,width,report["replaced_edge_fraction"],report["passed"],flush=True)
    valid=[w for w in plan["candidate_bin_widths_cells"] if all(r["passed"] for r in reports if r["bin_width_cells"]==w)]
    write_json(output/"decision.json",dict(selected_bin_width_cells=min(valid) if valid else None,
               ready=bool(valid),reports=reports,note="engineering fit-only check; no mechanism round or held-out scoring"))
