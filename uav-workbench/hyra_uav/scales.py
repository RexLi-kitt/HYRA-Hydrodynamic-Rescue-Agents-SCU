"""P2 formal scale audit: recompute hydraulics, never resize a water image."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from .core import Config, FloodProcess, Pose
from .experiments import FIXTURES, TERRAIN, run_episode, save_run, source_version, write_json


def refined_process(folder, mechanism, refinement=1, output_s=5):
    sys.path.insert(0, str(TERRAIN))
    from flood_solver import FloodSolver
    import flood_run
    from multilayer import MapWorld
    from map_sessions import save_process
    recipe=json.loads((FIXTURES/mechanism/"scene.json").read_text(encoding="utf-8"))["recipe"]
    n=32*refinement;dx=5/refinement
    # Preserve the outer domain [-2.5,157.5]^2 and the original analytic terrain.
    local=(np.arange(n)+.5)/refinement-.5
    zz,xx=np.meshgrid(local,local,indexing="ij")
    bed=.0005*((xx-15.5)**2+(zz-15.5)**2) if mechanism=="slow" else .006*xx+.003*zz
    source=np.full((32,32),-1)
    if recipe["sources"]:source[8,4]=0
    if mechanism=="async":source[23,26]=1
    source=np.repeat(np.repeat(source,refinement,0),refinement,1)
    layers=dict(source_id=source,drain_capacity=np.full((n,n),recipe["drains"]/refinement**2),hydrographs=recipe["sources"])
    if mechanism=="async":
        wall=np.zeros((n,n+1),bool);wall[:,16*refinement]=True;layers["wall_x"]=wall
    config=flood_run.settings(dict(resolution=n,duration_s=600,rain_mm_h=recipe["rain"],
            rain_duration_s=600 if recipe["rain"] else 0,infiltration_mm_h=recipe["infiltration"],
            boundary="closed",baseflow=False,warmup_s=0), (n,n))
    config["frames"]=round(600/output_s)+1
    origin=(.5/refinement-.5)*5
    world=MapWorld(FloodSolver(bed,np.zeros((n,n)),dx,boundary="closed",manning=.035),
          dict(config=config,coarsening_factor=1,origin_blocks=[origin,origin],terrain_source="p2_resolution_audit",
               terrain_id="p2-"+mechanism,initial_conditions=dict(duration_s=0,converged=False,reason="no_active_baseflow")),layers)
    frame_folder=folder/"generated/event/maps";frame_folder.mkdir(parents=True,exist_ok=False)
    frames=[]
    for i,t in enumerate(np.arange(0,600+output_s/2,output_s)):
        world.advance_to(float(t));u,v=world.solver.velocity();name=f"frame-{i:03d}.f32"
        np.stack((world.solver.h,u,v,world.solver.maximum)).astype("<f4").tofile(frame_folder/name)
        frames.append(dict(index=i,url="/generated/event/maps/"+name,**world.solver.summary()))
    save_process(folder,"event",world,frames)
    (frame_folder/"map-result.zip").replace(folder/"map-result.zip")
    scene=dict(id=f"{mechanism}-r{refinement}-dt{output_s}",family="p2-"+mechanism,
               kind="synthetic_scale_development",path=str((folder/"map-result.zip").resolve()),refinement=refinement,output_s=output_s)
    write_json(folder/"scene.json",scene)
    return scene


def formal_scales(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    plan=dict(stage="P2-formal",version=source_version(),config=asdict(Config()),
              candidates=[dict(refinement=1,output_s=5),dict(refinement=2,output_s=5),dict(refinement=1,output_s=2.5)],
              previous_budgets_s=[300,600],methods=["R0","R1","R2"],starts=[[0,0,0],[155,155,180],[0,155,270]],
              selection="600s preferred if R1 reaches90% at all starts and R0 at >=2/3 starts; 300s retained as resource pressure boundary",
              spatial_threshold=dict(max_common_grid_label_disagreement=.05,max_flooded_area_fraction_difference=.05),
              temporal_threshold=dict(max_common_time_label_disagreement=.02),
              note="engineering tolerances for this synthetic scope; not physical validation against reality; no B3 used in selection")
    write_json(folder/"plan.json",plan)
    rows=[];audits=[];comparisons=[]
    for mechanism in ("slow","rapid","async"):
        baseline=FloodProcess(FIXTURES/mechanism/"map-result.zip")
        audits.append(dict(scene=mechanism+"-base",audit=baseline.audit()))
        for refinement,interval in ((1,5),(2,5),(1,2.5)):
            if refinement==1 and interval==5:
                scene=dict(id=mechanism+"-base",family="p2-"+mechanism,path=str(baseline.path),refinement=1,output_s=5)
                process=baseline
            else:
                scene=refined_process(folder/"scenes"/f"{mechanism}-r{refinement}-dt{interval}",mechanism,refinement,interval)
                process=FloodProcess(scene["path"]);audits.append(dict(scene=scene["id"],audit=process.audit()))
                shared=process.depths[::round(5/interval)]
                if refinement==2:shared=shared.reshape(len(shared),32,2,32,2).mean(axis=(2,4))
                labels=shared>=process.threshold
                area=abs(labels.mean(axis=(1,2))-baseline.labels.mean(axis=(1,2)))
                comparisons.append(dict(scene=scene["id"],refinement=refinement,output_s=interval,
                        max_label_disagreement=float((labels!=baseline.labels).mean(axis=(1,2)).max()),
                        mean_label_disagreement=float((labels!=baseline.labels).mean()),max_area_fraction_difference=float(area.max()),
                        max_abs_mass_residual_m3=max(abs(f["balance"]["residual_m3"]) for f in process.process["frames"])))
            for method in plan["methods"]:
                for start,(x,z,heading) in enumerate(plan["starts"]):
                    offset=(1-.5/refinement-.5)*5
                    pose=Pose(x+offset,z+offset,heading)
                    config=Config(sample_s=interval)
                    run_id=f"{scene['id']}-{method}-s{start}"
                    t=time.perf_counter();result=run_episode(scene,method,config,pose)
                    save_run(result,folder/"runs"/run_id)
                    rows.append(dict(run_id=run_id,scene=scene["id"],mechanism=mechanism,refinement=refinement,output_s=interval,
                                     method=method,start=start,status="completed",elapsed_s=time.perf_counter()-t,**result["summary"]))
                    write_json(folder/"index.json",rows);print(run_id,flush=True)
    spatial=[c for c in comparisons if c["refinement"]==2]
    temporal=[c for c in comparisons if c["output_s"]==2.5]
    spatial_ok=all(c["max_label_disagreement"]<=.05 and c["max_area_fraction_difference"]<=.05 for c in spatial)
    temporal_ok=all(c["max_label_disagreement"]<=.02 for c in temporal)
    r1=[r for r in rows if r["refinement"]==1 and r["output_s"]==5 and r["method"]=="R1"]
    r0=[r for r in rows if r["refinement"]==1 and r["output_s"]==5 and r["method"]=="R0"]
    coverage_ok=all(r["first_90_percent_coverage_s"] is not None for r in r1) and all(sum(r["first_90_percent_coverage_s"] is not None for r in r0 if r["mechanism"]==m)>=2 for m in ("slow","rapid","async"))
    write_json(folder/"audits.json",audits)
    write_json(folder/"comparison.json",comparisons)
    write_json(folder/"decision.json",dict(spatial_ok=spatial_ok,temporal_ok=temporal_ok,coverage_ok=coverage_ok,
            frozen=spatial_ok and temporal_ok and coverage_ok,config=asdict(Config()),
            domain_outer_m=[-2.5,157.5],status="accepted_for_synthetic_development" if spatial_ok and temporal_ok and coverage_ok else "requires_review",
            limits="grid comparison uses 2x2 mean fine depth on common cells; subcell wet fractions can differ; synthetic scope only"))
    return folder
