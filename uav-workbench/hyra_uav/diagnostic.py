"""P3 fixed-replay screening. Features only consume arrived observation records."""
from dataclasses import asdict
from hashlib import sha256
import heapq
import json
import math
from pathlib import Path
import time
import sys
import zipfile

import numpy as np
from .core import Config, FloodProcess, Memory
from .experiments import TERRAIN, run_episode, save_run, source_version, write_json

GROUPS = ("B1", "B2", "B3", "A1", "A2")
SETTINGS = dict(seed=20261025, pool_size=16, graph_degree=8, history_length=4,
                fit_candidates_per_frame=64, ridge=10.0, fractions=[.05, .10, .20],
                primary_fraction=.10, duration_s=600, decision_interval_s=10,
                learner="standardized ridge least squares, uncalibrated risk score",
                graph="static 4-neighbour path cost dx*(1+abs(elevation_difference)/0.05m); walls/solids block paths; max path 50m",
                wrong_graph="directed double-edge swaps preserving in/out degree and per-target distance rings within common raw pool",
                ranking="ceil(fraction*n); ties resolved by ascending flattened cell index",
                budget="six new processes, 18 frozen replays, five fits, one fixed hyperparameter setting; no tuning on holdout")


def generate_scene(folder, mechanism, split):
    sys.path.insert(0, str(TERRAIN))
    from flood_solver import FloodSolver
    import flood_run
    from multilayer import MapWorld
    from map_sessions import save_process
    zz, xx = np.indices((32, 32), dtype=float)
    # Six distinct parent constructions; no parent or perturbation crosses split.
    beds = {
        ("slow", "fit"): .08*(1-np.cos(xx/5)) + .0004*(zz-15)**2,
        ("rapid", "fit"): .008*xx + .04*np.sin(zz/4) + .015*np.cos(xx/3),
        ("async", "fit"): .0006*(xx-8)**2 + .025*np.sin(zz/3),
        ("slow", "holdout"): .035*np.floor(xx/6) + .0007*(zz-12)**2,
        ("rapid", "holdout"): .001*((xx-18)**2-(zz-16)**2) + .15,
        ("async", "holdout"): .12*np.exp(-((xx-14)/3)**2) + .001*(zz-20)**2,
    }
    names = {("slow","fit"):"wave-basin",("rapid","fit"):"inclined-channel",("async","fit"):"curved-valley",
             ("slow","holdout"):"terraced-basin",("rapid","holdout"):"saddle-valley",("async","holdout"):"cross-ridge"}
    bed = beds[mechanism, split]
    config = flood_run.settings(dict(resolution=32, duration_s=600, boundary="closed", baseflow=False,
                    warmup_s=0, rain_mm_h=600 if mechanism=="slow" else 0,
                    rain_duration_s=600 if mechanism=="slow" else 0, infiltration_mm_h=0), (32,32))
    config["frames"] = 121
    layers = dict(source_id=np.full((32,32),-1), drain_capacity=np.full((32,32),0 if mechanism=="slow" else .04),hydrographs={})
    if mechanism != "slow":
        layers["source_id"][8,4] = 0
        layers["hydrographs"]["0"] = [[0,0],[100,30],[220,0]] if mechanism=="rapid" else [[0,12],[180,0]]
    if mechanism == "async":
        layers["source_id"][23,26] = 1
        layers["hydrographs"]["1"] = [[0,0],[240,12],[420,0]]
        wall = np.zeros((32,33),bool);wall[:,16]=True;layers["wall_x"]=wall
    meta = dict(config=config, coarsening_factor=1,origin_blocks=[0,0],terrain_source="synthetic_p3_development",
                terrain_id=names[mechanism,split],initial_conditions=dict(duration_s=0,converged=False,reason="no_active_baseflow"))
    world = MapWorld(FloodSolver(bed,np.zeros((32,32)),5,boundary="closed",manning=.035),meta,layers)
    frames=[];frame_folder=folder/"generated/event/maps";frame_folder.mkdir(parents=True,exist_ok=False)
    for i,t in enumerate(range(0,601,5)):
        world.advance_to(float(t));u,v=world.solver.velocity();name=f"frame-{i:03d}.f32"
        np.stack((world.solver.h,u,v,world.solver.maximum)).astype("<f4").tofile(frame_folder/name)
        frames.append(dict(index=i,url="/generated/event/maps/"+name,**world.solver.summary()))
    save_process(folder,"event",world,frames)
    (frame_folder/"map-result.zip").replace(folder/"map-result.zip")
    scene=dict(id=f"p3-{split}-{mechanism}",family=names[mechanism,split],parent_terrain=names[mechanism,split],
               mechanism=mechanism,split=split,kind="synthetic_hydraulic_development",path=str((folder/"map-result.zip").resolve()))
    write_json(folder/"scene.json",scene)
    return scene


def static_graph(process, pool_size=None, degree=None, rewire_fn=None):
    shape=process.shape;n=math.prod(shape);dx=process.dx
    z,x=np.indices(shape);coordinates=np.column_stack((z.ravel(),x.ravel()))
    distances=np.linalg.norm(coordinates[:,None]-coordinates[None,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    pool=np.argsort(distances,axis=1,kind="stable")[:,:pool_size or SETTINGS["pool_size"]]
    with zipfile.ZipFile(process.path) as archive:
        base=json.loads(archive.read("map_model.json"))["M_base"]
    wall_x=np.asarray(base.get("wall_x",np.zeros((shape[0],shape[1]+1))),bool)
    wall_y=np.asarray(base.get("wall_y",np.zeros((shape[0]+1,shape[1]))),bool)
    solid=np.asarray(base.get("solid",np.zeros(shape)),bool)
    strength=np.zeros(pool.shape)
    for target in range(n):
        row,col=divmod(target,shape[1]);costs={target:0.};queue=[(0.,target)]
        while queue:
            cost,node=heapq.heappop(queue)
            if cost!=costs[node] or cost>50:continue
            r,c=divmod(node,shape[1])
            for dr,dc in ((0,1),(0,-1),(1,0),(-1,0)):
                rr,cc=r+dr,c+dc
                if not (0<=rr<shape[0] and 0<=cc<shape[1]) or solid[rr,cc] or solid[r,c]:continue
                if (dc and wall_x[r,max(c,cc)]) or (dr and wall_y[max(r,rr),c]):continue
                new=cost+dx*(1+abs(process.bed[rr,cc]-process.bed[r,c])/.05)
                neighbour=rr*shape[1]+cc
                if new<=50 and new<costs.get(neighbour,float("inf")):
                    costs[neighbour]=new;heapq.heappush(queue,(new,neighbour))
        strength[target]=[math.exp(-costs[j]/25) if j in costs else 0 for j in pool[target]]
    graph=np.zeros_like(strength,bool)
    for i in range(n):
        selected=np.argsort(-strength[i],kind="stable")[:degree or SETTINGS["graph_degree"]]
        graph[i,selected]=strength[i,selected]>0
    wrong,swaps=(rewire_fn or rewire)(pool,graph,distances)
    return dict(pool=pool,distance=distances[np.arange(n)[:,None],pool],strength=strength,
                B3=graph,A2=wrong,swaps=swaps)


def rewire(pool, graph, distances, attempts=60000):
    n=len(pool);wrong=graph.copy();lookup=[{int(j):k for k,j in enumerate(row)} for row in pool]
    rng=np.random.default_rng(SETTINGS["seed"]);swaps=0
    for _ in range(attempts):
        a,b=rng.integers(n,size=2)
        aa,bb=np.flatnonzero(wrong[a]),np.flatnonzero(wrong[b])
        if a==b or not len(aa) or not len(bb):continue
        ka,kb=rng.choice(aa),rng.choice(bb);u,v=pool[a,ka],pool[b,kb]
        av,bu=lookup[a].get(int(v)),lookup[b].get(int(u))
        if u==v or av is None or bu is None or wrong[a,av] or wrong[b,bu]:continue
        if math.floor(distances[a,u])!=math.floor(distances[a,v]) or math.floor(distances[b,v])!=math.floor(distances[b,u]):continue
        wrong[a,ka]=wrong[b,kb]=False;wrong[a,av]=wrong[b,bu]=True;swaps+=1
    return wrong,swaps


def node_features(memory, now):
    n=memory.labels.size;features=np.zeros((n,19),dtype=np.float32)
    for i,history in enumerate(memory.history):
        if not history:continue
        for k,item in enumerate(history[-4:][::-1]):
            observed,arrival,label,quality=item
            if observed>arrival or arrival>now:raise ValueError("未来观测进入特征")
            features[i,3*k:3*k+3]=[label,1,(now-observed)/600]
        features[i,12:15]=[history[-1][2],(now-history[-1][0])/600,history[-1][3]]
        for previous,current in reversed(list(zip(history[:-1],history[1:]))):
            if current[2]!=previous[2]:
                features[i,15:19]=[current[2]-previous[2],(now-current[0])/600,(current[0]-previous[0])/600,1]
                break
    return features


def feature_groups(memory, now, graph, bed):
    nodes=node_features(memory,now);pool=graph["pool"];n=len(pool)
    static=np.stack((graph["distance"]/5,bed.ravel()[pool]-bed.ravel()[:,None],graph["strength"]),axis=-1)
    result={}
    for group in GROUPS:
        values=nodes.copy()
        if group=="A1":
            values[:,:12]=0;values[:,15:]=0  # remove raw short-history changes as well as aggregated evidence
        raw=np.concatenate((values,values[pool].reshape(n,-1),static.reshape(n,-1)),axis=1)
        weights=np.ones(pool.shape) if group=="B1" else (
            np.exp(-graph["distance"]) if group=="B2" else graph["A2" if group=="A2" else "B3"].astype(float))
        weights=weights/np.maximum(weights.sum(axis=1,keepdims=True),1e-12)
        aggregate=np.einsum("nk,nkf->nf",weights,values[pool])
        interactions=aggregate[:,[15,18]]*values[:,13:14]
        result[group]=np.concatenate((raw,aggregate,interactions),axis=1).astype(np.float32)
    return result


def replay_samples(replay, graph, bed):
    shape=tuple(replay["grid"]["shape"]);memory=Memory(shape,4);cursor=0;observations=replay["observations"]
    for snap,truth in zip(replay["snapshots"],replay["truth"]):
        now=snap["time_s"]
        refreshed=np.zeros(math.prod(shape),bool)
        while cursor<len(observations) and observations[cursor]["arrived_s"]<=now:
            obs=observations[cursor];memory.accept(obs,now)
            if obs["arrived_s"]==now:refreshed[obs["cells"]]=True
            cursor+=1
        if now==0 or now>=replay["config"]["duration_s"] or now%10:continue
        candidates=np.flatnonzero((memory.labels.ravel()>=0)&~refreshed)
        # Labels are created separately after the feature builder has consumed only legal memory/static priors.
        features=feature_groups(memory,now,graph,bed)
        y=(memory.labels.ravel()[candidates]!=np.asarray(truth)[candidates]).astype(float)
        yield now,candidates,(now-memory.observed_s.ravel()[candidates]),{g:f[candidates] for g,f in features.items()},y


def ranking(y, scores, cells, fraction):
    if not len(y):return dict(precision=None,recall=None,prevalence=None,n=0,positives=0,k=0)
    k=max(1,math.ceil(fraction*len(y)));selected=np.lexsort((cells,-scores))[:k];positives=int(y.sum())
    return dict(precision=float(y[selected].mean()),recall=float(y[selected].sum()/positives) if positives else None,
                prevalence=float(y.mean()),n=len(y),positives=positives,k=k)


def fit_ridge(x,y):
    mean=x.mean(axis=0,dtype=np.float64);scale=x.std(axis=0,dtype=np.float64);scale[scale<1e-6]=1
    normalized=(x-mean)/scale;bias=float(y.mean())
    weight=np.linalg.solve(normalized.T@normalized+SETTINGS["ridge"]*np.eye(x.shape[1]),normalized.T@(y-bias))
    return dict(mean=mean,scale=scale,weight=weight,bias=np.asarray(bias))


def diagnose(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    plan=dict(stage="P3",status="preregistered",settings=SETTINGS,version=source_version(),config=asdict(Config()),
              split={"fit":["wave-basin","inclined-channel","curved-valley"],
                     "holdout":["terraced-basin","saddle-valley","cross-ridge"]},
              limits="three held-out synthetic development events; linear screening, not full nonlinear strong-baseline confirmation; no final test")
    write_json(folder/"plan.json",plan)
    entries=[];graphs={};graph_reports=[]
    for split in ("fit","holdout"):
        for mechanism in ("slow","rapid","async"):
            scene=generate_scene(folder/"scenes"/f"{split}-{mechanism}",mechanism,split)
            process=FloodProcess(scene["path"]);graph=static_graph(process);graphs[scene["id"]]=(graph,process.bed)
            before=np.bincount(graph["pool"][graph["B3"]],minlength=process.bed.size)
            after=np.bincount(graph["pool"][graph["A2"]],minlength=process.bed.size)
            graph_reports.append(dict(scene=scene["id"],swaps=graph["swaps"],changed_edge_fraction=float(np.mean(graph["B3"]!=graph["A2"])),
                degree_preserved=bool(np.array_equal(before,after) and np.array_equal(graph["B3"].sum(1),graph["A2"].sum(1))),
                distance_rings_preserved=all(np.array_equal(np.sort(np.floor(graph["distance"][i,graph["B3"][i]])),np.sort(np.floor(graph["distance"][i,graph["A2"][i]]))) for i in range(len(graph["pool"]))),
                graph_sha256=sha256(graph["B3"].tobytes()).hexdigest(),source_sha256=process.digest,
                bed_sha256=sha256(process.bed.tobytes()).hexdigest(),changed_cells=process.audit()["changed_cells_ever"]))
            np.savez_compressed(folder/"scenes"/f"{split}-{mechanism}"/"static-graphs.npz",**{k:v for k,v in graph.items() if isinstance(v,np.ndarray)})
            for method in ("R0","R1","R2"):
                run_id=scene["id"]+"-"+method;result=run_episode(scene,method);save_run(result,folder/"replays"/run_id)
                entries.append(dict(run_id=run_id,scene=scene["id"],family=scene["family"],split=split,method=method,
                                    path=str((folder/"replays"/run_id/"replay.json").resolve()),
                                    replay_sha256=sha256((folder/"replays"/run_id/"replay.json").read_bytes()).hexdigest()))
                write_json(folder/"replay-index.json",entries);print("frozen",run_id,flush=True)
    write_json(folder/"graph-audit.json",graph_reports)
    write_json(folder/"freeze.json",dict(replays=entries,graphs=graph_reports,version=source_version()))
    datasets={g:[] for g in GROUPS};targets=[];rng=np.random.default_rng(SETTINGS["seed"])
    for entry in entries:
        if entry["split"]!="fit":continue
        replay=json.loads(Path(entry["path"]).read_text(encoding="utf-8"));graph,bed=graphs[entry["scene"]]
        for now,cells,ages,features,y in replay_samples(replay,graph,bed):
            chosen=np.sort(rng.choice(len(y),min(len(y),SETTINGS["fit_candidates_per_frame"]),replace=False))
            for g in GROUPS:datasets[g].append(features[g][chosen])
            targets.append(y[chosen])
    y=np.concatenate(targets);models={};fits=[]
    (folder/"models").mkdir()
    for g in GROUPS:
        t=time.perf_counter();x=np.concatenate(datasets.pop(g));model=fit_ridge(x,y);models[g]=model
        np.savez_compressed(folder/"models"/(g+".npz"),**model)
        fits.append(dict(method=g,samples=len(y),positives=int(y.sum()),feature_dimension=x.shape[1],parameters=x.shape[1]+1,elapsed_s=time.perf_counter()-t))
        print("fit",g,len(y),flush=True)
    write_json(folder/"fit-summary.json",fits)
    metrics=[]
    for entry in entries:
        if entry["split"]!="holdout":continue
        replay=json.loads(Path(entry["path"]).read_text(encoding="utf-8"));graph,bed=graphs[entry["scene"]]
        for now,cells,ages,features,y in replay_samples(replay,graph,bed):
            for g in GROUPS:
                m=models[g];scores=((features[g]-m["mean"])/m["scale"])@m["weight"]+m["bias"]
                for band,mask in (("all",np.ones(len(y),bool)),("age_0_60",ages<60),("age_60_180",(ages>=60)&(ages<180)),("age_180_plus",ages>=180)):
                    for fraction in SETTINGS["fractions"]:
                        metrics.append(dict(scene=entry["scene"],route=entry["method"],time_s=now,group=g,band=band,fraction=fraction,
                                            **ranking(y[mask],scores[mask],cells[mask],fraction)))
        print("scored",entry["run_id"],flush=True)
    write_json(folder/"metrics.json",metrics)
    summary=[]
    for scene in [e["scene"] for e in entries if e["split"]=="holdout" and e["method"]=="R0"]:
        for g in GROUPS:
            for fraction in SETTINGS["fractions"]:
                # Equal route weights inside an event. Uniform decision intervals imply equal time weights.
                selected=[m for m in metrics if m["scene"]==scene and m["group"]==g and m["band"]=="all" and m["fraction"]==fraction]
                means={}
                for field in ("precision","recall","prevalence"):
                    routes=[[m[field] for m in selected if m["route"]==route and m[field] is not None] for route in ("R0","R1","R2")]
                    means[field]=float(np.mean([np.mean(r) for r in routes if r])) if any(routes) else None
                summary.append(dict(scene=scene,group=g,fraction=fraction,**means,empty_times=sum(m["n"]==0 for m in selected),no_positive_times=sum(m["positives"]==0 for m in selected)))
    write_json(folder/"summary.json",dict(status="screening_completed",events=summary,fits=fits,graphs=graph_reports,
                elapsed_s=time.perf_counter()-started,gate="not_automatically_passed_linear_screening_only",limits=plan["limits"]))
    return folder
