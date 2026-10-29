"""Bounded P3 nonlinear study with independent prepare, fit and holdout stages.

Selection sees only fit/selection families. Holdout hydraulics are not generated
until the chosen models and all source/config hashes have been frozen.
"""
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from .core import Config, FloodProcess
from .diagnostic import ranking
from .experiments import ROOT, TERRAIN, run_episode, save_run, source_version, write_json
from .relations import weighted_relations
from .strong_features import GROUPS, cache, model_inputs

FAMILIES = {
    1: {"fit":["harmonic-channel","elliptic-basin"],"selection":["tilted-shelf"],
        "holdout":["double-saddle","broken-terrace","offset-ridge"]},
    2: {"fit":["cross-basin","diagonal-trough"],"selection":["split-ridge"],
        "holdout":["radial-notch","warped-valley","oblique-steps"]},
}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dependencies():
    # Optional local dependency directory, never auto-install or use a remote service.
    local=ROOT/".runtime/research-packages"
    if local.is_dir() and str(local) not in sys.path:sys.path.append(str(local))
    import sklearn,joblib
    from sklearn.ensemble import ExtraTreesClassifier
    return sklearn,joblib,ExtraTreesClassifier


def ensure_sources(plan):
    actual=source_version()["source_files_sha256"]
    if set(actual)!=set(plan["version"]["source_files_sha256"]):raise ValueError("研究源文件集合已变化")
    for name,expected in plan["version"]["source_files_sha256"].items():
        if actual.get(name)!=expected:raise ValueError("研究源文件已变化，不能混合版本续跑："+name)


def make_plan(round_id, output):
    if round_id not in FAMILIES:raise ValueError("仅允许两轮新增机制开发实验")
    if not load_json(ROOT/"generated/p2-formal-20261025/decision.json")["frozen"]:
        raise ValueError("P2尺度尚未冻结")
    preflight=load_json(ROOT/"generated/p3-weighted-scale-preflight-20261025/decision.json")
    if not preflight["ready"]:raise ValueError("错误联系拟合集预检未通过")
    cases=[]
    for split,families in FAMILIES[round_id].items():
        for family in families:
            for mechanism in ("slow","rapid","async"):
                case_id=f"r{round_id}-{split}-{family}-{mechanism}"
                seed=int(sha256(case_id.encode()).hexdigest()[:8],16)
                cases.append(dict(id=case_id,split=split,family=f"r{round_id}-"+family,
                                  terrain_recipe=family,mechanism=mechanism,seed=seed,parent_terrain=case_id))
    sklearn,_,_=dependencies()
    return dict(schema="hyra-p3-nonlinear-v1",round=round_id,version=source_version(),cases=cases,
        config=asdict(Config()),groups=list(GROUPS),seeds=[11,29,47],fractions=[.05,.1,.2],
        strict_change_ablation=(round_id==2),
        question=("unchanged B3 and strong baselines on new terrain families; strict latest-only A1 necessity test, no round-1 holdout tuning" if round_id==2 else "shared-information nonlinear evidence-organisation diagnostic"),
        pool_size=64,cost_scale_m=preflight["selected_cost_scale_m"],fit_candidates_per_time=32,
        selection_cap=None,holdout_cap=None,learner="ExtraTreesClassifier",sklearn_version=sklearn.__version__,
        learner_candidates=[dict(min_samples_leaf=5,max_features=.25),dict(min_samples_leaf=20,max_features=.25)],
        selection_trees=64,final_trees=128,max_depth=18,n_jobs=2,
        selection_rule="mean precision@10% equally over time, then route, then event; ties prefer first configuration",
        final_fit="fit split only, selection split never enters final fitting; all three seeds reported equally",
        A1=("target and every neighbour retain latest observation only; earlier histories and derived changes absent; intentional information ablation, same feature dimension and model budget" if round_id==2 else "same raw histories; zero propagated explicit change channels and their age interactions only; representation ablation, not absence of inferable information"),
        A2="weighted terrain factors permuted within identical lattice distance and positive support; all unweighted degrees and distance-stratified outgoing mass exact; incoming weighted mass difference reported",
        gate=dict(min_mean_increment=.005,min_positive_events=7,min_positive_per_mechanism=2,
                  require_positive_each_mechanism=True,min_A2_increment=.005,min_A1_increment=0.,
                  min_valid_control_events=8),
        resource_budget=dict(new_events=len(cases),routes_per_event=3,selection_fits=10,final_fits=15,
                             tuning_candidates_per_group=2,selected_checkpoints_per_seed=1),
        limits="synthetic development only; nine holdout events grouped in three new terrain families; seeds and routes are not independent events",
        output=str(output.resolve()))


def register(round_id, output):
    ledger=ROOT/"research/rounds.json"
    old=load_json(ledger) if ledger.exists() else {"maximum_rounds":2,"rounds":[]}
    if any(r["round"]==round_id for r in old["rounds"]):
        raise ValueError("此轮已登记；请从原目录继续，不能另建同轮或增加机制轮数")
    if round_id==2:
        first=next((r for r in old["rounds"] if r["round"]==1),None)
        if not first or not (Path(first["output"])/"summary.json").exists():raise ValueError("须先完成第一轮")
    old["rounds"].append(dict(round=round_id,output=str(output.resolve()),state="preregistered"))
    write_json(ledger,old)


def terrain(recipe,seed):
    z,x=np.indices((32,32),dtype=float);rng=np.random.default_rng(seed)
    phase=rng.uniform(-.6,.6);cx,cz=rng.uniform(11,20,size=2)
    functions={
        "harmonic-channel":.03*np.sin(x/3+phase)+.04*np.cos(z/5+phase)+.005*x,
        "elliptic-basin":.0005*((x-cx)**2+.7*(z-cz)**2)+.035*np.sin((x+z)/4+phase),
        "tilted-shelf":.016*np.floor((x+phase)/5)+.0007*(z-cz)**2+.025*np.sin(z/3),
        "double-saddle":.0006*((x-cx)**2-.6*(z-cz)**2)+.025*np.sin((x-z)/4+phase)+.18,
        "broken-terrace":.03*np.floor((z+phase)/4)+.004*x+.04*np.sin(x/4+z/9+phase),
        "offset-ridge":.14*np.exp(-((x-cx-.3*(z-cz))/3)**2)+.0006*(z-cz)**2+.02*np.sin(x/4+phase),
        "cross-basin":.00045*((x-cx)**2+(z-cz)**2)+.025*np.sin(x/2)*np.cos(z/3+phase),
        "diagonal-trough":.001*(x-z-phase*4)**2+.005*(x+z)+.02*np.cos(x/4),
        "split-ridge":.12*np.exp(-((z-cz)/3)**2)+.006*x+.025*np.sin(x/5+phase),
        "radial-notch":.008*np.sqrt((x-cx)**2+(z-cz)**2)+.025*np.sin(np.arctan2(z-cz,x-cx)*3+phase),
        "warped-valley":.0008*(z-cz-3*np.sin(x/5+phase))**2+.004*x+.02*np.cos(z/4),
        "oblique-steps":.025*np.floor((x+.6*z+phase)/5)+.02*np.cos((x-z)/4+phase),
    }
    return functions[recipe]


def generate_case(case,folder):
    sys.path.insert(0,str(TERRAIN))
    import flood_run
    from flood_solver import FloodSolver
    from multilayer import MapWorld
    from map_sessions import save_process
    bed=terrain(case["terrain_recipe"],case["seed"]);mechanism=case["mechanism"]
    rng=np.random.default_rng(case["seed"]+17)
    source=np.full((32,32),-1);graphs={}
    if mechanism!="slow":
        source[8,4]=0
        graphs["0"]=[[0,0],[100,float(rng.uniform(24,32))],[220,0]] if mechanism=="rapid" else [[0,12],[180,0]]
    layers=dict(source_id=source,hydrographs=graphs,drain_capacity=np.full((32,32),0 if mechanism=="slow" else .04))
    if mechanism=="async":
        source[23,26]=1;graphs["1"]=[[0,0],[240,12],[420,0]]
        wall=np.zeros((32,33),bool);wall[:,16]=True;layers["wall_x"]=wall
    config=flood_run.settings(dict(resolution=32,duration_s=600,rain_mm_h=600 if mechanism=="slow" else 0,
        rain_duration_s=600 if mechanism=="slow" else 0,infiltration_mm_h=0,boundary="closed",baseflow=False,warmup_s=0),(32,32))
    config["frames"]=121
    meta=dict(config=config,coarsening_factor=1,origin_blocks=[0,0],terrain_source="nonlinear_p3_synthetic_development",
              terrain_id=case["parent_terrain"],initial_conditions=dict(duration_s=0,converged=False,reason="no_active_baseflow"))
    world=MapWorld(FloodSolver(bed,np.zeros((32,32)),5,boundary="closed",manning=.035),meta,layers)
    frame_folder=folder/"generated/event/maps";frame_folder.mkdir(parents=True,exist_ok=False);frames=[]
    for i,t in enumerate(range(0,601,5)):
        world.advance_to(float(t));u,v=world.solver.velocity();name=f"frame-{i:03d}.f32"
        np.stack((world.solver.h,u,v,world.solver.maximum)).astype("<f4").tofile(frame_folder/name)
        frames.append(dict(index=i,url="/generated/event/maps/"+name,**world.solver.summary()))
    save_process(folder,"event",world,frames);(frame_folder/"map-result.zip").replace(folder/"map-result.zip")
    scene=dict(**case,path=str((folder/"map-result.zip").resolve()),kind="nonlinear_synthetic_hydraulic_development")
    write_json(folder/"scene.json",scene)
    return scene


def prepare_cases(root,plan,splits):
    index_path=root/"index.json";entries=load_json(index_path) if index_path.exists() else []
    audit_path=root/"case-audits.json";audits=load_json(audit_path) if audit_path.exists() else []
    (root/"cache").mkdir(exist_ok=True)
    for case in plan["cases"]:
        if case["split"] not in splits:continue
        scene_dir=root/"scenes"/case["id"]
        if (scene_dir/"scene.json").exists():scene=load_json(scene_dir/"scene.json")
        else:scene=generate_case(case,scene_dir)
        process=FloodProcess(scene["path"])
        graph=weighted_relations(process,pool_size=plan["pool_size"],terrain_cost_scale_m=plan["cost_scale_m"])
        if not any(a["id"]==case["id"] for a in audits):
            audits.append(dict(**case,source_sha256=process.digest,bed_sha256=sha256(process.bed.tobytes()).hexdigest(),
                               relation=graph["audit"],audit=process.audit(),
                               max_mass_residual_m3=max(abs(f["balance"]["residual_m3"]) for f in process.process["frames"])))
            write_json(audit_path,audits)
        for route in ("R0","R1","R2"):
            run_id=case["id"]+"-"+route
            existing=next((e for e in entries if e["run_id"]==run_id),None)
            if existing:
                if digest(root/existing["replay"])!=existing["replay_sha256"] or digest(root/existing["cache"])!=existing["cache_sha256"]:
                    raise ValueError("已冻结数据发生变化："+run_id)
                continue
            run_dir=root/"replays"/run_id
            if (run_dir/"replay.json").exists():replay=load_json(run_dir/"replay.json")
            else:
                replay=run_episode(scene,route,Config(**plan["config"]));save_run(replay,run_dir)
            cache_path=root/"cache"/(run_id+".npz")
            cap=plan["fit_candidates_per_time"] if case["split"]=="fit" else None
            stats=cache(replay,graph,process.bed,cache_path,cap=cap,seed=case["seed"],
                        strict_change_ablation=plan.get("strict_change_ablation",False))
            entry=dict(**case,route=route,run_id=run_id,replay=str((run_dir/"replay.json").relative_to(root)),
                       replay_sha256=digest(run_dir/"replay.json"),cache=str(cache_path.relative_to(root)),cache_sha256=digest(cache_path),**stats)
            entries.append(entry);write_json(index_path,entries);print("prepared",run_id,stats["rows"],flush=True)
    return entries


def prepare(root,round_id):
    root=Path(root)
    if (root/"plan.json").exists():
        plan=load_json(root/"plan.json")
        if plan["round"]!=round_id:raise ValueError("轮次不一致")
        ensure_sources(plan)
    else:
        root.mkdir(parents=True,exist_ok=False);plan=make_plan(round_id,root);register(round_id,root)
        write_json(root/"plan.json",plan)
    entries=prepare_cases(root,plan,("fit","selection"))
    audits=load_json(root/"case-audits.json")
    if len({a["bed_sha256"] for a in audits})!=len(audits):raise ValueError("母地形数组重复")
    write_json(root/"prepared.json",dict(entries=entries,source_digest=plan["version"]["source_digest"],
                                        family_sets={s:sorted({e["family"] for e in entries if e["split"]==s}) for s in ("fit","selection")}))


def score_records(data,scores,entry,group,seed,fractions,bands=True):
    records=[]
    for now in data["decision_times_s"]:
        time_mask=data["time_s"]==now
        masks=[("all",time_mask)]
        if bands:masks.extend((name,time_mask&mask) for name,mask in (
            ("age_0_60",data["age_s"]<60),("age_60_180",(data["age_s"]>=60)&(data["age_s"]<180)),("age_180_plus",data["age_s"]>=180)))
        for band,mask in masks:
            for fraction in fractions:
                records.append(dict(case=entry["id"],family=entry["family"],mechanism=entry["mechanism"],route=entry["route"],
                    group=group,seed=seed,time_s=float(now),band=band,fraction=fraction,
                    **ranking(data["y"][mask],scores[mask],data["cells"][mask],fraction)))
    return records


def event_values(records,field="precision",fraction=.1,band="all"):
    selected=[r for r in records if r["fraction"]==fraction and r["band"]==band]
    result={}
    for case in sorted({r["case"] for r in selected}):
        case_rows=[r for r in selected if r["case"]==case];seeds=[]
        for seed in sorted({r["seed"] for r in case_rows}):
            routes=[]
            for route in ("R0","R1","R2"):
                values=[r[field] for r in case_rows if r["seed"]==seed and r["route"]==route and r[field] is not None]
                if values:routes.append(float(np.mean(values)))
            if routes:seeds.append(float(np.mean(routes)))
        result[case]=float(np.mean(seeds)) if seeds else None
    return result


def risk(model,x):
    if 1 not in model.classes_:return np.zeros(len(x))
    return model.predict_proba(x)[:,list(model.classes_).index(1)]


def fit(root):
    root=Path(root);plan=load_json(root/"plan.json");ensure_sources(plan)
    if (root/"model-freeze.json").exists():raise ValueError("模型已冻结，禁止重新拟合后复用同轮留出")
    sklearn,joblib,ExtraTrees=dependencies()
    if sklearn.__version__!=plan["sklearn_version"]:raise ValueError("学习库版本变化")
    entries=load_json(root/"index.json");fit_entries=[e for e in entries if e["split"]=="fit"]
    selection_entries=[e for e in entries if e["split"]=="selection"]
    if len(fit_entries)!=18 or len(selection_entries)!=9:raise ValueError("拟合/选型回放不完整")
    verify_entries(root,entries)
    targets=[];raw=[];raw_a1=[];tails={g:[] for g in GROUPS}
    for e in fit_entries:
        with np.load(root/e["cache"],allow_pickle=False) as data:
            targets.append(data["y"]);raw.append(data["common"])
            if plan.get("strict_change_ablation",False):raw_a1.append(data["common_A1"])
            for group in GROUPS:tails[group].append(data[group])
    y=np.concatenate(targets);common=np.concatenate(raw);del targets,raw
    common_a1=np.concatenate(raw_a1) if raw_a1 else None;del raw_a1
    (root/"models").mkdir(exist_ok=True)
    trials=load_json(root/"selection.json") if (root/"selection.json").exists() else []
    models=load_json(root/"fit-index.json") if (root/"fit-index.json").exists() else [];choices=[]
    for row in models:
        if digest(root/row["path"])!=row["sha256"]:raise ValueError("续跑模型校验失败")
    # Work with one group's input matrix and forest at a time to bound memory.
    for group in GROUPS:
        group_raw=common_a1 if group=="A1" and common_a1 is not None else common
        x=np.concatenate((group_raw,np.concatenate(tails.pop(group))),axis=1)
        trial_rows=[]
        for candidate_id,params in enumerate(plan["learner_candidates"]):
            previous=next((r for r in trials if r["group"]==group and r["candidate"]==candidate_id),None)
            if previous:
                trial_rows.append(previous);continue
            started=time.perf_counter()
            model=ExtraTrees(n_estimators=plan["selection_trees"],max_depth=plan["max_depth"],n_jobs=plan["n_jobs"],random_state=11,**params)
            model.fit(x,y);records=[]
            for entry in selection_entries:
                with np.load(root/entry["cache"],allow_pickle=False) as data:
                    scores=risk(model,model_inputs(data,group))
                    records.extend(score_records(data,scores,entry,group,11,[.1],bands=False))
            values=event_values(records);mean=float(np.mean([v for v in values.values() if v is not None]))
            row=dict(group=group,candidate=candidate_id,params=params,selection_event_precision=values,
                     selection_mean_precision=mean,elapsed_s=time.perf_counter()-started,trees=len(model.estimators_),
                     total_nodes=sum(t.tree_.node_count for t in model.estimators_))
            trials.append(row);trial_rows.append(row);write_json(root/"selection.json",trials)
            print("selection",group,candidate_id,mean,flush=True)
        chosen=max(trial_rows,key=lambda r:(r["selection_mean_precision"],-r["candidate"]))
        choices.append(dict(group=group,candidate=chosen["candidate"],params=chosen["params"]))
        for seed in plan["seeds"]:
            if any(r["group"]==group and r["seed"]==seed for r in models):continue
            started=time.perf_counter()
            model=ExtraTrees(n_estimators=plan["final_trees"],max_depth=plan["max_depth"],n_jobs=plan["n_jobs"],random_state=seed,**chosen["params"])
            model.fit(x,y);path=root/"models"/f"{group}-{seed}.joblib";joblib.dump(model,path,compress=3)
            row=dict(group=group,seed=seed,path=str(path.relative_to(root)),sha256=digest(path),
                     fit_rows=len(y),positives=int(y.sum()),feature_dimension=x.shape[1],elapsed_s=time.perf_counter()-started,
                     trees=len(model.estimators_),total_nodes=sum(t.tree_.node_count for t in model.estimators_),
                     leaf_parameters=sum(int(np.sum(t.tree_.children_left<0))*2 for t in model.estimators_))
            models.append(row);write_json(root/"fit-index.json",models);print("fitted",group,seed,flush=True)
        del x
    write_json(root/"model-freeze.json",dict(version=source_version(),choices=choices,models=models,
               datasets=[dict(run_id=e["run_id"],cache_sha256=e["cache_sha256"],replay_sha256=e["replay_sha256"]) for e in entries],
               plan_sha256=digest(root/"plan.json")))


def verify_entries(root,entries):
    for entry in entries:
        for kind in ("cache","replay"):
            if digest(root/entry[kind])!=entry[kind+"_sha256"]:raise ValueError("冻结输入校验失败："+entry["run_id"])


def expected_rows(plan):
    holdout=sum(c["split"]=="holdout" for c in plan["cases"])
    decisions=len(range(10,int(plan["config"]["duration_s"]),10))
    return holdout*3*decisions*len(plan["groups"])*len(plan["seeds"])*len(plan["fractions"])*4


def evaluate(root):
    root=Path(root);plan=load_json(root/"plan.json");ensure_sources(plan)
    freeze=load_json(root/"model-freeze.json")
    if freeze["plan_sha256"]!=digest(root/"plan.json"):raise ValueError("预登记已变化")
    for model in freeze["models"]:
        if digest(root/model["path"])!=model["sha256"]:raise ValueError("模型已变化")
    entries=prepare_cases(root,plan,("holdout",));holdout=[e for e in entries if e["split"]=="holdout"]
    if len(holdout)!=27:raise ValueError("新留出回放数量不完整")
    verify_entries(root,entries)
    current={e["run_id"]:e for e in entries}
    for entry in freeze["datasets"]:
        if any(entry[k]!=current[entry["run_id"]][k] for k in ("cache_sha256","replay_sha256")):raise ValueError("拟合输入已变化")
    audits=load_json(root/"case-audits.json")
    if len({a["bed_sha256"] for a in audits})!=len(audits):raise ValueError("跨集合母地形数组重复")
    _,joblib,_=dependencies();metrics=root/"metrics.jsonl";shards=root/"scores";shards.mkdir(exist_ok=True)
    if metrics.exists():
        summarize(root);return
    for model_row in freeze["models"]:
        group=model_row["group"];seed=model_row["seed"]
        model=joblib.load(root/model_row["path"])
        for entry in holdout:
            shard=shards/f"{group}-{seed}-{entry['run_id']}.jsonl"
            if shard.exists():continue
            partial=shard.with_suffix(".partial")
            with partial.open("w",encoding="utf-8") as output:
                with np.load(root/entry["cache"],allow_pickle=False) as data:
                    records=score_records(data,risk(model,model_inputs(data,group)),entry,group,seed,plan["fractions"])
                    for record in records:output.write(json.dumps(record,allow_nan=False,separators=(",",":"))+"\n")
            partial.replace(shard)
        print("scored",group,seed,flush=True)
    partial=metrics.with_suffix(".partial")
    with partial.open("wb") as output:
        for model_row in freeze["models"]:
            for entry in holdout:
                output.write((shards/f"{model_row['group']}-{model_row['seed']}-{entry['run_id']}.jsonl").read_bytes())
    partial.replace(metrics)
    summarize(root)


def summarize(root):
    root=Path(root);plan=load_json(root/"plan.json")
    records=[json.loads(line) for line in (root/"metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(records)!=expected_rows(plan):raise ValueError("留出评分尚未完整，拒绝生成完成结论")
    keys={(r['case'],r['route'],r['group'],r['seed'],r['time_s'],r['band'],r['fraction']) for r in records}
    if len(keys)!=len(records):raise ValueError("留出评分含重复记录")
    events=[]
    for group in GROUPS:
        for fraction in plan["fractions"]:
            for band in ("all","age_0_60","age_60_180","age_180_plus"):
                group_rows=[r for r in records if r["group"]==group and r["fraction"]==fraction and r["band"]==band]
                precision=event_values(group_rows,"precision",fraction,band);recall=event_values(group_rows,"recall",fraction,band)
                prevalence=event_values(group_rows,"prevalence",fraction,band)
                for case,value in precision.items():
                    selected=[r for r in group_rows if r["case"]==case];first=selected[0]
                    events.append(dict(case=case,family=first["family"],mechanism=first["mechanism"],group=group,
                         fraction=fraction,band=band,precision=value,recall=recall[case],prevalence=prevalence[case],
                         empty_times=sum(r["n"]==0 for r in selected),no_positive_times=sum(r["positives"]==0 for r in selected)))
    main=[r for r in events if r["fraction"]==.1 and r["band"]=="all"];paired=[]
    for row in main:
        if row["group"]!="B3":continue
        peers={g:next(r["precision"] for r in main if r["case"]==row["case"] and r["group"]==g) for g in GROUPS}
        paired.append(dict(case=row["case"],family=row["family"],mechanism=row["mechanism"],precision=peers,
                           **{f"delta_{g}":peers["B3"]-peers[g] for g in ("B1","B2","A1","A2")}))
    audits=[a for a in load_json(root/"case-audits.json") if a["split"]=="holdout"]
    valid_control=sum(a["relation"]["mean_operator_total_variation"]>=.05 and a["relation"]["changed_positive_weight_fraction"]>=.5 and
                      a["relation"]["support_exact"] and a["relation"]["distance_weighted_mass_exact"] for a in audits)
    criteria=plan["gate"];means={g:float(np.mean([r[f"delta_{g}"] for r in paired])) for g in ("B1","B2","A1","A2")}
    mechanism_deltas={m:{g:float(np.mean([r[f"delta_{g}"] for r in paired if r["mechanism"]==m])) for g in ("B1","B2")} for m in ("slow","rapid","async")}
    positive=sum(r["delta_B1"]>0 and r["delta_B2"]>0 for r in paired)
    gate=(all(means[g]>=criteria["min_mean_increment"] for g in ("B1","B2")) and positive>=criteria["min_positive_events"]
          and all(sum(r["delta_B1"]>0 and r["delta_B2"]>0 for r in paired if r["mechanism"]==m)>=criteria["min_positive_per_mechanism"] for m in mechanism_deltas)
          and all(d[g]>0 for d in mechanism_deltas.values() for g in ("B1","B2"))
          and means["A2"]>=criteria["min_A2_increment"] and means["A1"]>=criteria["min_A1_increment"] and valid_control>=criteria["min_valid_control_events"])
    write_json(root/"summary.json",dict(status="completed",round=plan["round"],events=events,paired=paired,mean_deltas=means,
         mechanism_deltas=mechanism_deltas,positive_events=positive,valid_control_events=valid_control,gate_passed=gate,
         metrics_rows=len(records),expected_metrics_rows=expected_rows(plan),
         limits=plan["limits"],next_action="additional_predeclared_confirmation_or_P4_budget_audit" if gate else "adjust_once_with_new_holdout_or_deliver_negative_report"))


def main():
    import argparse
    parser=argparse.ArgumentParser(description="P3同信息同预算非线性实验")
    parser.add_argument("stage",choices=("prepare","fit","evaluate","summarize"));parser.add_argument("output",type=Path)
    parser.add_argument("--round",type=int,default=1);args=parser.parse_args()
    write_json(ROOT/"generated/strong-study-current.json",dict(stage=args.stage,output=str(args.output.resolve()),pid=os.getpid(),state="running"))
    try:
        if args.stage=="prepare":prepare(args.output,args.round)
        else:{"fit":fit,"evaluate":evaluate,"summarize":summarize}[args.stage](args.output)
    except BaseException as error:
        write_json(ROOT/"generated/strong-study-current.json",dict(stage=args.stage,output=str(args.output.resolve()),pid=os.getpid(),state="stopped",reason=type(error).__name__+": "+str(error)))
        raise
    write_json(ROOT/"generated/strong-study-current.json",dict(stage=args.stage,output=str(args.output.resolve()),pid=os.getpid(),state="completed"))


if __name__=="__main__":main()
