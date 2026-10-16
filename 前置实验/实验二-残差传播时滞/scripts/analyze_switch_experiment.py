"""Frozen offline anomaly and finite-action diagnostics for paired shifts."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.pipeline import load_config
from hyra_validation.data import load_events,prepare_observations
from hyra_validation.innovation import score_innovations
from hyra_validation.envs import RescueEnv
from hyra_validation.diagnostics import score_fixed_order
from run_action_value_diagnostic import distances_from


def anomaly(events,config,run):
    all_rows=[];thresholds=[];reports=[]
    footprint=[y*10+x for y in range(5) for x in range(5)]
    for noise in (0.,.02):
        rows=[]
        for split,sequence in events.items():
            for event in sequence:
                cfg=dict(config["observations"],noise_std_m=noise,route_regions=[footprint])
                scores=score_innovations(prepare_observations(event,cfg))
                meta=event["metadata"]
                rows.extend(dict(r,event_id=meta["event_id"],family=meta["event_family_id"],
                    condition=meta["condition"],split=split,noise_std_m=noise) for r in scores)
        for metric in ("absolute_residual_m","innovation_score"):
            train=[r[metric] for r in rows if r["split"]=="train" and r["condition"]=="constant" and r["time_s"]<120]
            threshold=float(np.quantile(train,.99))
            thresholds.append(dict(noise_std_m=noise,metric=metric,threshold=threshold,calibration_rows=len(train)))
            for split in ("validation","test"):
                for condition in ("constant","switch"):
                    for window,lo,hi in [("pre",50,110),("post",140,180),("late",240,300)]:
                        selected=[r for r in rows if r["split"]==split and r["condition"]==condition and lo<=r["time_s"]<=hi]
                        families=sorted({r["family"] for r in selected})
                        per=[float(np.mean([r[metric]>threshold for r in selected if r["family"]==f])) for f in families]
                        reports.append(dict(noise_std_m=noise,metric=metric,split=split,condition=condition,window=window,
                            threshold=threshold,samples=len(selected),family_mean_exceedance=float(np.mean(per)),
                            per_family=dict(zip(families,per))))
        all_rows.extend(rows)
    write_json(run/"innovation_rows.json",all_rows)
    write_json(run/"innovation_thresholds.json",thresholds)
    write_json(run/"innovation_summary.json",reports)


def opportunities(events,config,run):
    rows=[];excluded=0
    for split in ("train","validation"):
        for event in events[split]:
            meta=event["metadata"]
            for layout in range(32):
                env=RescueEnv(event,config,dict(weights=np.eye(100).tolist(),frozen=True),"B0_raw",815+layout)
                rng=np.random.default_rng(9815+layout)
                origin=int(rng.choice(np.flatnonzero(~env.blocked)))
                d=distances_from(env,origin);reachable=np.array([r for r in d if r!=origin])
                if len(reachable)<2:continue
                a,b=map(int,rng.choice(reachable,2,replace=False));between=distances_from(env,a)[b]
                orders=[([a,b],[d[a],between]),([b,a],[d[b],between])]
                duration=max(sum(ds) for _,ds in orders)*6+24
                for now in range(24,241,6):
                    f=int(np.flatnonzero(event["release_s"]<=now)[-1])
                    if now+duration>min(300,event["forecast_valid_s"][f,-1]):excluded+=1;continue
                    prior=[];truth=[]
                    for order,ds in orders:
                        kwargs=dict(order=order,distances=ds,now=now,step_s=6,service_s=12,limit=.8)
                        prior.append(score_fixed_order(event["forecast_valid_s"][f],event["forecast_depth_m"][f],**kwargs)["service_yield"])
                        truth.append(score_fixed_order(event["time_s"],event["truth_depth_m"],**kwargs)["service_yield"])
                    chosen=int(np.argmax(prior))
                    rows.append(dict(event_id=meta["event_id"],family=meta["event_family_id"],condition=meta["condition"],
                        split=split,layout=layout,time_s=now,targets=[a,b],origin=origin,prior_yields=prior,true_yields=truth,
                        chosen=chosen,gap=max(truth)-truth[chosen],order_sensitive=truth[0]!=truth[1]))
            print(meta["event_id"],"opportunity diagnosed",flush=True)
    summaries=[]
    for condition in ("constant","switch"):
        for window in ("all","post"):
            selected=[r for r in rows if r["condition"]==condition and (window=="all" or r["time_s"]>=120)]
            families=sorted({r["family"] for r in selected})
            summaries.append(dict(condition=condition,window=window,cases=len(selected),independent_families=len(families),
                order_sensitive_cases=sum(r["order_sensitive"] for r in selected),positive_gap_cases=sum(r["gap"]>0 for r in selected),
                event_mean_gap=float(np.mean([np.mean([r["gap"] for r in selected if r["family"]==f]) for f in families]))))
    indexed={(r["family"],r["layout"],r["time_s"],r["condition"]):r for r in rows}
    for r in rows:
        if r["condition"]=="switch":
            c=indexed[r["family"],r["layout"],r["time_s"],"constant"]
            assert r["true_yields"]==c["true_yields"] and r["targets"]==c["targets"]
    write_json(run/"opportunity_cases.json",rows)
    write_json(run/"opportunity_summary.json",dict(rows=summaries,excluded=excluded,paired_truth_and_geometry_verified=True))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    config=load_config(args.config);events=load_events(ROOT/config["manifest"])
    run=new_run("switch_diagnostic",dict(config=config,online_use=False,scale_floor_m=.02,history_length=3))
    anomaly(events,config,run);opportunities(events,config,run)
    write_json(run/"completion.json",dict(completed=True,online_use=False))
    print("DIAGNOSTIC",run,flush=True)
