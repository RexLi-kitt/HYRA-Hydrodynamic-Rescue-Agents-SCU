"""Risk/opportunity and causal-data audits after the exploration information run."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from hyra_validation.artifacts import ROOT,write_json
from hyra_validation.data import load_events,current_forecast
from hyra_validation.lead import TARGETS,sensor_records,features
from hyra_validation.envs import RescueEnv,interval_valid
from hyra_validation.diagnostics import score_fixed_order
from run_action_value_diagnostic import distances_from


def read(p):return json.loads(Path(p).read_text(encoding="utf-8"))


def audits(events,run):
    rows=[]
    for split,sequence in events.items():
        for e in sequence:
            errors=[]
            for f,release in enumerate(e["release_s"]):
                ti=int(np.flatnonzero(e["nominal_time_s"]==release)[0])
                errors.append(float(np.max(abs(e["forecast_depth_m"][f,0]-e["nominal_depth_m"][ti]))))
            assert max(errors)<1e-8
            rows.append(dict(event=e["metadata"]["event_id"],split=split,restart_max_error_m=max(errors)))
    assert len({e["metadata"]["event_family_id"] for es in events.values() for e in es})==12
    write_json(run/"causal_data_audit.json",dict(events=rows,passed=True,families=12,
        fixed_observation_lead_s=20,label_forecast="Current publication, never future publication"))


def risk(events,run):
    readers=read(run/"information_readers.json");graph=read(run/"physical_graph.json")["graph"]
    index={(r["moving"],r["noise"],r["delay"],r["tau"],r["method"]):r for r in readers}
    methods=sorted({r["method"] for r in readers});scores=[];calibrations=[]
    for moving in (False,True):
        for noise,delay in ((0,0),(.02,0),(0,10),(.02,10)):
            raw={s:[] for s in ("validation","test")}
            for split in raw:
                for event in events[split]:
                    rec=sensor_records(event,noise,delay,moving)
                    for now in range(80,241,10):
                        times=np.arange(now+12,now+25,2)
                        for region in TARGETS:
                            label=int(not interval_valid(event,region,now+12,now+24,.8))
                            for method in methods:
                                depths=[]
                                for tau in (10,20,30):
                                    model=index[moving,noise,delay,tau,method]
                                    x=features(event,rec,region,now,tau,graph,method)
                                    p=np.r_[1,(x-np.array(model["mean"]))/np.array(model["std"])]@np.array(model["weights"])
                                    depths.append(current_forecast(event,now,tau)[region]+p)
                                maximum=float(np.max(np.interp(times-now,[10,20,30],depths)))
                                raw[split].append(dict(family=event["metadata"]["event_family_id"],condition=event["metadata"]["condition"],
                                    method=method,label=label,margin=maximum-.8,time_s=now,region=region))
            for method in methods:
                val=[r for r in raw["validation"] if r["method"]==method]
                x=np.array([r["margin"]/.02 for r in val]);y=np.array([r["label"] for r in val])
                def loss(w):
                    z=w[0]*x+w[1]
                    return np.mean(np.logaddexp(0,z)-y*z)+.001*w[0]**2
                fit=minimize(loss,[1,0],bounds=[(0,None),(None,None)])
                assert fit.success
                a,b=fit.x
                negatives=[expit(a*r["margin"]/.02+b) for r in val if r["condition"]=="C0_stable" and not r["label"]]
                threshold=float(np.quantile(negatives,.95)) if negatives else 1.
                calibrations.append(dict(moving=moving,noise=noise,delay=delay,method=method,slope=a,intercept=b,threshold=threshold,
                    validation_positives=int(y.sum()),test_used=False))
                selected=[r for r in raw["test"] if r["method"]==method]
                for family,condition in sorted({(r["family"],r["condition"]) for r in selected}):
                    rr=[r for r in selected if r["family"]==family and r["condition"]==condition]
                    yy=np.array([r["label"] for r in rr]);pp=expit(a*np.array([r["margin"]/.02 for r in rr])+b);flag=pp>threshold
                    scores.append(dict(moving=moving,noise=noise,delay=delay,method=method,family=family,condition=condition,
                        brier=float(np.mean((pp-yy)**2)),positives=int(yy.sum()),samples=len(yy),
                        recall=float(flag[yy==1].mean()) if yy.any() else None,
                        false_positive_rate=float(flag[yy==0].mean()) if (yy==0).any() else None))
            print("risk",moving,noise,delay,"complete",flush=True)
            write_json(run/"risk_scores.json",scores)
    write_json(run/"risk_calibration.json",calibrations)


def opportunities(events,run):
    config=read(ROOT/"configs/pilot.v2.json");rows=[];excluded=0
    for split in ("train","validation"):
        for event in events[split]:
            for layout in range(32):
                env=RescueEnv(event,config,dict(weights=np.eye(100).tolist(),frozen=True),"B0_raw",815+layout)
                rng=np.random.default_rng(9815+layout);origin=int(rng.choice(np.flatnonzero(~env.blocked)))
                d=distances_from(env,origin);available=np.array([r for r in d if r!=origin])
                if len(available)<2:continue
                a,b=map(int,rng.choice(available,2,replace=False));between=distances_from(env,a)[b]
                orders=[([a,b],[d[a],between]),([b,a],[d[b],between])]
                duration=max(sum(ds) for _,ds in orders)*6+24
                for now in range(24,241,6):
                    f=int(np.flatnonzero(event["release_s"]<=now)[-1])
                    if now+duration>min(300,event["forecast_valid_s"][f,-1]):excluded+=1;continue
                    forecast=[];truth=[]
                    for order,ds in orders:
                        kwargs=dict(order=order,distances=ds,now=now,step_s=6,service_s=12,limit=.8)
                        forecast.append(score_fixed_order(event["forecast_valid_s"][f],event["forecast_depth_m"][f],**kwargs)["service_yield"])
                        truth.append(score_fixed_order(event["time_s"],event["truth_depth_m"],**kwargs)["service_yield"])
                    chosen=int(np.argmax(forecast))
                    rows.append(dict(family=event["metadata"]["event_family_id"],condition=event["metadata"]["condition"],layout=layout,time_s=now,
                        gap=max(truth)-truth[chosen],order_sensitive=truth[0]!=truth[1]))
    summaries=[]
    for condition in sorted({r["condition"] for r in rows}):
        rr=[r for r in rows if r["condition"]==condition];families=sorted({r["family"] for r in rr})
        summaries.append(dict(condition=condition,cases=len(rr),order_sensitive=sum(r["order_sensitive"] for r in rr),
            positive_gap=sum(r["gap"]>0 for r in rr),family_mean_gap=float(np.mean([np.mean([r["gap"] for r in rr if r["family"]==f]) for f in families]))))
    write_json(run/"opportunity_cases.json",rows);write_json(run/"opportunity_summary.json",dict(rows=summaries,excluded=excluded,
        known_map=True,finite_two_orders_only=True,geometry_before_truth=True,policy_effect_tested=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("run");args=parser.parse_args();run=Path(args.run)
    events=load_events(run/"manifest.json");audits(events,run);opportunities(events,run);risk(events,run)
    write_json(run/"additional_diagnostics_completion.json",dict(completed=True,risk_and_opportunity=True,formal_test=False))
