"""Known-map, two-target fixed-order information-value diagnostic only."""
import sys
from collections import deque
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.pipeline import load_config
from hyra_validation.data import load_events
from hyra_validation.envs import MOVES,RescueEnv
from hyra_validation.diagnostics import score_fixed_order


def distances_from(env,start):
    d={int(start):0};queue=deque([int(start)])
    while queue:
        r=queue.popleft();x,y=r%env.n,r//env.n
        for dx,dy in MOVES:
            xx,yy=x+int(dx),y+int(dy)
            if not (0<=xx<env.n and 0<=yy<env.n):continue
            rr=yy*env.n+xx
            if env.blocked[rr] or rr in d:continue
            if dx and dy and (env.blocked[y*env.n+xx] or env.blocked[yy*env.n+x]):continue
            d[rr]=d[r]+1;queue.append(rr)
    return d


if __name__=="__main__":
    config=load_config(ROOT/"configs/pilot.v2.json")
    settings=dict(config=config,seed=815,layouts=32,times=list(range(24,241,6)),
        split_use=["train","validation"],known_targets=2,single_agent=True,
        geometry="Known static obstacle map in offline subproblem; other drones absent",
        candidates="two service orders on shortest known paths; generated without water truth",
        tie_break="first fixed order, never truth",forecast="latest issue published by decision; entire schedule within its horizon",
        claim="finite-candidate opportunity diagnosis, not global optimality or new online planner")
    run=new_run("action_value_diagnostic",settings)
    events=load_events(ROOT/config["manifest"])
    rows=[];excluded=0
    rc=config["representation"]
    for split in settings["split_use"]:
        for event in events[split]:
            eid=event["metadata"]["event_id"]
            for layout in range(settings["layouts"]):
                env=RescueEnv(event,config,dict(weights=np.eye(100).tolist(),frozen=True),"B0_raw",815+layout)
                rng=np.random.default_rng(9815+layout)
                starts=np.flatnonzero(~env.blocked)
                origin=int(rng.choice(starts));from_origin=distances_from(env,origin)
                reachable=np.array([r for r in from_origin if r!=origin])
                if len(reachable)<2:continue
                first,second=map(int,rng.choice(reachable,2,replace=False))
                between=distances_from(env,first)[second]
                d0=[from_origin[first],between];d1=[from_origin[second],between]
                max_duration=max(sum(d0),sum(d1))*config["environment"]["step_s"]+2*config["environment"]["service_s"]
                for now in settings["times"]:
                    f=int(np.flatnonzero(event["release_s"]<=now)[-1])
                    if now+max_duration>min(300,event["forecast_valid_s"][f,-1]):
                        excluded+=1;continue
                    forecast_scores=[];truth_scores=[]
                    for order,distance in [([first,second],d0),([second,first],d1)]:
                        common=dict(order=order,distances=distance,now=now,step_s=config["environment"]["step_s"],
                            service_s=config["environment"]["service_s"],limit=config["environment"]["depth_limit_m"])
                        forecast_scores.append(score_fixed_order(event["forecast_valid_s"][f],event["forecast_depth_m"][f],**common))
                        truth_scores.append(score_fixed_order(event["time_s"],event["truth_depth_m"],**common))
                    prior=int(np.argmax([s["service_yield"] for s in forecast_scores]))
                    oracle=int(np.argmax([s["service_yield"] for s in truth_scores]))
                    true_values=[s["service_yield"] for s in truth_scores]
                    rows.append(dict(event_id=eid,split=split,layout=layout,decision_time_s=now,origin=origin,
                        targets=[first,second],forecast_release_s=float(event["release_s"][f]),
                        prior_order=prior,truth_best_order=oracle,prior_predicted_yields=[s["service_yield"] for s in forecast_scores],
                        true_yields=true_values,prior_selected_true_yield=true_values[prior],finite_candidate_best_yield=true_values[oracle],
                        gap=true_values[oracle]-true_values[prior],ordering_changes_true_yield=true_values[0]!=true_values[1]))
            print(eid,split,"diagnosed",flush=True)
    per_event=[]
    for eid in sorted({r["event_id"] for r in rows}):
        subset=[r for r in rows if r["event_id"]==eid]
        sensitive=[r for r in subset if r["ordering_changes_true_yield"]]
        per_event.append(dict(event_id=eid,cases=len(subset),order_sensitive_cases=len(sensitive),
            positive_gap_cases=sum(r["gap"]>0 for r in subset),mean_gap=float(np.mean([r["gap"] for r in subset])),
            mean_gap_in_order_sensitive_cases=float(np.mean([r["gap"] for r in sensitive])) if sensitive else None))
    write_json(run/"cases.json",rows);write_json(run/"event_summary.json",per_event)
    summary=dict(cases=len(rows),excluded_before_truth_by_geometry_or_forecast_horizon=excluded,
        independent_families=len(per_event),order_sensitive_cases=sum(r["ordering_changes_true_yield"] for r in rows),
        positive_gap_cases=sum(r["gap"]>0 for r in rows),
        event_mean_gap=float(np.mean([e["mean_gap"] for e in per_event])),
        note="Training/validation diagnostic only; time points and layouts are not independent floods; no residual efficacy claim")
    write_json(run/"summary.json",summary)
    write_json(run/"completion.json",dict(completed=True,new_online_module=False,test_events_used=False))
    print(run,flush=True)
