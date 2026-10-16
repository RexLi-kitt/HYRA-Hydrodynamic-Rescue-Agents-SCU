"""Offline optimistic service margin at the first usable evidence decision."""
import json
import sys
from collections import deque
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.envs import RescueEnv,MOVES,interval_valid


def shortest_steps(env,positions,target):
    goal=(target%env.n,target//env.n)
    queue=deque([(tuple(p),0) for p in positions]);seen={tuple(p) for p in positions}
    while queue:
        (x,y),steps=queue.popleft()
        if (x,y)==goal:return steps
        for dx,dy in MOVES:
            xx,yy=x+int(dx),y+int(dy)
            if not (0<=xx<env.n and 0<=yy<env.n) or env.blocked[yy*env.n+xx]:continue
            if dx and dy and (env.blocked[y*env.n+xx] or env.blocked[yy*env.n+x]):continue
            if (xx,yy) not in seen:
                seen.add((xx,yy));queue.append(((xx,yy),steps+1))
    return None


if __name__=="__main__":
    source=Path(sys.argv[1]).resolve()
    config=json.loads((source/"config.json").read_text(encoding="utf-8"))
    graph=json.loads((source/"frozen_structure.json").read_text(encoding="utf-8"))
    events=load_events(ROOT/config["manifest"])["test"]
    by_id={e["metadata"]["event_id"]:e for e in events}
    run=new_run("action_margin",dict(training_run=str(source),
        interpretation="optimistic necessity check; ignores agent conflicts, busy duration and competing goals; truth scoring only"))
    scores=json.loads((source/"closed_loop_scores.json").read_text(encoding="utf-8"))
    rows=[]
    for score in scores:
        name=f"{score['method']}_seed{score['training_seed']}_{score['event_id']}_layout{score['layout']}_trace.json"
        trace=json.loads((source/name).read_text(encoding="utf-8"))
        event=by_id[score["event_id"]]
        env=RescueEnv(event,config,graph,score["method"],score["scene_seed"])
        arrivals=[r["arrival_time_s"] for r in trace["observations"] if r["residual_m"] is not None]
        if not arrivals:continue
        first_arrival=min(arrivals)
        first_decision=float(np.ceil(first_arrival/env.ec["step_s"])*env.ec["step_s"])
        decisions=[r for r in trace["actions_and_services"] if r["type"]=="decision" and r["time_s"]<=first_decision]
        positions=np.array(decisions[-1]["positions"]) if decisions else env.positions
        visible=set(env._visible(env.positions))
        for decision in decisions:visible.update(env._visible(np.array(decision["positions"])))
        completed_before={r["region"] for r in trace["actions_and_services"] if r["type"]=="completed" and r["time_s"]<=first_decision}
        for region in trace["targets"]:
            travel=shortest_steps(env,positions,region)
            last=score["last_feasible_start_s"][str(region)]
            margin=None if travel is None or last is None else last-first_decision-travel*env.ec["step_s"]
            rows.append(dict(method=score["method"],seed=score["training_seed"],event_id=score["event_id"],layout=score["layout"],
                region=region,first_evidence_arrival_s=first_arrival,first_usable_decision_s=first_decision,
                previously_completed=region in completed_before,discovered_at_evidence=region in visible,
                shortest_remaining_steps=travel,last_feasible_start_s=last,optimistic_margin_s=margin))
    remaining=[r for r in rows if not r["previously_completed"]]
    feasible=[r for r in remaining if r["optimistic_margin_s"] is not None and r["optimistic_margin_s"]>=0]
    write_json(run/"target_margins.json",rows)
    write_json(run/"summary.json",dict(target_instances=len(rows),remaining_target_instances=len(remaining),
        remaining_with_nonnegative_optimistic_margin=len(feasible),fraction=len(feasible)/max(len(remaining),1),
        earliest_usable_decision_s=min(r["first_usable_decision_s"] for r in rows),
        median_finite_margin_s=float(np.median([r["optimistic_margin_s"] for r in remaining if r["optimistic_margin_s"] is not None])),
        note="Repeated targets are not independent statistical samples; this is descriptive timing only"))
    write_json(run/"completion.json",dict(completed=True,online_use=False))
    print(run,flush=True)
