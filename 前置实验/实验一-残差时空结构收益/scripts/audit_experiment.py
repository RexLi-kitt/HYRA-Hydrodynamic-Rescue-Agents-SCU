"""Post-run scientific checks, finite loss and saved-trace service verification."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from hyra_validation.artifacts import ROOT,new_run,write_json
from hyra_validation.data import load_events
from hyra_validation.envs import interval_valid
from hyra_validation.evaluation import compare_paired_events


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("training_run");args=parser.parse_args()
    source=Path(args.training_run).resolve()
    config=json.loads((source/"config.json").read_text(encoding="utf-8"))
    events=load_events(ROOT/config["manifest"])
    run=new_run("experiment_audit",dict(training_run=str(source),config=config))
    by_id={e["metadata"]["event_id"]:e for e in events["test"]}
    scores=json.loads((source/"closed_loop_scores.json").read_text(encoding="utf-8"))
    model_reports=[];trace_reports=[]
    for method in config["methods"]:
        for seed in config["training"]["seeds"]:
            directory=source/f"{method}_seed{seed}"
            stats=json.loads((directory/"learner_stats.json").read_text(encoding="utf-8"))
            losses=[s["value"] for s in stats if s["key"]=="loss"]
            selection=json.loads((directory/"validation_selection.json").read_text(encoding="utf-8"))
            capacity=json.loads((directory/"capacity.json").read_text(encoding="utf-8"))
            model_reports.append(dict(method=method,seed=seed,loss_updates=len(losses),finite_losses=bool(np.isfinite(losses).all()),
                first_loss=losses[0],last_loss=losses[-1],test_used_for_selection=selection["test_used"],**capacity))
    for score in scores:
        name=f"{score['method']}_seed{score['training_seed']}_{score['event_id']}_layout{score['layout']}_trace.json"
        trace=json.loads((source/name).read_text(encoding="utf-8"))
        done=[a for a in trace["actions_and_services"] if a["type"]=="completed"]
        failed=[a for a in trace["actions_and_services"] if a["type"]=="failed"]
        starts=[a for a in trace["actions_and_services"] if a["type"]=="started"]
        if config.get("protocol_version"):
            assert not any(a["agent"]==b["agent"] and a["time_s"]==b["time_s"] for a in failed for b in starts), "Failure immediately restarted"
        assert len(done)==score["completed"]
        assert len({a["region"] for a in done})==len(done)
        assert score["target_denominator"]==len(trace["targets"])
        assert trace["enhancement_execution"] is False and trace["training_state_read_by_actor"] is False
        for entry in done:
            assert entry["time_s"]-entry["start_s"]>=config["environment"]["service_s"]
            assert interval_valid(by_id[score["event_id"]],entry["region"],entry["start_s"],
                entry["time_s"],config["environment"]["depth_limit_m"])
        for record in trace["observations"]:
            assert record["prior_release_s"]==record["valid_time_s"]-config["representation"]["lead_time_s"]
            assert record["arrival_time_s"]>=record["valid_time_s"]
        final_scoring={str(r):("completed" if trace["final_status"][str(r)]=="completed" else
            "opportunity_expired" if score["last_feasible_start_s"][str(r)] is None or
            score["last_feasible_start_s"][str(r)]<score["steps"]*config["environment"]["step_s"] else
            trace["final_status"][str(r)]) for r in trace["targets"]}
        trace_reports.append(dict(trace=name,completed_interval_checks=len(done),failed_attempts_checked=len(failed),
            no_same_step_restart_checked=bool(config.get("protocol_version")),all_passed=True,scoring_only_final_states=final_scoring))
    assert len({(r["actor_parameters"],r["mixer_parameters"],tuple(r["observation_shape"])) for r in model_reports})==1
    assert all(r["finite_losses"] and not r["test_used_for_selection"] for r in model_reports)
    comparisons=compare_paired_events(scores,dict(metric="weighted_service_yield",seed=config["evaluation"]["bootstrap_seed"]))
    write_json(run/"model_checks.json",model_reports);write_json(run/"trace_checks.json",trace_reports)
    write_json(run/"paired_with_training_variation.json",comparisons)
    write_json(run/"completion.json",dict(completed=True,models_checked=len(model_reports),traces_checked=len(trace_reports),
        successful_intervals_checked=sum(r["completed_interval_checks"] for r in trace_reports),all_passed=True))
    print(run,flush=True)
