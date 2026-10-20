"""Paired engineering replay on the complete retained design candidate matrix."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
import json
import time

import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .diagnostics import load_field
from .phase_probe import mission_view
from .planning import RollingPlanner
from .protocol import load_protocol
from .records import save_observations
from .simulation import EpisodeConfig, run_episode
from .windows import NominalPredictor, service_start_intervals

VARIANTS = {
    "exploration": (True, False, False),
    "routes": (False, True, False),
    "combined": (True, True, False),
    "combined_truth_diagnostic": (True, True, True),
}


class QueryPredictor(NominalPredictor):
    def __init__(self, *args, privileged=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.privileged, self.queries, self.publications = privileged, [], []

    def issue(self, now, arrived_observations):
        super().issue(now, arrived_observations)
        self.publications.append((self.version, self.issued_at, self.valid_until))

    def _evidence(self, estimate):
        return replace(estimate, evidence="privileged_public_grid_future_truth_diagnostic", evidence_supported=True) if self.privileged else estimate

    def estimate(self, cell, now):
        self.queries.append((now, *cell, self.version, 0))  # Known task.
        return self._evidence(super().estimate(cell, now))

    def estimate_many(self, cells, now):
        self.queries.extend((now, *cell, self.version, 1) for cell in cells)  # Public frontier.
        return [self._evidence(row) for row in super().estimate_many(cells, now)]


def missed_categories(result, intervals):
    counts = dict(no_complete_window=0, undiscovered=0, discovered_after_close=0, window_remained_at_discovery=0)
    for target in result["targets"]:
        if target["served_at"] is not None:
            continue
        windows = intervals[tuple(target["cell"])]
        if not windows:
            counts["no_complete_window"] += 1
        elif target["discovered_at"] is None:
            counts["undiscovered"] += 1
        elif target["discovered_at"] > windows[-1][1] + 1e-7:
            counts["discovered_after_close"] += 1
        else:
            counts["window_remained_at_discovery"] += 1
    return counts


def audit_episode(result, predictor, truth, c):
    """Independent saved-curve scoring and causal checks on every replay."""
    discovered = {tuple(t["cell"]): t["discovered_at"] for t in result["targets"]}
    for now, row, col, version, role in predictor.queries:
        if not (0 <= row < c.map_shape[0] and 0 <= col < c.map_shape[1]):
            raise AssertionError("Query outside public geometry")
        if role == 0 and (discovered.get((row, col)) is None or discovered[row, col] > now + 1e-7):
            raise AssertionError("A task query preceded target discovery")
    for row in result["raw_observations"]:
        if row["sample_time"] > row["arrival_time"] + 1e-9 or row["arrival_time"] > c.duration_s + 1e-7:
            raise AssertionError("Noncausal observation")
    publications = {int(v): (a, b) for v, a, b in predictor.publications}
    checked_services, checked_visits = 0, 0
    payload = {i: c.payload for i in range(c.n_drones)}
    for event in result["events"]:
        if event["event"] == "service_complete":
            tid, start = event["target"], event["start_s"]
            end = start + c.service_duration_s
            row, col = result["targets"][tid]["cell"]
            curve = truth["depths_m"][:, row * c.map_shape[1] + col].astype(float)
            knots = np.r_[start, truth["times_s"][(truth["times_s"] > start) & (truth["times_s"] < end)], end]
            correct = bool(np.max(np.interp(knots, truth["times_s"], curve)) <= c.max_service_depth_m)
            if correct != event["success"] or abs(event["time_s"] - end) > 1e-7:
                raise AssertionError("Complete-service score or duration mismatch")
            if result["targets"][tid]["discovered_at"] > start + 1e-7:
                raise AssertionError("Service preceded discovery")
            if correct:
                payload[event["drone"]] -= 1
            if payload[event["drone"]] < 0 or payload[event["drone"]] != event["payload_remaining"]:
                raise AssertionError("Payload accounting mismatch")
            checked_services += 1
        if event["event"] == "plan":
            used, drones = set(), set()
            for route in event.get("planner_diagnostics", {}).get("selected_routes", []):
                if route["drone"] in drones or used.intersection(route["targets"]) or len(route["targets"]) > route["payload"]:
                    raise AssertionError("Route allocation/payload conflict")
                drones.add(route["drone"])
                used.update(route["targets"])
                position, finish = np.asarray(route["initial_position_m"]), event["time_s"]
                published, valid_until = publications[route["version"]]
                for tid, start in zip(route["targets"], route["starts_s"]):
                    target = result["targets"][tid]
                    if target["discovered_at"] is None or target["discovered_at"] > event["time_s"] + 1e-7:
                        raise AssertionError("Planned an undiscovered target")
                    row, col = target["cell"]
                    destination = np.asarray([(col + .5) * c.cell_size_m, (row + .5) * c.cell_size_m])
                    if start < finish + np.linalg.norm(destination - position) / c.speed_mps - 1e-7:
                        raise AssertionError("Route violates travel time")
                    if start < published - 1e-7 or start + c.service_duration_s > min(valid_until, c.duration_s) + 1e-7:
                        raise AssertionError("Route extrapolates forecast")
                    end = start + c.service_duration_s
                    if hasattr(predictor, "published_windows"):
                        estimate = predictor.published_windows[route["version"], (row, col)]
                        if estimate.can_start(start) is not True:
                            raise AssertionError("Route violates the actual published model forecast")
                    else:
                        curve = predictor.depths[:, row, col]
                        knots = np.r_[start, predictor.times[(predictor.times > start) & (predictor.times < end)], end]
                        if np.max(np.interp(knots, predictor.times, curve)) > c.max_service_depth_m + 1e-9:
                            raise AssertionError("Route violates its published forecast window")
                    finish, position = end, destination
                    checked_visits += 1
    return dict(status="passed", observations=len(result["raw_observations"]), queries=len(predictor.queries),
                frontier_queries=sum(q[4] == 1 for q in predictor.queries),
                complete_services_rescored=checked_services, planned_visits_checked=checked_visits,
                true_service_score_mismatches=0, query_or_causality_violations=0)


def _job(run_path, source_job):
    run = Path(run_path)
    job_directory = run / "jobs" / source_job.name
    job_directory.mkdir(parents=True)
    records, totals = [], dict(episodes=0, observations=0, queries=0, frontier_queries=0, complete_services_rescored=0, planned_visits_checked=0)
    nominal_cache = {}
    started = time.perf_counter()
    for physical in sorted(source_job.glob("condition_*")):
        for source_marker in sorted(physical.glob("phase*/phase_case.json")):
            case = load_protocol(source_marker)
            phase = case["phase_s"]
            truth_record, nominal_record = case["truth"]["manifest"], case["nominal"]["manifest"]
            if nominal_record["field_path"] not in nominal_cache:
                if sha256(nominal_record["field_path"]) != nominal_record["field_sha256"]:
                    raise AssertionError("Changed historical nominal field")
                nominal_cache[nominal_record["field_path"]] = load_field(nominal_record["field_path"])
            if sha256(truth_record["field_path"]) != truth_record["field_sha256"]:
                raise AssertionError("Changed historical true field")
            truth = mission_view(load_field(truth_record["field_path"]), phase)
            nominal = mission_view(nominal_cache[nominal_record["field_path"]], phase)
            for pair in case["pairs"]:
                if sha256(pair["nominal"]["episode_path"]) != pair["nominal"]["episode_sha256"]:
                    raise AssertionError("Changed historical baseline")
                old = load_protocol(pair["nominal"]["episode_path"])
                c = EpisodeConfig(**old["execution_record"]["config"])
                intervals = {tuple(t["cell"]): service_start_intervals(truth["times_s"],
                    truth["depths_m"][:, t["cell"][0] * c.map_shape[1] + t["cell"][1]].astype(float), c.service_duration_s,
                    c.max_service_depth_m) for t in old["targets"]}
                row = dict(family_id=case["family"]["family_id"], side=case["side"], phase_s=phase,
                           condition=case["condition"], layout_index=pair["layout_index"], delay_s=pair["delay_s"],
                           targets=c.n_targets, baseline=old["summary"], baseline_missed=missed_categories(old, intervals),
                           source_case=str(source_marker), baseline_episode=pair["nominal"], variants={})
                for name, (explore, multi, privileged) in VARIANTS.items():
                    directory = job_directory / physical.name / f"phase{phase:g}" / f"layout{pair['layout_index']}_delay{pair['delay_s']:g}" / name
                    directory.mkdir(parents=True)
                    predictor = QueryPredictor(truth["times_s"],
                        (truth if privileged else nominal)["depths_m"].reshape(-1, *c.map_shape),
                        horizon_s=c.forecast_horizon_s, service_duration_s=c.service_duration_s,
                        threshold_m=c.max_service_depth_m, privileged=privileged)
                    result = run_episode(truth["depths_m"].reshape(-1, *c.map_shape), c, times=truth["times_s"],
                                         predictor=predictor, planner=RollingPlanner(forecast_exploration=explore, multi_visit=multi))
                    if [t["cell"] for t in result["targets"]] != [t["cell"] for t in old["targets"]]:
                        raise AssertionError("Paired hidden layouts differ")
                    audit = audit_episode(result, predictor, truth, c)
                    result["execution_record"] = dict(config=asdict(c), privileged_future_truth=privileged,
                        formal_test=False, trained_model=False, training_data_eligible=not privileged,
                        hidden_target_map_available_to_planner=False, frontier_prediction_queries_enabled=explore,
                        source_case=str(source_marker), baseline_episode=pair["nominal"],
                        optimal_over_all_routes=False, diagnostic_not_H_E_P=True)
                    queries = np.asarray(predictor.queries, float).reshape(-1, 5)
                    query_path = directory / "prediction_queries.npz"
                    np.savez_compressed(query_path, time_s=queries[:, 0], row=queries[:, 1].astype(np.int16),
                        col=queries[:, 2].astype(np.int16), version=queries[:, 3].astype(np.int32),
                        role=queries[:, 4].astype(np.int8), publications=np.asarray(predictor.publications))
                    result["query_log"] = dict(path=str(query_path), sha256=sha256(query_path), count=len(queries),
                                               roles={"0": "known_task", "1": "public_frontier"})
                    result["observation_log"] = save_observations(directory / "observations.npz", result.pop("raw_observations"))
                    write_json(directory / "episode.json", result)
                    write_json(directory / "audit.json", audit)
                    row["variants"][name] = dict(summary=result["summary"], missed=missed_categories(result, intervals),
                        audit=audit, episode_path=str(directory / "episode.json"), episode_sha256=sha256(directory / "episode.json"),
                        query_sha256=result["query_log"]["sha256"], observation_sha256=result["observation_log"]["sha256"])
                    totals["episodes"] += 1
                    for key in audit:
                        if key in totals:
                            totals[key] += audit[key]
                records.append(row)
                write_json(job_directory / "progress.json", dict(pairs=len(records), episodes=totals["episodes"]))
    result = dict(job_id=source_job.name, status="completed", records=records, audit=totals, elapsed_wall_s=time.perf_counter() - started)
    write_json(job_directory / "job_result.json", result)
    return result


def summarize(run, jobs):
    rows = [row for job in jobs for row in job["records"]]
    groups = []
    for phase in (450., 600.):
        for side, delay in ((32, 0.), (50, 0.), (50, 10.), (70, 0.)):
            subset = [r for r in rows if (r["phase_s"], r["side"], r["delay_s"]) == (phase, side, delay)]
            if not subset:
                continue
            family_ids = sorted({r["family_id"] for r in subset})
            group = dict(phase_s=phase, side=side, delay_s=delay, pairs=len(subset), families=len(family_ids), variants={})
            def rate(method, family):
                selected = [r for r in subset if r["family_id"] == family]
                return float(np.mean([(r["baseline"] if method == "baseline" else r["variants"][method]["summary"])["successful_services"] / r["targets"] for r in selected]))
            group["baseline_rate"] = float(np.mean([rate("baseline", f) for f in family_ids]))
            for name in VARIANTS:
                means = [rate(name, f) for f in family_ids]
                changes = [r["variants"][name]["summary"]["successful_services"] - r["baseline"]["successful_services"] for r in subset]
                group["variants"][name] = dict(rate=float(np.mean(means)), delta_vs_baseline_pp=100 * (float(np.mean(means)) - group["baseline_rate"]),
                    positive_pairs=sum(d > 0 for d in changes), negative_pairs=sum(d < 0 for d in changes), unchanged_pairs=sum(d == 0 for d in changes),
                    positive_families=sum(rate(name, f) > rate("baseline", f) + 1e-12 for f in family_ids),
                    timeouts=sum(r["variants"][name]["summary"]["planning_timeouts"] for r in subset),
                    missed={key: sum(r["variants"][name]["missed"][key] for r in subset) for key in subset[0]["baseline_missed"]})
            group["matched_combined_truth_minus_nominal_pp"] = 100 * (group["variants"]["combined_truth_diagnostic"]["rate"] - group["variants"]["combined"]["rate"])
            groups.append(group)
    totals = {key: sum(job["audit"][key] for job in jobs) for key in (jobs[0]["audit"] if jobs else [])}
    summary = dict(status="completed" if len(rows) == 1152 else "partial", completed_jobs=len(jobs), pairs=len(rows), episodes=totals.get("episodes", 0),
        groups=groups, audit=totals, formal_test_episodes=0, trained_H_E_P=False, phase_selected=None,
        all_retained_conditions_included=len(rows) == 1152, filtering_by_outcome=False,
        algorithm="bounded beam route generation, binary disjoint route selection, first-visit execution, 60s published forecasts",
        limitations=["candidate route pool is incomplete", "route-end coverage is a proxy for future exploration",
                     "truth diagnostic is privileged information, not P", "not trained or a formal frozen experiment"])
    write_json(Path(run) / "planner_replay_summary.json", summary)
    return summary


def run_replay(root, source, workers=6):
    root, source = Path(root).resolve(), Path(source).resolve()
    source_summary = load_protocol(source / "phase_matrix_summary.json")
    if source_summary["status"] != "completed" or source_summary["completed_pairs"] != 1152:
        raise ValueError("Full retained design matrix required")
    run = new_run(root, "planner_replay", dict(source=str(source), workers=workers))
    jobs = sorted((source / "jobs").glob("*_side*"))
    write_json(run / "replay_spec.json", dict(source=str(source), source_summary_sha256=sha256(source / "phase_matrix_summary.json"),
        planned_pairs=1152, planned_episodes=4608, variants=VARIANTS, jobs=[p.name for p in jobs],
        selection="all retained design families, phases, scales, conditions, layouts and delays", outcomes_filtered=False,
        planner_budget_s=.1, forecast_horizon_s=60, formal_test=False, selected_phase=None))
    print(f"Replay directory: {run}", flush=True)
    completed, errors, started = [], [], time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_job, str(run), job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    completed.append(future.result())
                    print(f"Replay {len(completed)}/{len(jobs)} jobs: {job.name}", flush=True)
                except BaseException as error:
                    errors.append(dict(job=job.name, error=repr(error)))
                    print(f"Replay failed: {job.name}: {error!r}", flush=True)
                summary = summarize(run, completed)
                write_json(run / "progress.json", dict(status="running", completed_jobs=len(completed), pairs=summary["pairs"], errors=errors))
        summary = summarize(run, completed)
        if errors or summary["status"] != "completed":
            raise RuntimeError(f"Replay incomplete: {errors!r}")
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(root / "results/latest_planner_replay.json", dict(run=str(run), status="completed", formal_test=False))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return run
    except BaseException as error:
        finish_run(root, run, status="failed", error=repr(error), elapsed_wall_s=time.perf_counter() - started)
        raise
