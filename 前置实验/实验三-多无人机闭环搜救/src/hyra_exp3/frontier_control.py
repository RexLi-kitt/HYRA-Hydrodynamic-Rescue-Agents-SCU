"""Matched frontier-pool control: remove water weighting, preserve everything else."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import json
import time
import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .diagnostics import load_field
from .phase_probe import mission_view
from .planner_replay import QueryPredictor, audit_episode, missed_categories
from .planning import RollingPlanner
from .protocol import load_protocol
from .records import save_observations
from .simulation import EpisodeConfig, run_episode
from .windows import WindowEstimate, service_start_intervals


class NeutralFrontierPredictor(QueryPredictor):
    """Known-task forecasts are unchanged; frontier water weights are neutral.

    Synthetic zero-depth curves mean weight one inside published support, and
    the same unknown weight outside support. They are not hydraulic estimates.
    No hidden target positions or true water values are used here.
    """
    def estimate_many(self, cells, now):
        if self.issued_at is None or now < self.issued_at - 1e-9:
            raise ValueError("Forecast not published yet")
        self.queries.extend((now, *cell, self.version, 1) for cell in cells)
        end = self.valid_until - self.service_duration_s
        intervals = ((self.issued_at, end),) if end >= self.issued_at - 1e-9 else ()
        stamps = (self.issued_at, self.valid_until) if self.valid_until > self.issued_at else (self.issued_at,)
        return [WindowEstimate(tuple(map(int, cell)), self.issued_at, self.version,
            stamps, (0.,) * len(stamps), intervals, False, self.valid_until,
            evidence="synthetic_neutral_frontier_weight_not_a_water_prediction", service_duration_s=self.service_duration_s) for cell in cells]


def _job(run_path, job_path):
    source = load_protocol(job_path)
    destination = Path(run_path) / "jobs" / source["job_id"]
    destination.mkdir(parents=True)
    results, audit_totals, nominal_cache = [], {}, {}
    truth_path, full_truth = None, None
    for row in source["records"]:
        original = row["variants"]["combined"]
        if sha256(original["episode_path"]) != original["episode_sha256"]:
            raise AssertionError("Combined controller record changed")
        combined = load_protocol(original["episode_path"])
        config = EpisodeConfig(**combined["execution_record"]["config"])
        case = load_protocol(row["source_case"])
        true_info, nom_info = case["truth"]["manifest"], case["nominal"]["manifest"]
        if truth_path != true_info["field_path"]:
            truth_path = true_info["field_path"]
            if sha256(truth_path) != true_info["field_sha256"]:
                raise AssertionError("True field changed")
            full_truth = load_field(truth_path)
        if nom_info["field_path"] not in nominal_cache:
            if sha256(nom_info["field_path"]) != nom_info["field_sha256"]:
                raise AssertionError("Nominal field changed")
            nominal_cache[nom_info["field_path"]] = load_field(nom_info["field_path"])
        truth = mission_view(full_truth, row["phase_s"])
        nominal = mission_view(nominal_cache[nom_info["field_path"]], row["phase_s"])
        predictor = NeutralFrontierPredictor(nominal["times_s"], nominal["depths_m"].reshape(-1, *config.map_shape),
            horizon_s=config.forecast_horizon_s, service_duration_s=config.service_duration_s, threshold_m=config.max_service_depth_m)
        result = run_episode(truth["depths_m"].reshape(-1, *config.map_shape), config, times=truth["times_s"],
                             predictor=predictor, planner=RollingPlanner())
        if [t["cell"] for t in result["targets"]] != [t["cell"] for t in combined["targets"]]:
            raise AssertionError("Control target pairing differs")
        audit = audit_episode(result, predictor, truth, config)
        for key in ("observations", "queries", "frontier_queries", "complete_services_rescored", "planned_visits_checked"):
            audit_totals[key] = audit_totals.get(key, 0) + audit[key]
        intervals = {tuple(t["cell"]): service_start_intervals(truth["times_s"],
            truth["depths_m"][:, t["cell"][0] * config.map_shape[1] + t["cell"][1]].astype(float),
            config.service_duration_s, config.max_service_depth_m) for t in result["targets"]}
        directory = destination / f"pair{len(results):03d}"
        directory.mkdir()
        result["summary"]["method"] = "same_route_and_frontier_candidates_without_frontier_water_weight"
        result["execution_record"] = dict(config=asdict(config), known_target_forecasts="unchanged_independent_nominal",
            frontier_water_weight="neutral_1_inside_support_same_unknown_outside", same_candidate_generation_and_routes=True,
            privileged_future_truth=False, formal_test=False, trained_model=False, training_data_eligible=False,
            matched_combined=original, synthetic_frontier_curves_not_predictions=True)
        queries = np.asarray(predictor.queries, float).reshape(-1, 5)
        path = directory / "prediction_queries.npz"
        np.savez_compressed(path, time_s=queries[:, 0], row=queries[:, 1].astype(np.int16), col=queries[:, 2].astype(np.int16),
                            version=queries[:, 3].astype(np.int32), role=queries[:, 4].astype(np.int8), publications=np.asarray(predictor.publications))
        result["query_log"] = dict(path=str(path), sha256=sha256(path), count=len(queries))
        result["observation_log"] = save_observations(directory / "observations.npz", result.pop("raw_observations"))
        write_json(directory / "episode.json", result)
        write_json(directory / "audit.json", audit)
        results.append(dict(family_id=row["family_id"], side=row["side"], phase_s=row["phase_s"], delay_s=row["delay_s"],
            targets=row["targets"], source_combined=original, neutral_summary=result["summary"],
            missed=missed_categories(result, intervals), audit=audit,
            episode_path=str(directory / "episode.json"), episode_sha256=sha256(directory / "episode.json")))
        write_json(destination / "progress.json", dict(pairs=len(results)))
    output = dict(job_id=source["job_id"], records=results, audit=audit_totals)
    write_json(destination / "job_result.json", output)
    return output


def run_control(root, replay, workers=6):
    root, replay = Path(root), Path(replay)
    summary = load_protocol(replay / "planner_replay_summary.json")
    if summary["status"] != "completed":
        raise ValueError("Full replay required")
    for filename in ("planning.py", "simulation.py", "windows.py", "planner_replay.py"):
        if sha256(root / "src/hyra_exp3" / filename) != sha256(replay / "source_snapshot/src/hyra_exp3" / filename):
            raise AssertionError("Matched control requires identical execution code")
    run = new_run(root, "matched_frontier_control", dict(source=str(replay), workers=workers))
    write_json(run / "control_spec.json", dict(planned_pairs=1152, source=str(replay),
        source_summary_sha256=sha256(replay / "planner_replay_summary.json"),
        only_change="neutralize_frontier_water_weights_keep_candidates_known_task_forecasts_and_route_planner", formal_test=False))
    print(f"Matched control directory: {run}", flush=True)
    completed, started = [], time.perf_counter()
    try:
        paths = sorted((replay / "jobs").glob("*/job_result.json"))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, str(run), path) for path in paths]
            for future in as_completed(futures):
                completed.append(future.result())
                print(f"Matched control {len(completed)}/{len(paths)} jobs", flush=True)
        records = [r for job in completed for r in job["records"]]
        if len(records) != 1152:
            raise AssertionError("Full matched control incomplete")
        groups = []
        for phase in (450., 600.):
            for side, delay in ((32, 0.), (50, 0.), (50, 10.), (70, 0.)):
                subset = [r for r in records if (r["phase_s"], r["side"], r["delay_s"]) == (phase, side, delay)]
                differences = [(r["source_combined"]["summary"]["successful_services"] - r["neutral_summary"]["successful_services"]) / r["targets"] for r in subset]
                families = sorted({r["family_id"] for r in subset})
                family_differences = [float(np.mean([d for d, r in zip(differences, subset) if r["family_id"] == family])) for family in families]
                groups.append(dict(phase_s=phase, side=side, delay_s=delay, pairs=len(subset),
                    neutral_rate=float(np.mean([r["neutral_summary"]["successful_services"] / r["targets"] for r in subset])),
                    nominal_weighted_rate=float(np.mean([r["source_combined"]["summary"]["successful_services"] / r["targets"] for r in subset])),
                    nominal_water_weight_gain_pp=100 * float(np.mean(family_differences)), family_gain_pp=[100*d for d in family_differences],
                    positive_pairs=sum(d > 0 for d in differences), negative_pairs=sum(d < 0 for d in differences),
                    positive_families=sum(d > 1e-12 for d in family_differences),
                    neutral_timeouts=sum(r["neutral_summary"]["planning_timeouts"] for r in subset)))
        audit = {key: sum(j["audit"][key] for j in completed) for key in completed[0]["audit"]}
        result = dict(status="completed", pairs=len(records), episodes=len(records), groups=groups, audit=audit,
                      formal_test=False, trained_H_E_P=False, selected_phase=None)
        write_json(run / "matched_control_summary.json", result)
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(root / "results/latest_matched_frontier_control.json", dict(run=str(run), status="completed"))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return run
    except BaseException as error:
        finish_run(root, run, status="failed", error=repr(error), elapsed_wall_s=time.perf_counter() - started)
        raise
