"""Full design-family validation of two preselected, unfrozen mission phases."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np

from .artifacts import environment, finish_run, new_run, peak_memory_bytes, sha256, write_json
from .design import design_jobs, execution_hashes, _cached_field, target_opportunities
from .phase_probe import geometric_upper_bound, mission_view
from .protocol import initialize, load_protocol
from .records import save_observations
from .simulation import EpisodeConfig, run_episode
from .windows import NominalPredictor

PHASES = [450., 600.]


class TruthWindowDiagnostic(NominalPredictor):
    """Privileged true forecast, queried by the planner only at discovered targets.

    This is an upper-information diagnostic, never H/E/P or a training dataset.
    It preserves the 60s publication horizon, sensing and hidden-target rules.
    The greedy planner is not guaranteed to attain an optimal oracle solution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queries = []

    def estimate(self, cell, now):
        estimate = super().estimate(cell, now)
        self.queries.append((float(now), int(cell[0]), int(cell[1]), self.version))
        return replace(estimate, evidence="privileged_future_truth_diagnostic_only", evidence_supported=True)


def oracle_episode(directory, truth, config):
    predictor = TruthWindowDiagnostic(truth["times_s"], truth["depths_m"].reshape(-1, *config.map_shape),
                                     horizon_s=config.forecast_horizon_s, service_duration_s=config.service_duration_s,
                                     threshold_m=config.max_service_depth_m)
    started = time.perf_counter()
    result = run_episode(truth["depths_m"].reshape(-1, *config.map_shape), config,
                         times=truth["times_s"], predictor=predictor)
    discovered = {tuple(t["cell"]): t["discovered_at"] for t in result["targets"]}
    for now, row, col, _ in predictor.queries:
        if (row, col) not in discovered or discovered[row, col] is None or discovered[row, col] > now + 1e-7:
            raise AssertionError("Privileged diagnostic queried an undiscovered target")
    query_path = directory / "oracle_queries.npz"
    queries = np.asarray(predictor.queries, float).reshape(-1, 4)
    np.savez_compressed(query_path, time_s=queries[:, 0], row=queries[:, 1].astype(np.int32),
                        col=queries[:, 2].astype(np.int32), version=queries[:, 3].astype(np.int32))
    result["summary"]["method"] = "truth_forecast_known_targets_60s_diagnostic"
    result["execution_record"] = dict(elapsed_wall_s=time.perf_counter() - started, config=asdict(config),
        label="privileged_diagnostic_not_H_E_P", formal_test=False, trained_model=False,
        privileged_future_truth=True, optimal_oracle=False, training_data_eligible=False,
        all_queries_after_discovery=True, query_log=dict(path=str(query_path), sha256=sha256(query_path), count=len(queries)))
    result["observation_log"] = save_observations(directory / "observations.npz", result.pop("raw_observations"))
    write_json(directory / "episode.json", result)
    write_json(directory / "episode_storage.json", dict(
        episode_json_bytes=(directory / "episode.json").stat().st_size,
        observation_npz_bytes=result["observation_log"]["bytes"],
        total_episode_log_bytes=(directory / "episode.json").stat().st_size + result["observation_log"]["bytes"] + query_path.stat().st_size))
    return result


def _load_or_run(directory, truth, nominal, delay, seed, oracle=False, config=None):
    from .pipeline import _episode
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "result_record.json"
    if marker.exists():
        record = load_protocol(marker)
        if sha256(record["episode_path"]) != record["episode_sha256"]:
            raise ValueError(f"Episode integrity differs: {directory}")
        result = load_protocol(record["episode_path"])
        if sha256(result["observation_log"]["path"]) != result["observation_log"]["sha256"]:
            raise ValueError(f"Observation integrity differs: {directory}")
        query = result["execution_record"].get("query_log")
        if query and sha256(query["path"]) != query["sha256"]:
            raise ValueError(f"Diagnostic query integrity differs: {directory}")
        return result, record
    destination = directory
    if any(directory.iterdir()):
        import uuid
        destination = directory / "recovery_attempts" / uuid.uuid4().hex
        destination.mkdir(parents=True)
    result = oracle_episode(destination, truth, config) if oracle else _episode(destination, truth, nominal, delay, seed)
    record = dict(episode_path=str(destination / "episode.json"), episode_sha256=sha256(destination / "episode.json"))
    write_json(marker, record)
    return result, record


def _job(run_path, job):
    directory = Path(run_path) / "jobs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    completed, started = [], time.perf_counter()
    try:
        with (directory / "solver.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            nominal_cache = {}
            for index, condition in enumerate(job["conditions"]):
                key = (condition["rise_m"], condition["rise_duration_s"])
                common = dict(duration_s=1800., output_interval_s=2., mesh_spacing_m=10.,
                              family_seed=job["family"]["family_seed"], **condition)
                if key not in nominal_cache:
                    nominal_cache[key] = _cached_field(directory / f"nominal_r{key[0]:g}_d{key[1]:g}", job["side"],
                        **{**common, "perturbation_m": 0., "nominal": True})
                nominal_record, full_nominal = nominal_cache[key]
                physical_directory = directory / f"condition_{index:02d}"
                physical_directory.mkdir(exist_ok=True)
                phase_markers = [physical_directory / f"phase{phase:g}" / "phase_case.json" for phase in PHASES]
                if all(p.exists() for p in phase_markers):
                    completed.extend(map(str, phase_markers))
                    continue
                truth_record, full_truth = _cached_field(physical_directory / "truth", job["side"], nominal=False, **common)
                for phase, marker in zip(PHASES, phase_markers):
                    if marker.exists():
                        completed.append(str(marker))
                        continue
                    phase_directory = marker.parent
                    phase_directory.mkdir(exist_ok=True)
                    truth, nominal = mission_view(full_truth, phase), mission_view(full_nominal, phase)
                    pairs = []
                    for layout, seed in enumerate(job["layout_seeds"]):
                        for delay in job["delays_s"]:
                            pair_directory = phase_directory / f"layout{layout}_delay{delay:g}"
                            nominal_episode, nominal_info = _load_or_run(pair_directory / "nominal", truth, nominal, delay, seed)
                            config = EpisodeConfig(**nominal_episode["execution_record"]["config"])
                            oracle, oracle_info = _load_or_run(pair_directory / "truth_diagnostic", truth, nominal, delay, seed,
                                                              oracle=True, config=config)
                            if [t["cell"] for t in nominal_episode["targets"]] != [t["cell"] for t in oracle["targets"]]:
                                raise AssertionError("Diagnostic pair uses different hidden targets")
                            target_rows = target_opportunities(truth, nominal_episode)
                            bounds = geometric_upper_bound(target_rows, (job["side"], job["side"]), config.n_drones)
                            nominal_count = nominal_episode["summary"]["successful_services"]
                            oracle_count = oracle["summary"]["successful_services"]
                            if max(nominal_count, oracle_count) > bounds["optimistic_individually_feasible_targets"]:
                                raise AssertionError("Executed services exceed individual feasibility upper bound")
                            pairs.append(dict(layout_index=layout, layout_seed=seed, delay_s=delay, bounds=bounds,
                                nominal_summary=nominal_episode["summary"], oracle_summary=oracle["summary"],
                                achieved_oracle_minus_nominal_services=oracle_count - nominal_count,
                                achieved_oracle_minus_nominal_pp=100. * (oracle_count - nominal_count) / config.n_targets,
                                optimistic_remaining_services=bounds["optimistic_individually_feasible_targets"] - nominal_count,
                                nominal_targets=target_rows, nominal=nominal_info, oracle=oracle_info))
                    write_json(marker, dict(family=job["family"], side=job["side"], condition=condition, phase_s=phase,
                        mission_duration_s=1200., truth=truth_record, nominal=nominal_record, pairs=pairs,
                        original_protocol_changed=False, formal_test=False, stage="unfrozen_design_candidate",
                        oracle_note="Privileged true forecast only at discovered targets, same 60s horizon and greedy planner; not an optimal oracle or trained method"))
                    completed.append(str(marker))
                    write_json(directory / "progress.json", dict(status="running", phase_cases=len(completed), planned_phase_cases=24))
                    gc.collect()
                del full_truth
                gc.collect()
        result = dict(job_id=job["job_id"], status="completed", phase_cases=completed,
                      elapsed_wall_s=time.perf_counter() - started, peak_process_memory_bytes=peak_memory_bytes())
        write_json(directory / "job_result.json", result)
        write_json(directory / "progress.json", dict(status="completed", phase_cases=len(completed), planned_phase_cases=24))
        return result
    except BaseException as error:
        import traceback
        write_json(directory / "job_error.json", dict(error=repr(error), traceback=traceback.format_exc(), completed_phase_cases=len(completed)))
        raise


def summarize(run):
    run = Path(run)
    records, families = [], {}
    paths = sorted((run / "jobs").glob("*/condition_*/phase*/phase_case.json"))
    for path in paths:
        case = load_protocol(path)
        for pair in case["pairs"]:
            targets = pair["nominal_summary"]["hidden_targets"]
            record = dict(family_id=case["family"]["family_id"], side=case["side"], phase_s=case["phase_s"],
                condition=case["condition"], layout_index=pair["layout_index"], delay_s=pair["delay_s"],
                targets=targets, nominal=pair["nominal_summary"]["successful_services"], oracle=pair["oracle_summary"]["successful_services"],
                feasible_upper=pair["bounds"]["optimistic_individually_feasible_targets"],
                any_window=pair["bounds"]["targets_with_any_complete_window"],
                delta_pp=pair["achieved_oracle_minus_nominal_pp"],
                nominal_timeouts=pair["nominal_summary"]["planning_timeouts"], oracle_timeouts=pair["oracle_summary"]["planning_timeouts"])
            records.append(record)
            key = (record["family_id"], record["side"], record["phase_s"], record["delay_s"])
            families.setdefault(key, []).append(record)
    family_means = []
    for (family, side, phase, delay), rows in sorted(families.items()):
        family_means.append(dict(family_id=family, side=side, phase_s=phase, delay_s=delay, pairs=len(rows),
            nominal_rate=float(np.mean([r["nominal"] / r["targets"] for r in rows])),
            oracle_rate=float(np.mean([r["oracle"] / r["targets"] for r in rows])),
            any_window_rate=float(np.mean([r["any_window"] / r["targets"] for r in rows])),
            feasible_upper_rate=float(np.mean([r["feasible_upper"] / r["targets"] for r in rows])),
            achieved_oracle_delta_pp=float(np.mean([r["delta_pp"] for r in rows]))))
    groups = []
    for phase in PHASES:
        for side, delay in ((32, 0.), (50, 0.), (50, 10.), (70, 0.)):
            rows = [r for r in family_means if (r["side"], r["phase_s"], r["delay_s"]) == (side, phase, delay)]
            paired = [r for r in records if (r["side"], r["phase_s"], r["delay_s"]) == (side, phase, delay)]
            if rows:
                groups.append(dict(side=side, phase_s=phase, delay_s=delay, families=len(rows), pairs=len(paired),
                    nominal_rate=float(np.mean([r["nominal_rate"] for r in rows])),
                    oracle_rate=float(np.mean([r["oracle_rate"] for r in rows])),
                    feasible_upper_rate=float(np.mean([r["feasible_upper_rate"] for r in rows])),
                    any_window_rate=float(np.mean([r["any_window_rate"] for r in rows])),
                    achieved_oracle_delta_pp=float(np.mean([r["achieved_oracle_delta_pp"] for r in rows])),
                    nominal_ceiling_pairs=sum(r["nominal"] == r["targets"] for r in paired),
                    positive_oracle_pairs=sum(r["oracle"] > r["nominal"] for r in paired),
                    negative_oracle_pairs=sum(r["oracle"] < r["nominal"] for r in paired),
                    nominal_timeouts=sum(r["nominal_timeouts"] for r in paired), oracle_timeouts=sum(r["oracle_timeouts"] for r in paired)))
    spec = load_protocol(run / "phase_matrix_spec.json")
    summary = dict(status="completed" if len(paths) == spec["planned_phase_cases"] and len(records) == spec["planned_pairs"] else "partial",
        completed_phase_cases=len(paths), planned_phase_cases=spec["planned_phase_cases"], completed_pairs=len(records),
        planned_pairs=spec["planned_pairs"], completed_episodes=2 * len(records), formal_test=False,
        globally_selected_phase=None, trained_H_E_P=False, numerical_convergence_gate_passed=False,
        groups=groups, family_means=family_means, records=records,
        interpretation="Achieved privileged forecast diagnostic, not P-vs-B* and not an optimal joint oracle")
    write_json(run / "phase_matrix_summary.json", summary)
    return summary


def run_matrix(root, workers=6, resume=None):
    root = Path(root).resolve()
    protocol = initialize(root)
    jobs = design_jobs(protocol)
    packages = environment(root)["packages"]
    hashes = execution_hashes(root)
    hashes.update({name: sha256(root / "src" / "hyra_exp3" / name) for name in ("phase_matrix.py", "phase_probe.py", "artifacts.py")})
    hashes["entrypoint_run.py"] = sha256(root / "run.py")
    specification = dict(study_id=protocol["study_id"], stage="unfrozen_phase_design_validation", jobs=jobs,
        phases_s=PHASES, mission_duration_s=1200., physical_duration_s=1800., output_interval_s=2., mesh_spacing_m=10.,
        planned_phase_cases=432, planned_pairs=1152, planned_nominal_episodes=1152, planned_privileged_diagnostic_episodes=1152,
        source_sha256=hashes, numerical_environment=dict(python=sys.version, numpy=packages["numpy"], scipy=packages["scipy"], anuga=packages["anuga"]),
        formal_test=False, original_protocol_changed=False, phase_selected=False,
        oracle="true_future_only_for_discovered_targets_same_60s_horizon_not_optimal_not_training_data",
        selection="full_design_family_condition_scale_layout_delay_cross_no_outcome_filtering")
    if resume is None:
        run = new_run(root, "phase_matrix", dict(workers=workers, phases_s=PHASES))
        write_json(run / "phase_matrix_spec.json", specification)
    else:
        run = Path(resume).resolve()
        if not run.is_relative_to(root / "results" / "runs") or load_protocol(run / "phase_matrix_spec.json") != specification:
            raise ValueError("Resume requires matching execution code, numerical environment and registered candidate matrix")
        import uuid
        write_json(run / "resume_attempts" / (uuid.uuid4().hex + ".json"), dict(workers=workers))
    print(f"Phase matrix directory: {run}", flush=True)
    started = time.perf_counter()
    completed, errors, pending = [], [], []
    for job in jobs:
        result_path = run / "jobs" / job["job_id"] / "job_result.json"
        if result_path.exists():
            completed.append(load_protocol(result_path))
        else:
            pending.append(job)
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_job, str(run), job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    completed.append(future.result())
                    print(f"Phase matrix {len(completed)}/{len(jobs)} jobs: {job['job_id']}", flush=True)
                except BaseException as error:
                    errors.append(dict(job_id=job["job_id"], error=repr(error)))
                    print(f"Phase job failed {job['job_id']}: {error!r}", flush=True)
                summary = summarize(run)
                write_json(run / "matrix_progress.json", dict(status="running", completed_jobs=len(completed),
                           planned_jobs=len(jobs), errors=errors, completed_phase_cases=summary["completed_phase_cases"], completed_pairs=summary["completed_pairs"]))
        summary = summarize(run)
        if errors or summary["status"] != "completed":
            raise RuntimeError(f"Phase candidate matrix incomplete: {errors!r}")
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(run / "matrix_progress.json", dict(status="completed", completed_jobs=len(completed), planned_jobs=len(jobs), errors=[]))
        write_json(root / "results" / "latest_phase_matrix.json", dict(run=str(run), status="completed", formal_test=False))
        print(json.dumps({k: summary[k] for k in ("status", "completed_phase_cases", "completed_pairs", "completed_episodes", "groups")}, ensure_ascii=False, indent=2))
        return run
    except BaseException as error:
        import traceback
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(root, run, status="failed", error=repr(error), elapsed_wall_s=time.perf_counter() - started)
        raise
