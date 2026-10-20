"""Full, outcome-independent design-set opportunity check with resumable jobs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from .artifacts import peak_memory_bytes, sha256, write_json
from .diagnostics import load_field, window_distribution
from .protocol import family_split, load_protocol
from .windows import service_start_intervals

EXECUTION_FILES = ["hydraulics.py", "simulation.py", "windows.py", "diagnostics.py",
                   "records.py", "protocol.py", "pipeline.py", "design.py"]


def execution_hashes(root):
    return {name: sha256(Path(root) / "src" / "hyra_exp3" / name) for name in EXECUTION_FILES}


def design_jobs(protocol):
    families = [r for r in family_split(protocol)["families"] if r["split"] == "design"]
    conditions = [dict(rise_m=r, rise_duration_s=t, perturbation_m=p)
                  for r, t, p in itertools.product(protocol["hydrology"]["rise_m"],
                      protocol["hydrology"]["rise_duration_s"], protocol["hydrology"]["local_perturbation_m"])]
    jobs = []
    for family in families:
        for side in (70, 50, 32):
            layout_seeds = [int.from_bytes(hashlib.sha256(
                f"{protocol['study_id']}:{family['family_id']}:layout_noise:{i}".encode()).digest()[:4], "big")
                for i in range(protocol["evaluation"]["layout_noise_configs"])]
            jobs.append(dict(job_id=f"{family['family_id']}_side{side}", family=family,
                             side=side, conditions=conditions, layout_seeds=layout_seeds,
                             delays_s=[0., 10.] if side == 50 else [0.]))
    return jobs


def target_opportunities(truth, episode, service_duration=12., threshold=.8):
    """Post-episode oracle diagnostics, never fed to the planner or target sampler."""
    output = []
    width = episode["execution_record"]["config"]["map_shape"][1]
    end = float(truth["times_s"][-1])
    for row in episode["targets"]:
        cell = tuple(row["cell"])
        curve = truth["depths_m"][:, cell[0] * width + cell[1]]
        intervals = service_start_intervals(truth["times_s"], curve, service_duration, threshold)
        start = None if row["served_at"] is None else row["served_at"] - service_duration
        containing = next(((a, b) for a, b in intervals if start is not None and a - 1e-7 <= start <= b + 1e-7), None)
        output.append(dict(
            target=row["id"], cell=row["cell"], discovered_at_s=row["discovered_at"], served_at_s=row["served_at"],
            service_start_s=start, truth_safe_start_intervals_s=[list(p) for p in intervals],
            safe_start_margin_after_actual_service_s=None if containing is None else max(0., containing[1] - start),
            closing_time_censored_by_episode_end=bool(containing and abs(containing[1] - (end - service_duration)) < 1e-7),
            attempts=row["attempts"], failures=row["failures"],
            diagnosis_uses_truth_after_execution_only=True,
        ))
    return output


def _cached_field(directory, side, **kwargs):
    from .hydraulics import generate_field
    directory = Path(directory)
    record_path = directory / "cache_record.json"
    if record_path.exists():
        record = load_protocol(record_path)
        if record["arguments"] != dict(side=side, **kwargs):
            raise ValueError(f"Cached field specification differs: {directory}")
        path = Path(record["manifest"]["field_path"])
        if sha256(path) != record["manifest"]["field_sha256"]:
            raise ValueError(f"Cached field checksum differs: {directory}")
        return record, load_field(path)
    # Recover a field completed just before a process stopped, without overwriting
    # an existing solver result. Unknown partial output is preserved and rejected.
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = load_protocol(manifest_path)
        path = Path(manifest["field_path"])
        if sha256(path) != manifest["field_sha256"]:
            raise ValueError(f"Incomplete field checksum: {directory}")
    else:
        if (directory / "field.npz").exists():
            import uuid
            # Preserve an interrupted, uncommitted output and generate into a
            # fresh child directory rather than overwrite the original bytes.
            destination = directory / "recovery_attempts" / uuid.uuid4().hex
        else:
            destination = directory
        manifest = generate_field(side, destination, **kwargs)
    record = dict(arguments=dict(side=side, **kwargs), manifest=manifest)
    write_json(record_path, record)
    gc.collect()
    return record, load_field(manifest["field_path"])


def _run_job(run_path, job):
    from .pipeline import _episode
    directory = Path(run_path) / "jobs" / job["job_id"]
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    case_paths = []
    try:
        with (directory / "solver.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            nominal_cache = {}
            for condition_index, condition in enumerate(job["conditions"]):
                key = (condition["rise_m"], condition["rise_duration_s"])
                common = dict(duration_s=1200., output_interval_s=2., mesh_spacing_m=10.,
                              family_seed=job["family"]["family_seed"], **condition)
                if key not in nominal_cache:
                    nominal_args = {**common, "perturbation_m": 0., "nominal": True}
                    nominal_directory = directory / f"nominal_r{key[0]:g}_d{key[1]:g}"
                    nominal_cache[key] = _cached_field(nominal_directory, job["side"], **nominal_args)
                nominal_record, nominal = nominal_cache[key]
                case_directory = directory / f"condition_{condition_index:02d}"
                summary_path = case_directory / "case_summary.json"
                if summary_path.exists():
                    case_paths.append(str(summary_path))
                    continue
                case_directory.mkdir(parents=True, exist_ok=True)
                truth_record, truth = _cached_field(case_directory / "truth", job["side"], nominal=False, **common)
                distributions = window_distribution(truth)
                episodes = []
                for layout_index, layout_seed in enumerate(job["layout_seeds"]):
                    for delay in job["delays_s"]:
                        episode_directory = case_directory / f"layout{layout_index}_delay{delay:g}"
                        episode_directory.mkdir(exist_ok=True)
                        completion = episode_directory / "design_episode_summary.json"
                        if completion.exists():
                            episode_summary = load_protocol(completion)
                        else:
                            episode_path = episode_directory / "episode.json"
                            storage_path = episode_directory / "episode_storage.json"
                            if episode_path.exists() and storage_path.exists():
                                result = load_protocol(episode_path)
                                observation_log = result["observation_log"]
                                if sha256(observation_log["path"]) != observation_log["sha256"]:
                                    raise ValueError(f"Observation checksum differs: {episode_path}")
                            else:
                                destination = episode_directory
                                if any(episode_directory.iterdir()):
                                    import uuid
                                    destination = episode_directory / "recovery_attempts" / uuid.uuid4().hex
                                    destination.mkdir(parents=True)
                                result = _episode(destination, truth, nominal, delay, layout_seed)
                                episode_path = destination / "episode.json"
                                storage_path = destination / "episode_storage.json"
                            episode_summary = dict(layout_index=layout_index, layout_seed=layout_seed,
                                delay_s=delay, summary=result["summary"],
                                completion_headroom_percentage_points=100. * (1. - result["summary"]["success_rate"]),
                                targets=target_opportunities(truth, result),
                                episode_path=str(episode_path), storage=load_protocol(storage_path))
                            write_json(completion, episode_summary)
                        episodes.append(episode_summary)
                case = dict(family=job["family"], side=job["side"], condition=condition,
                            window_distribution=distributions, truth=truth_record, nominal=nominal_record,
                            episodes=episodes, formal_test=False,
                            baseline="nominal_only_greedy_not_trained_H_or_E", status="completed")
                write_json(summary_path, case)
                case_paths.append(str(summary_path))
                write_json(directory / "progress.json", dict(status="running", completed_conditions=len(case_paths),
                           planned_conditions=len(job["conditions"]), latest_condition=condition))
                del truth
                gc.collect()
        result = dict(job_id=job["job_id"], status="completed", cases=case_paths,
                      elapsed_wall_s=time.perf_counter() - started, peak_process_memory_bytes=peak_memory_bytes())
        write_json(directory / "job_result.json", result)
        write_json(directory / "progress.json", dict(status="completed", completed_conditions=len(case_paths),
                   planned_conditions=len(job["conditions"])))
        return result
    except BaseException as error:
        import traceback
        write_json(directory / "job_error.json", dict(error=repr(error), traceback=traceback.format_exc(), completed_conditions=len(case_paths)))
        raise


def summarize_design(run):
    run = Path(run)
    specification = load_protocol(run / "design_spec.json")
    cases = [load_protocol(path) for path in sorted((run / "jobs").glob("*/condition_*/case_summary.json"))]
    records, cells = [], {}
    for case in cases:
        for episode in case["episodes"]:
            key = (case["family"]["family_id"], case["side"], episode["delay_s"])
            cells.setdefault(key, []).append(episode["summary"]["success_rate"])
            records.append(dict(family_id=case["family"]["family_id"], side=case["side"], condition=case["condition"],
                                layout_index=episode["layout_index"], delay_s=episode["delay_s"],
                                success_rate=episode["summary"]["success_rate"],
                                planning_timeouts=episode["summary"]["planning_timeouts"],
                                headroom_pp=episode["completion_headroom_percentage_points"]))
    family_means = [dict(family_id=f, side=s, delay_s=d, episodes=len(values), mean_success_rate=float(np.mean(values)),
                         completion_ceiling_headroom_pp=100 * (1 - float(np.mean(values))))
                    for (f, s, d), values in sorted(cells.items())]
    grouped = []
    for side, delay in ((32, 0.), (50, 0.), (50, 10.), (70, 0.)):
        rows = [r for r in family_means if r["side"] == side and r["delay_s"] == delay]
        episodes = [r for r in records if r["side"] == side and r["delay_s"] == delay]
        if rows:
            grouped.append(dict(side=side, delay_s=delay, families=len(rows), episodes=len(episodes),
                                family_mean_success_rate=float(np.mean([r["mean_success_rate"] for r in rows])),
                                ceiling_episodes=sum(r["success_rate"] == 1. for r in episodes),
                                mean_headroom_pp=float(np.mean([r["completion_ceiling_headroom_pp"] for r in rows])),
                                planning_timeouts=sum(r["planning_timeouts"] for r in episodes)))
    summary = dict(status="completed" if len(cases) == specification["planned_cases"] else "partial",
                   completed_cases=len(cases), planned_cases=specification["planned_cases"],
                   completed_episodes=len(records), planned_episodes=specification["planned_episodes"],
                   cases_with_zero_completion_headroom=sum(all(e["summary"]["success_rate"] == 1. for e in c["episodes"]) for c in cases),
                   all_completed_episodes_at_ceiling=bool(records) and all(r["success_rate"] == 1. for r in records),
                   family_means=family_means, by_scale_delay=grouped, episodes=records,
                   formal_test=False, H_E_P_trained=False, numerical_convergence_gate_passed=False,
                   interpretation="Completion headroom relative to an untrained nominal-only engineering baseline; not a P-vs-B* effect estimate")
    write_json(run / "design_summary.json", summary)
    return summary


def run_design(root, run, workers=4):
    root, run = Path(root), Path(run)
    protocol = load_protocol(root / "configs" / "experiment3.v1.json")
    jobs = design_jobs(protocol)
    specification = dict(study_id=protocol["study_id"], execution_source_sha256=execution_hashes(root), jobs=jobs,
                         planned_cases=sum(len(j["conditions"]) for j in jobs),
                         planned_episodes=sum(len(j["conditions"]) * len(j["layout_seeds"]) * len(j["delays_s"]) for j in jobs),
                         selection="full_registered_design_matrix_no_outcome_filtering", formal_test=False,
                         mesh_spacing_m=10., output_interval_s=2., duration_s=1200.)
    spec_path = run / "design_spec.json"
    if spec_path.exists():
        if load_protocol(spec_path) != specification:
            raise ValueError("Design specification or execution code changed; start a new run instead of mixing results")
    else:
        write_json(spec_path, specification)
    pending, completed, errors = [], [], []
    for job in jobs:
        path = run / "jobs" / job["job_id"] / "job_result.json"
        if path.exists():
            completed.append(load_protocol(path))
        else:
            pending.append(job)
    write_json(run / "design_progress.json", dict(completed_jobs=len(completed), planned_jobs=len(jobs), status="running"))
    # The parent has no solver state. Each spawned worker owns one family/scale
    # folder; a nominal field is reused only within that unique owner's job.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_job, str(run), job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                completed.append(result)
                print(f"Design {len(completed)}/{len(jobs)} jobs: {job['job_id']}, {result['elapsed_wall_s']:.1f}s", flush=True)
            except BaseException as error:
                errors.append(dict(job_id=job["job_id"], error=repr(error)))
                print(f"Design job failed: {job['job_id']}: {error!r}", flush=True)
            write_json(run / "design_progress.json", dict(completed_jobs=len(completed), planned_jobs=len(jobs),
                       failed_jobs=errors, status="running"))
            summarize_design(run)
    summary = summarize_design(run)
    write_json(run / "design_progress.json", dict(completed_jobs=len(completed), planned_jobs=len(jobs), failed_jobs=errors,
               status="completed" if not errors and summary["status"] == "completed" else "partial"))
    if errors or summary["status"] != "completed":
        raise RuntimeError("Design matrix is incomplete; preserve errors and resume this run after resolving the cause")
    return summary
