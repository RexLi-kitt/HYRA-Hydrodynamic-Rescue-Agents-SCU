"""Independent full design-grid spatial and output-clock sensitivity checks."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gc
from pathlib import Path
import time
import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .design import _cached_field
from .diagnostics import compare_fields, load_field
from .execution_plan import register
from .phase_probe import mission_view
from .protocol import load_protocol


def metrics(coarse, reference):
    result = compare_fields(coarse, reference)
    indexes = np.searchsorted(reference["times_s"], coarse["times_s"])
    errors = np.abs(coarse["depths_m"].astype(float) - reference["depths_m"][indexes].astype(float))
    result["depth_p99_abs_m"] = float(np.quantile(errors, .99))
    return result


def passes(result, criteria):
    return all(result[key] <= criteria[key] for key in (
        "depth_mae_m", "depth_p99_abs_m", "complete_window_disagreement_fraction"))


def _job(run_path, source_job, criteria):
    root = Path(run_path) / "jobs" / source_job.name
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "job_result.json"
    if marker.exists():
        return load_protocol(marker)
    rows, nominal_cache, started = [], {}, time.perf_counter()
    with (root / "solver.log").open("a", encoding="utf-8") as stream, contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        for physical in sorted(source_job.glob("condition_*")):
            case = load_protocol(physical / "phase600/phase_case.json")
            case_marker = root / physical.name / "refinement_case.json"
            if case_marker.exists():
                rows.append(load_protocol(case_marker))
                continue
            row = dict(family=case["family"], side=case["side"], condition=case["condition"], source_case=str(physical), fields={})
            for kind in ("truth", "nominal"):
                info = case[kind]
                args = dict(info["arguments"])
                side = args.pop("side")
                coarse_path = info["manifest"]["field_path"]
                if sha256(coarse_path) != info["manifest"]["field_sha256"]:
                    raise AssertionError("Historical coarse field changed")
                key = (kind, args["rise_m"], args["rise_duration_s"])
                if kind == "nominal" and key in nominal_cache:
                    row["fields"][kind] = nominal_cache[key]
                    continue
                coarse = load_field(coarse_path)
                directory = root / (physical.name if kind == "truth" else f"nominal_r{args['rise_m']:g}_d{args['rise_duration_s']:g}")
                ref2_record, ref2 = _cached_field(directory / f"{kind}_mesh5_dt2", side,
                    **{**args, "mesh_spacing_m": 5., "output_interval_s": 2.})
                ref1_record, ref1 = _cached_field(directory / f"{kind}_mesh5_dt1", side,
                    **{**args, "mesh_spacing_m": 5., "output_interval_s": 1.})
                comparisons = []
                for phase in (0., 450., 600.):
                    a, b, c = (mission_view(field, phase) for field in (coarse, ref2, ref1))
                    spatial = metrics(a, b)
                    temporal = metrics(b, c)
                    joint = metrics(a, c)
                    comparisons.append(dict(phase_s=phase, spatial_10_vs_5=spatial, temporal_2_vs_1=temporal,
                        joint_10m2s_vs_5m1s=joint, tolerances_passed=all(passes(m, criteria) for m in (spatial, temporal, joint))))
                field_result = dict(coarse=info, reference_5m2s=ref2_record, reference_5m1s=ref1_record,
                                    comparisons=comparisons, tolerances_passed=all(c["tolerances_passed"] for c in comparisons))
                row["fields"][kind] = field_result
                if kind == "nominal":
                    nominal_cache[key] = field_result
                del coarse, ref2, ref1
                gc.collect()
            row["tolerances_passed"] = all(f["tolerances_passed"] for f in row["fields"].values())
            write_json(case_marker, row)
            rows.append(row)
            write_json(root / "progress.json", dict(cases=len(rows), planned_cases=12))
    result = dict(job_id=source_job.name, status="completed", cases=rows, elapsed_wall_s=time.perf_counter() - started)
    write_json(marker, result)
    return result


def run_refinement(root, source, workers=6, resume=None):
    root, source = Path(root).resolve(), Path(source).resolve()
    plan = register(root)
    hashes = {name: sha256(root / "src/hyra_exp3" / name) for name in (
        "refinement.py", "hydraulics.py", "diagnostics.py", "phase_probe.py", "design.py")}
    spec = dict(source=str(source), source_summary_sha256=sha256(source / "phase_matrix_summary.json"),
        criteria=plan["numerical_criteria"], cases=216, independent_new_fields=576, physical_duration_s=1800.,
        phases_s=[0., 450., 600.], source_hashes=hashes, selection="all_design_families_scales_and_hydrologies",
        no_model_outcome_filtering=True, formal_test=False)
    if resume is None:
        run = new_run(root, "design_refinement", dict(workers=workers, source=str(source)))
        write_json(run / "refinement_spec.json", spec)
    else:
        run = Path(resume).resolve()
        if load_protocol(run / "refinement_spec.json") != spec:
            raise ValueError("Resume code, source or tolerances differ")
    print(f"Refinement directory: {run}", flush=True)
    jobs = sorted((source / "jobs").glob("*_side*"))
    completed, errors, started = [], [], time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_job, str(run), job, spec["criteria"]): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    completed.append(future.result())
                    print(f"Refinement {len(completed)}/{len(jobs)} jobs: {job.name}", flush=True)
                except BaseException as error:
                    errors.append(dict(job=job.name, error=repr(error)))
                    print(f"Refinement job error {job.name}: {error!r}", flush=True)
                cases = [c for j in completed for c in j["cases"]]
                write_json(run / "progress.json", dict(completed_jobs=len(completed), cases=len(cases), errors=errors))
        cases = [c for j in completed for c in j["cases"]]
        summary = dict(status="completed" if len(cases) == 216 and not errors else "incomplete", cases=len(cases),
            tolerances_passed=bool(cases) and all(c["tolerances_passed"] for c in cases),
            failed_tolerance_cases=[dict(family=c["family"]["family_id"], side=c["side"], condition=c["condition"]) for c in cases if not c["tolerances_passed"]],
            criteria=spec["criteria"], errors=errors, formal_test=False,
            asymptotic_convergence_claimed=False, actual_model_gain_direction_not_yet_checked=True)
        write_json(run / "refinement_summary.json", summary)
        if summary["status"] != "completed":
            raise RuntimeError(f"Refinement incomplete: {errors!r}")
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(root / "results/latest_design_refinement.json", dict(run=str(run), **summary))
        print(summary, flush=True)
        return run
    except BaseException as error:
        finish_run(root, run, status="failed", error=repr(error))
        raise
