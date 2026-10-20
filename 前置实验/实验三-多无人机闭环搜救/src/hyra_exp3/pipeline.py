"""Reproducible development runs. Formal H/E/P evaluation is deliberately absent."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .diagnostics import compare_fields, load_field, window_distribution


def _field_path(manifest, directory):
    if "field_path" in manifest:
        return Path(manifest["field_path"])
    candidates = list(Path(directory).glob("*.npz"))
    if len(candidates) != 1:
        raise ValueError("Generator must return field_path or save exactly one field NPZ")
    return candidates[0]


def _design_seed(root, requested):
    # The preflight may only use design families. Neither validation nor test
    # truth is accessed while developing the implementation.
    split = json.loads((root / "manifests" / "family_split.json").read_text(encoding="utf-8"))
    if isinstance(split, dict) and "families" in split:
        families = split["families"]
    elif isinstance(split, list):
        families = split
    else:
        raise ValueError("Unsupported family split schema")
    design = [row for row in families if row["split"] == "design"]
    seeds = [int(row.get("family_seed", row.get("seed"))) for row in design]
    if requested is None:
        return seeds[0]
    if requested not in seeds:
        raise ValueError("This development entry point accepts only a registered design-family seed")
    return requested


def _generate(run, side, name, **kwargs):
    from .hydraulics import generate_field
    directory = run / name
    directory.mkdir(parents=True)
    print(f"ANUGA {name}: {side}x{side}, {kwargs}", flush=True)
    started = time.perf_counter()
    manifest = generate_field(side, directory, **kwargs)
    path = _field_path(manifest, directory)
    record = dict(manifest=manifest, elapsed_wall_s=time.perf_counter() - started,
                  field_path=str(path), field_sha256=sha256(path), field_bytes=path.stat().st_size)
    write_json(directory / "generation_record.json", record)
    print(f"Saved {name}: {record['elapsed_wall_s']:.2f}s, {record['field_bytes'] / 2**20:.2f} MiB", flush=True)
    return record, load_field(path)


def _episode(run, truth, nominal, delay, layout_seed):
    from .simulation import EpisodeConfig, run_episode
    if not np.array_equal(truth["times_s"], nominal["times_s"]):
        raise ValueError("Truth and nominal output clocks differ")
    if not np.allclose(truth["positions_m"], nominal["positions_m"], atol=1e-9, rtol=0):
        raise ValueError("Truth and nominal task grids differ")
    cells = truth["depths_m"].shape[1]
    side = int(round(cells ** .5))
    if side * side != cells or side not in (32, 50, 70):
        raise ValueError("Expected 1024, 2500, or 4900 task cells")
    drones = {32: 4, 50: 10, 70: 20}[side]
    config = EpisodeConfig(map_shape=(side, side), n_drones=drones, n_targets=drones * 3,
                           duration_s=float(truth["times_s"][-1]),
                           observation_delay_s=delay, seed=layout_seed)
    started = time.perf_counter()
    result = run_episode(truth["depths_m"].reshape(-1, side, side), config,
                         times=truth["times_s"], nominal_depths=nominal["depths_m"].reshape(-1, side, side))
    result["execution_record"] = dict(
        elapsed_wall_s=time.perf_counter() - started, config=asdict(config),
        label="nominal_only_engineering_preflight", trained_model=False,
        formal_test=False, hyra_effect_estimate=False,
    )
    from .records import save_observations
    result["observation_log"] = save_observations(run / "observations.npz", result.pop("raw_observations"))
    write_json(run / "episode.json", result)
    write_json(run / "episode_storage.json", dict(
        episode_json_bytes=(run / "episode.json").stat().st_size,
        observation_npz_bytes=result["observation_log"]["bytes"],
        total_episode_log_bytes=(run / "episode.json").stat().st_size + result["observation_log"]["bytes"],
    ))
    return result


def _audit(root, run):
    from .protocol import initialize, validate_protocol
    protocol = initialize(root)
    result = validate_protocol(protocol)
    write_json(run / "protocol_audit.json", result)
    return result


def execute(root, args):
    from .protocol import initialize
    initialize(root)
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if args.command == "design-check" and args.resume is not None:
        run = args.resume.resolve()
        if not run.is_relative_to((root / "results" / "runs").resolve()) or not (run / "design_spec.json").is_file():
            raise ValueError("Resume requires an existing design-check run within this experiment")
        import uuid
        write_json(run / "resume_attempts" / (uuid.uuid4().hex + ".json"), arguments)
    else:
        run = new_run(root, args.command, arguments)
    print(f"Run directory: {run}", flush=True)
    started = time.perf_counter()
    try:
        if args.command == "audit":
            result = _audit(root, run)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        elif args.command == "report":
            from .reporting import progress_report
            progress_report(root)
        elif args.command == "design-check":
            from .design import run_design
            summary = run_design(root, run, workers=args.workers)
            print(json.dumps({k: summary[k] for k in ("status", "completed_cases", "completed_episodes", "by_scale_delay")},
                             ensure_ascii=False, indent=2), flush=True)
        elif args.command == "test":
            import pytest
            code = pytest.main([str(root / "tests"), "-q", "--basetemp", str(run / "tmp"),
                                "--junitxml", str(run / "tests.xml")])
            write_json(run / "test_result.json", dict(exit_code=int(code), passed=code == 0))
            if code:
                raise RuntimeError(f"Regression tests failed: exit {code}")
        elif args.command == "preflight":
            audit = _audit(root, run)
            seed = _design_seed(root, args.family_seed)
            common = dict(duration_s=args.duration, output_interval_s=args.output_interval,
                          mesh_spacing_m=args.mesh_spacing, family_seed=seed,
                          rise_m=args.rise, rise_duration_s=args.rise_duration,
                          perturbation_m=args.perturbation)
            truth_record, truth = _generate(run, args.side, "truth", nominal=False, **common)
            nominal_record, nominal = _generate(run, args.side, "nominal", nominal=True, **common)
            write_json(run / "window_distribution.json", window_distribution(truth))
            result = _episode(run, truth, nominal, args.delay, args.layout_seed)
            write_json(run / "preflight_summary.json", dict(
                phase="engineering_preflight", family_seed=seed, formal_test=False,
                truth=truth_record, nominal=nominal_record,
                episode=result.get("summary", {}),
                episode_wall_s=result["execution_record"]["elapsed_wall_s"],
                limitations=["Nominal-only greedy planner, no H/E/P training or benefit comparison",
                             "Single design family and condition, not design-set completion",
                             "10m hydraulic mesh remains subject to spatial refinement"],
            ))
            print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2), flush=True)
        elif args.command == "numerical":
            seed = _design_seed(root, args.family_seed)
            common = dict(duration_s=args.duration, output_interval_s=1., family_seed=seed,
                          rise_m=args.rise, rise_duration_s=args.rise_duration,
                          perturbation_m=args.perturbation, nominal=False)
            coarse_record, coarse = _generate(run, args.side, "mesh10m_output1s", mesh_spacing_m=10., **common)
            fine_record, fine = _generate(run, args.side, "mesh5m_output1s", mesh_spacing_m=5., **common)
            temporal_args = {**common, "output_interval_s": 2.}
            temporal_record, temporal = _generate(run, args.side, "mesh10m_output2s", mesh_spacing_m=10., **temporal_args)
            report = dict(
                family_seed=seed, formal_test=False, design_only=True,
                spatial_10m_vs_5m=compare_fields(coarse, fine),
                temporal_2s_vs_1s=compare_fields(temporal, coarse),
                coarse_windows=window_distribution(coarse), reference_windows=window_distribution(fine),
                temporal_comparison="independent_ANUGA_solves_with_1s_and_2s_yieldstep",
                fields=[coarse_record, fine_record, temporal_record],
                status="diagnostic_only_pending_full_design_matrix_and_preregistered_tolerances",
            )
            write_json(run / "numerical_comparison.json", report)
            print(json.dumps({k: report[k] for k in ("spatial_10m_vs_5m", "temporal_2s_vs_1s")},
                             ensure_ascii=False, indent=2), flush=True)
        elif args.command == "episode":
            for path in (args.truth, args.nominal):
                if not path.is_file():
                    raise FileNotFoundError(path)
            write_json(run / "input_files.json", {str(path): sha256(path) for path in (args.truth, args.nominal)})
            result = _episode(run, load_field(args.truth), load_field(args.nominal), args.delay, args.layout_seed)
            print(json.dumps(result.get("summary", {}), ensure_ascii=False, indent=2), flush=True)
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
    except BaseException as error:
        (run / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(root, run, status="failed", error=repr(error), elapsed_wall_s=time.perf_counter() - started)
        raise
