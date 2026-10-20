"""Shared geometric patrol records and causal model frames, independent of method."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
from pathlib import Path
import time
import numpy as np

from .artifacts import finish_run, new_run, sha256, write_json
from .design import _cached_field
from .execution_plan import register
from .protocol import load_protocol
from .records import save_observations
from .simulation import stratified_starts


def patrol_routes(side, drones):
    starts = np.asarray(stratified_starts((side, side), drones), float)
    axis = np.unique(np.r_[np.arange(2, side, 5), side - 3]).astype(int)
    cells = np.asarray([(r, c) for r in axis for c in axis], int)
    owners = np.argmin(np.sum((cells[:, None] - starts[None]) ** 2, axis=2), axis=1)
    routes = []
    for rid in range(drones):
        selected = cells[owners == rid]
        ordered = []
        for index, row in enumerate(sorted(set(selected[:, 0]))):
            line = sorted((int(c) for r, c in selected if r == row), reverse=bool(index % 2))
            ordered.extend((int(row), col) for col in line)
        if not ordered:
            ordered = [tuple(map(int, starts[rid]))]
        nearest = int(np.argmin(np.sum((np.asarray(ordered) - starts[rid]) ** 2, axis=1)))
        routes.append(ordered[nearest:] + ordered[:nearest])
    return starts.astype(int), routes


def sampled_field(field, sample_times):
    times, depths = field["times_s"], field["depths_m"]
    if min(sample_times) < times[0] or max(sample_times) > times[-1]:
        raise ValueError("No hydraulic extrapolation")
    upper = np.clip(np.searchsorted(times, sample_times, side="right"), 1, len(times) - 1)
    lower = upper - 1
    weight = ((sample_times - times[lower]) / (times[upper] - times[lower]))[:, None]
    return ((1 - weight) * depths[lower] + weight * depths[upper]).astype(np.float32)


def collect_patrol(truth, nominal, side, drones, seed, duration_s=1200., noise_m=.02):
    times = np.arange(0., duration_s + 1e-8, 5.)
    true_sample, nominal_sample = sampled_field(truth, times), sampled_field(nominal, times)
    starts, routes = patrol_routes(side, drones)
    def xy(cell):
        return np.asarray([(cell[1] + .5) * 10., (cell[0] + .5) * 10.])
    positions = np.asarray([xy(cell) for cell in starts])
    destinations = np.zeros(drones, int)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
    means = np.zeros((len(times), side * side), np.float32)
    stds = np.zeros_like(means)
    counts = np.zeros(means.shape, np.uint8)
    records, trajectories = [], []
    for tick, now in enumerate(times):
        if tick:
            for rid in range(drones):
                remaining, hops = 25., 0
                while remaining > 1e-9:
                    target = xy(routes[rid][destinations[rid]])
                    vector = target - positions[rid]
                    distance = float(np.linalg.norm(vector))
                    if distance > remaining:
                        positions[rid] += vector * (remaining / distance)
                        break
                    positions[rid] = target
                    remaining -= distance
                    destinations[rid] = (destinations[rid] + 1) % len(routes[rid])
                    hops += 1
                    if len(routes[rid]) == 1 or hops > len(routes[rid]) + 2:
                        break
        flat, observed = [], []
        for rid, position in enumerate(positions):
            col, row = np.clip(np.floor(position / 10).astype(int), 0, side - 1)
            cells = [(r, c) for r in range(max(0, row - 2), min(side, row + 3))
                     for c in range(max(0, col - 2), min(side, col + 3))]
            indexes = np.asarray([r * side + c for r, c in cells])
            values = true_sample[tick, indexes] + rng.normal(0., noise_m, len(cells))
            flat.extend(indexes.tolist())
            observed.extend(values.tolist())
            records.extend(dict(cell=cell, sample_time=float(now), arrival_time=float(now), depth_m=float(value), drone=rid)
                           for cell, value in zip(cells, values))
            trajectories.append(dict(time_s=float(now), drone=rid, position_m=position.tolist()))
        n = np.bincount(flat, minlength=side * side)
        total = np.bincount(flat, weights=observed, minlength=side * side)
        squared = np.bincount(flat, weights=np.square(observed), minlength=side * side)
        valid = n > 0
        means[tick, valid] = total[valid] / n[valid]
        stds[tick, valid] = np.sqrt(np.maximum(0., squared[valid] / n[valid] - (total[valid] / n[valid]) ** 2))
        counts[tick] = n
    return dict(times_s=times, mean_depth_m=means, std_depth_m=stds, count=counts,
        nominal_sample_m=nominal_sample, label_truth_sample_m=true_sample,
        elevation_m=nominal["elevation_m"].astype(np.float32), positions_m=nominal["positions_m"].astype(np.float32)), records, trajectories


def causal_frame(panel, now, delay_s=0., history_s=60.):
    """Every group receives exactly these features; labels are never read."""
    if delay_s < 0 or history_s != 60. or now < 0:
        raise ValueError("Invalid shared history contract")
    requested = now - np.arange(12, -1, -1) * 5.
    indexes = np.searchsorted(panel["times_s"], np.maximum(requested, 0.))
    if np.any(indexes >= len(panel["times_s"])) or not np.allclose(panel["times_s"][indexes], np.maximum(requested, 0.)):
        raise ValueError("Frame time must lie on the shared 5s observation grid")
    legal = (requested >= 0) & (requested + delay_s <= now + 1e-9)
    raw = panel["mean_depth_m"][indexes].T
    std = panel["std_depth_m"][indexes].T
    count = panel["count"][indexes].T
    mask = (count > 0) & legal[None]
    nominal = panel["nominal_sample_m"][indexes].T
    values = np.stack((np.where(mask, raw, 0.), np.where(requested[None] >= 0, nominal, 0.),
        np.broadcast_to((now - requested) / history_s, raw.shape), mask.astype(float),
        np.where(mask, np.log1p(count) / np.log(21.), 0.), np.where(mask, std, 0.)), axis=-1)
    return values.astype(np.float32)


def _job(run_path, family, side, study_id):
    directory = Path(run_path) / "jobs" / f"{family['family_id']}_side{side}"
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "case.json"
    if marker.exists():
        result = load_protocol(marker)
        for panel in result["panels"]:
            if sha256(panel["path"]) != panel["sha256"] or sha256(panel["raw_observations"]["path"]) != panel["raw_observations"]["sha256"]:
                raise AssertionError("Shared records changed")
        return result
    with (directory / "solver.log").open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        common = dict(duration_s=1200., output_interval_s=1., mesh_spacing_m=5., family_seed=family["family_seed"], **family["hydrology"])
        truth_record, truth = _cached_field(directory / "truth", side, nominal=False, **common)
        nominal_record, nominal = _cached_field(directory / "nominal", side, nominal=True, **{**common, "perturbation_m": 0.})
        panels = []
        panel_directory = directory
        if list(directory.glob("panel*.npz")) or list(directory.glob("observations*.npz")):
            import uuid
            panel_directory = directory / "recovery_attempts" / uuid.uuid4().hex
            panel_directory.mkdir(parents=True)
        for layout in range(2):
            seed = int.from_bytes(hashlib.sha256(f"{study_id}:{family['family_id']}:common_patrol:{layout}".encode()).digest()[:4], "big")
            panel, records, trajectories = collect_patrol(truth, nominal, side, {32: 4, 50: 10, 70: 20}[side], seed)
            panel_path = panel_directory / f"panel{layout}.npz"
            with panel_path.open("xb") as stream:
                np.savez_compressed(stream, **panel)
            observation = save_observations(panel_directory / f"observations{layout}.npz", records)
            trajectory_path = panel_directory / f"trajectory{layout}.json"
            write_json(trajectory_path, trajectories)
            panels.append(dict(layout=layout, seed=seed, path=str(panel_path), sha256=sha256(panel_path),
                raw_observations=observation, trajectory_path=str(trajectory_path), trajectory_sha256=sha256(trajectory_path)))
    result = dict(family=family, side=side, truth=truth_record, nominal=nominal_record, panels=panels,
        sampling="fixed_geometry_patrol_no_targets_no_water_adaptation", fusion="per_cell_tick_mean_std_and_count_with_lossless_raw_records",
        shared_by_all_methods=True, labels_private_to_training_and_evaluation=True, history_s=60., sample_period_s=5.)
    write_json(marker, result)
    return result


def run_collection(root, workers=2, resume=None):
    root = Path(root).resolve()
    plan = register(root)
    families = [f for f in plan["allocation"] if f["split"] in ("train", "validation")]
    spec = dict(plan_sha256=sha256(root / "manifests/full_execution_plan.v1.json"), splits=["train", "validation"],
        expected_cases=108, expected_panels=216, mesh_spacing_m=5., output_interval_s=1.,
        mission_start_s=0., numerical_configuration_provisional_until_design_gate=True,
        source_hashes={name: sha256(root / "src/hyra_exp3" / name) for name in ("common_data.py", "hydraulics.py", "execution_plan.py")})
    if resume is None:
        run = new_run(root, "common_training_records", dict(workers=workers))
        write_json(run / "collection_spec.json", spec)
    else:
        run = Path(resume).resolve()
        if load_protocol(run / "collection_spec.json") != spec:
            raise ValueError("Collection resume code/config differs")
    print(f"Common records directory: {run}", flush=True)
    started, completed = time.perf_counter(), []
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_job, str(run), family, side, plan["study_id"]) for family in families for side in (32, 50, 70)]
            for future in as_completed(futures):
                completed.append(future.result())
                write_json(run / "progress.json", dict(cases=len(completed), planned_cases=108))
                print(f"Common records {len(completed)}/108 cases", flush=True)
        summary = dict(status="completed", cases=len(completed), panels=sum(len(r["panels"]) for r in completed),
                       training_families=24, validation_families=12, test_truth_accessed=False, records=completed)
        write_json(run / "common_data_summary.json", summary)
        finish_run(root, run, elapsed_wall_s=time.perf_counter() - started)
        write_json(root / "results/latest_common_data.json", dict(run=str(run), status="completed"))
        return run
    except BaseException as error:
        finish_run(root, run, status="failed", error=repr(error))
        raise
