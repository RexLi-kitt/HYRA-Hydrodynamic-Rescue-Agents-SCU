"""Independent ANUGA solves on a metric triangular mesh, sampled on task cells."""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
from scipy.spatial import Delaunay

from .artifacts import peak_memory_bytes, sha256, write_json


def family_parameters(seed):
    rng = np.random.default_rng(seed)
    return dict(bed_drop_m=float(rng.uniform(.22, .35)), ridge_m=float(rng.uniform(.025, .065)),
                friction=float(rng.uniform(.025, .04)), phase=float(rng.uniform(0, 2 * np.pi)),
                perturbation_center_fraction=float(rng.uniform(.25, .75)))


def terrain(x, y, length, params):
    x, y = np.asarray(x) / length, np.asarray(y) / length
    return (-params["bed_drop_m"] * x + params["ridge_m"] *
            np.sin(2 * np.pi * y + params["phase"]) * np.sin(np.pi * x))


def finite_rise(t, rise_m, rise_duration_s, onset_s=120.):
    return rise_m * np.clip((np.asarray(t) - onset_s) / rise_duration_s, 0., 1.)


def perturbation(t, amplitude, rise_duration_s):
    center, halfwidth = 120. + .5 * rise_duration_s, .2 * rise_duration_s
    phase = (np.asarray(t) - (center - halfwidth)) / (2 * halfwidth)
    return amplitude * np.where((phase >= 0) & (phase <= 1), np.sin(np.pi * phase) ** 2, 0.)


def generate_field(side, output_dir, *, duration_s=1200., output_interval_s=2.,
                   mesh_spacing_m=10., family_seed=202610180001, rise_m=.6,
                   rise_duration_s=600., perturbation_m=.1, nominal=False,formal_test=False,experiment_role='development'):
    """Save times_s[T], depths_m[T,N], elevation_m[N], positions_m[N,2].

    Nominal solves start independently with the same public terrain and initial
    stage. Their rise amplitude is .75 of the true scenario specification and
    their local boundary pulse is zero. No true state is copied into nominal.
    Geometry/forcing choices are development choices, not yet frozen settings.
    """
    if side not in (32, 50, 70):
        raise ValueError("Expected side 32/50/70")
    if not all(np.isfinite(v) and v > 0 for v in (duration_s, output_interval_s, mesh_spacing_m, rise_duration_s)):
        raise ValueError("Invalid duration, output interval, or mesh spacing")
    if rise_m not in (.6, .9) or perturbation_m not in (0., .1, .2):
        raise ValueError("Unsupported forcing level")
    steps = duration_s / output_interval_s
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("Duration must be an integer number of output intervals")
    import anuga

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "field.npz").exists():
        raise FileExistsError(output_dir / "field.npz")
    started = time.perf_counter()
    length = side * 10.
    mesh_n = int(math.ceil(length / mesh_spacing_m))
    params = family_parameters(family_seed)
    domain = anuga.rectangular_cross_domain(mesh_n, mesh_n, len1=length, len2=length)
    domain.set_store(False)
    domain.set_flow_algorithm("DE0")
    domain.set_quantity("elevation", lambda x, y: terrain(x, y, length, params))
    domain.set_quantity("friction", params["friction"])
    domain.set_quantity("stage", .15)
    amplitude = rise_m * (.75 if nominal else 1.)
    local_amplitude = 0. if nominal else perturbation_m
    def boundary(t, x, y):
        local = abs(y / length - params["perturbation_center_fraction"]) <= .1
        level = .15 + float(finite_rise(t, amplitude, rise_duration_s))
        if local:
            level += float(perturbation(t, local_amplitude, rise_duration_s))
        return [level, 0., 0.]
    wall = anuga.Reflective_boundary(domain)
    inlet = anuga.Time_space_boundary(domain, boundary)
    domain.set_boundary(dict(left=inlet, right=wall, top=wall, bottom=wall))

    yy, xx = np.meshgrid((np.arange(side) + .5) * 10., (np.arange(side) + .5) * 10., indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    centroids = np.asarray(domain.get_centroid_coordinates())
    triangulation = Delaunay(centroids)
    simplices = triangulation.find_simplex(positions)
    if np.any(simplices < 0):
        raise ValueError("Task cell outside centroid interpolation hull")
    transform = triangulation.transform[simplices]
    barycentric = np.einsum("nij,nj->ni", transform[:, :2], positions - transform[:, 2])
    weights = np.column_stack((barycentric, 1 - barycentric.sum(axis=1)))
    if np.min(weights) < -1e-9:
        raise ValueError("Nonconvex field sampling")
    vertices = triangulation.simplices[simplices]
    del triangulation
    def sample(values):
        return np.sum(values[vertices] * weights, axis=1)
    elevation = sample(domain.quantities["elevation"].centroid_values)
    times = np.empty(int(round(steps)) + 1, dtype=np.float64)
    depths = np.empty((len(times), side * side), dtype=np.float32)
    minimum_raw_depth = float("inf")
    observed_max = 0.
    count = 0
    integration_started = time.perf_counter()
    for index, t in enumerate(domain.evolve(yieldstep=output_interval_s, finaltime=duration_s)):
        raw = domain.quantities["stage"].centroid_values - domain.quantities["elevation"].centroid_values
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("Nonfinite ANUGA depth")
        minimum_raw_depth = min(minimum_raw_depth, float(raw.min()))
        observed_max = max(observed_max, float(raw.max()))
        if minimum_raw_depth < -1e-5:
            raise FloatingPointError(f"Negative ANUGA depth: {minimum_raw_depth}")
        times[index] = t
        depths[index] = np.maximum(0., sample(raw))
        count = index + 1
        if abs(t % 300.) < 1e-9:
            print(f"  ANUGA t={t:g}/{duration_s:g}s", flush=True)
    if count != len(times) or not np.isclose(times[-1], duration_s):
        raise RuntimeError("Incomplete ANUGA output")
    integration_s = time.perf_counter() - integration_started
    path = output_dir / "field.npz"
    np.savez_compressed(path, times_s=times, depths_m=depths, elevation_m=elevation,
                        positions_m=positions)
    record = dict(
        field_path=str(path), field_sha256=sha256(path), solver="ANUGA", solver_version=anuga.__version__,
        formal_test=bool(formal_test), status="independent_test" if formal_test else "development",
        experiment_role=experiment_role,kind="nominal" if nominal else "truth",
        side=side, task_cell_size_m=10., length_m=length, task_cells=side * side,
        hydraulic_triangles=int(len(centroids)), mesh_spacing_requested_m=mesh_spacing_m,
        actual_mesh_spacing_m=length / mesh_n, output_interval_s=output_interval_s, duration_s=duration_s,
        family_seed=family_seed, public_family_parameters=params,
        forcing=dict(onset_s=120., rise_m=amplitude, rise_duration_s=rise_duration_s,
                     post_rise="hold", local_perturbation_m=local_amplitude,
                     pulse="sin_squared_around_mid_rise_halfwidth_0.2_rise_duration",
                     boundary_support="left_boundary_within_0.1_domain_fraction_of_family_center"),
        independent_initial_condition=True, initial_stage_m=.15, algorithm="DE0",
        sampling="convex_linear_interpolation_of_ANUGA_triangle_centroid_depths",
        retained_data="task_grid_fields_only_not_full_hydraulic_restart_states",
        minimum_raw_depth_m=minimum_raw_depth, maximum_centroid_depth_m=observed_max,
        all_finite=bool(np.all(np.isfinite(depths))), integration_wall_s=integration_s,
        total_wall_s=time.perf_counter() - started, field_bytes=path.stat().st_size,
        uncompressed_depth_bytes=depths.nbytes, peak_process_memory_bytes=peak_memory_bytes(),
    )
    write_json(output_dir / "manifest.json", record)
    return record
