"""Numerical and window-label diagnostics; never filter scenes by method benefit."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_field(path):
    with np.load(Path(path), allow_pickle=False) as field:
        return {name: field[name] for name in field.files}


def safe_start_grid(times_s, depths_m, service_duration_s=12., threshold_m=.8):
    """Evaluate complete services under the saved piecewise-linear trajectory.

    Returns [valid start times, cells]. No extrapolation beyond last saved time.
    Checks both fractional endpoints and every interior knot, rather than only
    the water depth at the start/end of service.
    """
    times = np.asarray(times_s, float)
    depths = np.asarray(depths_m, float)
    if times.ndim != 1 or len(times) < 2 or depths.ndim != 2 or len(depths) != len(times):
        raise ValueError("Expected times[T] and depths[T,N]")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(depths)) or np.any(np.diff(times) <= 0):
        raise ValueError("Nonfinite or nonmonotone field")
    if service_duration_s <= 0:
        raise ValueError("Service duration must be positive")
    eligible = np.flatnonzero(times + service_duration_s <= times[-1] + 1e-9)
    safe = np.zeros((len(times), depths.shape[1]), bool)
    for start in eligible:
        end = times[start] + service_duration_s
        right = int(np.searchsorted(times, end, side="right"))
        last = min(right - 1, len(times) - 1)
        maximum = np.max(depths[start:last + 1], axis=0)
        if times[last] < end - 1e-9:
            weight = (end - times[last]) / (times[last + 1] - times[last])
            endpoint = depths[last] * (1 - weight) + depths[last + 1] * weight
            maximum = np.maximum(maximum, endpoint)
        safe[start] = maximum <= threshold_m
    return safe


def window_distribution(field, service_duration_s=12., threshold_m=.8):
    times, depths = field["times_s"], field["depths_m"]
    safe = safe_start_grid(times, depths, service_duration_s, threshold_m)
    eligible = times + service_duration_s <= times[-1] + 1e-9
    count = safe.sum(axis=0)
    ever = count > 0
    last = np.full(depths.shape[1], np.nan)
    if ever.any():
        last[ever] = times[len(times) - 1 - np.argmax(safe[::-1, ever], axis=0)]
    def quantiles(values):
        return {str(q): float(np.quantile(values, q)) for q in (0, .25, .5, .75, 1)} if len(values) else None
    return dict(
        cells=int(depths.shape[1]),
        interpolation="piecewise_linear_saved_ANUGA_outputs_not_continuous_solver_guarantee",
        service_duration_s=service_duration_s, threshold_m=threshold_m,
        cells_with_any_complete_window=int(ever.sum()), cells_without_window=int((~ever).sum()),
        cell_start_opportunity_fraction=float(safe[eligible].mean()),
        last_safe_start_s_quantiles=quantiles(last[ever]),
        safe_start_count_quantiles=quantiles(count),
        depth_m_quantiles=quantiles(depths.ravel()),
        design_only=True, scenario_selected_by_method_benefit=False,
    )


def compare_fields(coarse, reference, service_duration_s=12., threshold_m=.8):
    """Compare only common exact saved times and identical task positions."""
    if not np.allclose(coarse["positions_m"], reference["positions_m"], atol=1e-9, rtol=0):
        raise ValueError("Different task positions")
    ct, rt = coarse["times_s"], reference["times_s"]
    indices = np.searchsorted(rt, ct)
    if np.any(indices >= len(rt)) or not np.allclose(rt[indices], ct, atol=1e-9, rtol=0):
        raise ValueError("Reference lacks common output times")
    error = coarse["depths_m"].astype(float) - reference["depths_m"][indices].astype(float)
    cs = safe_start_grid(ct, coarse["depths_m"], service_duration_s, threshold_m)
    rs = safe_start_grid(rt, reference["depths_m"], service_duration_s, threshold_m)[indices]
    valid = ct + service_duration_s <= min(ct[-1], rt[-1]) + 1e-9
    return dict(
        common_times=int(len(ct)), cells=int(error.shape[1]),
        depth_mae_m=float(np.mean(np.abs(error))), depth_rmse_m=float(np.sqrt(np.mean(error ** 2))),
        depth_max_abs_m=float(np.max(np.abs(error))),
        complete_window_disagreement_fraction=float(np.mean(cs[valid] != rs[valid])),
        coarse_only_safe_fraction=float(np.mean(cs[valid] & ~rs[valid])),
        reference_only_safe_fraction=float(np.mean(~cs[valid] & rs[valid])),
        unit_of_independence="design_family_not_cell_or_time",
        formal_convergence_pass=False,
        note="Descriptive design diagnostic; one pair cannot establish numerical convergence.",
    )
