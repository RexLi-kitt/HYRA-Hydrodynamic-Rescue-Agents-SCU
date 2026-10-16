"""Schedule geometry and residual-amplitude diagnostics for protocol V1.

The geometry is evaluated once per distinct release timetable, not once per
event. Event-dependent amplitudes are kept separate and phases are averaged
within family before families receive equal weight. No statistic here measures
decision benefit, independent family evidence, or a realized rescue deadline.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyra_validation.artifacts import ROOT, new_run, write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import binned_history, scheduled_records
from hyra_validation.lead import MONITORS, sensor_records

TIMES = np.arange(24, 241, 6, dtype=float)
HORIZONS = (6, 12, 20, 30)
KERNELS = ("recent", "physical", "uniform", "wrong")
SCHEDULES = [dict(revisit=r, delay=d, phase=p,
                  name=f"r{r}_d{d}_p{int(p*4)}")
             for r in (2, 5, 10, 20, 40)
             for d in (0, 10) for p in (0, .25, .5, .75)]
EVENT_RUN = ROOT / "results/runs/20261015_014922_600331_lead_stage1"


def _stats(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, min=None, median=None, max=None)
    return dict(count=len(values), mean=float(values.mean()),
                min=float(values.min()), median=float(np.median(values)),
                max=float(values.max()))


def _profile(event):
    digest = hashlib.sha256()
    for key in ("release_s", "forecast_valid_s"):
        data = np.ascontiguousarray(event[key], dtype="<f8")
        digest.update(str(data.shape).encode())
        digest.update(data.tobytes())
    return digest.hexdigest()[:16]


def _kernel(age, center, method):
    valid = np.isfinite(age)
    out = np.zeros(age.shape, dtype=float)
    if not np.any(valid):
        return out
    if method == "recent":
        center = 0
    elif method == "wrong":
        center = (center+20) % 40
    out[valid] = 1 if method == "uniform" else np.exp(-.5*((age[valid]-center)/10)**2)
    return out / out.sum()


def _schedule_geometry(event, graph, schedule):
    """Timetable-only output; residual values never enter its statistics."""
    records = scheduled_records(event, schedule["revisit"], schedule["delay"], schedule["phase"])
    representative = MONITORS[0]
    seq = records[representative]
    batch_times = np.asarray([r["valid_time_s"] for r in seq])
    age = np.full((len(TIMES), 6), np.nan)
    indices = np.full(age.shape, -1, dtype=int)
    timeline = []
    for ti, now in enumerate(TIMES):
        bins = binned_history(records, representative, now)
        entries = []
        for bi, row in enumerate(bins):
            if row is not None:
                age[ti, bi] = now-row["valid_time_s"]
                indices[ti, bi] = int(np.searchsorted(batch_times, row["valid_time_s"]))
            entries.append(None if row is None else {k: row[k] for k in (
                "valid_time_s", "arrival_time_s", "matched_forecast_id",
                "prior_release_s", "actual_lead_s")})
        arrived = [r for r in seq if r["arrival_time_s"] <= now]
        timeline.append(dict(now_s=float(now), bins=entries,
            bin_ages_s=[None if not np.isfinite(a) else float(a) for a in age[ti]],
            arrived_observations=len(arrived)*len(MONITORS),
            arrived_missing_version_observations=sum(r["residual_m"] is None for r in arrived)*len(MONITORS),
            available_bins=int(np.isfinite(age[ti]).sum())))
    targets = sorted(map(int, graph))
    edges = [dict(target=t, source=int(link["source"]),
                  travel_s=float(link["travel_s"]), spatial_weight=float(link["weight"]))
             for t in targets for link in graph[str(t)]]
    if any(e["source"] not in MONITORS for e in edges):
        raise ValueError("Timing scan requires graph sources among fixed monitors")
    weights = np.zeros((len(TIMES), len(edges), len(HORIZONS), len(KERNELS), 6))
    latest_age = np.array([np.min(a[np.isfinite(a)]) if np.isfinite(a).any() else np.nan for a in age])
    remaining = np.array([e["travel_s"] for e in edges])[None, :]-latest_age[:, None]
    centers = np.maximum(np.array([e["travel_s"] for e in edges])[:, None]-np.array(HORIZONS)[None, :], 0)
    for ti in range(len(TIMES)):
        for ei in range(len(edges)):
            for hi in range(len(HORIZONS)):
                for ki, method in enumerate(KERNELS):
                    weights[ti, ei, hi, ki] = _kernel(age[ti], centers[ei, hi], method)
    available = np.isfinite(latest_age)
    kernel_rows = []
    for hi, tau in enumerate(HORIZONS):
        physical = weights[:, :, hi, 1]
        row = dict(horizon_s=tau, edges=len(edges), decision_times=len(TIMES),
            empty_history_time_fraction=float(np.mean(~available)),
            physical_center_s=_stats(centers[:, hi]),
            physical_center_zero_edge_fraction=float(np.mean(centers[:, hi] == 0)),
            nominal_remaining_after_horizon_s=_stats(remaining-tau),
            nominal_remaining_positive_fraction=float(np.mean((remaining[available]-tau) > 0)) if available.any() else None)
        for ki, method in enumerate(KERNELS):
            if method == "physical":
                continue
            comparator = weights[:, :, hi, ki]
            l1 = np.abs(physical-comparator).sum(axis=-1)[available]
            denom = np.linalg.norm(physical, axis=-1)*np.linalg.norm(comparator, axis=-1)
            cosine = np.divide((physical*comparator).sum(axis=-1), denom,
                               out=np.full(denom.shape, np.nan), where=denom > 0)[available]
            row[f"physical_vs_{method}_l1"] = _stats(l1)
            row[f"physical_vs_{method}_cosine"] = _stats(cosine)
            row[f"physical_equals_{method}_fraction"] = float(np.mean(l1 < 1e-12)) if l1.size else None
        kernel_rows.append(row)
    valid_records = [r for r in seq if r["residual_m"] is not None]
    summary = dict(**schedule, monitor_count=len(MONITORS),
        batches=len(seq), scheduled_observations=len(seq)*len(MONITORS),
        matched_observations=len(valid_records)*len(MONITORS),
        missing_version_observations=(len(seq)-len(valid_records))*len(MONITORS),
        arrivals_after_300_s=sum(r["arrival_time_s"] > 300 for r in seq)*len(MONITORS),
        actual_lead_s=_stats([r["actual_lead_s"] for r in valid_records]),
        selected_age_s=_stats(age), latest_age_s=_stats(latest_age),
        empty_bin_fraction=float(np.mean(~np.isfinite(age))),
        no_history_time_fraction=float(np.mean(~available)),
        nominal_latest_remaining_s=_stats(remaining),
        nominal_latest_remaining_positive_fraction=float(np.mean(remaining[available] > 0)) if available.any() else None,
        kernels=kernel_rows)
    return dict(summary=summary, timeline=timeline, edges=edges, targets=targets,
                age=age, indices=indices, weights=weights, remaining=remaining,
                centers=centers, batch_times=batch_times)


def _amplitudes(event, graph, schedule, geometry):
    records = scheduled_records(event, schedule["revisit"], schedule["delay"], schedule["phase"])
    indices = geometry["indices"]
    edge_values = np.zeros((len(TIMES), len(geometry["edges"]), 6))
    all_residuals = []
    for rows in records.values():
        all_residuals.extend(r["residual_m"] for r in rows if r["residual_m"] is not None)
    for ei, edge in enumerate(geometry["edges"]):
        sequence = records[edge["source"]]
        residuals = np.array([np.nan if r["residual_m"] is None else r["residual_m"] for r in sequence])
        present = indices >= 0
        edge_values[:, ei, :][present] = residuals[indices[present]]
    by_edge = np.einsum("tehkb,teb->tehk", geometry["weights"], edge_values)
    values = np.zeros((len(TIMES), len(geometry["targets"]), len(HORIZONS), len(KERNELS)))
    for ri, target in enumerate(geometry["targets"]):
        edge_ids = [i for i, edge in enumerate(geometry["edges"]) if edge["target"] == target]
        spatial = np.array([geometry["edges"][i]["spatial_weight"] for i in edge_ids])
        values[:, ri] = np.average(by_edge[:, edge_ids], axis=1, weights=spatial)
    values[~np.any(indices >= 0, axis=1)] = np.nan
    raw = np.asarray(all_residuals)
    rows = []
    for hi, tau in enumerate(HORIZONS):
        physical = values[:, :, hi, 1].ravel()
        row = dict(horizon_s=tau,
            available_target_times=int(np.isfinite(physical).sum()),
            raw_residual_mae_m=float(np.mean(np.abs(raw))) if raw.size else None,
            raw_residual_rms_m=float(np.sqrt(np.mean(raw**2))) if raw.size else None)
        for ki, method in enumerate(KERNELS):
            v = values[:, :, hi, ki].ravel()
            row[f"{method}_summary_mae_m"] = float(np.nanmean(np.abs(v))) if np.isfinite(v).any() else None
            if method == "physical":
                continue
            keep = np.isfinite(v) & np.isfinite(physical)
            delta = physical[keep]-v[keep]
            row[f"physical_minus_{method}_mae_m"] = float(np.mean(np.abs(delta))) if delta.size else None
            row[f"physical_minus_{method}_rms_m"] = float(np.sqrt(np.mean(delta**2))) if delta.size else None
            row[f"physical_minus_{method}_max_abs_m"] = float(np.max(np.abs(delta))) if delta.size else None
            row[f"physical_vs_{method}_correlation"] = (float(np.corrcoef(physical[keep], v[keep])[0, 1])
                if delta.size > 1 and np.std(physical[keep]) > 1e-12 and np.std(v[keep]) > 1e-12 else None)
        rows.append(row)
    return values, rows


def _aggregate(rows, keys, count_name):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    metrics = [k for k in rows[0] if k.endswith(("_m", "_correlation"))] if rows else []
    out = []
    for group, rr in sorted(groups.items()):
        row = dict(zip(keys, group)); row[count_name] = len(rr)
        for metric in metrics:
            values = [r[metric] for r in rr if r[metric] is not None]
            row[metric] = float(np.mean(values)) if values else None
        out.append(row)
    return out


def _legacy_audit(flat_events):
    checked = 0; worst = 0.
    for _, event in flat_events:
        new = scheduled_records(event, 10, 0, 0)
        old = sensor_records(event, 0, 0, False)
        for region in MONITORS:
            if len(new[region]) != len(old[region]):
                raise AssertionError("Legacy observation count differs")
            for a, b in zip(new[region], old[region]):
                for key in ("valid_time_s", "arrival_time_s", "matched_forecast_id"):
                    if a[key] != b[key]:
                        raise AssertionError(f"Legacy {key} differs")
                if (a["residual_m"] is None) != (b["residual_m"] is None):
                    raise AssertionError("Legacy residual missingness differs")
                if a["residual_m"] is not None:
                    worst = max(worst, abs(a["residual_m"]-b["residual_m"]))
                checked += 1
    if worst > 1e-12:
        raise AssertionError("R10/phase0 residual differs from old contract")
    return dict(checked_observations=checked, max_absolute_residual_difference_m=worst, passed=True)


def scan(events, graph, run):
    """Run all 40 conditions and write artifacts under ``run`` (a Path).

    ``events`` is the split -> event list returned by load_events; ``graph`` is
    the frozen physical graph. Returns a JSON-compatible completion summary.
    Existing scan artifacts are refused to protect prior run provenance.
    """
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if list(run.glob("decision_timing_*")):
        raise FileExistsError("Decision timing artifacts already exist in this run")
    flat_events = [(split, e) for split, ee in events.items() for e in ee]
    if not flat_events:
        raise ValueError("No events")
    profiles = {}
    event_profiles = []
    for split, event in flat_events:
        if event["time_s"][0] > 20 or event["time_s"][-1] < 300:
            raise ValueError("Truth time grid must cover the full observation schedule")
        key = _profile(event)
        profiles.setdefault(key, event)
        event_profiles.append(dict(event_id=event["metadata"]["event_id"], profile_id=key))
    geometry_cache = {}; schedule_rows = []; timeline_rows = []; edge_metadata = None
    npz_geometry = {}; amplitude_rows = []; amplitude_curves = []
    for schedule in SCHEDULES:
        for profile_id, event in profiles.items():
            geo = _schedule_geometry(event, graph, schedule)
            geometry_cache[profile_id, schedule["name"]] = geo
            schedule_rows.append(dict(profile_id=profile_id, **geo["summary"]))
            timeline_rows.extend(dict(profile_id=profile_id, schedule=schedule["name"], **r) for r in geo["timeline"])
            edge_metadata = geo["edges"]
            prefix = f"{profile_id}_{schedule['name']}"
            for field in ("age", "indices", "weights", "remaining", "centers", "batch_times"):
                npz_geometry[f"{prefix}_{field}"] = geo[field]
        curves = []
        for split, event in flat_events:
            geo = geometry_cache[_profile(event), schedule["name"]]
            curve, rr = _amplitudes(event, graph, schedule, geo)
            curves.append(curve)
            meta = event["metadata"]
            amplitude_rows.extend(dict(schedule=schedule["name"], revisit=schedule["revisit"],
                delay=schedule["delay"], phase=schedule["phase"], split=split,
                event_id=meta["event_id"], family=meta["event_family_id"],
                condition=meta["condition"], **r) for r in rr)
        amplitude_curves.append(np.asarray(curves))
        print("timing scan", schedule["name"], flush=True)
    family_rows = _aggregate(amplitude_rows,
        ("revisit", "delay", "split", "family", "condition", "horizon_s"), "phase_count")
    summary_rows = _aggregate(family_rows,
        ("revisit", "delay", "split", "condition", "horizon_s"), "family_count")
    summary = dict(completed=True, protocol="DECISION_VALUE_DIAGNOSTIC_V1",
        schedule_count=len(SCHEDULES), distinct_release_profiles=len(profiles),
        event_count=len(flat_events),
        family_counts={split: len({e["metadata"]["event_family_id"] for e in ee}) for split, ee in events.items()},
        horizons_s=list(HORIZONS), kernels=list(KERNELS),
        legacy_contract_audit=_legacy_audit(flat_events),
        geometry_rows=len(schedule_rows), event_amplitude_rows=len(amplitude_rows),
        semantics=dict(geometry="Schedule-only; not independent event/family replicates.",
            amplitudes="Each event averaged over target/time, phases averaged within family, families equal-weighted.",
            nominal_remaining="Frozen graph travel minus latest valid-observation age; horizon margin subtracts tau. Not a rescue deadline or observed information value.",
            count="All 24 fixed monitors per batch; denser schedules spend more observations.",
            correlation="Within-event target/time correlation; missing if either summary is constant.",
            split_test="Previously seen internal diagnostic families; not blind testing.",
            kernels="L1/cosine compare normalized temporal weights over identical six age bins, only nonempty histories."),
        files=dict(schedules="decision_timing_schedules.json", timeline="decision_timing_timeline.json",
            geometry="decision_timing_geometry.npz", amplitudes="decision_timing_event_amplitudes.json",
            family_amplitudes="decision_timing_family_amplitudes.json", amplitude_summary="decision_timing_amplitude_summary.json",
            curves="decision_timing_amplitude_curves.npz", axes="decision_timing_axes.json"))
    axes = dict(times_s=TIMES, horizons_s=HORIZONS, kernels=KERNELS,
        schedules=SCHEDULES, edges=edge_metadata, targets=sorted(map(int, graph)),
        event_profiles=event_profiles,
        event_order=[dict(split=s, event_id=e["metadata"]["event_id"]) for s, e in flat_events],
        geometry_axes=dict(age=["time", "bin"], indices=["time", "bin"],
            weights=["time", "edge", "horizon", "kernel", "bin"],
            remaining=["time", "edge"], centers=["edge", "horizon"]),
        amplitude_axes=["schedule", "event", "time", "target", "horizon", "kernel"],
        missing_encoding="JSON null; NPZ NaN for missing age/amplitude, -1 for missing record index; zero kernel weights only when no history.")
    write_json(run/summary["files"]["schedules"], schedule_rows)
    write_json(run/summary["files"]["timeline"], timeline_rows)
    write_json(run/summary["files"]["amplitudes"], amplitude_rows)
    write_json(run/summary["files"]["family_amplitudes"], family_rows)
    write_json(run/summary["files"]["amplitude_summary"], summary_rows)
    write_json(run/summary["files"]["axes"], axes)
    np.savez_compressed(run/summary["files"]["geometry"], **npz_geometry)
    np.savez_compressed(run/summary["files"]["curves"], values=np.asarray(amplitude_curves))
    write_json(run/"decision_timing_completion.json", summary)
    return summary


if __name__ == "__main__":
    events = load_events(EVENT_RUN/"manifest.json")
    graph = json.loads((EVENT_RUN/"physical_graph.json").read_text(encoding="utf-8"))["graph"]
    run = new_run("decision_timing_scan", dict(protocol="DECISION_VALUE_DIAGNOSTIC_V1",
        manifest=str(EVENT_RUN/"manifest.json"), schedules=SCHEDULES,
        note="Independent schedule geometry and family-aggregated amplitude diagnostics; no task benefit claim."))
    scan(events, graph, run)
    print(run, flush=True)
