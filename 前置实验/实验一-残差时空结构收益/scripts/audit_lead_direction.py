"""Training-only nominal pulse audit; windows are offline strata, never features.

Run with --source-run OLD --output-run NEW. The old run is read-only. This
diagnostic cannot establish reflection as the cause of a later response.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyra_validation.artifacts import write_json
from hyra_validation.lead import XY, ROWS, SOURCES, TARGETS, MONITORS
from run_lead_stage1 import arrival, response

PULSE_START_S = 90.0
DATA_ROLE = "training_nominal_pulse_response"


def response_summary(time_s, values_m):
    """Summarize a complete nominal trajectory, exclusively for offline audit.

    Arrival follows stage 1: 10% of the complete-trajectory absolute peak,
    minimum peak 1 mm. That uses future values and is NOT an online detector.
    First peak means the first local maximum after arrival above that same
    10% threshold, not an identified incident/reflected wave decomposition.
    """
    t = np.asarray(time_s)
    values = np.asarray(values_m)
    if t.ndim != 1 or values.shape != t.shape or np.any(np.diff(t) <= 0):
        raise ValueError("Expected matching one-dimensional, increasing time arrays")
    magnitude = np.abs(values)
    peak_index = int(np.argmax(magnitude))
    first = arrival(t, values)
    first_peak_index = None
    if first is not None:
        eligible = np.flatnonzero((t >= first) & (magnitude >= .1 * magnitude[peak_index]))
        for index in eligible:
            if 0 < index < len(t) - 1 and magnitude[index] >= magnitude[index-1] and magnitude[index] > magnitude[index+1]:
                first_peak_index = int(index)
                break
        if first_peak_index is None:
            first_peak_index = peak_index
    return dict(
        arrival_s=first,
        arrival_after_pulse_s=None if first is None else first-PULSE_START_S,
        peak_abs_m=float(magnitude[peak_index]),
        peak_signed_m=float(values[peak_index]),
        peak_time_s=float(t[peak_index]),
        first_peak_time_s=None if first_peak_index is None else float(t[first_peak_index]),
        first_peak_abs_m=None if first_peak_index is None else float(magnitude[first_peak_index]),
    )


def build_training_windows(rows):
    """Fit offline time windows from training nominal responses only.

    Do not call with C1 reference-minus-nominal observations. No internal
    diagnostic/test outcomes are accepted. Windows may be translated by a
    known simulated event onset for retrospective stratification only.
    """
    if not rows or any(row.get("split") != "train" or row.get("data_role") != DATA_ROLE for row in rows):
        raise ValueError("Windows require training nominal responses, never reference/test labels")
    by_target = {}
    for region in TARGETS:
        selected = [r for r in rows if r["region"] == region and r["role"] == "target"]
        starts, ends, reverse_starts, families = [], [], [], []
        for target in selected:
            reverse = [r for r in rows if r["family"] == target["family"] and r["center_y_m"] == target["center_y_m"] and r["role"] == "later_sensor" and r["arrival_s"] is not None]
            if target["arrival_s"] is None or target["first_peak_time_s"] is None or not reverse:
                continue
            starts.append(target["arrival_s"]-PULSE_START_S)
            ends.append(target["first_peak_time_s"]-PULSE_START_S)
            reverse_starts.append(min(r["arrival_s"] for r in reverse)-PULSE_START_S)
            families.append(target["family"])
        if starts:
            start = float(np.median(starts))
            end = float(np.median(ends))
            reverse_start = float(np.median(reverse_starts))
            by_target[str(region)] = dict(
                relative_to_pulse_start_s=True,
                early_response=[start, end],
                pre_later_sensor=[start, max(start, reverse_start)],
                independent_training_families=len(set(families)),
                training_families=sorted(set(families)),
                arrival_range_s=[float(min(starts)), float(max(starts))],
                first_peak_range_s=[float(min(ends)), float(max(ends))],
                later_sensor_first_arrival_range_s=[float(min(reverse_starts)), float(max(reverse_starts))],
            )
    return dict(
        by_target=by_target,
        definition="Per-target medians over independent training families; early_response is arrival to first significant local peak; pre_later_sensor ends at the first responding x75 sensor in the same branch",
        usage="Offline stratification only. Event onset and nominal complete-trajectory peak/arrival are never online policy/readout features",
        online_feature_allowed=False,
        causal_wave_isolation_proven=False,
        training_only=True,
    )


def classify_offline_window(time_s, pulse_start_s, region, windows):
    """Retrospective stratum; requires explicit offline-only window metadata."""
    if windows.get("online_feature_allowed") is not False or windows.get("training_only") is not True:
        raise ValueError("Only training-fitted offline windows may classify evaluation rows")
    window = windows["by_target"][str(region)]
    relative = time_s-pulse_start_s
    start, end = window["early_response"]
    reverse_start, reverse_end = window["pre_later_sensor"]
    return dict(
        early_response=bool(start <= relative <= end),
        pre_later_sensor=bool(reverse_start <= relative < reverse_end),
        later_response=bool(relative > end),
    )


def describe(values):
    values = [float(v) for v in values if v is not None]
    return dict(n=len(values), median=float(np.median(values)) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def audit(source_run, output_run):
    source_run = Path(source_run).resolve()
    output_run = Path(output_run).resolve()
    if source_run == output_run:
        raise ValueError("Source run is immutable; select a different output run")
    output_run.mkdir(parents=True, exist_ok=True)
    destination = output_run / "direction_audit.json"
    if destination.exists():
        raise FileExistsError(destination)
    calibration_path = source_run / "calibration.json"
    parameter_path = source_run / "family_parameters.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    training = [p for p in json.loads(parameter_path.read_text(encoding="utf-8")) if p["split"] == "train"]
    if len(training) != len({p["family"] for p in training}) or len(training) != 6:
        raise ValueError("Expected six independent training families")
    mesh = calibration["nmesh"]
    rows, pairs, files = [], [], []
    for family in training:
        name = f"lead_{family['family']:03}"
        for center, branch in ((25, ROWS[:3]), (75, ROWS[3:])):
            t, response_m = response(family["params"], mesh, center)
            filename = f"direction_response_{name}_center{center}.npz"
            path = output_run / filename
            if path.exists():
                raise FileExistsError(path)
            np.savez_compressed(path, time_s=t, response_m=response_m, xy_m=XY)
            files.append(filename)
            region_rows = {}
            for x, role in ((0, "upstream"), (3, "target"), (5, "target"), (7, "later_sensor")):
                for y in branch:
                    region = 10*y+x
                    row = dict(family=name, split="train", data_role=DATA_ROLE, center_y_m=center,
                               region=region, role=role, xy_m=XY[region].tolist(),
                               **response_summary(t, response_m[:, region]))
                    rows.append(row)
                    region_rows[region] = row
            for target in [10*y+x for x in (3,5) for y in branch]:
                for y in branch:
                    source = 10*y
                    up, down, later = [region_rows[r] for r in (source, target, source+7)]
                    times = [r["arrival_s"] for r in (up, down, later)]
                    pairs.append(dict(family=name, center_y_m=center, source=source, target=target,
                        original_reverse_source=source+7,
                        forward_arrival_difference_s=None if any(v is None for v in times[:2]) else times[1]-times[0],
                        reverse_sensor_after_target_s=None if any(v is None for v in times[1:]) else times[2]-times[1],
                        reverse_is_later=bool(times[1] is not None and times[2] is not None and times[2] > times[1])))
            print(f"direction nominal pulse: {name}, center {center} complete", flush=True)
    windows = build_training_windows(rows)
    roles = {}
    for role in ("upstream", "target", "later_sensor"):
        selected = [r for r in rows if r["role"] == role]
        roles[role] = {key:describe([r[key] for r in selected]) for key in
                      ("arrival_after_pulse_s", "peak_abs_m", "peak_time_s", "first_peak_time_s")}
    result = dict(
        source_run=str(source_run), mesh=mesh, pulse_start_s=PULSE_START_S,
        calibration_role=DATA_ROLE, training_families=[f"lead_{r['family']:03}" for r in training],
        source_parameters_sha256=hashlib.sha256(parameter_path.read_bytes()).hexdigest(),
        nominal_response_files=files, rows=rows, pairs=pairs, training_windows=windows,
        summary=dict(role_statistics=roles,
            forward_arrival_difference_s=describe([r["forward_arrival_difference_s"] for r in pairs]),
            reverse_sensor_after_target_s=describe([r["reverse_sensor_after_target_s"] for r in pairs]),
            reverse_later_pair_count=sum(r["reverse_is_later"] for r in pairs),
            total_pairs=len(pairs)),
        interpretation=dict(
            reverse_identity="Original I5 uses each graph source +7: x5 -> x75, same y; x75 is downstream of both x35 and x55 targets",
            common_trend_contains_upstream=True,
            common_trend_monitor_count=len(MONITORS), common_trend_upstream_count=len(SOURCES),
            common_trend_limitation="I1 includes all visible monitor residuals including x5 upstream; I3 versus I1 tests extra spatial/timing representation above shared upstream information, not presence versus absence of upstream data",
            reverse_control_limitation="The x75 sensors belong to the same responding hydraulic system and may contain information about a common forcing or later response; I5 is not a strict causally irrelevant negative control",
            reflection_attribution="Right/top/bottom boundaries are reflective, but this audit does not decompose incident/reflected waves or establish reflection as the source of readout performance",
            arrival_detector_limitation="Arrival threshold uses a complete nominal response peak and is available only for offline training calibration, never as an online event detector",
            window_limitation="First local peak is a descriptive cutoff, not proof that the selected interval is reflection-free; translating windows by simulated true event onset is retrospective stratification only",
        ),
        formal_test_opened=False, policy_trained=False,
    )
    write_json(destination, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-run", required=True)
    args = parser.parse_args()
    result = audit(args.source_run, args.output_run)
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
