"""Read-only verification of serialized replay records and their provenance."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from .artifacts import sha256, write_json
from .protocol import load_protocol


def _audit_job(path):
    return _audit_records(load_protocol(path))


def _audit_records(job):
    counts = dict(episodes=0, observation_files=0, queries=0, multi_visit_routes=0, route_visits=0,
                  checked_trajectory_segments=0, services=0)
    for row in job["records"]:
        old_record = row["baseline_episode"]
        if sha256(old_record["episode_path"]) != old_record["episode_sha256"]:
            raise AssertionError("Changed historical episode")
        old = load_protocol(old_record["episode_path"])
        for variant in row["variants"].values():
            path = Path(variant["episode_path"])
            if sha256(path) != variant["episode_sha256"]:
                raise AssertionError("Serialized episode changed")
            episode = load_protocol(path)
            c = episode["execution_record"]["config"]
            if c != old["execution_record"]["config"] or [t["cell"] for t in episode["targets"]] != [t["cell"] for t in old["targets"]]:
                raise AssertionError("Replay pairing differs")
            for key in ("observation_log", "query_log"):
                if sha256(episode[key]["path"]) != episode[key]["sha256"]:
                    raise AssertionError("Serialized raw log changed")
            with np.load(episode["observation_log"]["path"], allow_pickle=False) as observations:
                sample, arrival = observations["sample_time_s"], observations["arrival_time_s"]
                if np.any(arrival > c["duration_s"] + 1e-7) or not np.allclose(arrival - sample, c["observation_delay_s"], atol=1e-8, rtol=0):
                    raise AssertionError("Serialized observation causality differs")
                counts["observation_files"] += 1
            with np.load(episode["query_log"]["path"], allow_pickle=False) as q:
                pubs = q["publications"]
                if np.any(np.diff(pubs[:, 0]) != 1) or np.any(np.diff(pubs[:, 1]) < -1e-9):
                    raise AssertionError("Forecast publications are not monotone")
                indexes = np.searchsorted(pubs[:, 0], q["version"])
                if np.any(indexes >= len(pubs)) or np.any(pubs[indexes, 0] != q["version"]):
                    raise AssertionError("Unknown query publication")
                if not np.allclose(q["time_s"], pubs[indexes, 1], atol=1e-8, rtol=0):
                    raise AssertionError("Query not at its published issue time")
                if np.any(pubs[:, 2] > np.minimum(pubs[:, 1] + c["forecast_horizon_s"], c["duration_s"]) + 1e-8):
                    raise AssertionError("Forecast support exceeds limit")
                counts["queries"] += len(q["time_s"])
            latest = {}
            for point in episode["trajectories"]:
                rid = point["drone"]
                position, now = np.asarray(point["position_m"]), point["time_s"]
                if rid in latest:
                    before, previous = latest[rid]
                    if np.linalg.norm(position - before) > c["speed_mps"] * (now - previous) + 1e-5:
                        raise AssertionError("Serialized trajectory exceeds flight speed")
                    counts["checked_trajectory_segments"] += 1
                latest[rid] = (position, now)
            for event in episode["events"]:
                if event["event"] == "service_complete":
                    counts["services"] += 1
                if event["event"] == "plan":
                    for route in event.get("planner_diagnostics", {}).get("selected_routes", []):
                        counts["route_visits"] += len(route["targets"])
                        counts["multi_visit_routes"] += int(len(route["targets"]) > 1)
            counts["episodes"] += 1
    return counts


def _audit_control_job(path):
    job = load_protocol(path)
    return _audit_records(dict(records=[dict(baseline_episode=row["source_combined"],
                                            variants={"neutral": row}) for row in job["records"]]))


def audit_replay(source, workers=4):
    source = Path(source)
    summary = load_protocol(source / "planner_replay_summary.json")
    if summary["status"] != "completed":
        raise ValueError("Full replay not completed")
    for relative, expected in load_protocol(source / "source_sha256.json").items():
        if sha256(source / "source_snapshot" / relative) != expected:
            raise AssertionError("Frozen execution source differs")
    spec = load_protocol(source / "replay_spec.json")
    if sha256(Path(spec["source"]) / "phase_matrix_summary.json") != spec["source_summary_sha256"]:
        raise AssertionError("Historical source summary changed")
    totals = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_audit_job, path) for path in sorted((source / "jobs").glob("*/job_result.json"))]
        for n, future in enumerate(as_completed(futures), 1):
            for key, count in future.result().items():
                totals[key] = totals.get(key, 0) + count
            print(f"Serialized audit {n}/{len(futures)} jobs", flush=True)
    if totals["episodes"] != summary["episodes"] or totals["queries"] != summary["audit"]["queries"]:
        raise AssertionError("Audit counts differ from replay")
    result = dict(status="passed", **totals, frozen_source_hashes_passed=True, historical_baselines_preserved=True,
                  original_online_audit=summary["audit"], formal_test=False)
    write_json(source / "serialized_integrity_audit.json", result)
    return result


def audit_control(source, workers=4):
    source = Path(source)
    summary = load_protocol(source / "matched_control_summary.json")
    if summary["status"] != "completed":
        raise ValueError("Matched control incomplete")
    for relative, expected in load_protocol(source / "source_sha256.json").items():
        if sha256(source / "source_snapshot" / relative) != expected:
            raise AssertionError("Frozen control source differs")
    spec = load_protocol(source / "control_spec.json")
    if sha256(Path(spec["source"]) / "planner_replay_summary.json") != spec["source_summary_sha256"]:
        raise AssertionError("Matched source summary changed")
    totals = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_audit_control_job, path) for path in sorted((source / "jobs").glob("*/job_result.json"))]
        for n, future in enumerate(as_completed(futures), 1):
            for key, count in future.result().items():
                totals[key] = totals.get(key, 0) + count
            print(f"Matched serialized audit {n}/{len(futures)} jobs", flush=True)
    if totals["episodes"] != summary["episodes"] or totals["queries"] != summary["audit"]["queries"]:
        raise AssertionError("Matched audit counts differ")
    result = dict(status="passed", **totals, frozen_source_hashes_passed=True,
                  original_online_audit=summary["audit"], formal_test=False)
    write_json(source / "serialized_integrity_audit.json", result)
    return result
