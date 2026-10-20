"""Audit the completed design corpus independently of reported success scores."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
import numpy as np
from hyra_exp3.artifacts import sha256, write_json
from hyra_exp3.design import execution_hashes


def audit(source):
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    spec = read(source / "design_spec.json")
    errors, fields = [], {}
    expected = set()
    for job in spec["jobs"]:
        for condition in job["conditions"]:
            expected.add((job["family"]["family_id"], job["side"], condition["rise_m"],
                          condition["rise_duration_s"], condition["perturbation_m"]))
    seen, episode_count, observation_count, completed_targets = set(), 0, 0, 0
    if execution_hashes(ROOT) != spec["execution_source_sha256"]:
        errors.append("Current execution modules differ from the run specification")
    for path in sorted((source / "jobs").glob("*/condition_*/case_summary.json")):
        case = read(path)
        family, side, condition = case["family"], case["side"], case["condition"]
        key = (family["family_id"], side, condition["rise_m"], condition["rise_duration_s"], condition["perturbation_m"])
        if family["split"] != "design" or key not in expected or key in seen:
            errors.append(f"Unexpected or duplicate design case: {path}")
        seen.add(key)
        for kind in ("truth", "nominal"):
            manifest = case[kind]["manifest"]
            if manifest["kind"] != kind or not manifest["independent_initial_condition"] or manifest["family_seed"] != family["family_seed"]:
                errors.append(f"Invalid {kind} source: {path}")
            fields[manifest["field_path"]] = manifest["field_sha256"]
        job = next(j for j in spec["jobs"] if j["family"]["family_id"] == family["family_id"] and j["side"] == side)
        pairs = {(e["layout_index"], e["delay_s"]) for e in case["episodes"]}
        if pairs != {(i, d) for i in range(len(job["layout_seeds"])) for d in job["delays_s"]} or len(pairs) != len(case["episodes"]):
            errors.append(f"Missing or duplicate layout/delay: {path}")
        for result in case["episodes"]:
            episode_count += 1
            episode = read(result["episode_path"])
            config = episode["execution_record"]["config"]
            seed = job["layout_seeds"][result["layout_index"]]
            if config["seed"] != seed or config["observation_delay_s"] != result["delay_s"]:
                errors.append(f"Layout/noise seed or delay differs: {result['episode_path']}")
            rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
            expected_targets = np.argwhere(np.ones((side, side), bool))[rng.choice(side * side, config["n_targets"], replace=False)].tolist()
            if [t["cell"] for t in episode["targets"]] != expected_targets:
                errors.append(f"Target draw is not geometry-only: {result['episode_path']}")
            if len(episode["targets"]) != config["n_targets"] or not episode["summary"]["all_hidden_targets_in_denominator"]:
                errors.append(f"Hidden target denominator differs: {result['episode_path']}")
            observation = episode["observation_log"]
            if sha256(observation["path"]) != observation["sha256"]:
                errors.append(f"Observation checksum mismatch: {observation['path']}")
            with np.load(observation["path"], allow_pickle=False) as log:
                observation_count += len(log["sample_time_s"])
                if (np.any(log["sample_time_s"] < 0) or np.any(log["arrival_time_s"] > config["duration_s"] + 1e-7) or
                        not np.allclose(log["arrival_time_s"] - log["sample_time_s"], result["delay_s"], atol=1e-7, rtol=0)):
                    errors.append(f"Observation causal clock differs: {observation['path']}")
                if (np.any(log["row"] < 0) or np.any(log["row"] >= side) or np.any(log["col"] < 0) or np.any(log["col"] >= side) or
                        np.any(log["drone"] < 0) or np.any(log["drone"] >= config["n_drones"])):
                    errors.append(f"Observation location or drone invalid: {observation['path']}")
            successful = [e for e in episode["events"] if e["event"] == "service_complete" and e["success"]]
            if len({e["target"] for e in successful}) != len(successful) or len(successful) != episode["summary"]["successful_services"]:
                errors.append(f"Duplicate success or score mismatch: {result['episode_path']}")
            completed_targets += len(successful)
            for event in episode["events"]:
                if event["event"] == "service_complete" and abs(event["time_s"] - event["start_s"] - config["service_duration_s"]) > 1e-7:
                    errors.append(f"Incomplete service scoring: {result['episode_path']}")
                if event["event"] == "service_start":
                    discovered = episode["targets"][event["target"]]["discovered_at"]
                    if discovered is None or event["time_s"] < discovered - 1e-7:
                        errors.append(f"Service used an undiscovered target: {result['episode_path']}")
            for drone in episode["drones"]:
                used = sum(e["drone"] == drone["id"] for e in successful)
                if drone["payload"] != config["payload"] - used or drone["payload"] < 0:
                    errors.append(f"Payload scoring differs: {result['episode_path']}")
    for path, checksum in fields.items():
        if sha256(path) != checksum:
            errors.append(f"Physical field checksum mismatch: {path}")
    if seen != expected or len(seen) != 216 or episode_count != 576 or len(fields) != 288:
        errors.append("Full matrix coverage or unique physical field count differs")
    report = dict(passed=not errors, errors=errors, cases=len(seen), episodes=episode_count,
                  unique_physical_fields=len(fields), legal_arrived_observations=observation_count,
                  successful_target_instances=completed_targets, formal_test=False,
                  numerical_convergence_checked=False, raw_service_depth_rescored=False,
                  checks=["source_and_field_and_observation_integrity", "complete_design_matrix",
                          "design_family_isolation", "geometry_only_target_draws", "arrived_observation_times",
                          "hidden_target_denominator", "complete_service_duration", "unique_successes", "payload_accounting"])
    write_json(source / "design_integrity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    audit(parser.parse_args().source_run.resolve())
