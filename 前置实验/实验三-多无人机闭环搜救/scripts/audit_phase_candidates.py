"""Audit phase matrix provenance, oracle discovery rights, and saved-depth scoring."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
import numpy as np
from hyra_exp3.artifacts import sha256, write_json


def audit(source):
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    spec = read(source / "phase_matrix_spec.json")
    errors, fields, seen = [], {}, set()
    expected = {(j["family"]["family_id"], j["side"], c["rise_m"], c["rise_duration_s"], c["perturbation_m"], phase)
                for j in spec["jobs"] for c in j["conditions"] for phase in spec["phases_s"]}
    for name, checksum in spec["source_sha256"].items():
        path = ROOT / "run.py" if name == "entrypoint_run.py" else ROOT / "src" / "hyra_exp3" / name
        if sha256(path) != checksum:
            errors.append(f"Execution source changed: {name}")
    pairs = episodes = observations = queries = successes = rescored = strict_mismatches = boundary_events = 0
    mismatch_examples = []
    loaded_truth = None
    for path in sorted((source / "jobs").glob("*/condition_*/phase*/phase_case.json")):
        case = read(path)
        family, side, condition, phase = case["family"], case["side"], case["condition"], case["phase_s"]
        key = (family["family_id"], side, condition["rise_m"], condition["rise_duration_s"], condition["perturbation_m"], phase)
        if key not in expected or key in seen or family["split"] != "design":
            errors.append(f"Unexpected or duplicate phase case: {path}")
        seen.add(key)
        job = next(j for j in spec["jobs"] if j["family"]["family_id"] == family["family_id"] and j["side"] == side)
        pair_keys = {(p["layout_index"], p["delay_s"]) for p in case["pairs"]}
        if pair_keys != {(i, d) for i in range(2) for d in job["delays_s"]} or len(pair_keys) != len(case["pairs"]):
            errors.append(f"Missing or duplicate paired layout/delay: {path}")
        for kind in ("truth", "nominal"):
            manifest = case[kind]["manifest"]
            if (manifest["kind"] != kind or manifest["family_seed"] != family["family_seed"] or
                    not manifest["independent_initial_condition"] or manifest["duration_s"] != 1800):
                errors.append(f"Invalid field source: {path}, {kind}")
            fields[manifest["field_path"]] = manifest["field_sha256"]
        truth_path = case["truth"]["manifest"]["field_path"]
        if loaded_truth != truth_path:
            with np.load(truth_path, allow_pickle=False) as data:
                true_times, true_depths = data["times_s"], data["depths_m"]
            loaded_truth = truth_path
        for pair in case["pairs"]:
            pairs += 1
            paired_config = None
            seed = job["layout_seeds"][pair["layout_index"]]
            for method in ("nominal", "oracle"):
                info = pair[method]
                if sha256(info["episode_path"]) != info["episode_sha256"]:
                    errors.append(f"Episode checksum differs: {info['episode_path']}")
                episode = read(info["episode_path"])
                episodes += 1
                config = episode["execution_record"]["config"]
                if paired_config is None:
                    paired_config = config
                elif paired_config != config:
                    errors.append(f"Unmatched paired permissions: {path}")
                if config["seed"] != seed or config["observation_delay_s"] != pair["delay_s"] or config["duration_s"] != 1200:
                    errors.append(f"Wrong phase episode clock/seed: {path}")
                rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
                positions = np.argwhere(np.ones((side, side), bool))
                drawn = positions[rng.choice(side * side, config["n_targets"], replace=False)].tolist()
                if [t["cell"] for t in episode["targets"]] != drawn:
                    errors.append(f"Target draw changed or used water availability: {path}")
                observation = episode["observation_log"]
                if sha256(observation["path"]) != observation["sha256"]:
                    errors.append(f"Observation checksum differs: {observation['path']}")
                with np.load(observation["path"], allow_pickle=False) as log:
                    observations += len(log["sample_time_s"])
                    if (np.any(log["sample_time_s"] < 0) or np.any(log["arrival_time_s"] > 1200 + 1e-7) or
                            not np.allclose(log["arrival_time_s"] - log["sample_time_s"], pair["delay_s"], atol=1e-7, rtol=0)):
                        errors.append(f"Future or incorrectly delayed observation: {path}")
                successful = [e for e in episode["events"] if e["event"] == "service_complete" and e["success"]]
                if len({e["target"] for e in successful}) != len(successful) or len(successful) != episode["summary"]["successful_services"]:
                    errors.append(f"Duplicate success or score mismatch: {path}")
                successes += len(successful)
                for event in episode["events"]:
                    if event["event"] != "service_complete":
                        continue
                    rescored += 1
                    if abs(event["time_s"] - event["start_s"] - 12) > 1e-7:
                        errors.append(f"Incomplete service interval: {path}")
                    row, col = episode["targets"][event["target"]]["cell"]
                    start, end = phase + event["start_s"], phase + event["time_s"]
                    lo, hi = np.searchsorted(true_times, start, side="right"), np.searchsorted(true_times, end, side="left")
                    points = np.r_[start, true_times[lo:hi], end]
                    maximum = float(np.max(np.interp(points, true_times, true_depths[:, row * side + col])))
                    recomputed = maximum <= .8
                    boundary_events += abs(maximum - .8) <= 5e-7
                    if bool(event["success"]) != recomputed:
                        strict_mismatches += 1
                        if len(mismatch_examples) < 20:
                            mismatch_examples.append(dict(episode=info["episode_path"], target=event["target"], max_depth_m=maximum,
                                                          recorded=event["success"], recomputed=recomputed))
                        if abs(maximum - .8) > 5e-7:
                            errors.append(f"Service depth scoring mismatch outside numerical boundary band: {path}")
                for drone in episode["drones"]:
                    if drone["payload"] != 4 - sum(e["drone"] == drone["id"] for e in successful):
                        errors.append(f"Payload mismatch: {path}")
                if method == "oracle":
                    execution = episode["execution_record"]
                    if not execution["privileged_future_truth"] or execution["training_data_eligible"]:
                        errors.append(f"Privileged diagnostic mislabeled: {path}")
                    query = execution["query_log"]
                    if sha256(query["path"]) != query["sha256"]:
                        errors.append(f"Oracle query checksum differs: {path}")
                    discovered = np.full(side * side, np.inf)
                    for target in episode["targets"]:
                        if target["discovered_at"] is not None:
                            discovered[target["cell"][0] * side + target["cell"][1]] = target["discovered_at"]
                    with np.load(query["path"], allow_pickle=False) as log:
                        queries += len(log["time_s"])
                        if np.any(log["time_s"] < discovered[log["row"] * side + log["col"]] - 1e-7):
                            errors.append(f"Privileged predictor queried an undiscovered target: {path}")
    for path, checksum in fields.items():
        if sha256(path) != checksum:
            errors.append(f"Physical field checksum differs: {path}")
    if seen != expected or len(seen) != 432 or pairs != 1152 or episodes != 2304 or len(fields) != 288:
        errors.append("Candidate matrix coverage or unique field count is incomplete")
    report = dict(passed=not errors, errors=errors, phase_cases=len(seen), pairs=pairs, episodes=episodes,
                  unique_physical_fields=len(fields), legal_arrived_observations=observations,
                  discovered_target_oracle_queries=queries, successful_target_instances=successes,
                  service_intervals_independently_rescored=rescored,
                  strict_saved_depth_scoring_mismatches=strict_mismatches,
                  boundary_band_events=boundary_events, numerical_boundary_band_m=5e-7, mismatch_examples=mismatch_examples,
                  formal_test=False, spatial_convergence_checked=False, privileged_data_training_eligible=False)
    write_json(source / "phase_matrix_integrity_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    audit(parser.parse_args().source_run.resolve())
