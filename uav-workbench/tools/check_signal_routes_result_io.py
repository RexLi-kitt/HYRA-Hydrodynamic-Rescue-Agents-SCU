"""Exercise strict JSON, actual rescue events and the registered actor replay path.

Uses development events and an already completed recovery actor only. Does not
train, read confirmation scores, alter a checkpoint, or edit frozen sources.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
import json
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np

from signal_routes import evaluate
from signal_routes.belief import Parameters
from signal_routes.common import DEFAULT, read, write, digest, sources
from signal_routes.data import cases
from signal_routes.environment import RouteEnv
from scale100_env import config
from scale100_development import legal_rule
from scale100_rules import water_aware
from signal_routes_result_io import install_result_io_adapter, write_result_json


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def unit_checks(folder):
    path = folder / "scalars.json"
    payload = dict(integer=np.int64(7), floating=np.float32(0.5),
                   boolean=np.bool_(True), coordinates=(np.int64(3), np.int64(4)),
                   array=np.array([1, 2], dtype=np.int64), absent=None)
    write_result_json(path, payload)
    require(read(path) == dict(integer=7, floating=0.5, boolean=True,
                             coordinates=[3, 4], array=[1, 2], absent=None),
            "NumPy JSON conversion changed a value")
    original = path.read_bytes()
    for bad in (float("nan"), float("inf"), np.float64("nan"), np.complex128(1j)):
        try:
            write_result_json(path, dict(value=bad))
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError("Nonfinite or unsupported numeric value was accepted")
        require(path.read_bytes() == original, "Failed serialization replaced valid output")
    require(not list(folder.glob("*.tmp")), "Failed write left temporary files")
    return dict(passed=True, numpy_scalars_and_arrays=True,
                nonfinite_and_complex_rejected=True, failed_write_preserves_previous=True)


def run(study, output=None):
    study = Path(study).resolve()
    folder = Path(output).resolve() if output else study / "result-io-check"
    folder.mkdir(parents=True, exist_ok=True)
    before_sources = sources()
    actor = study / "recovery-full/R3-s11/actor.pt"
    frozen_paths = [study / "registration.json", study / "parameters.json",
                    study / "task-selection.json", study / "data/dataset.json",
                    actor, actor.with_suffix(".json")]
    frozen_hashes = {str(path): digest(path) for path in frozen_paths}
    original_actor_run = evaluate.actor_run
    original_view = RouteEnv.view
    original_common_writer = write
    receipt = install_result_io_adapter(study)
    require(install_result_io_adapter(study, receipt["adapter_sha256"]) == receipt,
            "Adapter installation is not idempotent")
    require(evaluate.actor_run is original_actor_run and RouteEnv.view is original_view,
            "I/O installation changed an inference entry point")
    from signal_routes import common
    require(common.write is original_common_writer, "Global writer was changed")
    try:
        install_result_io_adapter(expected_sha256="0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("Unexpected adapter hash was accepted")
    sentinel = lambda *_args: None
    evaluate.write = sentinel
    try:
        try:
            install_result_io_adapter()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Unknown writer was silently replaced")
        require(evaluate.write is sentinel, "Unknown writer was changed before rejection")
    finally:
        evaluate.write = write_result_json

    units = unit_checks(folder)
    task = read(study / "task-selection.json")
    candidate = next(row for row in task["rows"]
                     if row["duration_s"] == task["duration_s"] and row["metrics"]["rescued"] > 0)
    dev = list(cases(study / "data/dataset.json", "development"))
    row, case = next((r, c) for r, c in dev
                     if r["parent_terrain"] == candidate["parent"] and r["mechanism"] == candidate["mechanism"])
    parameters = Parameters(**read(study / "parameters.json")["parameters"]["Braw"])
    cfg = replace(config("E2", "B3", candidate["seed"]), duration_s=float(task["duration_s"]))
    env = RouteEnv(case, cfg, "Braw", parameters)
    env.reset()
    done = False
    while not done:
        actions = water_aware(env) if candidate["rule"] == "water_aware" else legal_rule(env, candidate["rule"])
        _, _, done = env.step(actions)
    rescued = [event for event in env.events if event["type"] == "rescued"]
    require(len(rescued) > 0, "Development replay had no successful rescue event")
    require(any(isinstance(value, np.integer) for event in rescued for value in event["cell"]),
            "Actual rescue events did not exercise the NumPy coordinate regression")
    raw = dict(events=env.events, actions=env.actions, observations=env.observations, metrics=env.metrics())
    try:
        json.dumps(raw, ensure_ascii=False, allow_nan=False)
    except TypeError:
        original_writer_reproduced = True
    else:
        raise AssertionError("Original serializer unexpectedly accepted the regression payload")
    rescue_path = folder / "successful-rule-rescue.json"
    evaluate.write(rescue_path, raw)
    saved = read(rescue_path)
    require(len([event for event in saved["events"] if event["type"] == "rescued"]) == len(rescued),
            "Successful rescue events were lost during serialization")
    require(all(type(value) is int for event in saved["events"] if "cell" in event for value in event["cell"]),
            "Event coordinates did not become JSON integers")

    replay_path = folder / "recovery-actor-development-replay.json"
    metrics = evaluate.actor_run(study, actor, "R3", 11, case, save_replay=replay_path)
    replay = read(replay_path)
    require(replay["replay_audit"] == dict(raw_records_equal=True, final_actor_view_equal=True, return_equal=True),
            "Registered actor replay audit failed")
    require(replay["source_sha256"] == case.digest, "Saved replay has a different event source")
    require(sources() == before_sources, "Frozen training sources changed")
    require({str(path): digest(path) for path in frozen_paths} == frozen_hashes,
            "Frozen study/model metadata changed")
    result = dict(schema="hyra-signal-routes-result-io-regression-v1", passed=True,
                  checker_sha256=digest(__file__), adapter=receipt, unit_checks=units,
                  development_only=True, confirmation_opened=False,
                  successful_rule_rescue=dict(parent=row["parent_terrain"], mechanism=row["mechanism"],
                      scenario_seed=candidate["seed"], rule=candidate["rule"], rescued=len(rescued),
                      original_json_type_error_reproduced=original_writer_reproduced,
                      replay=str(rescue_path), replay_sha256=digest(rescue_path)),
                  registered_actor_save_replay=dict(actor=str(actor), actor_sha256=digest(actor),
                      replay=str(replay_path), replay_sha256=digest(replay_path), metrics=metrics,
                      audit=replay["replay_audit"]),
                  frozen_sources_and_metadata_unchanged=True)
    write_result_json(study / "result-io-regression.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, default=DEFAULT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.study, args.output)
    print(json.dumps(dict(passed=result["passed"], adapter_sha256=result["adapter"]["adapter_sha256"],
                          rescued=result["successful_rule_rescue"]["rescued"],
                          registered_actor_replay_audit=result["registered_actor_save_replay"]["audit"])))


if __name__ == "__main__":
    main()
