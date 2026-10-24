"""Draft experiment contract, fresh family isolation, and explicit matrix arithmetic."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path

from .artifacts import write_json

SPLIT_COUNTS = {"design": 6, "train": 24, "validation": 12, "test": 36}
SCALES = {32: (4, 12), 50: (10, 30), 70: (20, 60)}
METHODS = {32: ["H", "E", "P"], 50: ["H", "E", "P", "P_wrong_links", "P_uniform_time"], 70: ["H", "E", "P"]}


def default_protocol():
    return dict(
        study_id="hyra-experiment3-20261018-v1", status="development_not_frozen", master_seed=2026101803,
        source_document="docs/protocols/实验三原始方案.md", family_counts=SPLIT_COUNTS,
        scales=[dict(side=side, cells=side * side, drones=n, hidden_targets=t, methods=METHODS[side])
                for side, (n, t) in SCALES.items()],
        episode=dict(duration_s=1200, cell_size_m=10, speed_mps=5, payload=4,
                     sensor_shape=[5, 5], sampling_period_s=5, planning_period_s=10,
                     service_duration_s=12, max_service_depth_m=.8, observation_noise_m=.02,
                     main_delay_s=0, extension_delay_s=10),
        hydrology=dict(onset_s=120, rise_m=[.6, .9], rise_duration_s=[600, 900],
                       local_perturbation_m=[0, .1, .2], nominal_rise_factor=.75,
                       post_rise="hold", design_conditions="full_factorial_per_design_family",
                       formal_condition_allocation="unresolved"),
        evaluation=dict(layout_noise_configs=2, model_training_seeds=[101, 202, 303],
                        primary_baseline="select_H_or_E_once_on_validation_then_freeze",
                        family_paired=True, bootstrap_unit=["family", "training_seed"],
                        engineering_target_percentage_points=3,
                        quoted_main_episodes=2376, quoted_delay_episodes=648,
                        quoted_total_episodes=3024, delay_methods="unresolved"),
        development_choices=dict(hydraulic_mesh_m=[10, 5], output_intervals_s=[2, 1],
                                 forecast_horizon_s=60, nominal_only_preflight=True,
                                 greedy_planner_is_formal_planner=False),
        formal_test_enabled=False, configuration_frozen=False,
        unresolved_before_freeze=["Assign one hydrology condition per family or cross all twelve within each family",
                                  "Specify whether delay extension uses three or five methods",
                                  "Complete six-family design audit and numerical tolerances",
                                  "Implement and train five capacity-matched information modules",
                                  "Freeze planner, training, fixed-route replay and validation B* selection"],
    )


def load_protocol(path):
    value=json.loads(Path(path).read_text(encoding="utf-8"))
    origin=os.environ.get('HYRA3_RELOCATE_FROM')
    if origin:
        value=relocate_paths(value,origin,Path(__file__).resolve().parents[2])
    return value


def relocate_paths(value,origin,destination):
    """Read archived absolute paths from an exported bundle without rewriting evidence bytes."""
    if isinstance(value,dict):return {k:relocate_paths(v,origin,destination) for k,v in value.items()}
    if isinstance(value,list):return [relocate_paths(v,origin,destination) for v in value]
    if isinstance(value,str):
        normalized=value.replace('\\','/');prefix=str(origin).replace('\\','/').rstrip('/')
        if normalized.lower()==prefix.lower():return str(Path(destination))
        if normalized.lower().startswith(prefix.lower()+'/'):
            return str(Path(destination)/Path(normalized[len(prefix)+1:]))
    return value


def family_split(protocol):
    families = []
    for split, count in protocol["family_counts"].items():
        for index in range(count):
            identifier = f"exp3-{split}-{index:03d}"
            namespace = f"{protocol['study_id']}:{protocol['master_seed']}:{identifier}"
            seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:4], "big")
            families.append(dict(family_id=identifier, split=split, family_seed=seed,
                                 index_within_split=index,
                                 isolation="all_scales_hydrologies_layouts_remain_in_this_split"))
    return dict(study_id=protocol["study_id"], status="development_registration_not_model_freeze",
                families=families, actual_test_simulations=0)


def validate_protocol(protocol):
    split = family_split(protocol)["families"]
    errors = []
    if protocol["family_counts"] != SPLIT_COUNTS:
        errors.append("Family counts differ from the source plan")
    if len({row["family_seed"] for row in split}) != len(split):
        errors.append("Family seed collision")
    if len({row["family_id"] for row in split}) != len(split):
        errors.append("Duplicate family ID")
    for row in protocol["scales"]:
        side = row["side"]
        if side not in SCALES or (row["drones"], row["hidden_targets"]) != SCALES[side] or row["cells"] != side * side:
            errors.append("Scale geometry differs from source plan")
    ev = protocol["evaluation"]
    repetitions = protocol["family_counts"]["test"] * ev["layout_noise_configs"] * len(ev["model_training_seeds"])
    conditions = len(protocol["hydrology"]["rise_m"]) * len(protocol["hydrology"]["rise_duration_s"]) * len(protocol["hydrology"]["local_perturbation_m"])
    main = repetitions * sum(len(row["methods"]) for row in protocol["scales"])
    return dict(
        structural_checks_passed=not errors, errors=errors,
        registered_families=len(split), family_counts=protocol["family_counts"], hydrology_combinations=conditions,
        quoted_counts=dict(main=2376, delay=648, total=3024),
        one_condition_per_family=dict(main=main, delay_three_methods=repetitions * 3,
                                      total_with_three_delay_methods=main + repetitions * 3,
                                      total_with_five_delay_methods=main + repetitions * 5),
        all_conditions_per_family=dict(main=main * conditions, delay_three_methods=repetitions * 3 * conditions,
                                      total_with_three_delay_methods=(main + repetitions * 3) * conditions,
                                      total_with_five_delay_methods=(main + repetitions * 5) * conditions),
        count_interpretation="3024 implies exactly one preassigned hydrology per family and three delayed methods",
        formal_test_ready=False, configuration_frozen=False, actual_test_simulations=0,
        unresolved_before_freeze=protocol["unresolved_before_freeze"],
    )


def initialize(root):
    root = Path(root)
    path = root / "configs" / "experiment3.v1.json"
    if path.exists():
        protocol = load_protocol(path)
    else:
        protocol = default_protocol()
        write_json(path, protocol)
    expected = family_split(protocol)
    split_path = root / "manifests" / "family_split.json"
    if split_path.exists():
        if load_protocol(split_path) != expected:
            raise ValueError("Registered split changed; create a new study version instead of overwriting")
    else:
        write_json(split_path, expected)
    conditions = [dict(rise_m=r, rise_duration_s=t, perturbation_m=p)
                  for r, t, p in itertools.product(protocol["hydrology"]["rise_m"],
                      protocol["hydrology"]["rise_duration_s"], protocol["hydrology"]["local_perturbation_m"])]
    write_json(root / "manifests" / "design_conditions.json", dict(
        stage="planned_not_executed", family_ids=[r["family_id"] for r in expected["families"] if r["split"] == "design"],
        conditions=conditions, scales=[32, 50, 70], selection_rule="do_not_filter_by_method_advantage",
        full_design_truth_fields=6 * len(conditions) * 3, formal_test=False))
    return protocol
