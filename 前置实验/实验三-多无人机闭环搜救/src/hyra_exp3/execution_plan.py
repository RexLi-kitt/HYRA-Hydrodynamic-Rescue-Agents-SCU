"""Full experiment scope and outcome-independent hydrology allocation."""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from .artifacts import sha256, write_json
from .protocol import load_protocol

METHODS = ("H", "E", "P", "P_wrong_links", "P_uniform_time")


def balanced_allocation(families, study_id):
    conditions = [dict(rise_m=r, rise_duration_s=d, perturbation_m=p)
                  for r, d, p in itertools.product((.6, .9), (600, 900), (0., .1, .2))]
    output = []
    for split in ("train", "validation", "test"):
        selected = sorted((f for f in families if f["split"] == split),
            key=lambda f: hashlib.sha256(f"{study_id}:formal-hydrology-v1:{f['family_id']}".encode()).digest())
        for index, family in enumerate(selected):
            output.append(dict(**family, condition_index=index % len(conditions),
                               hydrology=conditions[index % len(conditions)]))
    return output


def register(root):
    root = Path(root)
    protocol = load_protocol(root / "configs/experiment3.v1.json")
    families = load_protocol(root / "manifests/family_split.json")["families"]
    plan = dict(schema="hyra3-full-execution-plan-v1", study_id=protocol["study_id"],
        source_document_sha256=sha256(root / protocol["source_document"]),
        family_split_sha256=sha256(root / "manifests/family_split.json"),
        mission_start_physical_s=0., mission_duration_s=1200.,
        development_burn_in_phases_are_not_formal_conditions=True,
        hydrology_interpretation="one_balanced_preassigned_condition_per_family_as_required_by_3024_episodes",
        allocation=balanced_allocation(families, protocol["study_id"]),
        design_hydrology="all_12_conditions_per_family", train_scales=[32, 50, 70],
        methods=list(METHODS), endpoint_methods=["H", "E", "P"], delay_methods=["H", "E", "P"],
        training_seeds=[101, 202, 303], layout_noise_configs=2,
        main_test_episodes=2376, delayed_test_episodes=648, total_formal_test_episodes=3024,
        select_B_star_once_on_validation=True, select_config_without_test_access=True,
        formal_test_enabled=False, configuration_frozen=False,
        numerical_criteria=dict(depth_mae_m=.005, depth_p99_abs_m=.02,
            complete_window_disagreement_fraction=.005, comparison="all_design_families_conditions_scales",
            threshold_changes_in_executed_services_must_be_reported=True,
            finer_reference_is_not_an_exact_solution=True),
        required_deliverables=[
            "design_numerical_and_clock_checks", "planner_candidate_and_budget_sensitivity",
            "common_causal_training_records", "five_parameter_matched_models_three_training_seeds",
            "training_only_statistical_and_validated_physical_links_and_lags",
            "fixed_route_identical_record_prediction_and_window_evaluation",
            "validation_B_star_and_configuration_freeze", "single_independent_3024_episode_test",
            "paired_family_and_training_seed_95_percent_intervals_and_3pp_target",
            "causal_reassignment_counterfactuals", "numerical_direction_stability_of_final_gain",
            "independent_reproducible_package_frozen_configuration_raw_statistics_final_report"],
        interpretation_note="Preserves the source's original phase and exact episode arithmetic; never select scenarios by P gains")
    path = root / "manifests/full_execution_plan.v1.json"
    if path.exists() and load_protocol(path) != plan:
        raise ValueError("Registered execution plan changed; version explicitly")
    if not path.exists():
        write_json(path, plan)
    ledger = root / "COMPLETION.json"
    if not ledger.exists():
        write_json(ledger, dict(objective="完成实验三", complete=False, plan=str(path),
            requirements={name: dict(status="not_yet_verified", evidence=[]) for name in plan["required_deliverables"]},
            formal_test_episodes=0, models_trained=False))
    return plan
