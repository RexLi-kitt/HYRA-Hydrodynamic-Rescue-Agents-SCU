# HyRA 当前目标最终验收

P1/P2/P3、Sur-QMIX适配贯通、独立物理辅助首轮及本地交付均已按证据验收。科学门槛未通过不等于工程未完成；本轮作为负结果收束，不追加原P3第三轮或新机制第二轮。

| 要求 | 判定 | 证据 |
|---|---|---|
| P3_complete_two_rounds_strict_A1_effective_A2_permissions_budget_parent_audit | 已核验 | docs/evidence/20261025-p3-closeout/audit.json |
| P1_P2_engineering_scale_time_changes_costs | 已核验 | docs/P1_P2_ENGINEERING_20261025.md |
| Sur_original_paper_implementation_differences_declared | 已核验 | docs/SUR_QMIX_IMPLEMENTATION_AUDIT_20261025.md |
| Sur_actual_small_budget_train_execute_planned_resume | 已核验 | docs/evidence/20261025-sur-adaptation/preflight-audit.json |
| Frozen4groups_same_actor_two_equal_aux_heads3seeds_new_holdout | 已核验 | generated/sciml-round1/formal-plan.json |
| Actual_SWE_operator_internal_step_validation | 已核验 | docs/evidence/20261025-physics-operator/validation.json |
| All21events_labels_parent_source_separation | 已核验 | docs/evidence/20261025-sciml-development/data-audit-final.json |
| All1440000training_transitions_legal_images_separate_CTDE | 已核验 | docs/evidence/20261025-sciml-development/transition-audit-final.json |
| All135closed_loop_costs_changes_and_actual_actions | 已核验 | docs/evidence/20261025-sciml-development/trajectory-audit-final.json |
| All9720common_route_forecasts_no_future_input | 已核验 | docs/evidence/20261025-sciml-development/forecast-audit-final.json |
| Actual_initialization_budgets_selection_optimizer_target_RNG | 已核验 | docs/evidence/20261025-sciml-development/targets-audit-final.json |
| Exact_domains_NA_hierarchical_aggregation_gate | 已核验 | docs/evidence/20261025-sciml-development/statistics-audit-final.json |
| Prediction_physics_decision_cost_resources_distinct_negative_conclusion | 已核验 | docs/SCIML_FINAL_20261025.md |
| All69tests_and_frozen_sources | 已核验 | generated/sciml-learning-tests-20261025-current.log |
| Actor_only_CLI_actual_no_CTDE | 已核验 | docs/evidence/20261025-sciml-development/actor-cli-fit-validation.json |
| Safe_completed_resume_no_metadata_rewrite | 已核验 | docs/evidence/20261025-sciml-final/completed-resume-guard.json |
| Runnable3local_services_browser_desktop_mobile_and_tamper_checks | 已核验 | docs/evidence/20261025-team-viewer/system-qa.json |
| All_experiment_index_and_reproduce_commands | 已核验 | docs/REPRODUCE.md |
| Original2round_and_independent1round_seals_no_retraining | 已核验 | research/sciml_rounds.json |
| Baseline_source_configs_models_logs_all61690artifact_hashes | 已核验 | generated/delivery-20261025/package-audit.json |

全部保留实验产物61690个文件、47.76GiB已逐文件计算SHA256并保存清单；源码包2146文件、模型/协议包56文件独立回读验证。Git基线bundle验证通过，未提交、推送或外网部署。

Physics当前正确覆盖16.850%，Data19.434%，Sur20.647%，QMIX21.996%，R0为54.387%；Physics比Data低2.583个百分点、比Sur低3.797个百分点，0/9同时优于两者。预测改善未转化为实际任务收益。10/12后段漂移，不能声称已收敛或普遍否定SciML。

最终结论与配置见SCIML_FINAL_20261025.md、SCIML_PROTOCOL_20261025.md；运行/恢复/独立复现见REPRODUCE.md。交付入口为generated/delivery-20261025/README.md，包摘要见package-audit.json，全量路径和哈希见artifact-manifest.json。

此签收文件在不可变源码包之后生成，作为最终交付附件；不改包内源码或已冻结科学配置。历史阶段报告中的当时未完成状态保留，以本签收和最终结果为准。每五小时接续任务hyra已核实为PAUSED，保留原计划与提示，不再自动追加实验。
