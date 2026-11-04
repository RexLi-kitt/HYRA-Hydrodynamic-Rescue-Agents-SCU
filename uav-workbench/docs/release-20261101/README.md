# UAV rescue research snapshot — 2026-11-01

This change publishes the six-algorithm study, all fifteen pair candidates, and the completed held-out evaluation. It does not change the production flood solver or establish real-world rescue reliability.

## Start here

- [Completed strict-test summary](../strict-eval-20261101/strict-summary.csv), [full 5,630 trial records](../strict-eval-20261101/results.jsonl), and [pre-registered protocol](../strict-eval-20261101/protocol.md).
- [Complete nine-sheet workbook](../export-20261101/单双模型训练全量数据-20261101.xlsx) and [independent export audit](../export-20261101/export-audit.json).
- [Single-model training protocol](../singles1024-20261101/protocol.json), [pair experiment archive](../all-pairs-20261101/), and [implementation](../../experiments/flood1km-six-20261101/).
- [Ten development-validation-selected checkpoint parameter summaries](../top10-20261101/). These were chosen before examining final-test scores; they are not a new test-selected shortlist.
- [Evidence index](evidence-index.json) lists included artifacts and hashes of local-only weights, fields and trajectories. No failed result was removed from the text records. Absolute paths in archived scripts and manifests describe the original local runs, not portable installed entry points.

## Scope and results

Allowed implementations: HAPPO, HATRPO, HASAC, MAAC, MAT and COMA. MAPPO, QMIX, VDN, IPPO, DQN and DDPG are excluded. These are compact research implementations, not claims to reproduce the original papers.

All six single models completed 1,024 adaptation episodes. All fifteen pairs entered the 64-episode screen, five proceeded to 256 and two to 1,024. None qualified for the pre-registered 2,048 extension. Best checkpoints may precede the final episode; episode zero already includes the parent model's pretraining. Current adaptation totals 9,600 environment episodes; historical reused prefixes must not be added again.

The frozen test contains 50 unseen parent fields plus 20 independent direction/cluster-layout stress fields. Two pair assignments are averaged within each parent. The 5,630 trials include complete methods, no-coverage-planner ablations, pure-planner and random-exploration controls, and sampled-action supplements for the two representatives chosen before testing. Model weights are unchanged by evaluation; final-test scenes are now used and cannot remain an independent holdout for future tuning.

| Method with coverage/service rules | Main 50-field completion | Stress 20-field completion | Main potential conflicts per step |
| --- | ---: | ---: | ---: |
| MAT | 98% | 85% | 0.01323 |
| MAAC + COMA | 97% | 100% | 0.00431 |
| HATRPO | 96% | 100% | 0.00451 |
| Pure observed-information planner + service baseline | 98% | 100% | 0.00862 |

These are descriptive comparisons, not statistically established ranks or proof that learning beats planning. Completion requires observing every grid cell and completing service for all 20 targets. Potential-intent conflicts are not actual collision probabilities. The complete CSV retains every model, failure and control. Main-task completion without coverage planning is zero for all 21 learned candidates, so the high full-method completion cannot be attributed to the neural policies alone.

Only one independent training seed and one synthetic hydraulic-generator family were used. The stress test rotates fields/velocity vectors and clusters target layouts; it does not validate buildings, weather, communications, battery constraints or physical evacuation.

## Validation and execution notes

The archived [strict integrity audit](../strict-eval-20261101/integrity-audit.json) verifies the full unique task set, trajectory/metric reconciliation and unchanged weights. Information-boundary checks are archived beside it; passing those checks rules out the tested leakage paths, not every possible defect.

Independent trials were resumed from one to two, four and finally six CPU workers after TRAIN-only exact metric/trajectory equivalence tests. Completed records, seeds, roles, original aggregation order and cumulative time budget were preserved. Four/six/eight-worker timings were 18.42/15.52/14.79 seconds; the pre-registered marginal-gain rule selected six. The performance changes do not improve model quality.

All trials and aggregates completed before a subsequent workbook-export heap failure. The export was repaired with explicit temporary-object collection and a 6,144 MiB Node heap limit, with no global configuration change. All nine worksheets and 430 evaluation summaries, 16,642 per-trial evaluation records, 13,184 current/historical training records and 232 checkpoint entries remain. Exported table values and formula caches were independently verified. Historical export failures and scheduler handoffs remain in the archive.

The preparation README and earlier reports are historical snapshots. This release index, completed strict-test status and final export audit describe the final state. Fresh source regression results for this PR are recorded in `pr-validation.json`.

## Reuse limits

The experiment modules and their synthetic environment/learner unit tests can be reviewed in this checkout. Full historical replay additionally needs the local-only model weights, generated water fields and path remapping recorded in the evidence index. Archive scripts intentionally retain their original provenance paths; do not run them unchanged on another computer or directly restart a one-use evaluation into its existing output directory. No claim of clean-checkout, one-command reproduction of all historical training is made.
