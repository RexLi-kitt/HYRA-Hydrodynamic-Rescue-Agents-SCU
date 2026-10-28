# 无人机工作台工具导航

日常代码检查：从 `uav-workbench` 运行 `python tools/check_workbench.py`。已有完整本地研究资料与学习环境时，加 `--integration` 核验真实交互回放。该入口不训练、不选模。

| 用途 | 常用脚本 |
|---|---|
| 统一代码、HTTP与回放检查 | `check_workbench.py` |
| 独立clean actor与双机查看 | `run_team_policy.py`、`team_replay_viewer.py` |
| 开发数据和工程审计 | `bootstrap_research.py`、`study_input_audit.py`、`engineering_audit.py` |
| P3与尺度报告 | `p3_report.py`、`p3_closeout.py`、`scales_report.py`、`strong_report.py` |
| 研究索引与源码封存 | `experiment_index.py`、`freeze_archive.py` |
| 历史Sur-QMIX适配核查 | `run_sur_preflight_sequence.py`、`audit_sur_preflight.py`、`audit_team_replays.py` |
| 历史物理辅助评价与审计 | `evaluate_sciml.py`、`report_sciml.py`、`audit_sciml_*.py`、`validate_physics_operator.py` |
| 三路径训练、汇总与交付 | `run_signal_routes_study.py`、`report_signal_routes_study.py`、`finalize_signal_routes_study.py`、`audit_signal_routes_*.py` |
| 三路径诊断与结果格式适配 | `signal_routes_direction_diagnostic.py`、`signal_routes_independent_null.py`、`signal_routes_result_io.py`、`check_signal_routes_*.py` |
| 对应历史任务接续 | `finish_study.py`、`finish_sciml*.py`、`resume_sciml.py` |

研究脚本的参数与执行顺序以对应轮次登记为准，不能把带 `run`、`finish`、`resume` 的脚本当作普通代码检查运行。已封存实验不在本轮重训。100米实验的根目录脚本见[代码导航](../docs/CODE_MAP.md)；历史复现见[运行说明](../docs/REPRODUCE.md)。
