# 核心测试

修正轮28项测试通过，记录results/setup/core_tests_v2_final.xml。新增失败后可离开、可以选择重试、五组完整原始历史相同，以及两任务顺序与失败中断评分测试。恢复规则的两个回归测试在修复前均失败，记录results/setup/recovery_before.log；修复后通过。

此前22项测试遗漏失败后的移动权，旧轮结论更正见docs/FIRST_RUN_CORRECTION.md。复核时使用新basetemp目录：python -m pytest tests -q --basetemp=results/setup/pytest_review_unique。
