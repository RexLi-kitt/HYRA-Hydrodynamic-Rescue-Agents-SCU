"""Statistical units and attribution safeguards in the decision report."""
import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[1] / "scripts" / "finalize_decision_value.py"
spec = importlib.util.spec_from_file_location("finalize_decision_value", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def task_row(family, phase, value, cases=100, delay=0, method="physical"):
    return dict(schedule="unused", revisit=5, delay=delay, phase=phase,
                split="test", family=family, condition="C1_local", method=method,
                cases=cases, mean_yield=value, mean_delta_nominal=value,
                mean_regret=2-value, mean_baseline_regret=2,
                positive=cases//2, negative=0, changed=cases//2)


def test_phases_then_families_not_pooled_cases_and_delay_separate():
    source = []
    for delay in (0, 10):
        for family, cases, value in (("a", 1000, 0.), ("b", 1, 1.)):
            source.extend(task_row(family, p, value+delay, cases, delay)
                          for p in (0, .25, .5, .75))
    family = module.phase_average(source, "task")
    summary = module.family_summary(family)
    assert len(summary) == 2
    assert [r["mean_yield"] for r in summary] == [.5, 10.5]
    assert all(r["independent_families"] == 2 for r in summary)
    assert all(r["phase_count"] == 4 for r in summary)


def test_incomplete_or_duplicate_phases_cannot_silently_change_estimand():
    source = [task_row("a", p, 1) for p in (0, .25, .5)]
    with pytest.raises(ValueError, match="Missing/duplicate phases"):
        module.phase_average(source, "task")
    source.append(task_row("a", .5, 1))
    with pytest.raises(ValueError, match="Missing/duplicate phases"):
        module.phase_average(source, "task")


def test_paired_delta_preserves_family_values_and_rejects_unmatched_family():
    source = [task_row(f, p, v, method=m)
              for f, v in (("a", 1.), ("b", 2.))
              for p in (0, .25, .5, .75) for m in ("physical", "uniform")]
    for row in source:
        if row["method"] == "uniform":
            row["mean_yield"] -= .1 if row["family"] == "a" else -.2
    family = module.phase_average(source, "task")
    comparison = module.paired_comparisons(family, "mean_yield")[0]
    assert comparison["mean_delta"] == pytest.approx(-.05)
    assert comparison["favorable_families"] == 1
    assert comparison["adverse_families"] == 1
    assert len(comparison["family_values"]) == 2
    with pytest.raises(ValueError, match="Unpaired families"):
        module.paired_comparisons([r for r in family if not (r["family"] == "a" and r["method"] == "uniform")], "mean_yield")


def test_zero_opportunity_is_missing_capture_rate_not_zero():
    source = []
    for phase in (0, .25, .5, .75):
        row = task_row("a", phase, 0)
        row.update(correctable_cases=0, captured_cases=0, initial_gain_cases=0, late_only_gain_cases=0)
        source.append(row)
    family = module.phase_average(source, "window")
    summary = module.family_summary(family)[0]
    assert summary["capture_rate_correctable"] is None
    assert summary["correctable_rate_all"] == 0


def test_nominal_gain_does_not_imply_residual_attribution():
    source = [task_row(f, p, 1, method=m)
              for f in ("a", "b", "c") for p in (0, .25, .5, .75)
              for m in module.METHODS]
    family = module.phase_average(source, "task")
    summary = module.family_summary(family)
    comparison = module.paired_comparisons(family, "mean_yield")
    result = module.recommendation(summary, comparison, [], False)
    assert result["status"] == "residual_increment_not_consistent"
    assert result["qmix_ready"] is False
    assert result["formal_test"] is False
    assert len(result["independent_test_families"]) == 3
