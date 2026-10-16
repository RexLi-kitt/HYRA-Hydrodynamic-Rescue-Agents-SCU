import copy
import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("scan_decision_timing",
    Path(__file__).resolve().parents[1]/"scripts/scan_decision_timing.py")
SCAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN)


def graph():
    return {"23": [dict(source=10, travel_s=13., weight=.5),
                   dict(source=20, travel_s=23., weight=.5)]}


def test_full_grid_and_observation_costs(simple_event):
    assert len(SCAN.SCHEDULES) == 40
    schedule = dict(revisit=2, delay=10, phase=.25, name="test")
    geo = SCAN._schedule_geometry(simple_event, graph(), schedule)
    summary = geo["summary"]
    assert summary["batches"] == 140
    assert summary["scheduled_observations"] == 3360
    assert summary["arrivals_after_300_s"] == 120
    assert summary["actual_lead_s"]["min"] == 20.5
    assert summary["actual_lead_s"]["max"] == 28.5
    assert summary["missing_version_observations"] == 0
    for row in geo["timeline"]:
        assert all(b is None or b["arrival_time_s"] <= row["now_s"] for b in row["bins"])


def test_geometry_is_independent_of_residual_amplitude_and_kernel_collapse(simple_event):
    schedule = dict(revisit=5, delay=0, phase=0, name="test")
    a = SCAN._schedule_geometry(simple_event, graph(), schedule)
    changed = copy.deepcopy(simple_event)
    changed["truth_depth_m"] += 10
    changed["forecast_depth_m"] -= 5
    b = SCAN._schedule_geometry(changed, graph(), schedule)
    assert a["summary"] == b["summary"]
    np.testing.assert_array_equal(a["weights"], b["weights"])
    rows = a["summary"]["kernels"]
    assert rows[-1]["horizon_s"] == 30
    assert rows[-1]["physical_equals_recent_fraction"] == 1
    assert rows[0]["physical_vs_recent_l1"]["mean"] > 0
    assert rows[-1]["nominal_remaining_positive_fraction"] == 0


def test_constant_residual_has_different_weights_but_identical_aggregates(simple_event):
    simple_event["truth_depth_m"] += .3
    schedule = dict(revisit=5, delay=0, phase=0, name="test")
    geo = SCAN._schedule_geometry(simple_event, graph(), schedule)
    values, rows = SCAN._amplitudes(simple_event, graph(), schedule, geo)
    np.testing.assert_allclose(values, .3)
    assert rows[0]["physical_minus_recent_mae_m"] < 1e-15
    assert rows[0]["physical_vs_recent_correlation"] is None


def test_missing_history_remains_unavailable(simple_event):
    simple_event["release_s"] += 1000
    schedule = dict(revisit=10, delay=0, phase=0, name="test")
    geo = SCAN._schedule_geometry(simple_event, graph(), schedule)
    values, rows = SCAN._amplitudes(simple_event, graph(), schedule, geo)
    assert geo["summary"]["missing_version_observations"] == 696
    assert geo["summary"]["no_history_time_fraction"] == 1
    assert np.isnan(values).all()
    assert rows[0]["physical_minus_recent_mae_m"] is None
    assert geo["summary"]["kernels"][0]["physical_equals_recent_fraction"] is None


def test_vectorized_scan_matches_model_summary_on_varying_history(simple_event):
    from hyra_validation.decision_value import normalized_summary
    t = simple_event["time_s"]
    simple_event["truth_depth_m"][:, 10] += .1*np.sin(t/13)
    simple_event["truth_depth_m"][:, 20] += .2*np.cos(t/17)
    schedule = dict(revisit=5, delay=10, phase=.5, name="test")
    links = graph()["23"]
    geo = SCAN._schedule_geometry(simple_event, graph(), schedule)
    values, _ = SCAN._amplitudes(simple_event, graph(), schedule, geo)
    records = SCAN.scheduled_records(simple_event, 5, 10, .5)
    for ti in (3, 6, 20, 36):
        now = SCAN.TIMES[ti]
        histories = [SCAN.binned_history(records, link["source"], now) for link in links]
        for hi, tau in enumerate(SCAN.HORIZONS):
            for ki, kernel in enumerate(SCAN.KERNELS):
                expected = normalized_summary(histories, links, now, tau, kernel)[0]
                np.testing.assert_allclose(values[ti, 0, hi, ki], expected, atol=1e-14)


def test_phase_mean_then_equal_family_mean():
    rows = [dict(family="a", phase=i, error_m=v) for i, v in enumerate((0., 4., 8., 12.))]
    rows += [dict(family="b", phase=i, error_m=2.) for i in range(4)]
    families = SCAN._aggregate(rows, ("family",), "phase_count")
    assert [r["phase_count"] for r in families] == [4, 4]
    summary = SCAN._aggregate(families, (), "family_count")
    assert summary == [dict(family_count=2, error_m=4.)]
