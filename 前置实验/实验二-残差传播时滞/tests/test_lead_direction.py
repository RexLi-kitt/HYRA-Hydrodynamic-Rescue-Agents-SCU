import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_lead_direction import (DATA_ROLE, build_training_windows,
                                  classify_offline_window, response_summary)


def nominal_rows():
    rows = []
    for family, shift in (("train0", 0), ("train1", 2)):
        for region, role, arrival, peak in ((13, "target", 106, 124), (17, "later_sensor", 118, 136)):
            rows.append(dict(family=family, split="train", data_role=DATA_ROLE,
                             center_y_m=25, region=region, role=role,
                             arrival_s=arrival+shift, first_peak_time_s=peak+shift))
    return rows


def test_nominal_arrival_and_first_peak_are_offline_trajectory_statistics():
    times = np.arange(80, 141, 2)
    values = np.zeros(len(times))
    values[(times >= 100) & (times <= 110)] = np.arange(1, 7)*.002
    values[(times >= 112) & (times <= 120)] = np.arange(5, 0, -1)*.002
    values[times == 132] = .02
    result = response_summary(times, values)
    assert result["arrival_s"] == 100
    assert result["first_peak_time_s"] == 110
    assert result["peak_time_s"] == 132
    assert result["arrival_after_pulse_s"] == 10


def test_windows_are_training_only_and_do_not_accept_reference_or_test_labels():
    rows = nominal_rows()
    windows = build_training_windows(rows)
    assert windows["by_target"]["13"]["early_response"] == [17, 35]
    assert windows["by_target"]["13"]["pre_later_sensor"] == [17, 29]
    assert windows["online_feature_allowed"] is False
    assert windows["training_only"] is True
    with pytest.raises(ValueError, match="training nominal"):
        build_training_windows(rows + [dict(rows[0], split="test")])
    with pytest.raises(ValueError, match="training nominal"):
        build_training_windows([dict(row, data_role="reference_truth") for row in rows])


def test_offline_stratification_requires_explicit_nonfeature_role():
    windows = build_training_windows(nominal_rows())
    early = classify_offline_window(80, 60, 13, windows)
    assert early == dict(early_response=True, pre_later_sensor=True, later_response=False)
    assert classify_offline_window(96, 60, 13, windows)["later_response"]
    with pytest.raises(ValueError, match="offline"):
        classify_offline_window(80, 60, 13, dict(windows, online_feature_allowed=True))


def test_future_reference_changes_cannot_change_windows_or_current_features(simple_event):
    from hyra_validation.lead import features, sensor_records
    graph = {"13":[dict(source=10, travel_s=12, weight=1.)]}
    simple_event["metadata"]["family_index"] = 0
    records = sensor_records(simple_event, 0, 0, False)
    before = features(simple_event, records, 13, 100, 10, graph, "I3_physical_lag")
    windows_before = build_training_windows(nominal_rows())
    simple_event["truth_depth_m"][simple_event["time_s"] > 100] = 999
    records_after = sensor_records(simple_event, 0, 0, False)
    after = features(simple_event, records_after, 13, 100, 10, graph, "I3_physical_lag")
    np.testing.assert_array_equal(before, after)
    assert windows_before == build_training_windows(nominal_rows())
