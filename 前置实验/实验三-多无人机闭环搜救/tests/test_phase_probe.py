import numpy as np
import pytest

from hyra_exp3.phase_probe import mission_view, geometric_upper_bound


def test_phase_change_uses_actual_burn_in_outputs_and_preserves_mission_duration():
    times = np.arange(0., 2101., 2.)
    field = dict(times_s=times, depths_m=times[:, None] / 1000, positions_m=np.array([[5., 5.]]))
    view = mission_view(field, 900)
    assert view["times_s"][0] == 0 and view["times_s"][-1] == 1200
    assert view["depths_m"][0, 0] == .9 and view["depths_m"][-1, 0] == 2.1
    assert field["times_s"][0] == 0 and field["times_s"][-1] == 2100
    with pytest.raises(ValueError, match="without padding"):
        mission_view(field, 1000)


def test_geometry_bound_does_not_treat_every_safe_target_as_reachable_before_close():
    # Single drone starts at (5,5). The far target needs >14s of flight and
    # cannot be served in an interval ending at 5s, despite having a window.
    targets = [dict(cell=[5, 5], truth_safe_start_intervals_s=[[0., 20.]]),
               dict(cell=[0, 0], truth_safe_start_intervals_s=[[0., 5.]])]
    bounds = geometric_upper_bound(targets, (10, 10), 1)
    assert bounds["targets_with_any_complete_window"] == 2
    assert bounds["optimistic_individually_feasible_targets"] == 1
