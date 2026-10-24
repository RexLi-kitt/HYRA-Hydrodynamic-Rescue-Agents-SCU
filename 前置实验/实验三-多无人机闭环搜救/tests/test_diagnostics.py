import numpy as np
import pytest

from hyra_exp3.diagnostics import safe_start_grid, compare_fields


def test_interior_flood_peak_invalidates_service_and_end_of_support_is_unknown():
    times = np.arange(0., 21., 2.)
    depths = np.full((len(times), 1), .2)
    depths[times == 6] = .9
    starts = safe_start_grid(times, depths)
    assert not starts[0, 0]
    assert starts[times == 8, 0].item()
    assert not starts[times > 8].any()


def test_fractional_service_endpoint_is_checked():
    times = np.array([0., 10., 20.])
    depths = np.array([[.2], [.7], [1.2]])
    assert safe_start_grid(times, depths, 12)[0, 0]
    assert not safe_start_grid(times, depths, 12.1)[0, 0]


def test_temporal_comparison_sees_hidden_peak_between_coarse_outputs():
    times = np.arange(15.)
    depth = np.full((15, 1), .2)
    depth[5] = .9
    fine = dict(times_s=times, depths_m=depth, positions_m=np.array([[5, 5]]))
    coarse = {**fine, "times_s": times[::2], "depths_m": depth[::2]}
    result = compare_fields(coarse, fine)
    assert result["depth_mae_m"] == 0
    assert result["complete_window_disagreement_fraction"] > 0


def test_invalid_time_support_is_rejected():
    with pytest.raises(ValueError):
        safe_start_grid([0, 0], [[.2], [.3]])
