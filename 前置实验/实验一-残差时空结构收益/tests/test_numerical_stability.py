import numpy as np
import pytest

from hyra_validation.numerical_stability import (point_weights, read_points, interval_maximum,
    scalar_order, boundary_footprint, fixed_domain)
from hyra_validation.lead import hydraulic_domain


def test_fixed_point_interpolation_preserves_affine_depth():
    xy = np.array([[0, 0], [0, 2], [2, 0], [2, 2], [1, 1]], float)
    query = np.array([[.4, .7], [1.4, 1.7]])
    weights = point_weights(xy, query)
    values = .1+.02*xy[:, 0]+.03*xy[:, 1]
    got = read_points(values[None], np.zeros(5), weights)['linear'][0]
    np.testing.assert_allclose(got, .1+.02*query[:, 0]+.03*query[:, 1], atol=1e-14)
    assert np.min(weights['weights']) >= 0


def test_output_sampling_can_hide_internal_peak():
    times = np.array([0, 1, 2, 3, 4])
    values = np.array([.2, .9, .2, .2, .2])
    assert interval_maximum(times, values, 0, 4) == .9
    assert interval_maximum(times[::2], values[::2], 0, 4) == .2
    with pytest.raises(ValueError): interval_maximum(times, values, 0, 5)


def test_full_service_scoring_retains_one_second_peaks():
    times = np.arange(61)
    depth = np.full((61, 2), .2); depth[13, 0] = .9
    score = scalar_order(times, depth, [0, 1], [1, 1], 0, 60)
    assert score['count'] == 1 and score['starts'] == [6, 24]
    coarse = scalar_order(times[::2], depth[::2], [0, 1], [1, 1], 0, 60)
    assert coarse['count'] == 2


def test_boundary_support_is_fixed_across_meshes_and_exposes_legacy_shift():
    params = dict(slope=.0015, ridge=.07, rate=.003, onset=20, friction=.03)
    for n in (10, 20, 40):
        np.testing.assert_allclose(boundary_footprint(fixed_domain(params, n, 1, 20)), [10, 30])
        np.testing.assert_allclose(boundary_footprint(fixed_domain(params, n, 1, 70)), [60, 80])
    old10 = hydraulic_domain(params, 10, factor=1, pulse_amplitude=.1, pulse_center=25)
    old20 = hydraulic_domain(params, 20, factor=1, pulse_amplitude=.1, pulse_center=25)
    assert boundary_footprint(old10) == [10, 30]
    assert boundary_footprint(old20) == [15, 35]
