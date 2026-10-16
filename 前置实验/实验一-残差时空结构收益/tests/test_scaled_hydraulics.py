import numpy as np
from hyra_validation.lead import hydraulic_domain
from hyra_validation.numerical_stability import point_weights, boundary_footprint
from hyra_validation.scaled_hydraulics import sampling_xy, scaled_domain

PARAMS = dict(slope=.0015, ridge=.07, rate=.003, onset=20., friction=.03)


def test_length100_matches_existing_model_state_and_mesh():
    old = hydraulic_domain(PARAMS, 20, factor=.99, pulse_amplitude=.1, pulse_start=180, pulse_center=20)
    new = scaled_domain(PARAMS, 20, 100, factor=.99, amplitude_m=.1, anchor_s=180)
    np.testing.assert_array_equal(old.get_centroid_coordinates(), new.get_centroid_coordinates())
    for name in ('stage', 'elevation', 'xmomentum', 'ymomentum'):
        np.testing.assert_array_equal(old.quantities[name].centroid_values, new.quantities[name].centroid_values)
    assert boundary_footprint(old) == boundary_footprint(new) == [10., 30.]


def test_physical_support_and_fixed_points_hold_across_meshes_and_lengths():
    for length in (100, 200, 300):
        xy = sampling_xy(length)
        for mesh in (20, 40, 80):
            for center in (20, 70):
                domain = scaled_domain(PARAMS, mesh, length, amplitude_m=.1, center_y_m=center)
                assert boundary_footprint(domain) == [center-10, center+10]
                weights = point_weights(domain.get_centroid_coordinates(), xy)
                rebuilt = (domain.get_centroid_coordinates()[weights['vertices']]*weights['weights'][:, :, None]).sum(axis=1)
                np.testing.assert_allclose(rebuilt, xy, atol=1e-10)


def test_bed_total_relief_is_preserved_when_length_changes():
    a = scaled_domain(PARAMS, 20, 100)
    b = scaled_domain(PARAMS, 20, 300)
    np.testing.assert_allclose(a.quantities['elevation'].centroid_values,
                               b.quantities['elevation'].centroid_values, atol=1e-14)
