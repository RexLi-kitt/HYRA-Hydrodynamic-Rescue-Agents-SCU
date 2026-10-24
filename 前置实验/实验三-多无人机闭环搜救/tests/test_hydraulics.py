import numpy as np

from hyra_exp3.hydraulics import family_parameters, finite_rise, perturbation, generate_field


def test_rise_stops_and_nominal_amplitude_is_lower():
    times = np.array([0, 120, 420, 720, 1200, 2000])
    np.testing.assert_allclose(finite_rise(times, .6, 600), [0, 0, .3, .6, .6, .6])
    assert perturbation(0, .2, 600) == 0
    assert perturbation(420, .2, 600) == .2
    assert family_parameters(1) == family_parameters(1)
    assert family_parameters(1) != family_parameters(2)


def test_actual_anuga_small_clock_and_metric_task_grid(tmp_path):
    record = generate_field(32, tmp_path, duration_s=2, output_interval_s=1,
                            mesh_spacing_m=10, family_seed=1)
    assert record["solver"] == "ANUGA"
    assert record["hydraulic_triangles"] == 4 * 32 * 32
    assert record["all_finite"]
    with np.load(record["field_path"]) as field:
        assert field["depths_m"].shape == (3, 1024)
        np.testing.assert_array_equal(field["times_s"], [0, 1, 2])
        np.testing.assert_array_equal(field["positions_m"][:2], [[5, 5], [15, 5]])
