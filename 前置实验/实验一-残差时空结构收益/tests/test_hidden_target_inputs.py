import numpy as np
import pytest

from hyra_validation.hidden_target_inputs import SourceRecord, causal_history, aligned_summary


def record(index, source=10, valid=90., arrival=90., release=60., observed=.5, prior=.4):
    return SourceRecord(index, source, valid, arrival, release, observed, prior)


def test_delayed_future_and_old_records_cannot_enter_features():
    rows = [record(0), record(1, valid=95, arrival=105), record(2, valid=110, arrival=110),
            record(3, valid=40, arrival=40, release=20)]
    got = causal_history(rows, 100)
    assert got['record_ids'][0, 1] == 0
    assert np.count_nonzero(got['record_ids'] >= 0) == 1
    assert got['values'][0, 1, 2] == pytest.approx(.1)
    assert got['values'][0, 1, 4] == 30


def test_target_observations_and_future_forecast_vintages_are_rejected():
    with pytest.raises(ValueError, match='upstream'): causal_history([record(0, source=13)], 100)
    with pytest.raises(ValueError, match='version'): causal_history([record(0, release=80)], 100)
    with pytest.raises(ValueError, match='unique'): causal_history([record(0), record(0)], 100)


def test_newest_record_per_bin_and_arrival_equal_to_decision_is_available():
    got = causal_history([record(1, valid=94, arrival=100, release=70),
                          record(0, valid=91, arrival=91, release=60)], 100)
    assert got['record_ids'][0, 0] == 1
    assert np.count_nonzero(got['record_ids'] >= 0) == 1


def test_physics_and_recency_share_support_and_references_but_align_different_ages():
    history = causal_history([record(0, valid=100, arrival=100, release=80, observed=.4),
                              record(1, valid=80, arrival=80, release=60, observed=.6)], 100)
    links = [(10, 1., 26.)]
    physical = aligned_summary(history, links, [6., 30.], bandwidth_s=2.)
    recent = aligned_summary(history, links, [6., 30.], mode='recency', bandwidth_s=2.)
    np.testing.assert_allclose(physical['features'][:, 1:], recent['features'][:, 1:])
    np.testing.assert_array_equal(physical['record_ids'], recent['record_ids'])
    assert physical['features'][0, 0] == pytest.approx(.2)
    assert recent['features'][0, 0] == pytest.approx(0., abs=1e-15)
    np.testing.assert_allclose(physical['normalized_weights'][1], recent['normalized_weights'][1])


def test_missing_sources_do_not_create_fake_support_or_nan_weights():
    history = causal_history([], 100)
    got = aligned_summary(history, [(10, 1., 20.)], [6.])
    assert np.all(got['features'] == 0) and np.all(got['normalized_weights'] == 0)
