import numpy as np
import pytest

from hyra_exp3.records import save_observations


def test_observation_log_preserves_measured_values_and_causal_times(tmp_path):
    records = [dict(sample_time=2.125, arrival_time=12.125, depth_m=.123456789012345,
                    cell=(69, 68), drone=19)]
    info = save_observations(tmp_path / "observations.npz", records)
    with np.load(info["path"], allow_pickle=False) as data:
        assert data["depth_m"][0] == records[0]["depth_m"]
        assert data["sample_time_s"][0] == 2.125
        assert data["arrival_time_s"][0] == 12.125
        assert data["row"][0] == 69 and data["drone"][0] == 19
    assert info["records"] == 1
    with pytest.raises(ValueError):
        save_observations(tmp_path / "bad.npz", [{**records[0], "arrival_time": 1}])
