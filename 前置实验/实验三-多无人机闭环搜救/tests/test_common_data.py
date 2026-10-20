import numpy as np
from hyra_exp3.common_data import collect_patrol, causal_frame

def field(side, value):
    yy, xx = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    return dict(times_s=np.arange(121.), depths_m=np.full((121, side*side), value, np.float32),
                elevation_m=np.zeros(side*side), positions_m=np.column_stack((xx.ravel(), yy.ravel())) * 10 + 5)

def test_common_patrol_is_geometry_only_and_samples_continuous_positions():
    a, raw, trace = collect_patrol(field(32,.2), field(32,.15),32,4,42,duration_s=30)
    b, _, other = collect_patrol(field(32,.7),field(32,.3),32,4,42,duration_s=30)
    assert trace == other
    assert int(a["count"].sum()) == len(raw)
    for rid in range(4):
        positions = np.asarray([r["position_m"] for r in trace if r["drone"] == rid])
        assert np.all(np.linalg.norm(np.diff(positions,axis=0),axis=1) <= 25 + 1e-8)

def test_delayed_or_future_records_and_truth_labels_cannot_change_features():
    panel, _, _ = collect_patrol(field(32,.2),field(32,.15),32,4,42,duration_s=60)
    original = causal_frame(panel,20,delay_s=10)
    panel["mean_depth_m"][3:] += 100
    panel["label_truth_sample_m"][:] = 999
    assert np.array_equal(original,causal_frame(panel,20,delay_s=10))
    assert original.shape == (1024,13,6)
    assert not original[:,-2:,3].any()
