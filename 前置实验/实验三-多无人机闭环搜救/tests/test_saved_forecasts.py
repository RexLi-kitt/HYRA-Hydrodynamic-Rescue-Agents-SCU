import numpy as np
from hyra_exp3.formal_audit import SavedWindows
from hyra_exp3.simulation import EpisodeConfig


def test_readback_preserves_strict_float_threshold_and_unknown_horizon():
    data=dict(version=np.array([1,1,2]),row=np.array([0,0,0]),col=np.array([0,1,0]),
        publications=np.array([[1,0,15],[2,1200,1200]]),length=np.array([4,4,1]),
        future_times_s=np.array([[0,5,10,15],[0,5,10,15],[1200,np.nan,np.nan,np.nan]]),
        predicted_depth_m=np.array([[.79]*4,[np.float32(.8)]*4,[0,np.nan,np.nan,np.nan]],np.float32),
        evidence_supported=np.array([True,True,False]))
    saved=SavedWindows(data,EpisodeConfig())
    assert saved[1,(0,0)].can_start(0) is True
    assert saved[1,(0,0)].can_start(4) is None
    assert saved[1,(0,1)].can_start(0) is False
    assert saved[2,(0,0)].can_start(1200) is None
    assert saved.get((99,(0,0))) is None
