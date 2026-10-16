from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_local_gain_ablation import keep_latest,inputs


def test_latest_is_per_source_and_keeps_missing_history_missing():
    x=np.zeros((1,134));x[0,8:12]=[.3,.2,.4,1];x[0,16:20]=[.4,.2,.8,1]
    x[0,24:28]=[.5,.2,.1,1];x[0,28:32]=[.6,.2,.2,1];x[0,96:]=7
    y=keep_latest(x)
    assert y[0,8]==.3 and y[0,24]==.5
    assert not y[0,16:20].any() and not y[0,28:96].any()
    np.testing.assert_array_equal(x[:,96:],y[:,96:])


def test_signed_residual_replaces_absolute_values_without_new_information():
    x=np.zeros((1,134));x[0,:4]=[.25,.3,.1,1]
    result=inputs(dict(x=dict(raw_self=x,raw_full=x)))
    assert np.isclose(result['residual_latest'][0,0],-.05)
    assert result['residual_latest'][0,1]==0
    np.testing.assert_array_equal(result['self_latest'][0,:4],x[0,:4])
    np.testing.assert_array_equal(x[0,:4],[.25,.3,.1,1])
