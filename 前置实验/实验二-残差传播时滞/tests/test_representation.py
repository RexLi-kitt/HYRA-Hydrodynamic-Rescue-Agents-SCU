import numpy as np
import pytest
from hyra_validation.representation import build_representation,causal_history,fit_structure


def record(valid=20,arrival=20,residual=-2,region=0):
    return dict(region_id=region,valid_time_s=valid,arrival_time_s=arrival,
        observed_depth_m=3+residual,prior_depth_m=3,prior_release_s=valid-20,residual_m=residual)


def settings(method="B2_structure"):
    return dict(n_regions=2,history_length=2,max_age_s=60,decay_time_s=30,method=method)


def test_signed_fixed_denominator_decay_and_mask():
    graph=dict(weights=np.eye(2).tolist(),frozen=True)
    r=build_representation([record()],None,50,graph,settings())
    assert r["evidence"][0]==pytest.approx(-np.exp(-1))
    assert r["mask"].tolist()==[True,False]
    assert r["coverage"].tolist()==[.5,0]
    assert not build_representation([record()],None,81,graph,settings())["mask"].any()


def test_future_and_unarrived_do_not_change_inputs():
    graph=dict(weights=np.eye(2).tolist(),frozen=True)
    baseline=build_representation([record()],None,30,graph,settings())
    records=[record(),record(valid=40,arrival=40,residual=1000),record(valid=25,arrival=35,residual=1000)]
    actual=build_representation(records,None,30,graph,settings())
    np.testing.assert_array_equal(actual["features"],baseline["features"])


def test_out_of_order_and_duplicate_history():
    records=[record(30,50),record(40,40),record(20,55),record(40,45,residual=99)]
    h=causal_history(records,55,settings())[0]
    assert [r["valid_time_s"] for r in h]==[40,30]
    assert h[0]["residual_m"]==-2


def test_same_graph_and_expiry_uniform_vs_decay():
    graph=dict(weights=[[.5,-.5],[0,1]],frozen=True)
    records=[record(20,20,2,0),record(20,20,2,1)]
    decay=build_representation(records,None,50,graph,settings())
    uniform=build_representation(records,None,50,graph,settings("B2_uniform"))
    assert decay["evidence"][0]==0 and uniform["evidence"][0]==0
    assert uniform["evidence"][1]==1
    np.testing.assert_array_equal(decay["mask"],uniform["mask"])


def test_test_event_cannot_fit(simple_event,config):
    simple_event["metadata"]["split"]="test"
    with pytest.raises(ValueError,match="training"):
        fit_structure([simple_event],config["representation"])


def test_raw_preserves_both_depths_age_and_mask():
    raw=build_representation([record()],None,50,dict(frozen=True),settings("B0_raw"))
    np.testing.assert_array_equal(raw["features"][0,:4],[1,3,.5,1])


def test_v2_keeps_identical_raw_history_and_matched_slots():
    graph=dict(weights=[[.5,.5],[0,1]],frozen=True)
    records=[record(20,20,-2,0),record(30,30,1,1)]
    inputs={m:build_representation(records,None,50,graph,dict(settings(m),preserve_raw_history=True))
        for m in ["B0_raw","B_local","B1_distance","B2_structure","B2_uniform"]}
    baseline=inputs["B0_raw"]["features"]
    assert baseline.shape==(2,11)
    np.testing.assert_array_equal(baseline[:,8:],0)
    for representation in inputs.values():
        assert representation["features"].shape==baseline.shape
        np.testing.assert_array_equal(representation["features"][:,:8],baseline[:,:8])
    assert inputs["B_local"]["evidence"][1]==pytest.approx(.5*np.exp(-20/30))
    assert inputs["B_local"]["evidence"][0]!=inputs["B2_structure"]["evidence"][0]
