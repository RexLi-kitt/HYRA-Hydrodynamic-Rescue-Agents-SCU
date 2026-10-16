import json
import numpy as np
import pytest
from hyra_validation.data import match_prior,observation,current_forecast,load_events


def test_prior_is_fixed_lead_and_immutable(simple_event):
    e=simple_event
    r=observation(e,0,40,50,1.2,20)
    assert r["prior_release_s"]==20
    e["forecast_depth_m"][e["release_s"]>20]=999
    assert r["prior_depth_m"]==.2
    assert match_prior(e,0,40,20)["prior_depth_m"]==.2
    assert observation(e,0,41,41,1.2,20)["residual_m"] is None


def test_future_issues_never_in_current_forecast(simple_event):
    e=simple_event
    expected=current_forecast(e,24,20)
    e["forecast_depth_m"][e["release_s"]>24]=999
    np.testing.assert_array_equal(current_forecast(e,24,20),expected)


def test_family_leakage_rejected(tmp_path,simple_event):
    meta=simple_event.pop("metadata")
    for name in ("a","b","c"):
        np.savez(tmp_path/f"{name}.npz",**simple_event)
        (tmp_path/f"{name}.json").write_text(json.dumps(dict(meta,event_id=name)),encoding="utf-8")
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(train=["a.npz"],validation=["b.npz"],test=["c.npz"])),encoding="utf-8")
    with pytest.raises(ValueError,match="family"):
        load_events(manifest)
