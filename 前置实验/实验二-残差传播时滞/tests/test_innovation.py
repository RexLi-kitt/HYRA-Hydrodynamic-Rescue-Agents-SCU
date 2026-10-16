import pytest
from hyra_validation.innovation import score_innovations


def rec(t,value,arrival=None):
    return dict(region_id=0,valid_time_s=t,arrival_time_s=t if arrival is None else arrival,residual_m=value)


def test_current_value_is_excluded_and_persistent_bias_is_quiet():
    rows=score_innovations([rec(20,.2),rec(30,.2),rec(40,.2),rec(50,.2),rec(60,.5)])
    assert len(rows)==2
    assert rows[0]["innovation_score"]==pytest.approx(0)
    assert rows[1]["innovation_score"]==pytest.approx(225)
    assert rows[1]["history_times"]==[30,40,50]


def test_delayed_future_record_and_duplicate_never_enter_prior_history():
    rows=score_innovations([rec(60,100,35),rec(20,.2),rec(30,.2),rec(40,.2),rec(40,100,45),rec(50,.5)])
    scored=next(r for r in rows if r["time_s"]==50)
    assert scored["history_times"]==[20,30,40]
    assert scored["innovation_score"]==pytest.approx(225)


def test_stale_history_requires_new_warmup():
    assert not score_innovations([rec(20,.2),rec(30,.2),rec(40,.2),rec(150,.5)])
