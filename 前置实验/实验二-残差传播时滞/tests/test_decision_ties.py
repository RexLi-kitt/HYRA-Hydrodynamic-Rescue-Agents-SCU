from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_decision_ties import maximum,tie_weights


def test_margin_is_only_a_tie_breaker_and_never_overrides_service_count():
    scores=[dict(service_yield=2),dict(service_yield=1)]
    assert list(tie_weights(scores,[-.1,.5],'maximin_margin')) == [1,0]
    scores[1]['service_yield']=2
    assert list(tie_weights(scores,[-.1,.5],'maximin_margin')) == [0,1]


def test_symmetric_ties_have_exact_expected_yield():
    scores=[dict(service_yield=2),dict(service_yield=2)]
    weights=tie_weights(scores,[.1,.1],'uniform_tie_expectation')
    assert weights@np.array([1,2]) == 1.5
    assert list(tie_weights(scores,[.1,.1],'fixed_reverse')) == [0,1]


def test_interval_margin_uses_interior_peak_and_rejects_extrapolation():
    t=np.array([0.,2.,4.]);y=np.array([.2,.9,.3])
    assert maximum(t,y,1,3) == .9
    with pytest.raises(ValueError):maximum(t,y,1,5)
