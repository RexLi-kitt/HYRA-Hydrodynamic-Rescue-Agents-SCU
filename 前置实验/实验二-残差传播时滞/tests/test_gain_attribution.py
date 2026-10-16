import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from analyze_decision_gain import decompose_case, pulse_contrast


def test_increment_chain_preserves_losses_and_exact_total():
    values = {'forecast_only': 1, 'raw_self': 2, 'raw_full': 1, 'physical': 2}
    rows = {m: dict(baseline_true_yield=0, selected_true_yield=v) for m,v in values.items()}
    result = decompose_case(rows)
    assert result == dict(learned_forecast_correction=1, local_history=1,
                          upstream_history=-1, physical_representation=1)
    assert sum(result.values()) == 2


def test_pulse_contrast_keeps_treatment_truth_fixed():
    a = dict(targets=[1,2], orders=[([1,2],[1,1]),([2,1],[1,1])],
             selected_order=0, true_yields=[2,2])
    b = dict(a, selected_order=1, true_yields=[1,2])
    result = pulse_contrast(a,b)
    assert result['delta'] == 1
    assert result['control_input_yield'] == 1
    assert result['actual_input_yield'] == 2
    assert pulse_contrast(b,dict(b,selected_order=0))['delta'] == -1


def test_matched_pair_rejects_changed_geometry():
    a = dict(targets=[1,2],orders=[],selected_order=0,true_yields=[1,2])
    with pytest.raises(ValueError):
        pulse_contrast(a,dict(a,targets=[2,3]))
