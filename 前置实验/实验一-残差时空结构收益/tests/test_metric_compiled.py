import numpy as np
from hyra_validation.lead import TARGETS
from hyra_validation.metric_tasks import metric_cases
from hyra_validation.metric_vector import metric_vector_scores
from hyra_validation.metric_compiled import CompiledMetricCases


def test_compiled_windows_preserve_all_cases_and_exact_order_logic():
    cases,_=metric_cases(100,240);times=np.repeat(240+np.arange(0,31,6),12);targets=np.tile(TARGETS,6)
    compiled=CompiledMetricCases(cases,times,targets)
    curves=np.random.default_rng(2026101631).uniform(.7,.9,(7,72,26));curves[0]=.8
    expected=metric_vector_scores(curves,cases,times,targets);actual=compiled.scores(curves)
    for key in ('count','chosen'):np.testing.assert_array_equal(actual[key],expected[key])
    for key in ('margin','common_maxima'):np.testing.assert_allclose(actual[key],expected[key],atol=0,rtol=0)
    assert compiled.window_count<len(cases)*8
