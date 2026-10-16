import numpy as np
from hyra_validation.metric_vector import interval_maximum_float,metric_vector_scores
from hyra_validation.metric_tasks import metric_cases,metric_order,choose_metric


def test_fractional_window_endpoints_and_internal_peaks():
    curve=np.array([.1,.2,.9,.1,.1,.1,.1,.1])[None,None]
    assert interval_maximum_float(curve,1.25,6)[0,0]==.9
    np.testing.assert_allclose(interval_maximum_float(curve,4.5,6),.7)


def test_vector_scores_match_independent_scalar_at_every_case_for_random_curves():
    cases,_=metric_cases(100,180);targets=sorted({r for c in cases for r in c['targets']});times=sorted({c['now_s'] for c in cases})
    row_times=np.repeat(times,len(targets));row_targets=np.tile(targets,len(times))
    rng=np.random.default_rng(23);curves=.78+rng.normal(0,.035,(len(row_times),26))
    got=metric_vector_scores(curves,cases,row_times,row_targets)
    lookup={(t,r):i for i,(t,r) in enumerate(zip(row_times,row_targets))};scores=[]
    for case in cases:
        depth=np.zeros((26,100))
        for r in targets:depth[:,r]=curves[lookup[case['now_s'],r]]
        rows=[metric_order(case['now_s']+np.arange(0,51,2),depth,*order,case['now_s'],240) for order in case['orders']]
        scores.append(rows)
    np.testing.assert_array_equal(got['count'][0],[[r['count'] for r in row] for row in scores])
    np.testing.assert_allclose(got['maxima'][0],[[r['maxima'] for r in row] for row in scores],atol=1e-13)
    np.testing.assert_array_equal(got['chosen'][0],choose_metric(scores,cases))
