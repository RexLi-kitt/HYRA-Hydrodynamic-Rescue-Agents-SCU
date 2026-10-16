import copy
import numpy as np
from hyra_validation.lead import features, MONITORS
from hyra_validation.lead_diagnostic import matched_features, timing_rows, phased_records, NORMALIZED


def records():
    return {s: [dict(region_id=s, valid_time_s=t, arrival_time_s=t,
                    observed_depth_m=.2+v, prior_depth_m=.2, residual_m=v)
                for t, v in [(40,.01),(50,.03),(60,.02)]] for s in MONITORS}


def graph():
    return {'23': [dict(source=s, travel_s=t, weight=1/3) for s,t in [(10,7),(20,13),(30,23)]]}


def test_legacy_controls_exactly_reproduce_existing_features(simple_event):
    rr=records(); gg=graph()
    new=matched_features(simple_event,rr,23,60,10,gg)
    for method in ('I1_common','I2_unaligned','I3_physical_lag'):
        np.testing.assert_allclose(new[method],features(simple_event,rr,23,60,10,gg,method),atol=1e-15)


def test_normalization_preserves_constant_and_shares_support(simple_event):
    rr=records()
    for rows in rr.values():
        for r in rows:r['residual_m']=.07
    out=matched_features(simple_event,rr,23,60,10,graph())
    for method in NORMALIZED:
        np.testing.assert_allclose(out[method][-3],.07)
        np.testing.assert_array_equal(out[method][-2:],out[NORMALIZED[0]][-2:])


def test_no_future_observation_or_future_publication_affects_features(simple_event):
    rr=records(); before=matched_features(simple_event,rr,23,60,10,graph())
    rr=copy.deepcopy(rr)
    rr[20].extend([dict(region_id=20,valid_time_s=70,arrival_time_s=70,residual_m=999),
                   dict(region_id=20,valid_time_s=60,arrival_time_s=80,residual_m=999)])
    simple_event['forecast_depth_m'][simple_event['release_s']>60]=999
    simple_event['truth_depth_m'][simple_event['time_s']>60]=999
    simple_event.pop('_published_cache',None)
    after=matched_features(simple_event,rr,23,60,10,graph())
    for method in before:np.testing.assert_array_equal(before[method],after[method])


def test_long_horizon_physical_kernel_collapses_to_recent(simple_event):
    out=matched_features(simple_event,records(),23,60,30,graph())
    np.testing.assert_array_equal(out['N1_physical_norm'],out['N2_recent_norm'])
    np.testing.assert_array_equal(out['I3_physical_lag'],out['N0_recent'])


def test_timing_does_not_subtract_network_delay_twice():
    rr={20:[dict(region_id=20,valid_time_s=40,arrival_time_s=50,residual_m=.01)]}
    gg={'23':[dict(source=20,travel_s=23,weight=1)]}
    row=timing_rows(rr,gg,50,23,10,wait=6)[0]
    assert row['age_s']==10 and row['nominal_remaining_s']==7
    assert row['nominal_lead_s']==13 and row['horizon_margin_s']==3
    assert row['horizon_covered'] and not row['post_action_horizon_covered']
    missing=timing_rows(rr,gg,49,23,10)[0]
    assert not missing['observed'] and missing['nominal_remaining_s'] is None


def test_phase_shifts_keep_observation_budget_and_causal_matching(simple_event):
    simple_event['metadata']['family_index']=0
    arrays=[phased_records(simple_event,0,0,True,p) for p in range(4)]
    assert len({sum(map(len,r.values())) for r in arrays})==1
    for rr in arrays:
        for rows in rr.values():
            for r in rows:
                assert r['prior_release_s']==r['valid_time_s']-20
                assert r['arrival_time_s']==r['valid_time_s']


def test_normalization_does_not_reweight_spatial_sources(simple_event):
    rr=records()
    for source,value in [(10,.01),(20,.03),(30,.08)]:
        for row in rr[source]:row['residual_m']=value
    out=matched_features(simple_event,rr,23,60,10,graph())
    for method in NORMALIZED:
        np.testing.assert_allclose(out[method][-3],.04)


def test_missing_history_has_zero_support_not_evidence_of_accuracy(simple_event):
    out=matched_features(simple_event,{},23,60,10,graph())
    for method in NORMALIZED:np.testing.assert_array_equal(out[method][-3:],0)
    assert all(not r['observed'] for r in timing_rows({},graph(),60,23,10))
