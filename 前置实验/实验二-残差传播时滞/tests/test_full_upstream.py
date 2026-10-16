import numpy as np
from hyra_validation.hidden_target_inputs import SourceRecord,SOURCES
from hyra_validation.full_upstream import full_histories,ordinary_full


def test_all_twelve_actual_records_are_kept_and_latency_reduces_only_available_slots():
    records=[SourceRecord(j*6+i,s,float(t),float(t),float(t-20),.5,.4)
             for j,t in enumerate(range(125,181,5)) for i,s in enumerate(SOURCES)]
    h=full_histories(records,[180]);assert np.sum(h['values'][...,5])==72
    delayed=[SourceRecord(r.record_id,r.source,r.valid_s,r.valid_s+10,r.prior_release_s,r.observation_m,r.prior_m) for r in records]
    assert np.sum(full_histories(delayed,[180])['values'][...,5])==60


def test_full_filter_uses_earliest_raw_sample_that_legacy_thinning_would_drop():
    records=[SourceRecord(j*6+i,s,float(t),float(t),float(t-20),.4+(.1 if t==125 else 0),.4)
             for j,t in enumerate(range(125,181,5)) for i,s in enumerate(SOURCES)]
    models=dict(ewma_half_life_s=[40],kf_rw_process_variance_rate_m2_per_s=[1e-7],
        kf_cv_process_spectral_density_m2_per_s3=[],kf_measurement_std_m=[.02],kf_initial_bias_std_m=.05,kf_initial_bias_rate_std_m_per_s=.002)
    out=ordinary_full(full_histories(records,[180]),models)
    assert out['ewma_40'][0,0,0]>0 and out['kf_rw_q1e-07_r0.02'][0,0,0]>0
