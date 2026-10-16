import numpy as np
from hyra_validation.hidden_target_inputs import SourceRecord,SOURCES
from hyra_validation.upstream_baselines import source_histories,spatial_weights


def test_source_histories_preserve_current_age_and_delayed_source_ids():
    records=[SourceRecord(i,s,90.,100.,70.,.5,.4) for i,s in enumerate(SOURCES)]
    h=source_histories(records,[99,100])
    assert not h['available_sources'][0].any() and h['available_sources'][1].all()
    assert np.all(h['record_ids'][0]==-1)
    np.testing.assert_allclose(h['conventional'][6:,1,0],.1)
    np.testing.assert_allclose(h['conventional'][6:,1,1],10/60)
    np.testing.assert_allclose(h['conventional'][6:,1,2],20/30)


def test_spatial_transfer_preserves_constants_and_does_not_invent_missing_support():
    sources=np.array([(5,y) for y in (15,25,35,65,75,85)])
    targets=np.array([[35,15],[55,75]])
    available=np.array([[1,1,1,1,1,1],[0,1,0,0,0,0],[0,0,0,0,0,0]],bool)
    for kind in ('uniform','idw1','idw2'):
        w=spatial_weights(sources,targets,available,kind)
        np.testing.assert_allclose(w[:2].sum(axis=-1),1.)
        assert np.all(w[2]==0) and np.all(w[1,:,1]==1)
        np.testing.assert_allclose(np.einsum('nts,s->nt',w,np.full(6,.2))[:2],.2)
