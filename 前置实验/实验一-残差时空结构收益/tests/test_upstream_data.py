import numpy as np
import pytest

from hyra_validation.upstream_data import ForecastTable,source_records
from hyra_validation.hidden_target_inputs import SOURCES,causal_history


def table(ids=SOURCES):
    releases=np.array([100.,110.,120.]);valid=releases[:,None]+np.arange(0,61,2)
    depth=np.broadcast_to(np.array([.2,.3,.9])[:,None,None],(3,31,6))
    return ForecastTable(releases,valid,depth,ids)


def test_history_matches_actual_past_version_and_current_issue_excludes_future():
    f=table();data=source_records([0,200],np.full((2,6),.5),SOURCES,f,[135.],0,np.zeros((1,6)))
    assert all(r.prior_release_s==110 and r.prior_m==.3 for r in data['records'])
    assert all(r['actual_lead_s']==25 for r in data['references'])
    release,values=f.at_decision(115,[0,6,12])
    assert release==110 and np.all(values==.3)
    with pytest.raises(ValueError,match='version'):f.at_issue(0,[161.])


def test_paired_events_share_explicit_noise_and_only_source_observations_change():
    f=table();noise=np.full((1,6),.01)
    a=source_records([0,200],np.full((2,6),.5),SOURCES,f,[135.],10,noise)
    b=source_records([0,200],np.full((2,6),.6),SOURCES,f,[135.],10,noise)
    assert a['references']==b['references']
    np.testing.assert_allclose([y.observation_m-x.observation_m for x,y in zip(a['records'],b['records'])],.1)
    assert np.all(causal_history(a['records'],144)['record_ids']==-1)
    assert np.count_nonzero(causal_history(a['records'],145)['record_ids']>=0)==6


def test_wrong_network_and_nonexistent_historical_versions_are_rejected():
    f=table()
    with pytest.raises(ValueError,match='six'):source_records([0,200],np.full((2,6),.5),(13,23,33,63,73,83),f,[135.],0,np.zeros((1,6)))
    with pytest.raises(ValueError,match='not saved'):source_records([0,200],np.full((2,6),.5),SOURCES,f,[110.],0,np.zeros((1,6)))
    with pytest.raises(ValueError):source_records([0,200],np.full((2,6),.5),SOURCES,f,[135.],np.nan,np.zeros((1,6)))
    with pytest.raises(ValueError):f.depth_m[0,0,0]=1
