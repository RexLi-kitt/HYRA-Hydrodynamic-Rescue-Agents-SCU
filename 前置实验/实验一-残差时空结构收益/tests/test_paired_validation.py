import numpy as np
from hyra_validation import diagnostics


def test_two_conditions_of_one_family_do_not_count_as_two_families(monkeypatch,tmp_path):
    monkeypatch.setattr(diagnostics,"fit_structure",lambda *_:dict(weights=[[.5,.5],[.5,.5]],frozen=True))
    monkeypatch.setattr(diagnostics,"distance_structure",lambda *_:dict(weights=[[.5,.5],[.5,.5]],frozen=True))
    monkeypatch.setattr(diagnostics,"residual_samples",lambda event,_:event["corr"])
    monkeypatch.setattr(diagnostics,"correlation",lambda x:x)
    validation=[]
    for family in range(3):
        for condition in range(2):
            validation.append(dict(metadata=dict(event_id=f"{family}_{condition}",event_family_id=str(family)),
                corr=np.array([[1,.9 if family==0 else .1],[.9 if family==0 else .1,1]])))
    events=dict(train=[dict(xy_m=np.array([[0,0],[10,0]]))],validation=validation)
    graph,_=diagnostics.freeze_graph(events,dict(representation=dict(lead_time_s=20)),tmp_path)
    np.testing.assert_array_equal(graph["weights"],np.eye(2))
