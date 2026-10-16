import numpy as np
import pytest
from hyra_validation.metric_tasks import metric_cases,metric_order,choose_metric


def test_all_pairs_and_both_speeds_are_retained_in_100m_geometry():
    cases,coverage=metric_cases(100,180,offsets=(0,6,12))
    assert coverage['candidate_layouts']==66*4*2 and coverage['excluded_layouts']==0
    assert len(cases)==66*4*2*3
    assert {tuple(c['targets']) for c in cases}.__len__()==66
    assert any(c['cross_half'] for c in cases) and any(not c['cross_half'] for c in cases)
    for c in cases:
        assert c['now_s']+c['worst_duration_s']<=240+1e-10


def test_longer_domain_exclusions_are_geometry_only_and_all_fast_cases_remain():
    cases,coverage=metric_cases(300,180,offsets=(0,6,12))
    assert coverage['excluded_layouts']>0
    assert sum(c['speed_m_per_s']==10 for c in cases)==66*4*3
    assert all(c['worst_duration_s']<=48+1e-10 for c in cases)


def test_late_decisions_keep_absolute_deadline_and_count_all_clock_exclusions():
    cases,coverage=metric_cases(100,180)
    assert {c['decision_offset_s'] for c in cases}=={0,6,12,18,24,30}
    assert coverage['available_cases']+coverage['excluded_cases']==66*4*2*6
    assert coverage['deadline_excluded_cases']>0
    assert all(c['now_s']+c['worst_duration_s']<=240+1e-10 for c in cases)


def test_fractional_travel_complete_windows_and_abort_feedback_can_change_order_value():
    t=np.arange(0,61,2);depth=np.full((len(t),2),.2);depth[t==14,0]=.9
    a=metric_order(t,depth,[0,1],[2.25,2.5],0,60)
    b=metric_order(t,depth,[1,0],[1.25,2.5],0,60)
    assert a['count']==1 and a['starts']==[2.25,16.75]
    assert b['count']==2
    cases=[dict(orders=[([0,1],[2.25,2.5]),([1,0],[1.25,2.5])])]
    assert choose_metric([[a,b]],cases)[0]==1
    with pytest.raises(ValueError):metric_order(t,depth,[0,1],[40,10],0,60)
