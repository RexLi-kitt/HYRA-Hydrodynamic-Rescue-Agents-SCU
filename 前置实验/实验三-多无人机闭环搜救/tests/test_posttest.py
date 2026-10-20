from hyra_exp3.posttest import prefix_comparison


def test_reassignment_prefix_detects_timeout_before_intervention():
    plan=dict(event='plan',time_s=0.,planning_state=[],planner_diagnostics=dict(selected_routes=[]))
    intervention=dict(event='plan',time_s=5.,policy_interventions=[dict(drone=0)])
    full=dict(events=[plan,dict(event='plan',time_s=5.)])
    cf=dict(events=[plan,intervention])
    assert prefix_comparison(full,cf)['uncontaminated_before_first_intervention']
    cf['events'].insert(1,dict(event='planning_timeout',time_s=2.))
    assert not prefix_comparison(full,cf)['uncontaminated_before_first_intervention']


def test_no_intervention_requires_equal_full_decision_trace():
    a=dict(events=[dict(event='service_complete',time_s=12.,success=True)])
    b=dict(events=[dict(event='service_complete',time_s=12.,success=False)])
    verdict=prefix_comparison(a,b)
    assert verdict['no_intervention'] and not verdict['all_episode_decisions_equal_if_no_intervention']
