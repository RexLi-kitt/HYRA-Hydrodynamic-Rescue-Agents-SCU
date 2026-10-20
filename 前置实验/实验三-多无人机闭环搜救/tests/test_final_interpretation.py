from hyra_exp3.final_report import conclusion,window_timing_context


def test_ceiling_and_numerical_instability_cannot_be_reported_as_positive_HyRA_results():
    effect=dict(primary_positive=True,engineering_target_3pp_met=True)
    assert '天花板' in conclusion(effect,True,True)
    assert '尚不能' in conclusion(effect,False,False)
    assert '达到增加3' in conclusion(effect,False,True)
    effect['primary_positive']=False
    assert '不能宣称' in conclusion(effect,False,True)


def test_early_partial_service_does_not_count_as_all_tasks_finished_before_closure():
    def row(served,last):
        return dict(summary=dict(hidden_targets=2,successful_services=served,last_successful_service_s=last),
            outcome_analysis=dict(targets=[dict(true_safe_start_intervals=[[0.,100.]]),dict(true_safe_start_intervals=[[0.,200.]])]))
    context=window_timing_context([row(2,100.),row(2,120.),row(1,90.)])
    assert context['episodes']==3 and context['all_hidden_targets_initially_safe_episodes']==3
    assert context['all_targets_served_before_any_initial_true_window_closed_episodes']==1
    assert context['all_targets_served_initial_window_margin_s_quantiles'][0]==-8.
