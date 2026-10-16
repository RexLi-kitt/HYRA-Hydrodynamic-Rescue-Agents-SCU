from hyra_validation.diagnostics import score_fixed_order


def test_known_order_can_change_actual_completed_yield(simple_event):
    simple_event["truth_depth_m"][simple_event["time_s"]>=20,1]=1.
    args=dict(times=simple_event["time_s"],depths=simple_event["truth_depth_m"],distances=[1,1],
        now=0,step_s=6,service_s=12,limit=.8)
    assert score_fixed_order(order=[1,2],**args)["service_yield"]==2
    assert score_fixed_order(order=[2,1],**args)["service_yield"]==1


def test_failed_service_aborts_without_automatic_retry(simple_event):
    simple_event["truth_depth_m"][:,1]=1.
    result=score_fixed_order(simple_event["time_s"],simple_event["truth_depth_m"],[1,2],[1,1],0,6,12,.8)
    assert result["service_yield"]==1
    assert result["trace"][0]==dict(region=1,start_s=6,end_s=12,completed=False)
    assert result["trace"][1]==dict(region=2,start_s=18,end_s=30,completed=True)


def test_future_outside_released_forecast_never_succeeds(simple_event):
    result=score_fixed_order(simple_event["forecast_valid_s"][0],simple_event["forecast_depth_m"][0],
        [1],[1],60,6,12,.8)
    assert result["service_yield"]==0
