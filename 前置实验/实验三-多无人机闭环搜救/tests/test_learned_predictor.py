import numpy as np
import pytest
import torch
from hyra_exp3.learned_predictor import LearnedPredictor
from hyra_exp3.models import SharedForecastBackbone
from hyra_exp3.windows import service_start_intervals

def predictor():
    side=4;yy,xx=np.meshgrid(np.arange(side),np.arange(side),indexing='ij')
    z=np.zeros((16,16)).tolist()
    bank=dict(banks={'4':dict(train_gain=z,held_out_train_gain=z,lag_steps=z,held_out_train_improving_families=z)})
    return LearnedPredictor(np.arange(81.),np.full((81,side,side),.2),np.column_stack((xx.ravel(),yy.ravel()))*10+5,
        np.zeros(16),SharedForecastBackbone(),bank,'H')

def test_learned_interface_uses_no_truth_and_never_extrapolates_service_support():
    p=predictor()
    p.issue(0.,[])
    e=p.estimate((0,0),0)
    assert not e.evidence_supported
    assert e.can_start(49) is None
    assert e.version==1 and e.valid_until==60
    p.estimate_many([(3,3)],0.)
    assert p.queries[-1][-1]==1
    assert p.published_windows[1,(3,3)].version==1
    p.issue(70.5,[])
    e=p.estimate((0,0),70.5)
    assert e.valid_until==75.5 and e.can_start(70.5) is None

def test_learned_interface_rejects_unarrived_observations_and_accepts_fractional_clock():
    p=predictor()
    r=dict(cell=(0,0),sample_time=10.,arrival_time=20.,depth_m=.3,drone=0)
    with pytest.raises(ValueError):
        p.issue(10.,[r])
    p.issue(20.5,[r])
    assert p.estimate((0,0),20.5).evidence_supported
    assert np.isfinite(p.prediction).all()

def test_sensor_tick_roundoff_cannot_create_a_fourteenth_history_bin():
    p=predictor()
    now=20.-1e-10
    r=dict(cell=(0,0),sample_time=now,arrival_time=now,depth_m=.3,drone=0)
    p.issue(now,[r])
    assert p.issued_at==now and p.estimate((0,0),now).evidence_supported

def test_unused_publications_need_no_neural_compute_and_inputs_are_frozen():
    p=predictor()
    r=dict(cell=[0,0],sample_time=10.,arrival_time=10.,depth_m=.3,drone=0)
    p.issue(10.,[r])
    assert p.prediction is None
    r['depth_m']=999.;r['cell'][0]=3
    assert p._pending_observations[0]['depth_m']==.3 and p._pending_observations[0]['cell']==(0,0)
    p.estimate_many([(0,0),(1,1)],10.)
    values=p.prediction.copy()
    p.estimate((2,2),10.)
    assert np.array_equal(p.prediction,values) and not p._pending_observations

def test_fractional_roundoff_never_publishes_past_hydraulic_endpoint():
    p=predictor();now=65.+5e-9
    p.issue(now,[]);e=p.estimate((0,0),now)
    assert e.valid_until<=80. and e.future_times[-1]<=80.

@pytest.mark.parametrize('constant',[.2,.8,.9])
def test_safe_curve_fast_path_matches_exact_float64_window_contract(constant):
    class Constant(torch.nn.Module):
        def forward(self,history,static,regions,query,nominal,mask,graph,method):
            return nominal*0+constant
    p=predictor();p.model=Constant()
    p.issue(0.,[]);e=p.estimate((0,0),0.)
    assert e.service_intervals==service_start_intervals(e.future_times,e.future_depths)
    if constant==.8:assert e.can_start(0.) is False
