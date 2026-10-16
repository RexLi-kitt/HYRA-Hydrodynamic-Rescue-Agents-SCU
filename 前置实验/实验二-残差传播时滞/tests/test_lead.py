import numpy as np
from hyra_validation.lead import pulse,available_history,future_label
from hyra_validation.lead import hydraulic_domain,XY
from hyra_validation.adapters.anuga import simulate


def test_pulse_is_local_in_time_and_continuous():
    assert pulse(90)==pulse(130)==0
    assert pulse(100)==pulse(120)==.1
    assert abs(pulse(90+1e-6))<1e-10
    assert abs(pulse(130-1e-6))<1e-10


def test_future_label_locks_current_publication(simple_event):
    expected=future_label(simple_event,0,100,20)
    simple_event["forecast_depth_m"][simple_event["release_s"]>100]=999
    simple_event.pop("_published_cache")
    assert future_label(simple_event,0,100,20)==expected


def test_history_excludes_not_arrived_and_future_valid_times():
    records=[dict(region_id=0,valid_time_s=t,arrival_time_s=a,residual_m=1) for t,a in [(20,20),(30,60),(50,50)]]
    assert [r["valid_time_s"] for r in available_history(records,0,40)]==[20]


def test_local_pulse_updates_actual_solver_boundary_segments():
    params=dict(slope=.0015,ridge=.07,rate=.003,onset=25)
    domain=hydraulic_domain(params,pulse_amplitude=.1)
    assert len(domain.tag_boundary_cells["pulse"])==2
    assert len(domain.tag_boundary_cells["left"])==8
    _,reference,_,_=simulate(hydraulic_domain(params),160,XY)
    _,perturbed,_,_=simulate(domain,160,XY)
    assert np.max(abs(perturbed-reference))>.005
