from types import SimpleNamespace
import numpy as np
from hyra_exp3.simulation import EpisodeConfig
from hyra_exp3.windows import WindowEstimate
from hyra_exp3.outcomes import analyze_outcomes


def example(support=60,intervals=()):
    config=EpisodeConfig(map_shape=(2,2),n_drones=1,n_targets=2,duration_s=60)
    truth=dict(times_s=np.arange(61.),depths_m=np.full((61,4),.2))
    curve=(.2,.2) if intervals else (1.,1.)
    estimate=WindowEstimate((1,1),0.,1,(0.,float(support)),curve,intervals,True,float(support))
    predictor=SimpleNamespace(published_windows={(1,(1,1)):estimate})
    targets=[dict(id=0,cell=[0,0],discovered_at=None,served_at=None),dict(id=1,cell=[1,1],discovered_at=0.,served_at=None)]
    result=dict(targets=targets,summary=dict(successful_services=0),events=[dict(event='plan',time_s=0.,version=1,reason='periodic',
        planning_state=[dict(drone=0,position_m=[5.,5.],payload=1,mode='idle',target=None,planned_start=None)])])
    return truth,result,predictor,config


def test_failure_accounting_distinguishes_supported_error_from_unknown_horizon():
    t,r,p,c=example()
    a=analyze_outcomes(t,r,p,c)
    assert a['categories']['undiscovered']==1 and a['categories']['window_misclassification']==1
    t,r,p,c=example(support=10)
    a=analyze_outcomes(t,r,p,c)
    assert a['categories']['window_misclassification']==0 and a['categories']['resource_or_time_exhausted']==1
    assert sum(a['categories'].values())==2


def test_forecast_failed_service_and_execution_deviation_are_separate():
    t,r,p,c=example(intervals=((0.,48.),))
    t['depths_m'][6,3]=1.
    r['events'].append(dict(event='service_complete',target=1,success=False,prediction_version=1,start_s=0.))
    assert analyze_outcomes(t,r,p,c)['categories']['window_misclassification']==1
    t,r,p,c=example()
    t['depths_m'][6,3]=1.
    r['events'].append(dict(event='service_complete',target=1,success=False,prediction_version=1,start_s=0.))
    assert analyze_outcomes(t,r,p,c)['categories']['execution_failure']==1
