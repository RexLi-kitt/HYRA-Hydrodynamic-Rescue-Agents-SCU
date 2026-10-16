import copy
import numpy as np
from hyra_validation.envs import RescueEnv,interval_valid


def fixed_env(event,config):
    e=RescueEnv(event,config,dict(weights=np.eye(100).tolist(),frozen=True),"B2_structure",17)
    e.blocked[:]=False
    e.targets=np.array([0])
    e.status={0:"discovered"}
    e.positions=np.array([[0,0],[9,9],[9,8],[8,9]])
    e.busy=[dict(region=0,start=0,end=12,entry_distance=1),None,None,None]
    e.status[0]="servicing"
    return e


def test_complete_interval_required_and_resource_once(simple_event,config):
    e=fixed_env(simple_event,config)
    e.step([0,1,1,3])
    assert e.status[0]=="servicing" and e.metrics()["completed"]==0
    e.step([0,1,1,3])
    assert e.status[0]=="completed" and e.metrics()["completed"]==1
    assert e.resources[0]==config["environment"]["resources_per_agent"]-1
    e.step([0,1,1,3])
    assert e.resources[0]==config["environment"]["resources_per_agent"]-1


def test_mid_service_and_late_failure(simple_event,config):
    simple_event["truth_depth_m"][simple_event["time_s"]>=8,0]=1
    e=fixed_env(simple_event,config)
    e.step([0,1,1,3]); e.step([0,1,1,3])
    assert e.metrics()["completed"]==0
    assert e.failed_attempts==1 and e.resources[0]==2
    assert e.log[-3]["type"] in ("failed","started","decision")
    assert not interval_valid(simple_event,0,8,20,.8)


def test_future_truth_does_not_change_current_actor_obs(simple_event,config):
    e=fixed_env(simple_event,config)
    before=e.get_obs()
    e.event["truth_depth_m"][e.event["time_s"]>e.now]=999
    np.testing.assert_array_equal(e.get_obs(),before)


def test_methods_have_identical_capacity_and_task_state(simple_event,config):
    obs=[]
    for method in config["methods"]:
        e=RescueEnv(simple_event,config,dict(weights=np.eye(100).tolist(),frozen=True),method,13)
        obs.append(e.get_obs())
        assert e.metrics()["target_denominator"]==5
    assert len({x.shape for x in obs})==1
    for x in obs[1:]:
        np.testing.assert_array_equal(x,obs[0])


def test_unseen_targets_stay_in_denominator(simple_event,config):
    e=fixed_env(simple_event,config)
    e.targets=np.array([0,99])
    e.status[99]="undiscovered"
    assert e.metrics()["target_denominator"]==2


def test_busy_agent_stays_and_cannot_share_target(simple_event,config):
    e=fixed_env(simple_event,config)
    e.positions[1]=[1,0]
    e.step([1,0,1,3])
    np.testing.assert_array_equal(e.positions[0],[0,0])
    np.testing.assert_array_equal(e.positions[1],[1,0])


def test_failed_service_releases_agent_for_next_movement(simple_event,config):
    simple_event["truth_depth_m"][:,0]=1.
    e=fixed_env(simple_event,config)
    e.step([0,1,1,3])
    assert e.busy[0] is None
    assert e.status[0]=="discovered"
    failed_time=e.now
    assert not any(r["type"]=="started" and r["time_s"]==failed_time for r in e.log)
    e.step([1,1,1,3])
    np.testing.assert_array_equal(e.positions[0],[1,0])
    assert e.failed_attempts==1


def test_agent_can_choose_to_retry_after_failure(simple_event,config):
    simple_event["truth_depth_m"][:,0]=1.
    e=fixed_env(simple_event,config)
    e.step([0,1,1,3])
    assert e.busy[0] is None
    # A boundary-directed action stays in place and can initiate a new attempt.
    e.step([0,1,1,3])
    assert e.busy[0] is not None and e.busy[0]["start"]==12
    assert e.failed_attempts==1 and e.resources[0]==2
