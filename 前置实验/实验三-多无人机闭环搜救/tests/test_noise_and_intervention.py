from dataclasses import replace
import numpy as np
from hyra_exp3.noise import world_noise
from hyra_exp3.simulation import EpisodeConfig,run_episode


def test_world_noise_is_observation_order_independent_and_has_configured_scale():
    cells=np.arange(20000)
    values=world_noise(42,3,2,cells)
    assert np.array_equal(values[[31,0,100]],world_noise(42,3,2,[31,0,100]))
    assert not np.array_equal(values,world_noise(42,4,2,cells))
    assert abs(values.mean())<.0007 and abs(values.std()-.02)<.0007


class SwitchingPlanner:
    name='intervention_contract_test'
    def __init__(self):
        self.last_diagnostics={}
    def __call__(self,*,now,robots,known_targets,covered,allowed,predictor,config):
        self.last_diagnostics={}
        if not robots or not known_targets:
            return {}
        tid=max(known_targets) if now>=5 else min(known_targets)
        cell=known_targets[tid]
        destination=np.array([(cell[1]+.5)*10,(cell[0]+.5)*10])
        arrival=now+np.linalg.norm(destination-robots[0]['position'])/5
        start=predictor.estimate(cell,now).earliest_start(arrival)
        if start is None or start+12>config.duration_s:
            return {}
        return {robots[0]['id']:(cell,tid,start,predictor.version)}


def run(policy,period=10):
    times=np.arange(41.)
    field=np.full((41,10,10),.2)
    config=EpisodeConfig(map_shape=(10,10),n_drones=1,n_targets=2,duration_s=30,
        planner_budget_s=1.,noise_mode='counter_keyed',reassignment_policy=policy,planning_period_s=period)
    return run_episode(field,config,times=times,nominal_depths=field.copy(),targets=[(2,2),(4,4)],
                       starts=[(0,0)],planner=SwitchingPlanner())


def test_intervention_holds_a_feasible_task_only_between_periodic_replans():
    normal=run('full');held=run('defer_new_evidence')
    n=next(e for e in normal['events'] if e['event']=='plan' and e['time_s']==5)
    h=next(e for e in held['events'] if e['event']=='plan' and e['time_s']==5)
    assert n['reassignment_details'][0]['next_target']==1
    assert h['policy_interventions'][0]['held_target']==0 and h['reassignments']==0
    original={(r['sample_time'],r['drone'],tuple(r['cell'])):r['depth_m'] for r in normal['raw_observations']}
    compared=0
    for r in held['raw_observations']:
        key=r['sample_time'],r['drone'],tuple(r['cell'])
        if key in original:
            assert r['depth_m']==original[key]
            compared+=1
    assert compared>25
    periodic=run('defer_new_evidence',period=5)
    assert not any(e.get('policy_interventions') for e in periodic['events'])
