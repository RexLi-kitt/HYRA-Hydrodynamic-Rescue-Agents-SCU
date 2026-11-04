import copy
import unittest
import numpy as np
from environment import Config,Env,sample_water,first_hit,make_scenario

def water(u=0.,v=0.,h=.12):
    f=np.zeros((3,100,100));f[0]=h;f[1]=u;f[2]=v
    return np.broadcast_to(f,(301,3,100,100))

def scene(cfg,point=(5.,5.),fall=0):
    return dict(config=cfg.__dict__,traps=[],events=[dict(fall=fall,deadline=fall+cfg.window,xy=list(point))]+
                [dict(fall=cfg.duration-cfg.window,deadline=cfg.duration,xy=[400+i,500]) for i in range(19)])

class Tests(unittest.TestCase):
    def test_sampling_components_and_interpolation(self):
        np.testing.assert_allclose(sample_water(water(.3,-.2),2.5,np.array([[5,5],[500,500]])),[[.12,.3,-.2]]*2)

    def test_segment_catches_thin_obstacle(self):
        hit=first_hit(np.array([0.,0.]),np.array([100.,0.]),np.array([[50.,0.]]),1)
        self.assertAlmostEqual(hit,.49)

    def test_known_target_follow_and_supply_at_deadline(self):
        cfg=Config(duration=30,window=10);env=Env(water(.5),scene(cfg),cfg)
        for _ in range(2):
            _,preferred,_=env.control();self.assertEqual(preferred[0],8);env.step(preferred)
        self.assertEqual(env.status[0],2);self.assertEqual(env.delivered_at[0],10)
        self.assertAlmostEqual(env.person[0,0],10);self.assertAlmostEqual(env.pos[0,0],10)

    def test_hidden_expiry_counts_failure_not_removed(self):
        cfg=Config(duration=30,window=10);env=Env(water(),scene(cfg,(500,500)),cfg)
        for _ in range(2):env.step(env.control()[1])
        self.assertEqual(env.status[0],3);self.assertEqual(env.metrics()['expired_unseen'],1)
        self.assertFalse(env.metrics()['scene_success']);self.assertEqual(len(env.status),20)

    def test_approaching_moving_person_does_not_chase_radius_edge_forever(self):
        cfg=Config(duration=60,window=30);env=Env(water(.5),scene(cfg,(30,5)),cfg)
        for _ in range(3):env.step(env.control()[1])
        self.assertEqual(env.status[0],2)

    def test_future_event_does_not_exist_early(self):
        cfg=Config(duration=30,window=10);env=Env(water(),scene(cfg,fall=20),cfg)
        self.assertFalse(env.discovered.any());self.assertFalse((env.status==1).any())

    def test_hidden_truth_cannot_change_observation_or_action_mask(self):
        cfg=Config();env=Env(water(),scene(cfg,(500,500)),cfg);a=env.observe().copy();mask=env.control()[0].copy()
        env.person[~env.discovered]=[750,750];env.fall[:]=1000;env.deadline[:]=1400;env.water=water(2,1,.8)
        np.testing.assert_array_equal(a,env.observe());np.testing.assert_array_equal(mask,env.control()[0])

    def test_outside_counts_as_failure(self):
        cfg=Config(duration=30,window=10);env=Env(water(2),scene(cfg,(999,500)),cfg)
        env.step(env.control()[1]);self.assertEqual(env.status[0],4);self.assertEqual(env.metrics()['outside'],1)

    def test_person_traps_and_stays(self):
        cfg=Config(duration=30,window=10,trap_radius=1);s=scene(cfg,(500,500));s['traps']=[[503,500]]
        env=Env(water(3),s,cfg);env.step(env.control()[1]);self.assertTrue(env.stuck[0]);self.assertAlmostEqual(env.person[0,0],502)

    def test_generation_is_reproducible_and_within_time_budget(self):
        cfg=Config(duration=30,window=10);a=make_scenario(water(),123,cfg);b=make_scenario(water(),123,cfg)
        self.assertEqual(a,b);self.assertEqual(len(a['events']),20)
        self.assertTrue(all(e['fall']<=20 and e['deadline']<=30 for e in a['events']))

    def test_dry_scenario_rejected_not_fake_falls(self):
        with self.assertRaises(ValueError):make_scenario(water(h=0),3,Config(duration=30,window=10))

    def test_success_does_not_require_complete_map_coverage(self):
        env=Env(water(),scene(Config()));env.status[:]=2
        self.assertTrue(env.metrics()['scene_success']);self.assertLess(env.metrics()['coverage'],1)

    def test_action_validation(self):
        env=Env(water(),scene(Config()))
        for actions in [[-1]*4,[9]*4,[.5]*4,[1,2]]:
            with self.assertRaises(ValueError):env.step(actions)

if __name__=='__main__':unittest.main()
