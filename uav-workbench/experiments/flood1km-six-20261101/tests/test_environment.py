"""Prepared contract checks. These have NOT been executed during preparation."""
import unittest,json
import numpy as np
from flood1km.env import Env,MOVES,sector,min_separation,segment_circle

class GeometryTests(unittest.TestCase):
    def test_sector_edge_and_vertex(self):
        p=np.array([[0,0],[100,0],[0,100],[0,-100],[-1,0],[101,0]])
        np.testing.assert_array_equal(sector(p,np.zeros(2),np.array([1,0]),100),[1,1,1,1,0,0])
    def test_eight_distances(self):
        np.testing.assert_allclose(np.linalg.norm(MOVES[:8]*50,axis=1),50)
    def test_crossing_between_endpoints(self):
        self.assertAlmostEqual(min_separation(np.array([0.,25]),np.array([50.,25]),np.array([25.,0]),np.array([25.,50])),0)
    def test_exchange_positions(self):
        self.assertAlmostEqual(min_separation(np.array([0.,0]),np.array([50.,0]),np.array([50.,0]),np.array([0.,0])),0)
    def test_move_towards_hover(self):
        p=np.array([25.,0]);self.assertAlmostEqual(min_separation(np.zeros(2),np.array([50.,0]),p,p),0)
    def test_short_stop_still_collides(self):
        self.assertAlmostEqual(min_separation(np.array([0.,0]),np.array([10.,0]),np.array([10.,40]),np.array([10.,-10])),0)
    def test_service_entry(self):
        self.assertAlmostEqual(segment_circle(np.array([5.,5]),np.array([50.,0]),np.array([55.,5]),10),.8)

class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.water=np.zeros((301,3,100,100),np.float32)
    def make(self):
        people=np.array([[55.,5.]]+[[205.+30*(i%10),205.+30*(i//10)] for i in range(19)])
        return Env(self.water,people=people)
    def test_unique_initial_coverage(self):
        e=self.make();n=e.known.sum();e.sample();self.assertEqual(n,e.known.sum());self.assertEqual(e.t,0)
    def test_arrival_requires_two_following_hovers(self):
        e=self.make();e.status[0]=1
        e.step(np.array([3,8,8,8]));self.assertAlmostEqual(e.pos[0,0],45.);self.assertEqual(e.progress[0],0)
        e.step(np.array([8,8,8,8]));self.assertEqual(e.progress[0],5);self.assertEqual(e.status[0],1)
        e.step(np.array([8,8,8,8]));self.assertEqual(e.status[0],2);self.assertEqual(e.first,3)
    def test_moving_interrupts_service(self):
        e=self.make();e.pos[0]=[45,5];e.status[0]=1
        e.step(np.array([8,8,8,8]));e.step(np.array([2,8,8,8]));self.assertEqual(e.progress[0],0)
    def test_late_interval_water_interrupts(self):
        e=self.make();e.water=self.water.copy();e.water[1,0,0,5]=.1;e.pos[0]=[45,5];e.status[0]=1
        e.step(np.array([8,8,8,8]));self.assertEqual(e.progress[0],0)
    def test_speed_not_depth_identifies_service_failure(self):
        e=self.make();e.water=self.water.copy();e.water[1,1,0,5]=.4;e.pos[0]=[45,5];e.status[0]=1
        e.step(np.array([8,8,8,8]));self.assertEqual(e.progress[0],0)
    def test_one_person_one_rescue(self):
        e=self.make();e.pos[0]=[47,5];e.pos[1]=[63,5];e.status[0]=1
        e.step(np.full(4,8));e.step(np.full(4,8));self.assertEqual(e.metrics()['rescued'],1)
        self.assertEqual(sum(x['target']==0 for x in e.events),1)
    def test_last_step_partial_is_failure(self):
        e=self.make();e.pos[0]=[45,5];e.status[0]=1;e.t=299
        _,_,done,_=e.step(np.full(4,8));self.assertTrue(done);self.assertEqual(e.metrics()['rescued'],0);self.assertIsNone(e.first)
    def test_last_step_complete_is_success_event(self):
        e=self.make();e.pos[0]=[45,5];e.status[0]=1;e.t=299;e.owner[0]=0;e.progress[0]=5
        e.step(np.full(4,8));self.assertEqual(e.metrics()['rescued'],1);self.assertEqual(e.first,300)
    def test_future_water_hidden(self):
        e=self.make();f=self.make();f.water=self.water.copy();f.water[10:]=.5
        np.testing.assert_array_equal(e.observe(),f.observe());np.testing.assert_array_equal(e.available(),f.available())
    def test_unseen_person_coordinates_hidden(self):
        e=self.make();f=self.make();f.people[1:]+=300
        np.testing.assert_array_equal(e.observe(),f.observe())
    def test_observation_keeps_sampling_time(self):
        e=self.make();was=e.known.copy();measured=e.measured.copy();e.water=self.water.copy();e.water[1]=.1
        e.pos[:]=[[400,400],[600,400],[400,600],[600,600]];e.t=1;e.sample()
        self.assertTrue((e.last[was]==0).all());np.testing.assert_array_equal(e.measured[:,was],measured[:,was])
    def test_intent_conflict_is_recorded_and_resolved(self):
        e=self.make();e.pos[:2]=[[400,400],[450,400]];before=e.pos.copy()
        e.step(np.array([3,2,8,8]));self.assertEqual(e.conflicts,1);np.testing.assert_array_equal(e.pos[:2],before[:2])
    def test_missing_times_are_null(self):
        e=self.make();m=e.metrics();self.assertIsNone(m['FRT']);self.assertIsNone(m['CRT'])
    def test_metrics_json_after_boundary(self):
        e=self.make();e.step(np.array([2,3,2,3]));json.dumps(e.metrics())
    def test_reproducible_layout(self):
        a=Env(self.water,123);b=Env(self.water,123);np.testing.assert_array_equal(a.people,b.people)

if __name__=='__main__':unittest.main()
