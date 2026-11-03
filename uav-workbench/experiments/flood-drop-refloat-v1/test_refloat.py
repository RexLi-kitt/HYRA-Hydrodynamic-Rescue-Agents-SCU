import unittest,copy
import numpy as np
from refloat_environment import Env,Config
from dataclasses import asdict

def make(h=0,u=0,v=0):
    cfg=Config(duration=120,window=60,people=1,traps=0,trigger_depth=2,trigger_speed=1)
    s=dict(seed=33,config=asdict(cfg),events=[dict(fall=0,deadline=60,xy=[500,500],exposure_id=7)],traps=[])
    water=np.broadcast_to(np.array([h,u,v],float)[None,:,None,None],(301,3,100,100))
    return Env(water,s)
def wet(e,h,u=0,v=0):e.water=make(h,u,v).water

class TestRefloat(unittest.TestCase):
    def ground(self):
        e=make();e.move_people(0);self.assertEqual(e.stop_kind[0],1);return e
    def test_dry_stays_even_with_invalid_dry_velocity(self):
        e=self.ground();wet(e,0,2);e.move_people(1);self.assertTrue(e.stuck[0]);self.assertEqual(e.refloat_counts[0],0)
    def test_depth_boundary_and_deadline(self):
        e=self.ground();wet(e,2,.2);e.move_people(1);self.assertFalse(e.stuck[0]);self.assertAlmostEqual(e.person[0,0],500.2);self.assertEqual(e.deadline[0],60)
    def test_speed_magnitude_boundary(self):
        e=self.ground();wet(e,.1,.6,.8);e.move_people(1);self.assertEqual(e.refloat_counts[0],1)
    def test_random_per_minute_conversion(self):
        e=self.ground();wet(e,.1,.2);p=-np.expm1(np.log1p(-.05)/60)
        e.refloat_draw=lambda j,t:p+1e-8;e.move_people(1);self.assertTrue(e.stuck[0])
        e.refloat_draw=lambda j,t:p-1e-8;e.move_people(2);self.assertFalse(e.stuck[0]);self.assertEqual(e.events[-1]['reason'],'random')
        self.assertAlmostEqual(1-(1-p)**60,.05)
    def test_obstacle_remains_captured(self):
        e=make(.1,1);e.traps=np.array([[502.,500.]]);e.move_people(0);self.assertEqual(e.stop_kind[0],2)
        wet(e,3,2);xy=e.person.copy();e.move_people(1);np.testing.assert_array_equal(xy,e.person)
    def test_random_call_order_and_public_id_invariance(self):
        e=make();a=e.refloat_draw(0,7)
        for t in range(60):e.refloat_draw(0,t)
        self.assertEqual(a,e.refloat_draw(0,7));self.assertEqual(a,make().refloat_draw(0,7))
    def test_resolved_people_do_not_refloat(self):
        for status in (0,2,3,4):
            e=self.ground();e.status[:]=status;wet(e,3,2);e.move_people(1);self.assertEqual(e.refloat_counts[0],0)
    def test_resticking_and_second_refloat(self):
        e=self.ground();wet(e,2,.1);e.move_people(1);wet(e,0);e.move_people(2);wet(e,2,.1);e.move_people(3)
        self.assertEqual(e.refloat_counts[0],2);self.assertEqual(e.deadline[0],60)
    def test_full_step_expiry_does_not_restart(self):
        e=self.ground();wet(e,2,.1)
        while e.t<60:e.step(e.control()[1])
        self.assertEqual(e.status[0],3);self.assertEqual(e.deadline[0],60)

if __name__=='__main__':unittest.main()
