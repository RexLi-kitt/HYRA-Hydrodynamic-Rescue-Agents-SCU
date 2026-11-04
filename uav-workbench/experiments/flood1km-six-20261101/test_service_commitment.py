import unittest
import numpy as np
from flood1km.env import Env
from service_commitment import ServiceCommitment

class ControllerTests(unittest.TestCase):
    def env(self):
        water=np.broadcast_to(np.zeros((1,3,100,100),np.float32),(301,3,100,100))
        e=Env(water,0);e.people=np.array([[105.,5.]]+[[505.+(k%5)*20,505.+(k//5)*20] for k in range(19)])
        e.reset();return e
    def test_detection_is_not_service(self):
        e=self.env();c=ServiceCommitment('hold');a,_=c.select(e.observe(),e.available(),[8]*4)
        e.step(a);self.assertEqual(e.metrics()['rescued'],0);self.assertEqual(len(c.jobs),0)
    def test_two_complete_hovers_after_arrival(self):
        e=self.env();c=ServiceCommitment('commit')
        e.step(c.select(e.observe(),e.available(),[8]*4)[0])
        e.step(c.select(e.observe(),e.available(),[8]*4)[0])
        self.assertAlmostEqual(e.progress[0],0);self.assertEqual(e.status[0],1)
        e.step(c.select(e.observe(),e.available(),[0,8,8,8])[0]);self.assertEqual(e.progress[0],5)
        e.step(c.select(e.observe(),e.available(),[0,8,8,8])[0]);self.assertEqual(e.status[0],2)
    def test_unique_assignments(self):
        e=self.env();e.status[:]=1;c=ServiceCommitment('commit');c.select(e.observe(),e.available(),[8]*4)
        keys=[x['target'] for x in c.jobs.values()];self.assertEqual(len(keys),len(set(keys)))
    def test_hidden_people_irrelevant(self):
        e=self.env();before=e.observe();e.people[e.status==0]=[321.,654.]
        after=e.observe();np.testing.assert_array_equal(before,after)
        a=ServiceCommitment('commit').select(before,e.available(),[8]*4)[0]
        b=ServiceCommitment('commit').select(after,e.available(),[8]*4)[0];np.testing.assert_array_equal(a,b)
    def test_measured_unsafe_does_not_force_hold(self):
        e=self.env();e.pos[0]=[95.,5.];e.sample();e.measured[0,0,10]=.2
        a,_=ServiceCommitment('commit').select(e.observe(),e.available(),[0]*4);self.assertEqual(a[0],0)
    def test_future_unsafe_not_leaked_or_overridden(self):
        e=self.env();e.pos[0]=[95.,5.];e.sample();e.water=e.water.copy();e.water[1,0,0,10]=.2
        c=ServiceCommitment('commit');a,_=c.select(e.observe(),e.available(),[0]*4)
        self.assertEqual(a[0],8);e.step(a);self.assertEqual(e.progress[0],0);self.assertEqual(e.status[0],1)
    def test_stuck_job_released(self):
        e=self.env();c=ServiceCommitment('commit');obs=e.observe()
        for _ in range(4):c.select(obs,e.available(),[8]*4)
        self.assertGreater(c.releases,0)
    def test_success_releases_job(self):
        e=self.env();c=ServiceCommitment('commit');c.select(e.observe(),e.available(),[8]*4)
        e.status[0]=2;c.select(e.observe(),e.available(),[8]*4);self.assertFalse(c.jobs)
    def test_collision_still_interrupts_service(self):
        e=self.env();e.pos[0]=[95.,5.];e.pos[1]=[105.,5.];e.owner[0]=0;e.progress[0]=5;e.sample()
        c=ServiceCommitment('commit');a,_=c.select(e.observe(),e.available(),[8,2,8,8])
        e.step(a);self.assertGreater(e.conflicts,0);self.assertEqual(e.progress[0],0)
    def test_existing_owner_keeps_service(self):
        e=self.env();e.pos[1]=[95.,5.];e.owner[0]=1;e.progress[0]=5;e.sample()
        c=ServiceCommitment('commit');a,_=c.select(e.observe(),e.available(),[8]*4)
        self.assertEqual(c.jobs[1]['target'],(105.,5.));e.step(a);self.assertEqual(e.status[0],2)

if __name__=='__main__':unittest.main()
