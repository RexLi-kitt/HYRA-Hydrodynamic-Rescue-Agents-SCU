import unittest
import numpy as np
import torch
from flood1km.env import Env,rule_action
from flood1km.learners import distribution
from coverage_controller import CoverageWatchdog,service_mask
class CoverageTests(unittest.TestCase):
    def env(self):return Env(np.broadcast_to(np.zeros((1,3,100,100),np.float32),(301,3,100,100)),21)
    def test_observed_memory_is_conservative(self):
        e=self.env();p=CoverageWatchdog()
        for _ in range(90):
            p.observe(e.observe());self.assertFalse((p.known & ~e.known).any());e.step(rule_action(e))
    def test_no_early_takeover(self):
        e=self.env();p=CoverageWatchdog();a,info=p.select(e.observe(),e.available(),[8]*4,['learned']*4)
        self.assertFalse(info['active']);np.testing.assert_array_equal(a,[8]*4)
    def test_stall_takeover_preserves_service(self):
        e=self.env();p=CoverageWatchdog()
        for _ in range(7):a,info=p.select(e.observe(),e.available(),[8]*4,['hold','learned','learned','learned'])
        self.assertTrue(info['active']);self.assertEqual(a[0],8);self.assertGreater(info['planned_agents'],0)
    def test_forced_mask_has_zero_score_gradient(self):
        mask=torch.tensor(service_mask(np.ones((4,9),bool),[8,1,2,3],['hold','approach','learned','learned']))
        logits=torch.zeros((4,9),requires_grad=True);d=distribution(logits,mask)
        actions=d.sample();self.assertEqual(actions[0],8);self.assertEqual(actions[1],1)
        (-d.log_prob(actions).sum()).backward();self.assertEqual(float(logits.grad[:2].abs().sum()),0)
        self.assertTrue(torch.isfinite(logits.grad).all())
    def test_controller_reads_only_observation(self):
        e=self.env();obs=e.observe();e.water=None;e.people[e.status==0]=[321,654]
        np.testing.assert_array_equal(obs,e.observe())
if __name__=='__main__':unittest.main()
