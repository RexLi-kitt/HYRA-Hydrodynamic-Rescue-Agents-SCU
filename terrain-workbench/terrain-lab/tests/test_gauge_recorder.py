import unittest
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver
from gauge_recorder import GaugeRecorder

class RecorderTests(unittest.TestCase):
    def sim(self,refined=False):
        a=np.ones((4,4));s=FloodSolver(a*0,a,1,boundary='periodic',manning=.1,max_timestep=.01);s.mx[:]=1
        return RefinedFloodSolver.from_solver(s,np.ones_like(a,bool)) if refined else s
    def test_native_and_refined_no_state_change(self):
        for refined in (False,True):
            s,control=self.sim(refined),self.sim(refined)
            r=GaugeRecorder(s,{'g':(0,0)},origin_center_m=(0,0),run_id='a')
            for end in (.05,.1):s.advance(end,observer=r);control.advance(end)
            self.assertEqual(len(r.records['g']),s.steps+1)
            self.assertEqual(r.records['g'][0]['time_s'],0)
            for k in ('h','mx','mz'):np.testing.assert_array_equal(getattr(s,k),getattr(control,k))
            self.assertEqual(s.steps,control.steps);self.assertEqual(s.balance,control.balance)
            if refined:
                for k in ('h','mx','mz'):np.testing.assert_array_equal(s.leaf[k],control.leaf[k])
    def test_reset_and_other_solver_rejected(self):
        s=self.sim();r=GaugeRecorder(s,{'g':(0,0)},origin_center_m=(0,0),run_id='a');s.advance(.1,observer=r)
        with self.assertRaises(ValueError):r(self.sim())
        s.start_scenario()
        with self.assertRaises(ValueError):s.advance(.1,observer=r)
        self.assertEqual(s.time,0)
    def test_window_summary_and_missing_point(self):
        s=self.sim();r=GaugeRecorder(s,{'g':(0,0),'out':(-10,0)},origin_center_m=(0,0),run_id='a');s.advance(.1,observer=r)
        q=r.summarize('g',start_s=0,end_s=.1)
        self.assertEqual(q['sampled_peak_speed_m_s'],1);self.assertEqual(q['sampled_peak_time_s'],0)
        q=r.summarize('out',start_s=0,end_s=.1)
        self.assertEqual(q['missing_sample_count'],q['sample_count']);self.assertIsNone(q['sampled_peak_speed_m_s'])
