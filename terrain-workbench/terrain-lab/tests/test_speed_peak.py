import unittest
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver

class SpeedPeakTests(unittest.TestCase):
    def test_initial_peak_survives_drag_and_resets(self):
        for refined in (False,True):
            a=np.ones((4,4));s=FloodSolver(a*0,a,1,boundary='periodic',manning=.3)
            s.mx[:]=2
            if refined:s=RefinedFloodSolver.from_solver(s,np.ones_like(a,bool))
            s.step(.01)
            self.assertAlmostEqual(s.summary()['peak_speed_m_s'],2)
            self.assertEqual(s.peak_speed_time_s,0)
            self.assertLess(s.summary()['max_speed_m_s'],2)
            s.start_scenario()
            self.assertAlmostEqual(s.peak_speed_m_s,s.summary()['max_speed_m_s'])
            self.assertEqual(s.peak_speed_time_s,0)

    def test_leaf_peak_not_projected_average(self):
        a=np.ones((4,4));s=RefinedFloodSolver.from_solver(FloodSolver(a*0,a,1),np.ones_like(a,bool))
        s.leaf['mx'][0]=3;s._sync();s.advance(0)
        self.assertEqual(s.peak_speed_m_s,3)
        self.assertLess(s.velocity()[0].max(),3)

    def test_tracking_does_not_change_state(self):
        a=np.ones((4,4));solvers=[FloodSolver(a*0,a,1,boundary='periodic',manning=.1) for _ in range(2)]
        for s in solvers:s.mx[:]=1
        solvers[1]._record_speed_peak=lambda:None
        for s in solvers:s.advance(.1)
        for field in ('h','mx','mz'):np.testing.assert_array_equal(getattr(solvers[0],field),getattr(solvers[1],field))
