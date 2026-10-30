import unittest
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver

class RefinedOwnershipTests(unittest.TestCase):
    def base(self):
        s=FloodSolver(np.zeros((4,4)),np.ones((4,4)),1,boundary='closed',manning=0,max_timestep=.01)
        s.set_external_inflows([dict(id='inlet',enabled=True,cells=[dict(z=1,x=1,weight=1.)],hydrograph=[[0,.2],[.1,0]])])
        return s
    def test_independent_runs_do_not_double_count(self):
        base=self.base();mask=np.ones_like(base.h,bool)
        a=RefinedFloodSolver.from_solver(base,mask);b=RefinedFloodSolver.from_solver(base,mask)
        a.advance(.1)
        self.assertEqual(base.balance['external_inflow_m3'],0)
        self.assertEqual(b.balance['external_inflow_m3'],0)
        self.assertEqual(base.external_volume_by_id['inlet'],0)
        self.assertIsNone(base.diagnostics['min_dt_s'])
        b.advance(.1)
        for s in (a,b):
            self.assertAlmostEqual(s.balance['external_inflow_m3'],.02)
            self.assertAlmostEqual(s.external_volume_by_id['inlet'],.02)
            self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)
        np.testing.assert_array_equal(a.h,b.h)
        np.testing.assert_array_equal(a.mx,b.mx)
        a.start_scenario()
        self.assertAlmostEqual(b.balance['external_inflow_m3'],.02)
    def test_nested_forcing_and_arrays_are_owned(self):
        base=self.base();a=RefinedFloodSolver.from_solver(base,np.ones_like(base.h,bool))
        self.assertIs(a.acceleration,base.acceleration)
        for name in ('bed','manning','solid','wall_x','wall_y','drain_capacity','source_rate','external_rate','initial'):
            self.assertFalse(np.shares_memory(getattr(a,name),getattr(base,name)),name)
        a.external_schedule[0]['hydrograph'][0][1]=.9
        self.assertEqual(base.external_schedule[0]['hydrograph'][0][1],.2)
        base.manning[:]=.8
        self.assertTrue(np.all(a.manning==0))
        a.sea_edges['west'][:]=True
        self.assertFalse(base.sea_edges['west'].any())
