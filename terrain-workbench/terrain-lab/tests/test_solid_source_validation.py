import unittest,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
class SolidSourceTests(unittest.TestCase):
 def make(self):return FloodSolver(np.zeros((4,4)),np.zeros((4,4)),1,boundary='closed')
 def test_reject_source_after_building_without_replacing_valid_source(self):
  s=self.make();solid=np.zeros((4,4),bool);solid[1,1]=True;s.set_environment(solid=solid)
  s.set_baseflow([dict(z=2,x=2,discharge_m3_s=1)]);before=s.baseflow_rate.copy()
  with self.assertRaisesRegex(ValueError,'solid'):s.set_baseflow([dict(z=1,x=1,discharge_m3_s=1)])
  np.testing.assert_array_equal(s.baseflow_rate,before)
  s.advance(1);self.assertAlmostEqual(s.h.sum(),1,places=10);self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)
 def test_reject_building_after_source_without_mutating_environment(self):
  s=self.make();s.set_baseflow([dict(z=1,x=1,discharge_m3_s=1)])
  solid=np.zeros((4,4),bool);solid[1,1]=True
  with self.assertRaisesRegex(ValueError,'solid'):s.set_environment(solid=solid)
  self.assertFalse(s.solid.any());s.advance(1);self.assertAlmostEqual(s.h.sum(),1,places=10)
 def test_invalid_late_environment_field_does_not_apply_earlier_field(self):
  s=self.make();solid=np.ones((4,4),bool)
  with self.assertRaises(ValueError):s.set_environment(solid=solid,drain_capacity=np.full((4,4),-1.))
  self.assertFalse(s.solid.any())
if __name__=='__main__':unittest.main()
