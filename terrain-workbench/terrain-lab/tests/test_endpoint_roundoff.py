import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
class EndpointTests(unittest.TestCase):
 def make(self,**kw):return FloodSolver(np.zeros((2,2)),np.ones((2,2)),1,boundary='closed',**kw)
 def test_final_roundoff_remainder_is_integrated(self):
  s=self.make();s.time=1.-6e-15;before=s.time
  s.advance(1.,rain=1e8)
  self.assertEqual(s.time,1.);self.assertEqual(s.steps,1)
  self.assertAlmostEqual(s.balance['rainfall_m3'],4e8*(1-before),places=16)
  self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-12)
 def test_short_valid_requested_interval(self):
  s=self.make();s.advance(1e-12,rain=1e6)
  self.assertEqual(s.time,1e-12);self.assertAlmostEqual(s.balance['rainfall_m3'],4e-6,places=16)
 def test_small_cfl_still_rejected(self):
  s=self.make();s.mx[:]=1e12
  with self.assertRaisesRegex(RuntimeError,'limiting_cell'):s.advance(1.)
  self.assertEqual(s.steps,0)
 def test_step_limit_still_applies_to_tiny_final_interval(self):
  s=self.make(max_steps=1);s.steps=1;s.time=1.-6e-15
  with self.assertRaises(RuntimeError):s.advance(1.)
 def test_repeated_output_times(self):
  s=self.make(max_timestep=.00025)
  # Reproduce accumulated rounding close to the originally failing endpoint.
  s.time=1.30
  for t in np.linspace(1.31,1.50,20):s.advance(float(t))
  self.assertEqual(s.time,1.5);np.testing.assert_allclose(s.h,1,atol=1e-12)
if __name__=='__main__':unittest.main()
