"""Experimental outlet semantics: no claim of improved field accuracy."""
import unittest,sys,os
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
class BoundaryTrial(unittest.TestCase):
 def test_uniform_flow_is_preserved_with_extrapolation(self):
  for method in ('rusanov','knp'):
   for ux,uy in ((.5,0),(-.5,0),(0,.5),(0,-.5)):
    s=FloodSolver(np.zeros((8,12)),np.ones((8,12)),1,manning=0,flux_method=method,open_boundary_mode='extrapolated')
    s.mx[:]=ux;s.mz[:]=uy;s.advance(2)
    np.testing.assert_allclose(s.h,1,atol=1e-12)
    np.testing.assert_allclose(s.mx,ux,atol=1e-12)
    np.testing.assert_allclose(s.mz,uy,atol=1e-12)
    self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)
 def test_defaults_unchanged(self):
  a=FloodSolver(np.zeros((8,12)),np.ones((8,12)),1,manning=0)
  b=FloodSolver(np.zeros((8,12)),np.ones((8,12)),1,manning=0,open_boundary_mode='outflow_only')
  for s in (a,b):s.mx[:]=.5;s.advance(2)
  np.testing.assert_array_equal(a.h,b.h)
 def test_extrapolation_preserves_dry_bed_and_still_water(self):
  for h in (0.,1.):
   s=FloodSolver(np.zeros((8,12)),np.full((8,12),h),1,open_boundary_mode='extrapolated');s.advance(2)
   np.testing.assert_allclose(s.h,h,atol=1e-12)
 def test_invalid_combination_rejected(self):
  with self.assertRaises(ValueError):FloodSolver(np.zeros((2,2)),np.ones((2,2)),1,boundary='closed',open_boundary_mode='extrapolated')
if __name__=='__main__':unittest.main()
