import unittest,sys,copy
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver,G

class ExtendedPhysicsTests(unittest.TestCase):
 def solver(self,**kw):
  return FloodSolver(np.zeros((8,12)),np.zeros((8,12)),1,boundary='closed',**kw)
 def test_scheduled_source_solid_in_both_orders_is_atomic(self):
  for first in ('solid','schedule'):
   s=self.solver();solid=np.zeros(s.h.shape,bool);solid[2,3]=True
   schedule=[dict(id=1,cells=[[2,3]],hydrograph=[[0,1]])]
   if first=='solid':
    s.set_environment(solid=solid)
    with self.assertRaises(ValueError):s.set_source_schedule(schedule)
    self.assertEqual(s.source_schedule,[])
   else:
    s.set_source_schedule(schedule)
    with self.assertRaises(ValueError):s.set_environment(solid=solid)
    self.assertFalse(s.solid.any())
 def test_invalid_schedule_preserves_previous_source(self):
  good=dict(id=1,cells=[[2,3]],hydrograph=[[0,1]])
  changes=[dict(cells=[]),dict(cells=[[-1,0]]),dict(cells=[[True,0]]),dict(cells=[[2,3],[2,3]]),
           dict(hydrograph=[]),dict(hydrograph=[[1,1],[0,1]]),dict(hydrograph=[[0,-1]]),
           dict(hydrograph=[[0,float('nan')]]),dict(hydrograph=[[0,1],[0,2]])]
  for change in changes:
   s=self.solver();s.set_source_schedule([good]);before=copy.deepcopy(s.source_schedule)
   with self.assertRaises(ValueError):s.set_source_schedule([dict(good,**change)])
   self.assertEqual(s.source_schedule,before)
 def test_hydrograph_integral_and_input_mutation(self):
  s=self.solver();source=[dict(id=1,cells=[[2,3],[2,4]],hydrograph=[[.25,2],[.75,0]])]
  s.set_source_schedule(source);source[0]['hydrograph'][0][1]=200;source[0]['cells'][0][0]=7
  s.advance(1)
  self.assertAlmostEqual(s.h.sum(),1.,places=10)
  self.assertAlmostEqual(s.balance['scheduled_inflow_m3'],1.,places=10)
  self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)
 def test_manning_decay_matches_exact_uniform_solution(self):
  # dq/dt = -g*n²*|q|*q/h^(7/3), exact for constant h and no gradients.
  for method in ('rusanov','knp'):
   for dt in (.2,.05):
    h=.7;n=.04;ux=.8;uy=-.6;end=2.
    s=FloodSolver(np.zeros((6,8)),np.full((6,8),h),1,boundary='periodic',manning=n,flux_method=method,max_timestep=dt)
    s.mx[:]=h*ux;s.mz[:]=h*uy;s.advance(end)
    divisor=1+G*n*n*np.hypot(h*ux,h*uy)*end/h**(7/3)
    np.testing.assert_allclose(s.mx,h*ux/divisor,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(s.mz,h*uy/divisor,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(s.h,h,atol=1e-12)
 def test_uniform_rain_infiltration_matches_storage_balance(self):
  for h0,rain,sink in ((0.,.003,.001),(.002,0.,.003),(.01,.001,.001)):
   for method in ('rusanov','knp'):
    s=FloodSolver(np.zeros((4,6)),np.full((4,6),h0),1,boundary='closed',flux_method=method)
    s.advance(2,rain=rain,infiltration=sink)
    np.testing.assert_allclose(s.h,max(0,h0+2*(rain-sink)),atol=1e-12)
    self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-12)
 def test_transpose_symmetry_with_obstacles_and_momentum(self):
  z,x=np.indices((8,12));bed=.02*x+.03*z;h=np.maximum(.5-bed,0);solid=np.zeros_like(h,bool);solid[3:5,5:7]=True
  for method in ('rusanov','knp'):
   a=FloodSolver(bed,h,1,boundary='closed',flux_method=method)
   b=FloodSolver(bed.T,h.T,1,boundary='closed',flux_method=method)
   a.set_environment(solid=solid);b.set_environment(solid=solid.T)
   a.mx[:]=a.h*.1;a.mz[:]=-a.h*.03;b.mx[:]=a.mz.T;b.mz[:]=a.mx.T
   a.advance(.5);b.advance(.5)
   np.testing.assert_allclose(a.h,b.h.T,atol=1e-12)
   np.testing.assert_allclose(a.mx,b.mz.T,atol=1e-12)
   np.testing.assert_allclose(a.mz,b.mx.T,atol=1e-12)
if __name__=='__main__':unittest.main()
