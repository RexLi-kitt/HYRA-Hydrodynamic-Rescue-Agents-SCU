import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from okushiri_boundary import OkushiriBoundarySolver
class WaveBoundaryTests(unittest.TestCase):
    def make(self,wave=None):
        return OkushiriBoundarySolver(np.full((4,20),-.135),np.full((4,20),.135),.1,wave=wave if wave is not None else [[0,0],[1,0]],manning=0,max_timestep=.005)
    def test_momentum_and_other_walls(self):
        for q in [0.,.01,-.01]:
            s=self.make();s.mx[:]=q
            fx=s._flux(1,details=True);fz=s._flux(0,details=True)
            np.testing.assert_allclose(fx['mass'][:,0],q,rtol=0,atol=1e-12)
            np.testing.assert_allclose(fx['mass'][:,-1],0,atol=1e-12)
            np.testing.assert_allclose(fz['mass'][[0,-1]],0,atol=1e-12)
    def test_rest(self):
        s=self.make();s.advance(.2)
        np.testing.assert_allclose(s.h,.135,rtol=0,atol=1e-12)
        np.testing.assert_allclose(s.mx,0,atol=1e-12)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-12)
    def test_stage_times_and_mass(self):
        s=self.make([[0,0],[.1,.005],[.2,-.003],[1,0]])
        s.step(.005);self.assertEqual(s.last_wave_stage_times,[0.,.005])
        s.advance(.4)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-12)
        self.assertTrue(np.isfinite(s.h).all());self.assertGreaterEqual(s.h.min(),0)
        self.assertGreater(s.balance['sea_inflow_m3'],0)
    def test_reject_extrapolation(self):
        s=self.make()
        with self.assertRaises(ValueError):s.advance(1.1)
        self.assertEqual(s.time,0)
    def test_reject_dry_boundary(self):
        with self.assertRaises(ValueError):self.make([[0,0],[1,-.14]])
    def test_cfl_and_breakpoint(self):
        s=self.make([[0,0],[.003,.01],[1,0]])
        dt=s.timestep(.1);self.assertLessEqual(dt,.003)
        self.assertGreater(s.last_timestep['boundary_wave_bound_m_s'],0)
if __name__=='__main__':unittest.main()
