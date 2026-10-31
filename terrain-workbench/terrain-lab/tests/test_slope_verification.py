import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from round29_slope_audit import exact_depth,steady,uniform,G

class SlopeVerificationTests(unittest.TestCase):
    def test_uniform_thin_flow_gets_gravity_acceleration(self):
        for method in ('rusanov','knp'):
            r=uniform(.18,.1,2,method)
            self.assertLess(r['acceleration_relative_error'],1e-8)
            self.assertLess(r['depth_change'],1e-14)

    def test_supercritical_reference_and_spatial_refinement(self):
        x=np.linspace(0,10,201);h=exact_depth(x,.18);q=.01
        energy=h+q*q/(2*G*h*h)-.18*x
        np.testing.assert_allclose(energy,energy[0],atol=1e-12,rtol=0)
        self.assertTrue(np.all(q/h>np.sqrt(G*h)))
        for method in ('rusanov','knp'):
            runs=[steady(.18,dx,2,method) for dx in (.1,.05,.025)]
            for key in ('depth_mean_abs_m_s','discharge_mean_abs_m2_s2'):
                self.assertGreater(runs[0][key],runs[1][key])
                self.assertGreater(runs[1][key],runs[2][key])
            self.assertTrue(all(r['clipped_face_fraction']==0 for r in runs))
