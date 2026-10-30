import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver

class ReflectiveWallTests(unittest.TestCase):
    def test_invalid_mode(self):
        with self.assertRaises(ValueError):FloodSolver(np.zeros((4,4)),np.zeros((4,4)),1,wall_flux_mode='bad')
    def test_internal_wall_matches_closed_outer_boundary(self):
        for method in ('rusanov','knp'):
            for speed in (-.5,.5,2.):
                for axis in (0,1):
                    with self.subTest(method=method,speed=speed,axis=axis):
                        options=dict(boundary='closed',manning=0,flux_method=method,wall_flux_mode='reflective')
                        a=FloodSolver(np.zeros((6,6)),np.ones((6,6)),1,**options)
                        b=FloodSolver(np.zeros((6,6)),np.ones((6,6)),1,**options)
                        if axis==1:
                            a.mx[:]=speed;b.mx[:]=speed;wall=np.zeros((6,7),bool);wall[:,3]=True;b.set_environment(wall_x=wall)
                        else:
                            a.mz[:]=speed;b.mz[:]=speed;wall=np.zeros((7,6),bool);wall[3,:]=True;b.set_environment(wall_y=wall)
                        fa=a._flux(axis,details=True);fb=b._flux(axis,details=True)
                        momentum='advx' if axis==1 else 'advz'
                        outer=(fa[momentum]+fa['pressure'])[:,-1] if axis==1 else (fa[momentum]+fa['pressure'])[-1]
                        inner=fb['corr_l'][:,3] if axis==1 else fb['corr_l'][3]
                        np.testing.assert_allclose(inner,outer,atol=1e-12)
                        self.assertIsNone(b.acceleration)
    def test_still_water_and_collision_mass(self):
        for method in ('rusanov','knp'):
            for speed in (0.,1.):
                a=FloodSolver(np.zeros((8,16)),np.ones((8,16)),.1,boundary='closed',manning=0,
                              flux_method=method,wall_flux_mode='reflective')
                solid=np.zeros((8,16),bool);solid[2:6,8:10]=True;a.set_environment(solid=solid);a.mx[:]=a.h*speed
                a.advance(.5)
                self.assertLess(abs(a.summary()['balance']['residual_m3']),1e-10)
                self.assertTrue(np.isfinite(a.h).all());self.assertGreaterEqual(a.h.min(),0)
                if speed==0:
                    np.testing.assert_allclose(a.h[~solid],1,atol=1e-12);np.testing.assert_allclose(a.velocity(),0,atol=1e-12)
if __name__=='__main__':unittest.main()
