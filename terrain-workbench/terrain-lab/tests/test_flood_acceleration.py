"""Compiled native kernels must reproduce the reference physical scheme."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
try:
    import flood_acceleration
except ImportError:
    flood_acceleration=None


@unittest.skipUnless(flood_acceleration,'optional acceleration is not installed')
class AccelerationTests(unittest.TestCase):
    def pair(self,boundary,method,order,low_froude):
        z,x=np.indices((8,12));bed=.03*x+.01*z+10000.
        h=np.maximum(.25-.03*x-.01*z,0.)+.02*(x<4)
        options=dict(boundary=boundary,flux_method=method,spatial_order=order,
            low_froude=low_froude,sea_level=10000.2 if boundary=='ocean' else None,
            sea_edges={'west':np.ones(8,bool)} if boundary=='ocean' else None)
        with patch.dict(os.environ,HYRA_FLOOD_BACKEND='numpy'):
            a=FloodSolver(bed,h,1.,**options)
        with patch.dict(os.environ,HYRA_FLOOD_BACKEND='numba'):
            b=FloodSolver(bed,h,1.,**options)
        solid=np.zeros(h.shape,bool);solid[3,3]=True
        wall=np.zeros((8,13),bool);wall[2:5,7]=True
        for s in (a,b):
            s.set_environment(solid=solid,wall_x=wall)
            s.mx[:]=s.h*.08*np.sin(x);s.mz[:]=s.h*.06*np.cos(z)
            s.set_external_inflows([dict(id='source',enabled=True,
                delivery='boundary_discharge_with_momentum',hydrograph=[[0,.05],[.04,0]],
                inlet_velocity_m_s=.3,boundary_faces=[dict(edge='west',offset=1,z=1,x=0,width_m=.5,weight=1.,
                    momentum_direction_xz=[.8,.6])])])
        return a,b

    def test_reconstruction_flux_timestep_and_rk_state_match_reference(self):
        for boundary in ('closed','open','ocean','periodic'):
            for method in ('rusanov','knp'):
                for order in (1,2):
                    for low in (0,1,2):
                        with self.subTest(boundary=boundary,method=method,order=order,low=low):
                            a,b=self.pair(boundary,method,order,low)
                            ra,rb=a._reconstruction(),b._reconstruction()
                            for axis in (0,1):
                                for left,right in zip(ra[axis],rb[axis]):
                                    np.testing.assert_allclose(left,right,rtol=1e-13,atol=1e-14)
                                fa,fb=a._flux(axis,details=True),b._flux(axis,details=True)
                                for key in fa:np.testing.assert_allclose(fa[key],fb[key],rtol=1e-12,atol=1e-14)
                            self.assertAlmostEqual(a.timestep(.2,rain=.01),b.timestep(.2,rain=.01),places=14)
                            for _ in range(4):
                                dt=min(a.timestep(.01),b.timestep(.01))
                                for s in (a,b):s.step(dt,rain=.001,infiltration=.0001,inflow=.01,source=(2,2))
                            for key in ('h','mx','mz','qx','qy','maximum'):
                                np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-11,atol=1e-13)
                            for key in a.balance:self.assertAlmostEqual(a.balance[key],b.balance[key],places=11)


if __name__=='__main__':unittest.main()
