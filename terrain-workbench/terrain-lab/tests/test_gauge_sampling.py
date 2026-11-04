import unittest
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver
from gauge_sampling import sample_gauge

class GaugeSamplingTests(unittest.TestCase):
    def base(self):
        a=np.ones((3,3));s=FloodSolver(a*0,a,2);s.mx[:]=np.arange(9).reshape(3,3);return s
    def test_origin_and_half_open_edges(self):
        s=self.base();g=lambda x,z:sample_gauge(s,x,z,origin_center_m=(100,200))
        self.assertEqual(g(100,200)['u_m_s'],0)
        self.assertEqual(g(101,200)['u_m_s'],1)
        self.assertEqual(g(99,199)['status'],'wet')
        self.assertEqual(g(105,200)['status'],'outside')
        self.assertEqual(g(98.99,200)['status'],'outside')
    def test_dry_solid_and_stationary_distinct(self):
        s=self.base();s.h[0,1]=0;solid=np.zeros((3,3),bool);solid[1,0]=True;s.set_environment(solid=solid)
        wet=sample_gauge(s,0,0,origin_center_m=(0,0))
        self.assertEqual(wet['speed_m_s'],0)
        for x,z,status in ((2,0,'dry'),(0,2,'solid')):
            r=sample_gauge(s,x,z,origin_center_m=(0,0));self.assertEqual(r['status'],status);self.assertIsNone(r['speed_m_s'])
    def test_leaf_values_not_parent_average(self):
        s=RefinedFloodSolver.from_solver(self.base(),np.ones((3,3),bool))
        s.leaf['mx'][:4]=[1,2,3,4];s._sync()
        for x,z,v in ((-.5,-.5,1),(.5,-.5,2),(-.5,.5,3),(.5,.5,4)):
            r=sample_gauge(s,x,z,origin_center_m=(0,0));self.assertEqual(r['u_m_s'],v);self.assertEqual(r['cell_width_m'],1)
    def test_no_cross_wall_average(self):
        s=self.base();s.mx[:,0]=-2;s.mx[:,1]=8
        wall=np.zeros((3,4),bool);wall[:,1]=True;s.set_environment(wall_x=wall)
        self.assertEqual(sample_gauge(s,.9,0,origin_center_m=(0,0))['u_m_s'],-2)
        self.assertEqual(sample_gauge(s,1.1,0,origin_center_m=(0,0))['u_m_s'],8)
    def test_invalid_coordinates(self):
        for x in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):sample_gauge(self.base(),x,0,origin_center_m=(0,0))
