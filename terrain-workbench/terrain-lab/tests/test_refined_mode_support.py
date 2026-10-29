import unittest
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver

class RefinedModeSupportTests(unittest.TestCase):
    def test_default_supported(self):
        a=np.zeros((4,4))
        s=RefinedFloodSolver.from_solver(FloodSolver(a,a,1),a.astype(bool))
        s.advance(.01)
        self.assertEqual(s.h.sum(),0)

    def test_experimental_modes_rejected_without_mutation(self):
        for option in (dict(open_boundary_mode='extrapolated'),dict(wall_flux_mode='reflective')):
            for refine in (False,True):
                with self.subTest(option=option,refine=refine):
                    a=np.zeros((4,4));base=FloodSolver(a,a,1,**option)
                    with self.assertRaisesRegex(ValueError,'native grid'):
                        RefinedFloodSolver.from_solver(base,np.full(a.shape,refine))
                    self.assertEqual(base.time,0)
                    self.assertFalse(hasattr(base,'mesh'))
                    np.testing.assert_array_equal(base.h,a)
