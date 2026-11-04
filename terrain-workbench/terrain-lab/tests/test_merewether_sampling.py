import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from merewether import measurement_indices

class MeasurementIndicesTests(unittest.TestCase):
    def test_valid_centres_and_internal_faces(self):
        z,x=measurement_indices([[.25,1.75],[.5,1.5],[1.75,.25]],(4,4),.5)
        np.testing.assert_array_equal(z,[0,1,3]);np.testing.assert_array_equal(x,[0,1,3])
    def test_outside_and_excluded_faces(self):
        for point in [[-.01,1],[1,2.01],[2,1],[1,0],[1,-.01]]:
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError,'outside'):measurement_indices([point],(4,4),.5)
    def test_nonfinite_and_single_stage_point(self):
        for v in [float('nan'),float('inf')]:
            with self.assertRaises(ValueError):measurement_indices([[v,1]],(4,4),.5)
        z,x=measurement_indices([[0,2]],(4,4),.5)
        self.assertEqual((z.item(),x.item()),(0,0))
