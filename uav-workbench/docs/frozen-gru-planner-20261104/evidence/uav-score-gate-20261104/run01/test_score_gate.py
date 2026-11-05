import unittest
import numpy as np
from score_gate import strength
from neutral_residual import neutral_residual_score

class Contracts(unittest.TestCase):
    def test_continuous_neutral_limit_and_full_strength(self):
        base=np.ones((100,100));flat=np.ones((10,10));near=flat.copy();near[0,0]+=1e-6
        self.assertEqual(strength(flat,.1),0.)
        gated=neutral_residual_score(base,near,alpha=.25*strength(near,.1))
        self.assertAlmostEqual(gated[0,0],1.0000025)
        self.assertAlmostEqual(neutral_residual_score(base,near,alpha=.25)[0,0],1.25)
        high=flat.copy();high[0,0]+=1
        np.testing.assert_array_equal(neutral_residual_score(base,high,alpha=.25),
            neutral_residual_score(base,high,alpha=.25*strength(high,.1)))

    def test_rotation_and_invalid_inputs(self):
        p=np.arange(100).reshape(10,10)/100
        self.assertEqual(strength(p,.8),strength(np.rot90(p),.8))
        for ref in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):strength(p,ref)
        for bad in (np.full((10,10),np.nan),np.full((10,10),-1.)):
            with self.assertRaises(ValueError):strength(bad,.1)

    def test_neutral_before_nonuniform_transport(self):
        base=np.ones((100,100));p=np.ones((10,10));p[0,0]+=1e-8
        def transport(q):
            q[:,0]*=.5
            return q
        output=neutral_residual_score(base,p,alpha=.25*strength(p,.1),transport_fn=transport)
        self.assertLess(np.max(np.abs(output-base)),3e-8)

if __name__=='__main__':unittest.main()
