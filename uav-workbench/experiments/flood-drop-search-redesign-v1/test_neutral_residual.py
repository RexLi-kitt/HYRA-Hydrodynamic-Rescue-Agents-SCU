import unittest
import numpy as np
from neutral_residual import neutral_residual_score

class NeutralContract(unittest.TestCase):
    def setUp(self):
        self.b=np.random.default_rng(42).random((100,100))
    def test_disabled_exact(self):
        np.testing.assert_array_equal(neutral_residual_score(self.b,np.arange(100),alpha=0),self.b)
    def test_absent_exact(self):
        np.testing.assert_array_equal(neutral_residual_score(self.b,None,alpha=.25),self.b)
    def test_constant_before_transport(self):
        def no_call(p): raise AssertionError('constant cannot invoke transport')
        for c in (0.,.1,2.):
            np.testing.assert_array_equal(neutral_residual_score(self.b,np.full(100,c),alpha=.25,transport_fn=no_call),self.b)
    def test_bounded_positive_immutable(self):
        b=self.b.copy();p=np.arange(100,dtype=float);saved=p.copy()
        z=neutral_residual_score(self.b,p,alpha=.25)
        self.assertTrue(np.isfinite(z).all())
        self.assertTrue((z>=.75*self.b-1e-15).all() and (z<=1.25*self.b+1e-15).all())
        self.assertFalse(np.array_equal(z,self.b))
        np.testing.assert_array_equal(self.b,b);np.testing.assert_array_equal(p,saved)
    def test_reject_invalid(self):
        for a in (-.1,.251,float('nan')):
            with self.assertRaises(ValueError):neutral_residual_score(self.b,alpha=a)
        for p in (np.full(100,float('nan')),np.full(100,-1.),np.zeros(3)):
            with self.assertRaises(ValueError):neutral_residual_score(self.b,p,alpha=.1)
    def test_transport_invalid(self):
        with self.assertRaises(ValueError):
            neutral_residual_score(self.b,np.arange(100),alpha=.1,transport_fn=lambda p: p*np.nan)
if __name__=='__main__':unittest.main()

