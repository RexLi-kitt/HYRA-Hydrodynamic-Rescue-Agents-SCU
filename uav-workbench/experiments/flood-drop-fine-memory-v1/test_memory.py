import unittest,copy
from types import SimpleNamespace
import numpy as np
import torch
from memory import public_features,augment,MemoryGRU,FineHistory
class Contracts(unittest.TestCase):
    def env(self):
        rr,cc=np.mgrid[:100,:100]
        return SimpleNamespace(t=120,known=cc%10<2,last=np.full((100,100),90.),
            measured=np.ones((3,100,100)),cfg=SimpleNamespace(duration=2100,sensor=20,people=40,window=420,wet_depth=.01),
            discovered=np.zeros(0,bool),seen_xy=np.zeros((0,2)),belief=np.zeros(0))
    def test_fine_distinguishes_old_alias(self):
        a=self.env();b=copy.deepcopy(a);b.known=np.indices((100,100))[1]%10>=8
        x=public_features(a);y=public_features(b)
        np.testing.assert_array_equal(x[:14],y[:14]);self.assertFalse(np.array_equal(x[14:],y[14:]))
    def test_rotation_matches_fine_observation(self):
        a=self.env();x=public_features(a);b=copy.deepcopy(a)
        b.known=np.rot90(a.known);b.last=np.rot90(a.last)
        b.measured=np.rot90(a.measured,1,(-2,-1)).copy()
        b.measured[1:]=np.stack([b.measured[2].copy(),-b.measured[1].copy()])
        z,_=augment(torch.from_numpy(x)[None,None],torch.zeros(1,10,10),1,False)
        np.testing.assert_allclose(z[0,0].numpy(),public_features(b))
    def test_group_roundtrip(self):
        x=torch.randn(2,4,64,10,10);y=torch.rand(2,10,10);z=x.clone();q=y.clone()
        for _ in range(4):z,q=augment(z,q,1,False)
        torch.testing.assert_close(z,x);torch.testing.assert_close(q,y)
        z,q=augment(x,y,0,True);z,q=augment(z,q,0,True)
        torch.testing.assert_close(z,x);torch.testing.assert_close(q,y)
    def test_history_causal_and_hidden_independent(self):
        a=self.env();b=copy.deepcopy(a);b.person=np.full((40,2),888);b.water=None;b.deadline=np.arange(40)
        np.testing.assert_array_equal(public_features(a),public_features(b))
        h=FineHistory();x=h.get(a);a.t=121;a.known=~a.known
        np.testing.assert_array_equal(h.get(a),x)
        a.t=150;self.assertFalse(np.array_equal(h.get(a),x))
    def test_matched_capacity_and_finite_gradients(self):
        torch.set_num_threads(4)
        a=MemoryGRU(.1,True);b=MemoryGRU(.1,False)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        for m in (a,b):
            z=m(torch.randn(2,4,64,10,10));z.mean().backward()
            self.assertTrue(torch.isfinite(z).all() and (z>0).all())
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()

