import unittest
import numpy as np
import torch
from train import Ranker,choose,action_features,PREVIOUS,read,SegmentEnv,proposals,clone

class Contracts(unittest.TestCase):
    def test_default_centering_and_finite_gradient(self):
        for kind in ('linear','deep'):
            torch.manual_seed(15);m=Ranker(kind);x=torch.randn(4,225)
            self.assertTrue(torch.equal(m(x),torch.zeros(4)))
            opt=torch.optim.Adam(m.parameters(),lr=.001)
            for _ in range(3):
                opt.zero_grad();v=m(x)
                loss=torch.nn.functional.smooth_l1_loss(v,torch.tensor([0.,2.,-1.,1.]))
                loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters()))
                opt.step()
            self.assertEqual(m(x)[0].item(),0.)
            self.assertTrue(torch.isfinite(m(x)).all())

    def test_no_override_gate(self):
        self.assertEqual(choose(np.array([0.,0.,0.,0.]),0),0)
        self.assertEqual(choose(np.array([0.,10.,-1.,2.]),None),0)
        self.assertEqual(choose(np.array([0.,.5,-1.,.1]),.5),0)
        self.assertEqual(choose(np.array([0.,1.,-1.,.1]),.5),1)

    def test_public_action_features_and_clone(self):
        c=read(PREVIOUS/'scenarios.json')['fit'][12]
        e=SegmentEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'])
        ps=proposals(e);x=action_features(e,ps)
        self.assertEqual(x.shape,(4,225));self.assertFalse(np.array_equal(x[1],x[2]))
        b=clone(e);self.assertIs(e.water,b.water)
        b.person[:]=999;b.status[:]=3;b.fall[:]+=7;b.deadline[:]+=7;b.water=None
        self.assertEqual(ps,proposals(b));np.testing.assert_array_equal(x,action_features(b,ps))
        np.testing.assert_array_equal(e.pos,[[5.,5.],[995.,5.],[5.,995.],[995.,995.]])

if __name__=='__main__':unittest.main()
