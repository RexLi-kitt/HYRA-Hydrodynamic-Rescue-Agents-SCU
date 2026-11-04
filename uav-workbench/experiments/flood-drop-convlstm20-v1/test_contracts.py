import unittest,copy
from dataclasses import asdict
import numpy as np
import torch
from long_environment import Env,Config,sample_water,make_scenario
from long_flow import FlowFogEnv
from history_model import History,features
from models import maps,augment,DemandModel
from run import audit

class Contracts(unittest.TestCase):
    def setup_env(self,length=421):
        cfg=Config(duration=2100,window=420,people=1,traps=0,sensor=20.,step=1,trigger_depth=2.,trigger_speed=1.)
        w=np.broadcast_to(np.array([.1,0.,0.],np.float32)[None,:,None,None],(length,3,100,100))
        s=dict(seed=4,config=asdict(cfg),events=[dict(fall=1680,deadline=2100,xy=[500,500])],traps=[])
        return w,s
    def test_reject_short_water(self):
        w,s=self.setup_env(301)
        with self.assertRaises(ValueError):Env(w,s)
    def test_late_time_interpolation_no_last_frame_repeat(self):
        w=np.broadcast_to(np.arange(421,dtype=float)[:,None,None,None],(421,3,100,100))
        np.testing.assert_allclose(sample_water(w,1802,np.array([[500,500]])),360.4)
        np.testing.assert_allclose(sample_water(w,2100,np.array([[500,500]])),420)
        with self.assertRaises(ValueError):sample_water(w,2101,np.array([[500,500]]))
        with self.assertRaises(ValueError):sample_water(w,-1,np.array([[500,500]]))
    def test_late_observation_and_original_deadline(self):
        w,s=self.setup_env();e=Env(w,s);e.t=2099;e.status[:]=1;e.observe_world();e.step(e.control()[1])
        self.assertEqual(e.t,2100);self.assertEqual(e.status[0],3);self.assertEqual(e.deadline[0],2100)
    def test_public_history_ignores_hidden_truth(self):
        w,s=self.setup_env();e=FlowFogEnv(w,s,'convlstm_flowblend',lambda x:np.ones(100));e.t=120;e.observe_world();self.assertTrue(audit(e)['hidden_state_future_water_invariant'])
    def test_map_channel_order(self):
        z=np.arange(100*4*14,dtype=np.float32).reshape(100,4,14);a=maps(z)
        self.assertEqual(a[2,3,4,5],z[45,2,3]);self.assertTrue((a[:,8:10]==0).all())
    def test_rotate_vectors_and_labels(self):
        x=torch.zeros(1,4,14,10,10);x[:,:,3]=2;x[:,:,4]=3;y=torch.zeros(1,10,10);y[0,2,7]=1
        a,b=augment(x,y,1);self.assertTrue((a[:,:,3]==3).all());self.assertTrue((a[:,:,4]==-2).all());self.assertEqual(b[0,2,2],1)
        for _ in range(3):a,b=augment(a,b,1)
        torch.testing.assert_close(a,x);torch.testing.assert_close(b,y)
        a,b=augment(x,y,0,True);a,b=augment(a,b,0,True);torch.testing.assert_close(a,x);torch.testing.assert_close(b,y)
    def test_both_models_finite_gradients(self):
        torch.set_num_threads(2)
        for kind in ('gru','convlstm'):
            m=DemandModel(kind,.1);z=m(torch.randn(2,4,14,10,10));self.assertEqual(tuple(z.shape),(2,10,10));self.assertTrue(bool((z>0).all()));z.mean().backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_roster_windows_complete(self):
        w,s=self.setup_env();c=Config(**s['config']);ss=make_scenario(w,2026,c,fall_end=1680,pool_size=1)
        self.assertEqual(len(ss['events']),1);self.assertLessEqual(ss['events'][0]['deadline'],2100);self.assertEqual(ss['events'][0]['deadline']-ss['events'][0]['fall'],420)

if __name__=='__main__':unittest.main()
