import sys,unittest,math
from pathlib import Path
ROOT=Path(r'D:\HyRA\warehouse\HYRA-Hydrodynamic-Rescue-Agents-SCU\uav-workbench')
sys.path[:0]=[str(Path(__file__).parent),str(ROOT)]
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
from sur_rescue.gmrf import SAR,DualGMRF
from sur_rescue.environment import RescueEnv,RescueConfig,joint_motion,rule_actions,ACTIONS

class FakeHydro:
    shape=(10,10);dx=5.;digest='test';process={'grid':{'shape':[10,10],'dx_m':5.,'axis_order':['z','x']}}
    def index(self,t):return int(t)
    def labels(self,t):return np.indices(self.shape)[0]<int(t//5)%10

class RescueTests(unittest.TestCase):
    def test_sar_matches_independent_dense_solve_and_gaussian_precision(self):
        sar=SAR((3,4));n=12;columns=[]
        for i in range(n):
            v=np.eye(n)[i].reshape(3,4);columns.append(sar.apply(v).ravel())
        c=np.asarray(columns).T;rng=np.random.default_rng(9);noise=rng.normal(0,.1,(3,4));mean=rng.normal(size=(3,4))
        actual,d=sar.solve(mean,noise);expected=mean.ravel()+np.linalg.solve(np.eye(n)-c,noise.ravel())
        np.testing.assert_allclose(actual.ravel(),expected,atol=1e-8);self.assertLess(max(abs(np.linalg.eigvals(c))),1)
        precision=(np.eye(n)-c).T@(np.eye(n)-c)/.1**2;self.assertTrue(np.all(np.linalg.eigvalsh(precision)>0));self.assertLess(d['residual_inf'],1e-8)
        cut=np.zeros((3,4),bool);cut[1,1]=True;cut_c=np.column_stack([sar.apply(np.eye(n)[i].reshape(3,4),cut).ravel() for i in range(n)])
        self.assertTrue(np.all(cut_c<=c));self.assertTrue(np.all(cut_c>=0))
    def test_large_sparse_sar_and_signal_decay(self):
        sar=SAR((256,256));self.assertEqual(sar.weights.shape,(8,256,256))
        x,d=sar.solve(np.zeros((256,256)),np.ones((256,256))*.01);self.assertLess(d['iterations'],256)
        blocked=np.zeros((10,10),bool);blocked[4,4]=True;dual=DualGMRF((10,10),blocked,[(5,5)],11)
        signal=dual.sample_signal(np.arange(100)).reshape(10,10);self.assertEqual(signal[4,4],0);self.assertGreater(signal[5,5],signal[0,0])
        empty=DualGMRF((10,10),blocked,[],11);self.assertFalse(empty.sample_signal(np.arange(100)).any())
    def test_eight_exact_neighbour_displacements(self):
        blocked=np.zeros((10,10),bool)
        expected=[(5,4),(5,6),(4,5),(6,5),(4,4),(6,4),(4,6),(6,6)]
        for a,p in enumerate(expected):
            result,status,_=joint_motion([(5,5),(9,9)],[a,0],blocked,[False,True]);self.assertEqual(tuple(result[0]),p);self.assertEqual(status[0],'move')
    def test_boundary_obstacle_corner_and_crossing(self):
        b=np.zeros((10,10),bool);b[1,2]=True
        end,why,_=joint_motion([(0,0),(1,1)],[0,7],b,[False,False]);self.assertEqual(why,['boundary','obstacle'])
        end,why,c=joint_motion([(2,2),(2,3)],[7,5],b,[False,False]);self.assertEqual(c,1);np.testing.assert_array_equal(end,[(2,2),(2,3)])
        end,why,c=joint_motion([(2,2),(2,3)],[1,0],b,[False,False]);self.assertEqual(c,1);np.testing.assert_array_equal(end,[(2,2),(2,3)])
    def test_sensor_permissions_nine_channels_and_rescue_dwell(self):
        scenario=dict(obstacles=np.zeros((10,10),bool),targets=[(0,1),(5,8)])
        env=RescueEnv(FakeHydro(),RescueConfig(duration_s=20,service_s=10),scenario);view=env.reset()
        self.assertEqual(view.images.shape,(2,9,5,5));self.assertFalse(view.images.flags.writeable)
        self.assertFalse(hasattr(view,'training_state'));self.assertFalse(hasattr(view,'_targets'))
        self.assertNotIn((5,8,'undiscovered'),view.targets);self.assertTrue(np.all(view.signal[~view.explored]==0))
        view,_,_=env.step([1,1]);self.assertEqual(view.busy[0],10)
        first=view.positions[0];view,_,_=env.step([1,1]);self.assertEqual(view.positions[0],first);self.assertEqual(view.busy[0],5)
        view,_,_=env.step([1,1]);self.assertEqual(view.positions[0],first);self.assertIn((0,1,'rescued'),view.targets)
        self.assertEqual(env.snapshots[-1]['motion_status'][0],'rescuing')
        for row in env.observations:
            self.assertTrue(all(0<=c<100 for c in row['cells']))
    def test_terminal_and_actual_frame_only(self):
        env=RescueEnv(FakeHydro(),RescueConfig(duration_s=10));view=env.reset()
        _,_,done=env.step(rule_actions(view));self.assertFalse(done)
        _,_,done=env.step(rule_actions(env.view()));self.assertTrue(done)
        with self.assertRaises(ValueError):env.step([1,1])
        with self.assertRaises(ValueError):RescueConfig(duration_s=7).validate(FakeHydro())
        replay=env.result();self.assertEqual(len(replay['snapshots']),3);self.assertEqual(len(replay['actions']),2)
        self.assertEqual(replay['audit']['hydraulic_cache_max_frames'],2)

class LearningTests(unittest.TestCase):
    def test_network_eight_actions_enhancement_and_parameter_update(self):
        import torch
        from sur_rescue.learning import RescueAgent,Learner,execute,batch_episodes
        torch.set_num_threads(1);torch.manual_seed(11);actor=RescueAgent()
        env=RescueEnv(FakeHydro(),RescueConfig(duration_s=10));view=env.reset()
        q,h,s=actor(torch.from_numpy(view.images.copy()),torch.zeros(2,8),0,actor.hidden(2),True)
        self.assertEqual(tuple(q.shape),(2,8));self.assertTrue(torch.isfinite(q).all());self.assertTrue(((s>=0)&(s<=1)).all())
        actor.eval()
        with self.assertRaises(RuntimeError):actor(torch.from_numpy(view.images.copy()),torch.zeros(2,8),0,h,True)
        with self.assertRaises(ValueError):actor(torch.zeros(2,16,7,7),torch.zeros(2,4),0,h)
        _,data=execute(actor,env,collect=True);learner=Learner(len(env.training_state()),True);before=learner.actor.output.weight.detach().clone()
        result=learner.train(batch_episodes([data,data]));self.assertTrue(math.isfinite(result['loss']));self.assertFalse(torch.equal(before,learner.actor.output.weight))
        # Actual inference must remain operational when centralized truth is unavailable.
        env.training_state=lambda:(_ for _ in ()).throw(AssertionError('CTDE truth requested during execution'))
        execute(learner.actor,env,collect=False)

if __name__=='__main__':unittest.main(verbosity=2)
