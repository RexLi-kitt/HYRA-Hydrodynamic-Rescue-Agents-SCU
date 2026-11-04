"""Algorithm distinctions, information boundaries, finite updates and restoration."""
import copy,unittest
import numpy as np
import torch
from flood1km.learners import ALGORITHMS,Learner,CounterfactualCritic,AttentionCritic,TransformerPolicy,gae,td_lambda,distribution

class LearnerTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(1);np.random.seed(1);torch.set_num_threads(1)
    def rows(self,learner):
        result=[]
        for t in range(12):
            obs=np.random.normal(size=(4,12)).astype(np.float32);mask=np.ones((4,9),bool)
            action,logp,value=learner.act(obs,mask)
            result.append(dict(obs=obs,mask=mask,action=action,logp=logp,value=value,reward=float(t%3==0),done=float(t==11),
                               next_obs=obs*.9,next_mask=mask,next_action=np.full(4,8)))
        return result
    def test_all_six_registry(self):
        self.assertEqual(set(ALGORITHMS),{'happo','hatrpo','hasac','maac','mat','coma'})
    def test_unregistered_rejected(self):
        for name in ('mappo','qmix','vdn','ippo','dqn','ddpg'):
            with self.assertRaises(ValueError):Learner(name,12)
    def test_action_mask(self):
        for name in ALGORITHMS:
            model=Learner(name,12);mask=np.zeros((4,9),bool);mask[:,8]=True
            action,logs,_=model.act(np.zeros((4,12),np.float32),mask)
            np.testing.assert_array_equal(action,np.full(4,8));self.assertTrue(np.isfinite(logs).all())
    def test_counterfactual_critic_excludes_own_action(self):
        obs=torch.randn(3,4,12);actions=torch.zeros(3,4,dtype=torch.long)
        for model in (CounterfactualCritic(12),AttentionCritic(12)):
            before=model(obs,actions).detach();changed=actions.clone();changed[:,2]=5
            after=model(obs,changed).detach();torch.testing.assert_close(before[:,2],after[:,2])
    def test_mat_causal_actions(self):
        model=TransformerPolicy(12);obs=torch.randn(2,4,12);a=torch.zeros(2,4,dtype=torch.long)
        before,_=model(obs,a);a[:,2:]=7;after,_=model(obs,a)
        torch.testing.assert_close(before[:,:3],after[:,:3])
    def test_mat_sampling_matches_teacher_forcing(self):
        model=TransformerPolicy(12);obs=torch.randn(2,4,12);mask=torch.ones(2,4,9,dtype=torch.bool)
        a,logs,_=model.sample(obs,mask);logits,_=model(obs,a)
        torch.testing.assert_close(logs,distribution(logits,mask).log_prob(a))
    def test_terminal_no_bootstrap(self):
        r=torch.tensor([2.,3.]);v=torch.tensor([9.,9.]);done=torch.ones(2)
        _,returns=gae(r,v,done);torch.testing.assert_close(returns,r)
        target=td_lambda(r,torch.ones(2,4)*99,done);torch.testing.assert_close(target,r[:,None].expand(-1,4))
    def test_every_algorithm_updates_and_restores(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                model=Learner(name,12);rows=self.rows(model);before=copy.deepcopy(model.state_dict())
                stats=model.update_off_policy(rows) if name in ('hasac','maac') else model.update_on_policy(rows)
                model.check_finite();self.assertEqual(model.updates,1)
                self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.state_dict().items()))
                restored=Learner(name,12);restored.restore(copy.deepcopy(model.checkpoint()))
                for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k])
                # A second identical update checks optimizer state, not just parameter copying.
                state=torch.get_rng_state()
                update='update_off_policy' if name in ('hasac','maac') else 'update_on_policy'
                getattr(model,update)(rows);torch.set_rng_state(state);getattr(restored,update)(rows)
                for k,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[k])
    def test_trust_region_bound(self):
        model=Learner('hatrpo',12);stats=model.update_on_policy(self.rows(model))
        for k,v in stats.items():
            if k.startswith('actor_'):self.assertLessEqual(v['kl'],.010001)

if __name__=='__main__':unittest.main()
