"""Controls for the auxiliary optimization revision; no heldout scoring."""
import unittest,json
from pathlib import Path
from scale100_aux_tuning import *
DATA=Path(__file__).parent.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'

class AuxChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);rows=json.loads(DATA.read_text())['cases'];cls.cases=[FloodCase(r['path']) for r in rows if r['split']=='fit'];cls.prior=fit_prior(cls.cases)
        torch.manual_seed(11);env=ScaleEnv(cls.cases[0],config('E1','B3',11));env.reset();cls.state_dim=len(env.training_state());actor=Agent()
        _,data,_=rollout(actor,env,np.random.default_rng(11),.5,True);cls.batch=batch_episodes([data]*4)
    def create(self,method):
        torch.manual_seed(11);return TunedLearner(self.state_dim,method,2,self.prior)
    def test_fit_prior_is_from_given_fit_records(self):
        expected=sum(c.probes['before'].mean(axis=(0,2,3)) for c in self.cases)/3
        self.assertTrue(np.array_equal(expected,self.prior));self.assertTrue(all(c.case['split']=='fit' for c in self.cases))
    def test_d_and_p_have_identical_initial_parameters_and_optimizers(self):
        d=self.create('D');p=self.create('P');self.assertTrue(all(torch.equal(v,p.actor.state_dict()[k]) for k,v in d.actor.state_dict().items()))
        self.assertEqual([g['lr'] for g in d.optimizer.param_groups],[1e-4,1e-3]);self.assertEqual([g['lr'] for g in p.optimizer.param_groups],[1e-4,1e-3])
    def test_disabled_auxiliary_preserves_original_td_update(self):
        torch.manual_seed(11);original=Learner(self.state_dim,'B3');tuned=self.create('B3')
        a=original.train(self.batch,[self.cases[0]]*4);b=tuned.train(self.batch,[self.cases[0]]*4);self.assertEqual(a['td'],b['td'])
        for key,value in original.actor.state_dict().items():
            if not key.startswith('aux.'):self.assertTrue(torch.equal(value,tuned.actor.state_dict()[key]),key)
        self.assertTrue(all(torch.equal(v,tuned.mixer.state_dict()[k]) for k,v in original.mixer.state_dict().items()))
    def test_physics_gradients_reach_shared_representation(self):
        p=self.create('P');result=p.train(self.batch,[self.cases[0]]*4);g=result['shared_gradient_audit']
        self.assertGreater(g['td_shared_norm'],0.);self.assertGreater(g['data_shared_norm'],0.);self.assertGreater(g['physics_shared_norm'],0.)
        self.assertTrue(all(math.isfinite(v) for v in g.values()))

if __name__=='__main__':unittest.main(verbosity=2)
