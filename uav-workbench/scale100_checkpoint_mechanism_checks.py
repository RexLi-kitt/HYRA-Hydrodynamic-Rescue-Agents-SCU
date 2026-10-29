"""Validate a late-checkpoint path probe, including an intentionally cut path."""
import unittest
from scale100_checkpoint_mechanism import *


class MechanismChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        plan = read(FORMAL / 'plan.json')
        entry = next(r for r in plan['inputs'] if r['split'] == 'fit')
        cls.case = FloodCase(entry['path'])
        cls.weights = ROOT / 'generated/scale100-aux-resilient-20261026/E1-P-s11/actor.pt'

    def actor(self):
        actor = make_actor('aux_resilient_v4')
        actor.load_state_dict(torch.load(self.weights, weights_only=True, map_location='cpu'))
        return actor.eval()

    def state(self, actor):
        env = ScaleEnv(self.case, config('E1', 'P', 11))
        view = env.reset()
        return probe_state(actor, self.case, view, torch.zeros(2,8), actor.hidden(2), 0)[0]

    def test_healthy_saved_prototype_has_both_gradient_paths(self):
        value = self.state(self.actor())
        self.assertGreater(value['data_to_latent_norm'], 0)
        self.assertGreater(value['physics_to_latent_norm'], 0)
        self.assertGreater(value['data_to_shared_norm'], 0)
        self.assertGreater(value['physics_to_shared_norm'], 0)
        self.assertFalse(value['physics_nonzero_loss_zero_shared_gradient'])

    def test_cut_latent_path_is_detected_despite_nonzero_losses(self):
        actor = self.actor()
        with torch.no_grad():
            actor.aux[0].weight[:, :64].zero_()
        value = self.state(actor)
        self.assertGreater(value['data_loss'], 0)
        self.assertGreater(value['physics_loss'], 0)
        self.assertEqual(value['data_to_latent_norm'], 0)
        self.assertEqual(value['physics_to_shared_norm'], 0)
        self.assertTrue(value['data_nonzero_loss_zero_shared_gradient'])
        self.assertTrue(value['physics_nonzero_loss_zero_shared_gradient'])

    def test_full_fit_reference_does_not_update_or_accumulate_gradients(self):
        actor = self.actor()
        before = fingerprint(actor)
        rows = checkpoint_probe(actor, self.case, 'E1')
        self.assertEqual([r['time_s'] for r in rows], [0., 300., 600., 900.])
        self.assertEqual(fingerprint(actor), before)
        self.assertTrue(all(p.grad is None for p in actor.parameters()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
