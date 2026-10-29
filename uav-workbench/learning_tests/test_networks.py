import io
import unittest

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import torch

from hyra_learning.networks import SearchAgent, make_mixer


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(29)
        self.agent = SearchAgent()
        self.images = torch.randn(6, 16, 7, 7)
        self.poses = torch.randn(6, 6)
        self.previous = torch.zeros(6, 4)
        self.hidden = self.agent.initial_hidden(6)

    def test_sur_is_train_only_and_valid_padding_does_not_change_it(self):
        self.agent.train()
        valid = torch.tensor([True, True, True, True, False, False])
        q, hidden, _, surprise = self.agent(self.images, self.poses, self.previous, 3, self.hidden,
                                           enhance=True, valid=valid)
        truncated = self.agent(self.images[:4], self.poses[:4], self.previous[:4], 3, self.hidden[:4], enhance=True)
        torch.testing.assert_close(surprise[:4], truncated[-1])
        torch.testing.assert_close(q[:4], truncated[0])
        self.assertTrue(torch.equal(surprise[4:], torch.zeros(2)))
        self.assertEqual(q.shape, (6, 4))
        self.assertTrue(torch.all(surprise >= 0) and torch.all(surprise <= 1))
        q.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.agent.parameters() if p.grad is not None))
        self.agent.eval()
        with self.assertRaises(RuntimeError):
            self.agent(self.images, self.poses, self.previous, 3, self.hidden, enhance=True)

    def test_virtual_embedding_gradient_matches_finite_difference(self):
        latent = self.agent.encode(self.images[:1], self.poses[:1]).detach().requires_grad_(True)
        # Use one fixed selected action to avoid crossing a max-action tie.
        q, _ = self.agent.decode(latent, self.previous[:1], 2, self.hidden[:1])
        action = int(q.argmax())
        gradient = torch.autograd.grad(q[0, action], latent)[0]
        coordinate = int(gradient.abs().argmax())
        epsilon = .001
        plus, minus = latent.detach().clone(), latent.detach().clone()
        plus[0, coordinate] += epsilon
        minus[0, coordinate] -= epsilon
        value_plus = self.agent.decode(plus, self.previous[:1], 2, self.hidden[:1])[0][0, action]
        value_minus = self.agent.decode(minus, self.previous[:1], 2, self.hidden[:1])[0][0, action]
        numeric = (value_plus-value_minus)/(2*epsilon)
        torch.testing.assert_close(numeric, gradient[0, coordinate], atol=2e-5, rtol=.01)

    def test_clean_actor_roundtrip_and_virtual_forward_does_not_advance_hidden_twice(self):
        self.agent.eval()
        with torch.no_grad():
            expected = self.agent(self.images, self.poses, self.previous, 1, self.hidden)
        stream = io.BytesIO()
        torch.save(self.agent.state_dict(), stream)
        stream.seek(0)
        restored = SearchAgent()
        restored.load_state_dict(torch.load(stream, weights_only=True))
        restored.eval()
        with torch.no_grad():
            actual = restored(self.images, self.poses, self.previous, 1, self.hidden)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right)
        self.agent.train()
        _, state, latent, surprise = self.agent(self.images, self.poses, self.previous, 1, self.hidden, enhance=True)
        _, single = self.agent.decode(latent+surprise[:, None]*self.agent.direction, self.previous, 1, self.hidden)
        torch.testing.assert_close(state, single)

    def test_mixer_is_monotone_and_auxiliary_gradient_can_reach_actor(self):
        mixer = make_mixer(2, 20)
        q = torch.randn(3, 4, 2, requires_grad=True)
        states = torch.randn(3, 4, 20)
        gradient = torch.autograd.grad(mixer(q, states).sum(), q)[0]
        self.assertTrue(torch.all(gradient >= -1e-8))
        _, _, latent, _ = self.agent(self.images, self.poses, self.previous, 0, self.hidden)
        latent.square().mean().backward()
        self.assertGreater(float(self.agent.image_embed.weight.grad.abs().sum()), 0)

    def test_cuda_forward_and_backward_if_available(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA not available')
        agent = self.agent.cuda().train()
        output = agent(self.images.cuda(), self.poses.cuda(), self.previous.cuda(), 0,
                       self.hidden.cuda(), enhance=True)
        output[0].square().mean().backward()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(output[0]).all())


if __name__ == '__main__':
    unittest.main()
