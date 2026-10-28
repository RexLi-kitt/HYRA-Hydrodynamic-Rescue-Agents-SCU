import io
import json
from pathlib import Path
import unittest

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch

from hyra_learning.training import EpisodeReplay, Learner, memory_usage


def episode(length, generator):
    n = 2
    return dict(images=torch.randn(length+1, n, 16, 7, 7, generator=generator),
                poses=torch.randn(length+1, n, 6, generator=generator),
                states=torch.randn(length+1, 20, generator=generator),
                actions=torch.randint(4, (length, n), generator=generator),
                rewards=torch.full((length,), .01),
                terminal=torch.tensor([0.]*(length-1)+[1.]))


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.plan = json.loads((Path(__file__).resolve().parents[1]/'learning_configs/sur_preflight_20261025.json').read_text(encoding='utf-8'))

    def test_optimizer_replay_and_rng_resume_produce_same_next_update(self):
        for surprise in (False, True):
            with self.subTest(surprise=surprise):
                learner = Learner(2, 20, self.plan, 'cpu', surprise)
                replay, rng = EpisodeReplay(8), np.random.default_rng(17)
                generator = torch.Generator().manual_seed(31)
                for n in (2, 3, 4, 2):
                    replay.add(episode(n, generator))
                learner.train_batch(replay.sample(3, rng, 'cpu'))
                stream = io.BytesIO()
                torch.save(learner.checkpoint(replay, rng, 1, []), stream)
                reference_batch = replay.sample(3, rng, 'cpu')
                reference_stats = learner.train_batch(reference_batch)
                reference_rng = rng.integers(0, 100000, 8)
                reference_torch = torch.rand(8)
                stream.seek(0)
                restored = Learner(2, 20, self.plan, 'cpu', surprise)
                restored_replay, restored_rng = EpisodeReplay(8), np.random.default_rng(999)
                restored.restore(torch.load(stream, weights_only=True), restored_replay, restored_rng)
                next_batch = restored_replay.sample(3, restored_rng, 'cpu')
                for key in reference_batch:
                    torch.testing.assert_close(reference_batch[key], next_batch[key], atol=0, rtol=0)
                actual_stats = restored.train_batch(next_batch)
                self.assertEqual(reference_stats, actual_stats)
                for left, right in zip(learner.parameters, restored.parameters):
                    torch.testing.assert_close(left, right, atol=0, rtol=0)
                np.testing.assert_array_equal(reference_rng, restored_rng.integers(0, 100000, 8))
                torch.testing.assert_close(reference_torch, torch.rand(8), atol=0, rtol=0)

    def test_padding_cannot_enter_td_or_sur_normalization(self):
        replay, rng = EpisodeReplay(4), np.random.default_rng(19)
        generator = torch.Generator().manual_seed(37)
        replay.add(episode(2, generator))
        replay.add(episode(4, generator))
        batch = replay.sample(2, rng, 'cpu')
        learner = Learner(2, 20, self.plan, 'cpu', True)
        # Put extreme finite garbage only in padding. Registered finite horizon
        # masks the last valid transition, so padded next states have no value.
        changed = {k: v.clone() for k, v in batch.items()}
        for b, length in enumerate(batch['lengths']):
            for key in ('images', 'poses', 'states'):
                changed[key][b, int(length)+1:] = 100
            changed['rewards'][b, int(length):] = 100
        import copy
        another = copy.deepcopy(learner)
        left, right = learner.train_batch(batch), another.train_batch(changed)
        self.assertEqual(left, right)
        for a, b in zip(learner.parameters, another.parameters):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_process_native_memory_is_measured(self):
        result = memory_usage()
        self.assertGreater(result['process_peak_working_set_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
