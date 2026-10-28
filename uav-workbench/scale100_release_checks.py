"""Exercise publication audit using development scenes, never holdout scoring."""
import copy
import tempfile
import unittest
from scale100_release_audit import *
from scale100_forecast_audit import DATA


class ReleaseChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        entry = next(r for r in read(DATA)['cases'] if r['split'] == 'development')
        cls.case = FloodCase(entry['path'])
        cls.temp = tempfile.TemporaryDirectory(prefix='hyra-release-development-')
        cls.path = Path(cls.temp.name)
        from scale100_evaluate_sealed import run_trial
        cls.rule_row = run_trial(cls.path, 'dev-rule', cls.case, 'E2', 'water_aware', 11,
                                split='development_smoke')
        with gzip.open(cls.path / cls.rule_row['replay_path'], 'rt', encoding='utf-8') as stream:
            cls.rule_replay = json.load(stream)
        model = Path(__file__).resolve().parent / 'generated/scale100-aux-resilient-20261026/E2-D-s11'
        cls.actor = make_actor('aux_resilient_v4')
        cls.actor.load_state_dict(torch.load(model / 'actor.pt', weights_only=True, map_location='cpu'))
        cls.actor.eval()
        cls.actor_row = run_trial(cls.path, 'dev-actor', cls.case, 'E2', 'D', 11,
                                 cls.actor, digest(model / 'actor.pt'), 11,
                                 split='development_smoke')
        with gzip.open(cls.path / cls.actor_row['replay_path'], 'rt', encoding='utf-8') as stream:
            cls.actor_replay = json.load(stream)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_complete_rule_and_actor_reproduction(self):
        for payload, row, actor in ((self.rule_replay, self.rule_row, None),
                                     (self.actor_replay, self.actor_row, self.actor)):
            proof = verify_replay(payload, row, self.case, actor)
            self.assertEqual(proof['records_reproduced'], 362)
            self.assertTrue(proof['actor_features_checked'])

    def test_wrong_model_same_shape_is_rejected(self):
        other = make_actor('aux_resilient_v4')
        path = Path(__file__).resolve().parent / 'generated/scale100-aux-resilient-20261026/E2-P-s11/actor.pt'
        other.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        with self.assertRaisesRegex(AssertionError, 'policy'):
            verify_replay(self.actor_replay, self.actor_row, self.case, other)

    def test_changed_action_is_rejected(self):
        payload = copy.deepcopy(self.rule_replay)
        value = payload['actions'][0]['actions'][0]
        payload['actions'][0]['actions'][0] = ACTIONS[(ACTIONS.index(value) + 1) % 8]
        with self.assertRaisesRegex(AssertionError, 'policy'):
            verify_replay(payload, self.rule_row, self.case)

    def test_changed_measurement_is_rejected(self):
        payload = copy.deepcopy(self.rule_replay)
        payload['observations'][0]['depth_m'][0] += 1
        with self.assertRaisesRegex(AssertionError, 'observations'):
            verify_replay(payload, self.rule_row, self.case)

    def test_wrong_attribution_is_rejected(self):
        payload = copy.deepcopy(self.actor_replay)
        payload['training_seed'] = 29
        with self.assertRaisesRegex(AssertionError, 'attribution'):
            verify_replay(payload, self.actor_row, self.case, self.actor)

    def test_release_cannot_read_holdout_while_training(self):
        with tempfile.TemporaryDirectory(prefix='hyra-incomplete-training-') as directory:
            path = Path(directory)
            (path / 'queue.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
            with self.assertRaisesRegex(AssertionError, 'All36'):
                execute(path, path / 'must-not-open')

    def test_actual_completed_formal_training_integrity(self):
        formal = Path(__file__).resolve().parent / 'generated/scale100-formal-20261026'
        plan = read(formal / 'plan.json')
        # First three completed jobs; final production entry audits the full36.
        plan['jobs'] = [j for j in plan['jobs']
                        if j['id'] in ('E1-B0-s11', 'E2-B0-s11', 'E1-B1-s11')]
        rows = training_integrity(formal, plan)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['episodes'] == 1024 and r['updates'] == 1021 for r in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
