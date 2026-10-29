"""Forecast integration and attribution checks, development data only."""
import tempfile
import unittest
from unittest.mock import patch
import scale100_formal_forecasts as diagnostic
from scale100_formal_forecasts import *


class FormalForecastChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        plan = read(FORMAL / 'plan.json')
        entry = next(r for r in plan['inputs'] if r['split'] == 'development')
        cls.case = FloodCase(entry['path'])
        cls.prior = fit_prior(plan)
        cls.actors = {}
        prototype = ROOT / 'generated/scale100-aux-resilient-20261026'
        for method in ('D', 'P'):
            actor = make_actor('aux_resilient_v4')
            actor.load_state_dict(torch.load(prototype / f'E1-{method}-s11/actor.pt', weights_only=True))
            actor.eval()
            cls.actors[(method, 11)] = actor
        cls.points, cls.stream = reference(cls.case, 'E1', 11, cls.actors, cls.prior)
        cls.previous = read(prototype / 'E1-matched-forecast.json')

    def test_full_reference_matches_original_development_diagnostic(self):
        self.assertEqual(len(self.points), 95)
        for row in self.points:
            old = next(r for r in self.previous['rows'] if r['seed'] == 11
                       and r['time_s'] == row['time_s'] and r['method'] == row['method'])
            for key in row:
                if key not in ATTRIBUTION:
                    self.assertEqual(row[key], old[key])
        self.assertTrue(all(p.grad is None for actor in self.actors.values() for p in actor.parameters()))

    def test_actor_iteration_order_does_not_change_reference_inputs(self):
        points, stream = reference(self.case, 'E1', 11, dict(reversed(list(self.actors.items()))), self.prior)
        self.assertEqual(stream, self.stream)
        identity = lambda r: (r['method'], r['time_s'])
        self.assertEqual(sorted(points, key=identity), sorted(self.points, key=identity))

    def test_baselines_do_not_duplicate_training_seed_samples(self):
        for method in BASELINES:
            rows = [r for r in self.points if r['method'] == method]
            self.assertEqual(len(rows), 19)
            self.assertTrue(all(r['training_seed'] is None for r in rows))
        for method in ('D', 'P'):
            self.assertTrue(all(r['training_seed'] == 11 for r in self.points if r['method'] == method))

    def test_parent_equal_weighting_and_paired_differences(self):
        rows = []
        for parent, count, difference in (('A', 3, .1), ('B', 1, .5)):
            for seed in range(count):
                for method in ('D', 'P'):
                    rows.append(dict(task='E1', parent=parent, scenario_seed=seed,
                        training_seed=11, method=method, time_s=0., actual_microstep_s=.5,
                        error=.2 + (difference if method == 'P' else 0.)))
        result = aggregate(rows)
        self.assertAlmostEqual(result['contrasts'][0]['means']['error'], .3)
        missing = rows[:-1]
        with self.assertRaisesRegex(AssertionError, 'missing'):
            aggregate(missing)

    def test_no_holdout_before_policy_audit(self):
        with tempfile.TemporaryDirectory(prefix='hyra-forecast-guard-') as directory:
            path = Path(directory)
            (path / 'queue.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
            with patch.object(diagnostic, 'FORMAL', path):
                with self.assertRaisesRegex(AssertionError, 'All36'):
                    execute(path / 'must-not-create.json')
                self.assertFalse((path / 'must-not-create.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
