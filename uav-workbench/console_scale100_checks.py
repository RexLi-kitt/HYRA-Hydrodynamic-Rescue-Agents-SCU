"""Contract rejection checks; full development rollouts are a separate CLI."""
import json
import subprocess
import sys
import unittest
from unittest.mock import patch
import console_scale100_contract as contract


class ContractChecks(unittest.TestCase):
    def payload(self, task='exploration', group='B3'):
        return dict(scene=next(s['id'] for s in contract.scene_records()
                               if s['split'] == 'development'),
                    task=task, group=group, mode='rule')

    def test_server_import_never_loads_learning_runtime(self):
        code = ('import sys; import console_scale100_contract; '
                'assert not any(n in sys.modules for n in '
                '("torch", "numpy", "scale100_train", "scale100_backend"))')
        subprocess.run([sys.executable, '-B', '-c', code], cwd=contract.ROOT, check=True)

    def test_holdout_is_absent_and_cannot_be_selected(self):
        self.assertEqual(len(contract.scene_records()), 4)
        self.assertTrue(all(s['split'] != 'holdout' for s in contract.scene_records()))
        data = self.payload()
        data['scene'] = 'scale100:procedural-parent-503'
        with self.assertRaises(ValueError):
            contract.validate(data)

    def test_rule_has_no_model_attribution_for_active_choices(self):
        for task in contract.TASKS:
            for group in contract.ACTIVE_GROUPS:
                data = self.payload(task, group)
                data['config'] = dict(seed=47)
                result = contract.validate(data)
                self.assertIsNone(result['group'])
                self.assertEqual(result['requested_group'], group)
                self.assertEqual(result['config']['seed'], 47)
                self.assertEqual(result['training_seed'], 11)
                self.assertEqual(result['config']['n_targets'], 0 if task == 'exploration' else 5)

    def test_catalog_and_requests_only_allow_three_active_groups(self):
        self.assertEqual([g['id'] for g in contract.catalog()['groups']], ['B3', 'D', 'P'])
        for group in ('B0', 'B1', 'B2'):
            with self.subTest(group=group), self.assertRaises(ValueError):
                contract.validate(self.payload(group=group))

    def test_wrong_environment_is_rejected(self):
        for key, value in dict(duration_s=250, step_s=10, sense_width=7,
                               service_s=5, obstacle_fraction=0, n_agents=4,
                               service_depth_limit_m=.1, n_targets=5).items():
            data = self.payload()
            data['config'] = {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                contract.validate(data)

    def test_unknown_or_boolean_arguments_are_rejected(self):
        for changes in (dict(config=dict(seed=True)), dict(config=dict(step_s=True)),
                        dict(training_seed=True), dict(training_seed=17),
                        dict(config=dict(typo=5)), dict(checkpoint='legacy'),
                        dict(task='E1'), dict(group='QMIX8')):
            data = self.payload()
            data.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                contract.validate(data)

    def test_unreleased_models_cannot_run(self):
        for task in contract.TASKS:
            for group in contract.ACTIVE_GROUPS:
                data = self.payload(task, group)
                data['mode'] = 'model'
                # Independent of whether a release is published in a later run.
                with patch.object(contract, 'release_record', side_effect=ValueError('unreleased')):
                    with self.assertRaisesRegex(ValueError, 'unreleased'):
                        contract.validate(data)

    def test_scene_tampering_is_rejected(self):
        with patch.object(contract, 'digest', return_value='wrong-source'):
            with self.assertRaisesRegex(ValueError, '摘要'):
                contract.validate(self.payload())


if __name__ == '__main__':
    unittest.main(verbosity=2)
