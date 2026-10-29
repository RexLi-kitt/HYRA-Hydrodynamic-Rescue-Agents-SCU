"""Path relocation must also work in GitHub's repeated repository directory."""
from pathlib import Path
import unittest
from unittest.mock import patch

import runtime_assets as assets


class RuntimePathTests(unittest.TestCase):
    def test_current_checkout_and_historical_paths_use_same_shipped_input(self):
        relative = 'uav-workbench/generated/scale100-formal-20261026/plan.json'
        expected = assets.BUNDLE / relative
        self.assertTrue(expected.is_file())
        for value in (assets.REPO / relative, relative,
                      'D:\\HyRA\\HYRA-Hydrodynamic-Rescue-Agents-SCU\\' + relative.replace('/', '\\'),
                      '/home/runner/work/HYRA-Hydrodynamic-Rescue-Agents-SCU/'
                      'HYRA-Hydrodynamic-Rescue-Agents-SCU/' + relative):
            with self.subTest(path=value):
                self.assertEqual(assets.runtime_path(value), expected)

    def test_repeated_current_repo_name_is_removed_as_a_whole(self):
        repo = Path('work/HYRA-Hydrodynamic-Rescue-Agents-SCU/HYRA-Hydrodynamic-Rescue-Agents-SCU').resolve()
        relative = 'uav-workbench/generated/scale100-formal-20261026/plan.json'
        with patch.object(assets, 'REPO', repo):
            self.assertEqual(assets.runtime_path(repo / relative), assets.BUNDLE / relative)
