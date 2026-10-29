"""Publication gates and plotting-runtime smoke; no holdout policy scoring."""
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import scale100_publish as publication
import scale100_report as report


class DeliveryChecks(unittest.TestCase):
    def test_training_gate_prevents_any_publication(self):
        with tempfile.TemporaryDirectory(prefix='hyra-publication-test-') as directory:
            path = Path(directory)
            (path / 'queue.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
            release = path / 'must-not-publish.json'
            with patch.object(publication, 'FORMAL', path), patch.object(publication.contract, 'RELEASE', release):
                with self.assertRaisesRegex(AssertionError, 'All36'):
                    publication.publish()
            self.assertFalse(release.exists())

    def test_evaluation_gate_prevents_reading_release_proofs(self):
        with tempfile.TemporaryDirectory(prefix='hyra-incomplete-evaluation-') as directory:
            path = Path(directory)
            (path / 'queue.json').write_text(json.dumps(dict(status='completed')), encoding='utf-8')
            (path / 'status.json').write_text(json.dumps(dict(status='running')), encoding='utf-8')
            with patch.object(publication, 'FORMAL', path), patch.object(publication, 'EVALUATION', path):
                with self.assertRaisesRegex(AssertionError, 'All252'):
                    publication.validated_bundle()

    def test_report_gate_creates_no_plausible_partial_report(self):
        with tempfile.TemporaryDirectory(prefix='hyra-report-guard-') as directory:
            output = Path(directory) / 'must-not-create'
            with patch.object(report, 'validated_bundle', side_effect=AssertionError('incomplete study')):
                with self.assertRaisesRegex(AssertionError, 'incomplete study'):
                    report.execute(output, Path(directory) / 'missing-forecast.json')
            self.assertFalse(output.exists())

    def test_small_nonzero_gradients_are_not_printed_as_zero(self):
        self.assertGreater(float(report.scientific(1.2e-8)), 0)
        self.assertEqual(report.scientific(None), '—')

    def test_matplotlib_runtime_renders_png_and_svg(self):
        # Explicitly synthetic plotting fixture, isolated from result artifacts.
        parents, training = [], []
        for task in ('E1', 'E2'):
            for method in report.METHODS:
                for index in range(2):
                    parents.append(dict(task=task, method=method, parent=f'test-{index}',
                                        first_coverage_auc=.3 + index*.1, rescue_fraction=.4 + index*.2))
            for method in report.METHODS[:6]:
                for seed in (11, 29, 47):
                    training.append(dict(task=task, method=method, seed=seed,
                                         returns=[index/1024 for index in range(1024)]))
        with tempfile.TemporaryDirectory(prefix='hyra-figure-smoke-') as directory:
            path = Path(directory)
            report.figures(dict(parent_tables=parents), training, path)
            from PIL import Image
            for name in ('primary-results', 'training-curves'):
                with Image.open(path / (name + '.png')) as image:
                    self.assertGreaterEqual(image.width, 2000)
                    self.assertGreaterEqual(image.height, 700)
                self.assertIn('<svg', (path / (name + '.svg')).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
