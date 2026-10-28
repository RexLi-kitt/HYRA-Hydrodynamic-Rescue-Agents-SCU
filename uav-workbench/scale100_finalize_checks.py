"""One-shot continuation guards; no training or holdout evaluation is launched."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import scale100_finalize as finalization
import scale100_model_acceptance as acceptance


class FinalizeChecks(unittest.TestCase):
    def test_wait_requires_actual_upstream_liveness(self):
        with tempfile.TemporaryDirectory(prefix='hyra-finalize-test-') as directory:
            state = Path(directory) / 'upstream.json'
            state.write_text(json.dumps(dict(status='waiting', pid=123)), encoding='utf-8')
            with patch.object(finalization, 'UPSTREAM', state), \
                 patch.object(finalization, 'process_identity', return_value=None):
                with self.assertRaisesRegex(ValueError, 'verified live'):
                    finalization.inspect()

    def test_terminal_upstream_does_not_need_a_live_old_pid(self):
        with tempfile.TemporaryDirectory(prefix='hyra-terminal-upstream-') as directory:
            state = Path(directory) / 'upstream.json'
            state.write_text(json.dumps(dict(status='completed', pid=123)), encoding='utf-8')
            with patch.object(finalization, 'UPSTREAM', state), \
                 patch.object(finalization, 'process_identity', return_value=None), \
                 patch.object(finalization, 'bindings', return_value={}):
                self.assertEqual(finalization.inspect()['upstream_status'], 'completed')

    def test_source_change_is_detected_before_dispatch(self):
        with tempfile.TemporaryDirectory(prefix='hyra-finalization-binding-') as directory:
            path = Path(directory) / 'source.py'
            path.write_text('original', encoding='utf-8')
            value = {str(path):finalization.sha(path)}
            finalization.verify(value)
            path.write_text('changed', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                finalization.verify(value)

    def test_model_acceptance_cannot_run_before_release(self):
        with tempfile.TemporaryDirectory(prefix='hyra-acceptance-guard-') as directory:
            output = Path(directory) / 'must-not-create'
            with patch.object(acceptance, 'release_record', side_effect=ValueError('not published')):
                with self.assertRaisesRegex(ValueError, 'not published'):
                    acceptance.execute(output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
