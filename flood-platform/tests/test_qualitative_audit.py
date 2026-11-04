"""Regression guards for evidence classification, not hydraulic accuracy tests."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

source = Path(__file__).resolve().parents[1] / 'benchmarks' / 'compare_okushiri_qualitative.py'
spec = importlib.util.spec_from_file_location('qualitative_audit', source)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class QualitativeEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.t = np.arange(451) * .05
        self.observed = np.column_stack([self.t, 2*self.t, 3*self.t]) * .001
        self.reference = np.column_stack([self.t, self.observed*100])

    def test_opposite_trends_are_rejected(self):
        up = audit.phase(self.t, self.t, 0, 1, 1e-5)['label']
        down = audit.phase(self.t, -self.t, 0, 1, 1e-5)['label']
        self.assertEqual((up, down), ('rising', 'falling'))
        self.assertEqual(audit.compare(up, down), 'inconsistent')

    def test_flat_and_sub_resolution_are_unknown_not_passed(self):
        for y in [np.zeros(451), self.t*1e-6]:
            result = audit.phase(self.t, y, 0, 1, 1e-5)
            self.assertEqual(result['label'], 'unresolved')
            self.assertEqual(audit.compare(result['label'], result['label']), 'unknown')

    def test_internal_reversal_is_retained(self):
        values = self.t.copy()
        values[10] = 10
        result = audit.phase(self.t, values, 0, 1, 1e-5)
        self.assertEqual(result['label'], 'rising')
        self.assertTrue(result['internal_reversal'])

    def test_last_short_window_is_not_dropped(self):
        rows = audit.classify_windows(self.t, self.observed, self.observed, 1, 1e-5, .25)
        self.assertEqual(len(rows), 69)
        self.assertEqual({r['station'] for r in rows[-3:]}, set(audit.STATIONS))
        for row in rows[-3:]:
            self.assertEqual((row['start_s'], row['end_s']), (22, 22.5))
            self.assertEqual(row['observed']['sample_count'], 11)

    def test_time_shift_and_wrong_units_are_rejected(self):
        self.assertLess(audit.validate_series(self.t, self.observed, self.observed, self.reference), 1e-12)
        with self.assertRaisesRegex(ValueError, 'time shift'):
            audit.validate_series(self.t+.05, self.observed, self.observed, self.reference)
        with self.assertRaisesRegex(ValueError, 'unit/value'):
            audit.validate_series(self.t, self.observed, self.observed*100, self.reference)

    def test_incomplete_and_nonfinite_series_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            audit.validate_series(self.t[:-1], self.observed[:-1], self.observed[:-1], self.reference)
        bad = self.observed.copy(); bad[100, 1] = np.nan
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.validate_series(self.t, bad, self.observed, self.reference)

    def test_tied_peak_is_not_assigned_false_order(self):
        y = np.zeros(451); y[100] = y[110] = 1
        peak = audit.peak_interval(self.t, y, 1e-5, .05)
        self.assertEqual(peak['near_max_times_s'], [5, 5.5])
        order = audit.order(peak['sample_support_s'], [5.2, 5.3])
        self.assertEqual(order, 'unresolved')
        self.assertEqual(audit.compare(order, order), 'unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
