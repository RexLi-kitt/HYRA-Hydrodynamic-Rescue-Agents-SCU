"""Revision identity and readiness checks using actual development artifacts."""
import unittest,json
from pathlib import Path
from copy import deepcopy
from scale100_backend import *
from scale100_formal import validate_method_decision,validate_aux_review

ROOT=Path(__file__).parent
class BackendChecks(unittest.TestCase):
    def test_same_tensor_shape_cannot_hide_wrong_training_revision(self):
        old=json.loads((ROOT/'generated/scale100-budget-pilot-20261026/E1-D-s11/actor.json').read_text());new=json.loads((ROOT/'generated/scale100-aux-tuning-20261026/E1-D-s11/actor.json').read_text())
        validate_meta(old,'absolute_head_v1');validate_meta(new,'aux_optimization_v2')
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_meta(old,'aux_optimization_v2')
        with self.assertRaisesRegex(ValueError,'mismatch'):validate_meta(new,'absolute_head_v1')
    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unregistered'):source_hashes('other')
    def test_tuned_decision_requires_matching_source_fingerprint(self):
        ready=dict(status='ready_for_formal',episodes_per_job=1024,selected_training_backend='aux_optimization_v2',training_sources=source_hashes('aux_optimization_v2'));validate_method_decision(ready,1024)
        ready['training_sources']=source_hashes('absolute_head_v1')
        with self.assertRaisesRegex(AssertionError,'different training sources'):validate_method_decision(ready,1024)
    def test_partial_auxiliary_review_cannot_promote_method(self):
        value=dict(complete=False,completed=1,missing_jobs=['E1-P-s11'])
        with self.assertRaisesRegex(AssertionError,'incomplete'):validate_aux_review(value)
    def test_both_task_forecast_diagnostics_are_required(self):
        rows=[dict(job=f'{task}-{method}-s11',episodes=256,updates=253,full_trajectory_audits_passed=True,shared_auxiliary_mechanism_live_at_end=True) for task in ('E1','E2') for method in ('D','P')]
        value=dict(complete=True,completed=4,missing_jobs=[],rows=rows,matched_initial_tensors=True,holdout_policy_evaluated=False,matched_forecast_paths={'E1':'fixture_only'})
        with self.assertRaisesRegex(AssertionError,'Both task'):validate_aux_review(value)
    def test_gradient_collapse_blocks_formal_even_if_trajectories_pass(self):
        rows=[dict(job=f'{task}-{method}-s11',episodes=256,updates=253,full_trajectory_audits_passed=True,shared_auxiliary_mechanism_live_at_end=True) for task in ('E1','E2') for method in ('D','P')];rows[1]['shared_auxiliary_mechanism_live_at_end']=False
        value=dict(complete=True,completed=4,missing_jobs=[],rows=rows,matched_initial_tensors=True,holdout_policy_evaluated=False,matched_forecast_paths={'E1':'fixture','E2':'fixture'})
        with self.assertRaisesRegex(AssertionError,'collapsed'):validate_aux_review(value)
    def test_parameterless_decoder_activation_is_restored_by_backend(self):
        actor=make_actor('aux_resilient_v4');self.assertIsInstance(actor.aux[1],base.nn.LeakyReLU);self.assertEqual(actor.aux[1].negative_slope,.05)

if __name__=='__main__':unittest.main(verbosity=2)
