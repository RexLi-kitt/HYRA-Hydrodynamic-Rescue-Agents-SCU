"""Formal-start guards and model selection semantics; no formal training is run."""
import unittest
from copy import deepcopy
from scale100_formal import validate_review,validate_method_decision,selection_key,process_alive,METHODS,os,sources

def fixture():
    return dict(complete=True,completed=12,missing_jobs=[],matched_initial_tensors=True,holdout_policy_evaluated=False,rows=[dict(job=f'{task}-{method}-s11',episodes=256,updates=253,full_trajectory_audits_passed=True,all_development_points=[dict(episode=ep) for ep in (64,128,256)]) for task in ('E1','E2') for method in METHODS])

class FormalChecks(unittest.TestCase):
    def test_incomplete_development_cannot_start_formal(self):
        value=fixture();value['complete']=False;value['completed']=2;value['rows']=value['rows'][:2];value['missing_jobs']=['pending']
        with self.assertRaisesRegex(AssertionError,'incomplete'):validate_review(value)
    def test_wrong_or_duplicate_groups_rejected(self):
        value=fixture();value['rows'][-1]=deepcopy(value['rows'][0])
        with self.assertRaisesRegex(AssertionError,'duplicated'):validate_review(value)
    def test_holdout_tuning_rejected(self):
        value=fixture();value['holdout_policy_evaluated']=True
        with self.assertRaisesRegex(AssertionError,'holdout'):validate_review(value)
    def test_failed_replay_audit_rejected(self):
        value=fixture();value['rows'][0]['full_trajectory_audits_passed']=False
        with self.assertRaisesRegex(AssertionError,'auditing'):validate_review(value)
    def test_rescue_selection_preserves_primary_outcome(self):
        low=dict(episode=256,rescue_fraction=.4,first_coverage_auc=.99,balanced_current_correct_coverage_auc=.99)
        high=dict(episode=512,rescue_fraction=.6,first_coverage_auc=.1,balanced_current_correct_coverage_auc=.1)
        self.assertGreater(selection_key(high,'E2'),selection_key(low,'E2'))
        self.assertGreater(selection_key(low,'E1'),selection_key(high,'E1'))
        tied=dict(high,episode=1024);self.assertGreater(selection_key(high,'E2'),selection_key(tied,'E2'))
    def test_current_process_is_live(self):
        self.assertTrue(process_alive(os.getpid()));self.assertFalse(process_alive(None))
    def test_underfit_method_cannot_be_declared_formal(self):
        with self.assertRaisesRegex(AssertionError,'unresolved'):validate_method_decision(dict(status='not_ready_auxiliary_underfit'),1024)
    def test_formal_budget_must_match_reviewed_decision(self):
        decision=dict(status='ready_for_formal',episodes_per_job=512,selected_training_backend='absolute_head_v1',training_sources=sources())
        with self.assertRaisesRegex(AssertionError,'budget differs'):validate_method_decision(decision,1024)

if __name__=='__main__':unittest.main(verbosity=2)
