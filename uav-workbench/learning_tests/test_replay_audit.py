"""Check independent audit catches corrupted factual replay evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'))
from evaluate_sciml import read,actor_episode,FloodProcess
from audit_team_replays import audit_replay


class ReplayAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=read(ROOT/'generated/sciml-round1/formal-plan.json')
        cls.process=FloodProcess(plan['inputs'][0]['path'])
        _,env=actor_episode(None,cls.process,plan,'R0');cls.replay=env.result()

    def test_independent_reconstruction_matches(self):
        result=audit_replay(self.replay,self.process)
        self.assertEqual((result['frames'],result['observations'],len(result['views'])),(121,242,61))

    def test_rejects_false_observation_timestamp(self):
        replay=deepcopy(self.replay);replay['observations'][8]['observed_s']+=5
        with self.assertRaises(AssertionError):audit_replay(replay,self.process)

    def test_rejects_memory_refresh_without_observation(self):
        replay=deepcopy(self.replay);replay['snapshots'][3]['observed_s'][1023]=15
        with self.assertRaises(AssertionError):audit_replay(replay,self.process)

    def test_rejects_changed_path_cost_and_censored_event(self):
        replay=deepcopy(self.replay);replay['summary']['distance_m']+=1
        with self.assertRaises(AssertionError):audit_replay(replay,self.process)
        replay=deepcopy(self.replay);self.assertTrue(replay['changes'])
        replay['changes'].pop()
        with self.assertRaises(AssertionError):audit_replay(replay,self.process)


if __name__=='__main__':unittest.main()
