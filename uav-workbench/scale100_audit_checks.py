"""Falsification checks for the independent run audit (not actor training tests)."""
import json,unittest
from pathlib import Path
from scale100_run_audit import *
from scale100_rules import water_aware
ROOT=Path(__file__).resolve().parent

class AuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows=json.loads((ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json').read_text())['cases'];case=FloodCase(next(r['path'] for r in rows if r['split']=='development'))
        cls.env=ScaleEnv(case,config('E2','B3',11));v=cls.env.reset();images=[v.images.copy()];done=False
        while not done:
            v,_,done=cls.env.step(water_aware(cls.env));images.append(v.images.copy())
        cls.images=np.asarray(images)
    def test_valid_e2_dwell_and_observations(self):
        result=audit(self.env,self.images);self.assertGreater(result['rescued'],0);self.assertEqual(result['rescued'],self.env.metrics()['rescued']);self.assertEqual(result['change_cell_events'],result['post_change_observation_events']+result['changes_superseded_without_observation']+result['changes_right_censored'])
    def test_rejects_future_record(self):
        row=self.env.observations[0];value=row['observed_s'];row['observed_s']=5.
        try:
            with self.assertRaisesRegex(AssertionError,'Future'):audit(self.env,self.images)
        finally:row['observed_s']=value
    def test_rejects_unmeasured_actor_water(self):
        bad=self.images.copy();bad[0,0,9,0,0]+=1.
        with self.assertRaisesRegex(AssertionError,'Actor water'):audit(self.env,bad)
    def test_rejects_hidden_personnel(self):
        snap=self.env.snapshots[0];original=snap['targets'];known={(p['row'],p['col']) for p in original};hidden=next(p for p in self.env._targets if p not in known)
        snap['targets']=[*original,dict(row=hidden[0],col=hidden[1],status='discovered')]
        try:
            with self.assertRaisesRegex(AssertionError,'Hidden target'):audit(self.env,self.images)
        finally:snap['targets']=original
    def test_rejects_teleport(self):
        pose=self.env.snapshots[1]['poses'][0];old=pose['x_m'];pose['x_m']+=50.
        try:
            with self.assertRaisesRegex(AssertionError,'Motion exceeds'):audit(self.env,self.images)
        finally:pose['x_m']=old
    def test_rejects_premature_rescue(self):
        snap,target=next((s,t) for s in self.env.snapshots for t in s['targets'] if t['status']=='discovered')
        old=target['status'];target['status']='rescued'
        try:
            with self.assertRaisesRegex(AssertionError,'full dwell|Rescue dwell|service window'):audit(self.env,self.images)
        finally:target['status']=old

if __name__=='__main__':unittest.main(verbosity=2)
