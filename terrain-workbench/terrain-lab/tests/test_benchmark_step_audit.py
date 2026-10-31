import sys
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from benchmark_step_audit import AcceptedStepAudit
from benchmark_restart import restore_audit_metadata
from okushiri_boundary import OkushiriBoundarySolver


class AcceptedStepAuditTests(unittest.TestCase):
    def make(self):
        return OkushiriBoundarySolver(np.full((4,20),-.135),np.full((4,20),.135),.1,
            wave=[[0,0],[.1,.005],[.2,-.003],[1,0]],manning=0,max_timestep=.005)

    def test_duplicate_output_callbacks_do_not_add_steps(self):
        s=SimpleNamespace(steps=0,time=0.,last_timestep={})
        dt=[];record=AcceptedStepAudit(s,dt);record(s)
        s.steps=1;s.time=.01;s.last_timestep={'dt_s':.01}
        record(s);record(s);record(s)
        self.assertEqual(dt,[.01])

    def test_fix_preserves_wave_state_bitwise(self):
        old=self.make();new=self.make();legacy=[];correct=[]
        def previous(s):
            if s.last_timestep:legacy.append(s.last_timestep['dt_s'])
        record=AcceptedStepAudit(new,correct)
        for t in np.arange(9)*.05:
            old.advance(float(t),observer=previous);new.advance(float(t),observer=record)
        self.assertGreater(len(legacy),old.steps)
        self.assertGreater(math.fsum(legacy),old.time)
        self.assertEqual(len(correct),new.steps)
        self.assertAlmostEqual(math.fsum(correct),new.time,places=12)
        for name in ('h','mx','mz','maximum'):
            np.testing.assert_array_equal(getattr(old,name),getattr(new,name))
        self.assertEqual(old.summary(),new.summary())

    def test_restored_observer_matches_uninterrupted(self):
        full=self.make();first=self.make();all_dt=[];first_dt=[]
        a=AcceptedStepAudit(full,all_dt);b=AcceptedStepAudit(first,first_dt)
        for t in np.arange(5)*.05:
            full.advance(float(t),observer=a);first.advance(float(t),observer=b)
        resumed=self.make()
        for name in ('h','mx','mz','maximum'):
            setattr(resumed,name,getattr(first,name).copy())
        restore_audit_metadata(resumed,first.summary());resumed._wave_time=resumed.time
        continued=first_dt.copy();c=AcceptedStepAudit(resumed,continued)
        for t in np.arange(4,9)*.05:
            full.advance(float(t),observer=a);resumed.advance(float(t),observer=c)
        self.assertEqual(all_dt,continued)
        for name in ('h','mx','mz','maximum'):
            np.testing.assert_array_equal(getattr(full,name),getattr(resumed,name))
        self.assertEqual(full.summary(),resumed.summary())

    def test_missed_step_rejected(self):
        s=SimpleNamespace(steps=0,time=0.,last_timestep={});r=AcceptedStepAudit(s,[])
        s.steps=2;s.time=.02;s.last_timestep={'dt_s':.01}
        with self.assertRaisesRegex(ValueError,'Missed'):r(s)

    def test_duplicate_or_wrong_duration_history_rejected(self):
        s=SimpleNamespace(steps=2,time=.02)
        with self.assertRaisesRegex(ValueError,'length'):AcceptedStepAudit(s,[.01,.01,.01])
        with self.assertRaisesRegex(ValueError,'duration'):AcceptedStepAudit(s,[.02,.01])

    def test_nonfinite_or_nonpositive_dt_rejected(self):
        for dt in [float('nan'),float('inf'),0.,-.1]:
            s=SimpleNamespace(steps=0,time=0.,last_timestep={});r=AcceptedStepAudit(s,[])
            s.steps=1;s.time=.1;s.last_timestep={'dt_s':dt}
            with self.assertRaisesRegex(ValueError,'Invalid'):r(s)


if __name__=='__main__':unittest.main(verbosity=2)
