import copy
import unittest
import numpy as np
from flood_solver import FloodSolver
from benchmark_restart import restore_audit_metadata


class RestartTests(unittest.TestCase):
    def sim(self):
        a = np.ones((4, 4))
        s = FloodSolver(a*0, a, 1, boundary='periodic', manning=.3, max_timestep=.01)
        s.mx[:] = 2
        s.set_external_inflows([
            dict(id='a', enabled=True, cells=[dict(z=1,x=1,weight=1.)], hydrograph=[[0,.1],[.1,0]]),
            dict(id='b', enabled=True, cells=[dict(z=2,x=2,weight=1.)], hydrograph=[[0,.2],[.15,0]])])
        return s

    def test_restart_preserves_fields_counters_and_earlier_peak(self):
        continuous = self.sim(); continuous.advance(.1)
        frame = continuous.summary()
        resumed = self.sim()
        for name in ('h','mx','mz','maximum'):
            getattr(resumed,name)[:] = getattr(continuous,name)
        restore_audit_metadata(resumed, frame)
        continuous.advance(.2); resumed.advance(.2)
        for name in ('h','mx','mz','maximum'):
            np.testing.assert_array_equal(getattr(resumed,name),getattr(continuous,name))
        self.assertEqual(resumed.summary()['balance'],continuous.summary()['balance'])
        self.assertEqual(resumed.external_volume_by_id,continuous.external_volume_by_id)
        self.assertAlmostEqual(resumed.external_volume_by_id['a'],.01)
        self.assertAlmostEqual(resumed.external_volume_by_id['b'],.03)
        self.assertEqual(resumed.peak_speed_m_s,2.)
        self.assertEqual(resumed.peak_speed_time_s,0.)
        frame['external_inflow_by_id_m3']['a'] = 999
        self.assertNotEqual(resumed.external_volume_by_id['a'],999)

    def test_missing_or_invalid_metadata_rejected_without_partial_mutation(self):
        original = self.sim(); original.advance(.1)
        for key,value in [('peak_speed_m_s',None),('external_inflow_by_id_m3',None),
                          ('peak_speed_time_s',999),('external_inflow_by_id_m3',{})]:
            frame = copy.deepcopy(original.summary())
            if value is None:del frame[key]
            else:frame[key]=value
            target=self.sim(); before=target.summary()
            with self.assertRaises(ValueError):restore_audit_metadata(target,frame)
            self.assertEqual(before,target.summary())
