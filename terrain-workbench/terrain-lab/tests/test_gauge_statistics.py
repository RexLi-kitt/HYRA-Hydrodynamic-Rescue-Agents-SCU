import unittest
from gauge_statistics import summarize_gauge_window

def row(t,v,status='wet',x=0):
    return dict(time_s=t,x_m=x,z_m=0,origin_center_m=[0,0],method='containing_cell',status=status,speed_m_s=v)

class GaugeWindowTests(unittest.TestCase):
    def test_window_excludes_larger_outside_peak_and_keeps_first_tie(self):
        r=summarize_gauge_window([row(0,99),row(1,3),row(2,3),row(3,50)],start_s=1,end_s=2)
        self.assertEqual(r['sampled_peak_speed_m_s'],3);self.assertEqual(r['sampled_peak_time_s'],1)
        self.assertTrue(r['endpoint_samples_present']);self.assertEqual(r['max_sample_gap_s'],1)
    def test_missing_is_explicit_and_not_zero(self):
        r=summarize_gauge_window([row(0,None,'dry'),row(1,0),row(2,None,'solid')],start_s=0,end_s=2)
        self.assertEqual(r['missing_sample_count'],2);self.assertEqual(r['wet_sample_count'],1)
        self.assertEqual(r['sampled_peak_speed_m_s'],0)
        r=summarize_gauge_window([row(0,None,'outside')],start_s=1,end_s=2)
        self.assertIsNone(r['sampled_peak_speed_m_s']);self.assertFalse(r['endpoint_samples_present'])
    def test_invalid_series_rejected(self):
        for rows in ([row(1,1),row(1,2)],[row(2,1),row(1,2)],[row(0,1),row(1,2,x=1)],[row(0,0,'dry')],[row(0,float('nan'))]):
            with self.assertRaises(ValueError):summarize_gauge_window(rows,start_s=0,end_s=2)
