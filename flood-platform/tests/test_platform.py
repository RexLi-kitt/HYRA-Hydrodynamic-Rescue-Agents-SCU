from pathlib import Path
import json,sys,tempfile,unittest,types
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
from make_cases import make
from hyra_flood.contracts import load_case,load_run,read_json,write_json,sha,new_output
from hyra_flood.engines import legacy_transform,run,_legacy
from hyra_flood.evaluation import strict_speed_score,inundation_score,compare_runs,gauge_series
from hyra_flood.rescue import FloodFeed

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);make(self.root,'lake');self.path=self.root/'lake/case.json'
    def tearDown(self):self.tmp.cleanup()
    def change(self,fn):
        c=read_json(self.path);fn(c);write_json(self.path,c)
    def test_valid_case_and_coordinate_transform(self):
        c,f,_=load_case(self.path);self.assertEqual(f['depth_m'].shape,(12,40))
        a=np.arange(12.).reshape(3,4);v={'depth_m':a,'qx_m2_s':2*a,'qy_m2_s':3*a}
        first=legacy_transform(v);second=legacy_transform(first)
        for k in v:np.testing.assert_array_equal(v[k],second[k])
        self.assertEqual(first['qy_m2_s'][0,0],-3*a[-1,0])
    def test_cached_wrong_engine_is_rejected(self):
        fake=types.SimpleNamespace(__file__=str(self.root/'wrong/flood_solver.py'),FloodSolver=object)
        with patch.dict(sys.modules,{'flood_solver':fake}),patch.object(sys,'path',list(sys.path)):
            with self.assertRaisesRegex(RuntimeError,'cached'):_legacy({}, {}, self.root/'expected')
    def test_si_units_required(self):
        self.change(lambda c:c['units'].update(length='cm'))
        with self.assertRaises(ValueError):load_case(self.path)
    def test_source_hash_required(self):
        with (self.root/'lake/input.npz').open('ab') as f:f.write(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256'):load_case(self.path)
    def test_unsupported_features_not_dropped(self):
        self.change(lambda c:c.update(rainfall_mm_h=100))
        with self.assertRaisesRegex(ValueError,'unknown'):load_case(self.path)
    def test_unimplemented_boundary_rejected(self):
        self.change(lambda c:c['boundaries']['west'].update(kind='hydrograph'))
        with self.assertRaises(NotImplementedError):load_case(self.path)
    def test_duplicate_or_outside_gauges_rejected(self):
        c=read_json(self.path);self.change(lambda c:c['gauges'].append(c['gauges'][0]))
        with self.assertRaises(ValueError):load_case(self.path)
        c['gauges'][0]['x_m']=104.;write_json(self.path,c)
        with self.assertRaises(ValueError):load_case(self.path)
    def test_escaping_data_path_rejected(self):
        self.change(lambda c:c['arrays'].update(path='../external.npz'))
        with self.assertRaisesRegex(ValueError,'inside'):load_case(self.path)
    def test_no_overwrite_and_no_backend_fallback(self):
        with self.assertRaises(FileExistsError):new_output(self.root)
        with self.assertRaises(NotImplementedError):run(self.path,'local_ns',self.root/'never')
        self.assertFalse((self.root/'never').exists())

class Evaluation(unittest.TestCase):
    def test_zero_measurement_uses_absolute_standard(self):
        score=strict_speed_score(['a','b','c'],[1.,0.,2.],[1.04,.021,2.2])
        self.assertEqual(score['passed'],1);self.assertAlmostEqual(score['points'][1]['tolerance_m_s'],.02)
    def test_no_missing_or_duplicate_points(self):
        for ids,o,p in [(['a','a'],[1,1],[1,1]),(['a'],[1],[np.nan]),(['a'],[1,2],[1])]:
            with self.assertRaises(ValueError):strict_speed_score(ids,o,p)
    def test_unknown_not_dry_or_overall_accuracy(self):
        score=inundation_score([1,1,0,0,-1],[1,-1,1,0,0])
        self.assertEqual(score['tp'],1);self.assertEqual(score['fn'],0);self.assertEqual(score['fp'],1)
        self.assertEqual(score['observed_wet_unresolved'],1);self.assertEqual(score['unknown_observation'],1)
        self.assertEqual(score['wet_recall_lower_bound'],.5);self.assertNotIn('accuracy',score)
    def test_uncertain_wet_cannot_raise_recall(self):
        score=inundation_score([1,0],[2,0]);self.assertEqual(score['wet_recall_lower_bound'],0.)
        self.assertIsNone(score['classified_csi'])

class Feed(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);make(self.root,'lake')
        c,_,fingerprint=load_case(self.root/'lake/case.json');self.case=c
        h=np.stack([np.full((12,40),v) for v in (.0,.1,.2,.3,.4)])
        z=np.zeros_like(h);self.arrays=dict(time_s=np.array([0,.05,.1,.15,.2]),bed_m=z[0],depth_m=h,qx_m2_s=z,qy_m2_s=z,
          native_volume_m3=h.sum(axis=(1,2))*.01,net_boundary_volume_m3=h.sum(axis=(1,2))*.01,net_source_volume_m3=np.zeros(5),accepted_dt_s=np.full(40,.005))
        self.folder=self.root/'result';self.folder.mkdir();self.save(fingerprint)
    def save(self,fingerprint='fixture'):
        np.savez(self.folder/'fields.npz',**self.arrays)
        write_json(self.folder/'run.json',dict(schema='hyra-flood-run-v1',complete=True,case=self.case,case_sha256=fingerprint,fields_sha256=sha(self.folder/'fields.npz'),backend='fixture',native_cells=480))
    def tearDown(self):self.tmp.cleanup()
    def test_only_past_frame_and_unknown_footprint(self):
        f=FloodFeed(self.folder);visible=np.zeros((12,40),bool);visible[0,0]=True
        obs=f.observe(.075,visible,.05)
        self.assertEqual(obs['source_frame_time_s'],.05);self.assertAlmostEqual(obs['depth_m'][0,0],.1)
        self.assertTrue(np.isnan(obs['depth_m'][1,1]));self.assertEqual(obs['water_class'][1,1],-1)
        with self.assertRaises(ValueError):f.observe(.21,visible,.05)
    def test_ambiguity_band_is_explicit(self):
        obs=FloodFeed(self.folder).observe(.05,np.ones((12,40),bool),.1,.01)
        self.assertTrue((obs['water_class']==2).all())
    def test_gauges_keep_dry_velocity_unknown(self):
        g=gauge_series(self.folder,.05)
        self.assertEqual(len(g['time_s']),5)
        self.assertIsNone(g['gauges'][0]['u_east_m_s'][0])
        self.assertFalse(g['gauges'][0]['velocity_valid'][0])
        self.assertEqual(g['gauges'][0]['u_east_m_s'][1],0.)
    def test_partial_time_window_rejected(self):
        self.arrays['time_s']=self.arrays['time_s'][:-1];self.save()
        with self.assertRaises(ValueError):load_run(self.folder)
    def test_changed_data_rejected_even_with_updated_hash(self):
        self.arrays['depth_m'][1,0,0]+=1;self.save()
        with self.assertRaisesRegex(ValueError,'projection'):load_run(self.folder)
    def test_comparing_different_inputs_rejected(self):
        other=self.root/'other';other.mkdir()
        np.savez(other/'fields.npz',**self.arrays)
        m=read_json(self.folder/'run.json');m['case_sha256']='different';m['fields_sha256']=sha(other/'fields.npz');write_json(other/'run.json',m)
        with self.assertRaisesRegex(ValueError,'Different case'):compare_runs(self.folder,other)

if __name__=='__main__':unittest.main()
