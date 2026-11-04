import unittest,copy,json
from pathlib import Path
import numpy as np
import validate as v

class ValidationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(17);self.water=rng.normal(size=(301,3,100,100)).astype(np.float32);self.water[:,0]=np.abs(self.water[:,0])
        self.scenario=dict(traps=[[225,375],[765,145]],events=[dict(xy=[325,455],fall=30,deadline=210,field=[.5,.2,-.3])])
    def test_rotate_vector_and_position(self):
        np.testing.assert_array_equal(v.xy_transform([200,300],'rotate90'),[300,800]);np.testing.assert_array_equal(v.vector_transform([2,3],'rotate90'),[3,-2])
        a,s=v.transformed(self.water,self.scenario,'rotate90')
        original=v.f.sample_water(self.water,42,np.array([[325.,455.]]))
        mapped=v.f.sample_water(a,42,np.asarray([s['events'][0]['xy']]))
        np.testing.assert_allclose(mapped[:,0],original[:,0],atol=1e-6);np.testing.assert_allclose(mapped[:,1:],v.vector_transform(original[:,1:],'rotate90'),atol=1e-6)
    def test_four_rotations_recover(self):
        a=self.water;s=copy.deepcopy(self.scenario)
        for _ in range(4):a,s=v.transformed(a,s,'rotate90')
        np.testing.assert_array_equal(a,self.water);self.assertEqual(s,self.scenario)
    def test_double_mirror_recover(self):
        a,s=v.transformed(self.water,self.scenario,'mirror_x');a,s=v.transformed(a,s,'mirror_x')
        np.testing.assert_array_equal(a,self.water);self.assertEqual(s,self.scenario)
    def test_cluster_bootstrap_known_paired_difference(self):
        rows=[]
        for parent in ('a','b','c'):
            for layout in range(2):
                for m,n in zip(v.METHODS,(20,24,32)):
                    rows.append(dict(parent=parent,method=m,people=40,supplied=n,unseen_expired=0,refloat_events=0))
        result=v.statistics(rows);self.assertEqual(result['parent_count'],3)
        np.testing.assert_allclose(result['gaps']['people_rate']['ci95'],[.2,.2],atol=1e-12)
        self.assertFalse(result['point_goal_met'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ValidationTests);result=unittest.TextTestRunner(verbosity=2).run(suite)
    Path(__file__).with_name('readiness.json').write_text(json.dumps(dict(tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors)),indent=2),encoding='utf8')
    raise SystemExit(not result.wasSuccessful())
