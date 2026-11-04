import importlib.util,sys,unittest,copy
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('flow_fog',Path(__file__).with_name('experiment.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
from refloat_environment import Config
from dataclasses import asdict

class Tests(unittest.TestCase):
    def setUp(self):
        self.water=np.zeros((301,3,100,100),np.float32);self.water[:,0]=.1
        self.scenario=dict(seed=2,config=asdict(Config(people=2,window=240,duration=1320,sensor=60,trigger_depth=2,trigger_speed=1)),traps=[],
            events=[dict(fall=0,deadline=240,xy=[550.,500.],exposure_id=1),dict(fall=0,deadline=240,xy=[505.,500.],exposure_id=2)])
    def test_transport_zero_and_mass(self):
        a=np.zeros((10,10));a[5,5]=4;z=np.zeros_like(a)
        np.testing.assert_array_equal(module.transport(a,z,z,60),a)
        b=module.transport(a,np.ones_like(a),z,50);self.assertAlmostEqual(b.sum(),4);self.assertAlmostEqual(b[5,6],2)
    def test_outflow_not_edge_pileup(self):
        a=np.zeros((10,10));a[5,9]=4;z=np.zeros_like(a)
        self.assertAlmostEqual(module.transport(a,np.ones_like(a),z,100).sum(),0)
    def test_fog_disks_and_person_visibility(self):
        s=copy.deepcopy(self.scenario);s['patch_fog']=True;e=module.FlowFogEnv(self.water,s,'coverage100')
        e.pos[:]=[500,500];e.known[:]=False;e.last[:]=-1;e.measured[:]=0;e.discovered[:]=False;e.observe_world()
        self.assertEqual(e.radii().tolist(),[20]*4);self.assertEqual(e.visible.tolist(),[False,True])
        expected=np.sum((e.cells-[500,500])**2,axis=-1)<=400
        np.testing.assert_array_equal(e.known,expected);self.assertTrue((e.measured[:,~expected]==0).all())
    def test_clear_trajectory_matches_original(self):
        a=module.FlowFogEnv(self.water,self.scenario,'coverage100');b=module.SearchEnv(self.water,self.scenario,'coverage100')
        for _ in range(30):
            np.testing.assert_array_equal(a.choices()[1],b.choices()[1]);a.step(a.choices()[1]);b.step(b.choices()[1])
            for key in ('pos','person','status','known','last','measured'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
    def test_predicted_score_no_hidden_person_input(self):
        e=module.FlowFogEnv(self.water,self.scenario,'gru-32_flowblend',lambda x:np.ones(100)*.1)
        stale=np.ones((100,100));risk=stale.copy();before=e.search_score(stale,risk)
        e.person[:]=999;e.fall[:]=100;e.deadline[:]=900;e.status[:]=2
        np.testing.assert_array_equal(e.search_score(stale,risk),before)
    def test_window_variant_retains_events_and_deadlines(self):
        # Real 40-person roster: no selecting by outcome or changing fall time/xy.
        e=copy.deepcopy(self.scenario['events'][0]);s=copy.deepcopy(self.scenario);s['events']=[dict(e,exposure_id=i) for i in range(40)]
        c=dict(id='check',scenario=s);p=dict(name='new',sensor=20,window=480,patch=False);v=module.variant(c,p)
        self.assertEqual(len(v['scenario']['events']),40)
        for before,after in zip(c['scenario']['events'],v['scenario']['events']):
            self.assertEqual(before['xy'],after['xy']);self.assertEqual(before['fall'],after['fall']);self.assertEqual(after['deadline']-after['fall'],480)

if __name__=='__main__':unittest.main()
