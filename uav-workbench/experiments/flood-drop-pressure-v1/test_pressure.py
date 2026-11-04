import copy,unittest
import numpy as np
from pressure import Config,Env,variant,paired_summary

class Tests(unittest.TestCase):
    def case(self):
        return dict(id='same',field={'parent':'p'},scenario=dict(config=Config().__dict__,traps=[[500,500]],
            events=[dict(fall=100+i*5,deadline=460+i*5,xy=[400+i,300],trigger='threshold',field=[.12,0,0]) for i in range(20)]))
    def test_variant_preserves_roster_falls_locations_and_traps(self):
        a=self.case();backup=copy.deepcopy(a);b=variant(a,240,75.)
        self.assertEqual(a,backup);self.assertEqual(len(b['scenario']['events']),20)
        self.assertEqual(a['scenario']['traps'],b['scenario']['traps'])
        for x,y in zip(a['scenario']['events'],b['scenario']['events']):
            self.assertEqual({k:v for k,v in x.items() if k!='deadline'},{k:v for k,v in y.items() if k!='deadline'})
            self.assertEqual(y['deadline']-x['deadline'],-120)
    def test_rejects_silent_truncation(self):
        a=self.case();a['scenario']['events'][0]['fall']=1300
        with self.assertRaises(ValueError):variant(a,300,100.)
    def test_reference_matches_original_observation_control_and_step(self):
        a=self.case();water=np.broadcast_to(np.full((3,100,100),.1),(301,3,100,100))
        ea=Env(water,a['scenario']);eb=Env(water,variant(a,360,100.)['scenario'])
        for _ in range(3):
            np.testing.assert_array_equal(ea.observe(),eb.observe())
            for x,y in zip(ea.control(),eb.control()):np.testing.assert_array_equal(x,y)
            ea.step(ea.control()[1]);eb.step(eb.control()[1]);self.assertEqual(ea.metrics(),eb.metrics())
    def test_difference_is_percentage_points_and_preserves_pairing(self):
        rows=[]
        for parent in range(6):
            for i in range(4):
                for m,success in [('planner',i<2),('maac',i<3)]:
                    rows.append(dict(parent=parent,case=f'{parent}-{i}',level='reference',method=m,scene_success=success))
        result=paired_summary(rows,'maac')
        self.assertEqual(result['difference_percentage_points'],25)
        self.assertEqual(result['parent_cluster_bootstrap_95_interval'],[25,25])

if __name__=='__main__':unittest.main()
