import sys,unittest,json,math
from pathlib import Path
ROOT=Path(r'D:\HyRA\warehouse\HYRA-Hydrodynamic-Rescue-Agents-SCU\uav-workbench')
sys.path.insert(0,str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
from workbench_tasks import ExplorationConfig,ExplorationEnv,make_env
from workbench_api import catalog,validate,group_record
from sur_rescue.hydraulics import file_digest

class Hydro:
    shape=(10,10);dx=5.;process={'grid':{'shape':[10,10],'dx_m':5.}};digest='unit'
    def index(self,t):return int(t/5)
    def labels(self,t):return np.indices(self.shape)[0]<int(t/5)

class WorkbenchTests(unittest.TestCase):
    def test_exploration_has_no_personnel_signal_dwell_or_timeout_rescue_penalty(self):
        histories=[]
        for service in (5,100):
            env=ExplorationEnv(Hydro(),ExplorationConfig(duration_s=10,service_s=service,obstacle_fraction=0))
            view=env.reset();self.assertEqual(view.targets,());self.assertEqual(view.images.shape,(2,9,5,5));self.assertEqual(env._targets,[])
            rewards=[]
            for i in range(2):
                view,reward,done=env.step([1,1]);rewards.append(reward)
                self.assertFalse(view.signal.any());self.assertEqual(view.busy,(0,0));self.assertEqual(view.targets,())
            self.assertTrue(done);self.assertTrue(all(r>-10 for r in rewards));histories.append(rewards)
            r=env.result();self.assertEqual(r['events'],[]);self.assertEqual(r['summary']['total_targets'],0);self.assertIsNone(r['summary']['rescue_fraction']);self.assertFalse(r['audit']['rescue_reward_enabled'])
            self.assertTrue(all(not s['targets'] and not any(s['gmrf_signal']) for s in r['snapshots']))
            with self.assertRaises(ValueError):env.step([1,1])
        self.assertEqual(histories[0],histories[1])
    def test_initial_complete_and_invalid_exploration_configuration(self):
        source=Hydro();source.shape=(2,2);source.process={'grid':{'shape':[2,2],'dx_m':5.}};source.labels=lambda t:np.zeros((2,2),bool)
        env=ExplorationEnv(source,ExplorationConfig(duration_s=5,obstacle_fraction=0));env.reset();self.assertTrue(env._done)
        r=env.result();self.assertTrue(math.isfinite(r['summary']['time_average_correct_coverage']));self.assertEqual(len(r['snapshots']),1)
        with self.assertRaises(ValueError):ExplorationConfig(n_targets=1).validate(source)
        with self.assertRaises(ValueError):make_env(source,'unknown',{})
    def test_exactly_six_groups_and_task_matched_capabilities(self):
        c=catalog();self.assertEqual([g['id'] for g in c['groups']],['B0','B1','B2','B3','D','P'])
        self.assertEqual(len(c['tasks']),2);self.assertTrue(all(not g['tasks']['exploration']['ready'] for g in c['groups']))
        self.assertEqual([g['id'] for g in c['groups'] if g['tasks']['exploration_rescue']['ready']],['B1','B3'])
        self.assertEqual(len(c['actions']),8)
    def test_rule_checks_are_never_attributed_to_selected_model_group(self):
        c=catalog();job=validate(dict(scene='existing-demo',task='exploration',group='P',mode='rule',config=c['defaults']['exploration']))
        self.assertIsNone(job['group']);self.assertEqual(job['requested_group'],'P');self.assertEqual(job['config']['n_targets'],0)
    def test_unavailable_models_and_wrong_task_are_rejected(self):
        c=catalog()
        for g in ('B0','B2','D','P'):
            with self.subTest(group=g),self.assertRaises(ValueError):validate(dict(scene='existing-demo',task='exploration_rescue',group=g,mode='model',config=c['defaults']['exploration_rescue']))
        with self.assertRaises(ValueError):validate(dict(scene='existing-demo',task='exploration',group='B3',mode='model',config=c['defaults']['exploration']))
        with self.assertRaises(ValueError):validate(dict(scene='existing-demo',task='exploration',group='B3',mode='rule',config=dict(c['defaults']['exploration'],n_targets=5)))
    def test_original_scientific_and_eight_action_checkpoint_sources_preserved(self):
        for g in ('B1','B3'):
            _,meta=group_record(g);self.assertEqual(meta['n_actions'],8);self.assertEqual(meta['trained_episodes'],12)
        plan=json.loads((ROOT/'generated/sciml-round1/formal-plan.json').read_text(encoding='utf-8'))
        self.assertEqual(len(plan['sources']),35)
        self.assertTrue(all(file_digest(p)==h for p,h in plan['sources'].items()))

if __name__=='__main__':unittest.main(verbosity=2)
