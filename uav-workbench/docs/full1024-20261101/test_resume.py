import copy,json,tempfile,time,unittest
from pathlib import Path
import numpy as np
import torch
import continue_full1024 as run

ROOT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')

def assert_nested(test,a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        test.assertEqual(set(a),set(b))
        for k in a:assert_nested(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_nested(test,x,y)
    else:test.assertEqual(a,b)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        cls.parent=ROOT/'uav6-priority-20261101'
        cls.results=json.loads((cls.parent/'results.json').read_text(encoding='utf-8'))
        items=json.loads((ROOT/'uav6-pilot-20261101/data/dataset.json').read_text(encoding='utf-8'))['cases']
        cls.cases={'train':[(Path(x['path']).name,np.load(x['path'],mmap_mode='r')) for x in items if x['split']=='train']}

    def check_algorithm(self,a):
        m=json.loads((self.parent/a/'latest.json').read_text(encoding='utf-8'));p=Path(m['path'])
        self.assertEqual(run.core.sha(p),m['sha256'])
        old=self.results[a]
        best=dict(episode=old['best_episode'],path=old['best_checkpoint'],validation=old['best_validation'])
        identity=dict(algorithm=a,guard=old['guard'],test_only=True)
        s=run.Session(a,p,copy.deepcopy(best),self.cases,identity)
        start=s.episode
        s.train_one();s.train_one()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='resume-test-') as directory:
            cp=Path(directory)/'split.pt';s.save(cp)
            expected=[s.train_one(),s.train_one()]
            expected_state=copy.deepcopy(s.model.checkpoint());expected_rng=run.core.rng_state()
            expected_steps=s.steps
            resumed=run.Session(a,cp,copy.deepcopy(best),self.cases,identity)
            actual=[resumed.train_one(),resumed.train_one()]
            self.assertEqual(resumed.episode,start+4)
            self.assertEqual(resumed.steps,expected_steps)
            assert_nested(self,expected,actual)
            assert_nested(self,expected_state,resumed.model.checkpoint())
            assert_nested(self,expected_rng,run.core.rng_state())
            self.assertEqual(len(s.replay),len(resumed.replay))
            if s.replay:
                assert_nested(self,s.replay[-1],resumed.replay[-1])
        self.assertEqual(run.core.sha(p),m['sha256'])

    def test_hasac_resume_matches_uninterrupted_real_updates(self):self.check_algorithm('hasac')
    def test_hatrpo_resume_matches_uninterrupted_real_updates(self):self.check_algorithm('hatrpo')

    def test_post_validation_best_takes_precedence(self):
        # No training: demonstrate that a saved pre-validation selection cannot
        # silently erase a later recorded best at the same checkpoint boundary.
        a='hatrpo';m=json.loads((self.parent/a/'latest.json').read_text(encoding='utf-8'))
        old=self.results[a];best=dict(episode=old['best_episode'],path=old['best_checkpoint'],validation=old['best_validation'])
        s=run.Session(a,Path(m['path']),best,self.cases,dict(algorithm=a,guard=False))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='best-test-') as d:
            cp=Path(d)/'state.pt';s.save(cp)
            newer=dict(best,episode=704,path='post-validation-metadata-example')
            restored=run.Session(a,cp,newer,self.cases,s.identity)
            self.assertEqual(restored.best['episode'],704)


if __name__=='__main__':
    begin=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ResumeTests))
    (Path(__file__).parent/'preflight.json').write_text(json.dumps(dict(tests_run=result.testsRun,
        passed=result.wasSuccessful(),elapsed_s=time.perf_counter()-begin,
        checks=['real HASAC split/resume versus uninterrupted updates','real HATRPO split/resume versus uninterrupted updates',
                'identical parameters optimizers RNG replay counters and episode records','post-validation best metadata precedence'],
        scope='test-only temporary branches; original checkpoints unchanged; no main episode count increment'),indent=2),encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
