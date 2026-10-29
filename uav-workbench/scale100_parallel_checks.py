"""Concurrency checks on development data; no formal/holdout training launched."""
import unittest,hashlib,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scale100_formal_parallel import available_memory_gib,apply_event,worker_limit
from scale100_train import *
from scale100_evaluate_sealed import replay_bytes

ROOT=Path(__file__).parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-dynamic-900s-20261026/dataset.json'

def evaluate(identity):
    task,seed=identity;row=next(r for r in json.loads(DATA.read_text())['cases'] if r['split']=='development');case=FloodCase(row['path']);actor=Agent();actor.load_state_dict(torch.load(ROOT/f'generated/scale100-budget-pilot-20261026/{task}-B0-s11/actor.pt',weights_only=True));actor.eval()
    env=ScaleEnv(case,config(task,'B0',seed));total,_,metrics=rollout(actor,env,np.random.default_rng(seed),0.,False)
    return hashlib.sha256(replay_bytes(dict(return_=total,metrics=metrics,trajectory=env.trajectory,actions=env.actions,events=env.events))).hexdigest()

class ParallelChecks(unittest.TestCase):
    def test_parallel_evaluation_reproduces_serial_trajectories(self):
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);tasks=[('E1',11),('E2',29)];serial=[evaluate(t) for t in tasks]
        with ThreadPoolExecutor(max_workers=2) as pool:parallel=list(pool.map(evaluate,tasks))
        self.assertEqual(serial,parallel)
    def test_late_event_cannot_resurrect_completed_job(self):
        status=dict(active_jobs={'pending':dict(stage='starting')},completed=['finished'],formal_training_started=False);apply_event(status,'finished',dict(stage='training',child_pid=123))
        self.assertNotIn('finished',status['active_jobs']);self.assertFalse(status['formal_training_started'])
        apply_event(status,'pending',dict(stage='training',child_pid=124));self.assertTrue(status['formal_training_started']);self.assertEqual(status['active_jobs']['pending']['child_pid'],124)
    def test_actual_memory_guard_reads_available_memory(self):
        self.assertGreater(available_memory_gib(),0.)
    def test_actual_worker_limit_branch(self):
        self.assertEqual(worker_limit(4.5,3),3);self.assertEqual(worker_limit(3.2,3),2);self.assertEqual(worker_limit(2.,3),0);self.assertEqual(worker_limit(float('inf'),3),3);self.assertEqual(worker_limit(5.,3,1),1)

if __name__=='__main__':unittest.main(verbosity=2)
