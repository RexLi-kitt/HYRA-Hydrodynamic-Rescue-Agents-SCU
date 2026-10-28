"""Evaluation correctness checks using development data and synthetic statistics."""
import tempfile,unittest
from scale100_evaluate_sealed import *
from scale100_forecast_audit import DATA

class SealedChecks(unittest.TestCase):
    def test_holdout_cannot_open_before_all_formal_jobs_finish(self):
        with tempfile.TemporaryDirectory(prefix='hyra-seal-test-') as d:
            path=Path(d);(path/'queue.json').write_text(json.dumps(dict(status='running')))
            with self.assertRaisesRegex(AssertionError,'All36'):seal_models(path,{},path)
    def test_replay_has_deterministic_compression_and_numpy_scalars(self):
        with tempfile.TemporaryDirectory(prefix='hyra-replay-test-') as d:
            path=Path(d);payload=dict(cell=np.int64(3),water=np.float32(.1));a=path/'a.gz';b=path/'b.gz';save_replay(a,payload);save_replay(b,payload)
            self.assertEqual(digest(a),digest(b))
            with gzip.open(a,'rt',encoding='utf-8') as f:self.assertEqual(json.load(f)['cell'],3)
    def test_parent_weighting_does_not_count_scenarios_as_events(self):
        rows=[]
        for parent,seeds,delta in [('A',[11,29,47],.1),('B',[11],.5)]:
            for seed in seeds:
                for method in ('D','P'):
                    values={k:.2+(delta if method=='P' else 0.) for k in METRICS};rows.append(dict(parent=parent,task='E1',method=method,training_seed=11,scenario_seed=seed,metrics=values))
        result=aggregate(rows);contrast=next(r for r in result['contrasts'] if r['name']=='physics_increment')
        self.assertEqual(result['independent_parent_count'],2);self.assertAlmostEqual(contrast['mean_difference']['first_coverage_auc'],.3)
        self.assertEqual([r['pairs'] for r in contrast['parent_differences']],[3,1])
    def test_nonpaired_seeds_are_not_subtracted(self):
        rows=[dict(parent='A',task='E1',method=m,training_seed=11,scenario_seed=s,metrics={k:.2 for k in METRICS}) for m,s in [('D',11),('P',29)]]
        self.assertFalse(any(r['name']=='physics_increment' for r in aggregate(rows)['contrasts']))
    def test_actual_development_rescue_replay_and_resume(self):
        torch.set_num_threads(1);entry=next(r for r in json.loads(DATA.read_text())['cases'] if r['split']=='development');case=FloodCase(entry['path'])
        with tempfile.TemporaryDirectory(prefix='hyra-rescue-replay-') as d:
            path=Path(d);args=(path,'development-E2-water',case,'E2','water_aware',11)
            first=run_trial(*args,split='development_smoke');self.assertTrue(first['metrics']['audit_passed']);self.assertGreater(first['metrics']['rescued'],0)
            with gzip.open(path/first['replay_path'],'rt',encoding='utf-8') as f:replay=json.load(f)
            self.assertEqual(len(replay['snapshots']),181);self.assertEqual(len(replay['observations']),362);self.assertTrue(any(e['type']=='rescued' for e in replay['events']))
            self.assertEqual(first,run_trial(*args,split='development_smoke'))
            corrupt=path/first['replay_path'];original=corrupt.read_bytes();corrupt.write_bytes(b'corrupt')
            try:
                with self.assertRaisesRegex(AssertionError,'corrupted'):run_trial(*args,split='development_smoke')
            finally:corrupt.write_bytes(original)

if __name__=='__main__':unittest.main(verbosity=2)
