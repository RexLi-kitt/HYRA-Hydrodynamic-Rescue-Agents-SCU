import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from hyra_uav import strong_study as study
from hyra_uav.experiments import write_json
from test_core import TEST_TEMP


class StrongStudyTests(unittest.TestCase):
    def plan(self,root,round_id=1):
        def decisions(path):
            return {'frozen':True} if 'p2-formal' in str(path) else {'ready':True,'selected_cost_scale_m':2.5}
        with patch.object(study,'load_json',side_effect=decisions):
            return study.make_plan(round_id,root)

    def test_second_round_is_new_families_with_strict_ablation_same_budget(self):
        first=self.plan(Path('unused'));second=self.plan(Path('unused'),2)
        self.assertFalse(first['strict_change_ablation']);self.assertTrue(second['strict_change_ablation'])
        self.assertFalse({c['terrain_recipe'] for c in first['cases']}&{c['terrain_recipe'] for c in second['cases']})
        self.assertEqual(first['resource_budget'],second['resource_budget'])
        self.assertEqual(first['learner_candidates'],second['learner_candidates'])
        self.assertEqual(first['gate'],second['gate'])
        beds=[study.terrain(c['terrain_recipe'],c['seed']).tobytes() for c in second['cases']]
        self.assertEqual(len(set(beds)),18)

    def test_family_split_and_budget_are_frozen_before_holdout(self):
        plan=self.plan(Path("unused"))
        splits=[{c["family"] for c in plan["cases"] if c["split"]==s} for s in ("fit","selection","holdout")]
        self.assertTrue(all(not a&b for i,a in enumerate(splits) for b in splits[i+1:]))
        self.assertEqual([len(a) for a in splits],[2,1,3])
        self.assertEqual(plan["resource_budget"]["selection_fits"],10)
        self.assertEqual(plan["resource_budget"]["final_fits"],15)
        self.assertEqual(study.expected_rows(plan),286740)
        beds=[study.terrain(c["terrain_recipe"],c["seed"]).tobytes() for c in plan["cases"]]
        self.assertEqual(len(set(beds)),18)
        self.assertTrue(all(c["parent_terrain"]==c["id"] for c in plan["cases"]))

    def test_nonlinear_learner_recovers_history_interactions(self):
        _,_,ExtraTrees=study.dependencies()
        rng=np.random.default_rng(81);x=rng.uniform(-1,1,(2000,2));y=(x[:,0]*x[:,1]>0).astype(int)
        model=ExtraTrees(n_estimators=64,max_depth=18,min_samples_leaf=5,max_features=.25,random_state=11,n_jobs=2)
        model.fit(x[:1500],y[:1500])
        self.assertGreater(np.mean((study.risk(model,x[1500:])>=.5)==y[1500:]),.95)

    def test_na_and_route_seed_aggregation_do_not_make_fake_events(self):
        rows=[]
        for seed in (11,29):
            for route,value in (("R0",.2),("R1",.6),("R2",None)):
                rows.append(dict(case="one",seed=seed,route=route,precision=value,fraction=.1,band="all"))
        self.assertAlmostEqual(study.event_values(rows)["one"],.4)
        rows=[dict(r,precision=None) for r in rows]
        self.assertIsNone(study.event_values(rows)["one"])

    def test_partial_score_cannot_be_reported_as_completed(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as directory:
            root=Path(directory);write_json(root/"plan.json",self.plan(root))
            (root/"metrics.jsonl").write_text("",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"尚未完整"):study.summarize(root)
            self.assertFalse((root/"summary.json").exists())


if __name__=="__main__":unittest.main()
