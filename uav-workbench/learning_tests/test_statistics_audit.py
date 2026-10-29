"""Synthetic full-domain aggregation with deliberately unequal NA route counts."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
from evaluate_sciml import summarize,write,read
import audit_sciml_statistics as statistics
from test_core import TEST_TEMP


class StatisticsAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(dir=TEST_TEMP);cls.root=Path(cls.temp.name)
        methods=['QMIX','Sur-QMIX','Sur-QMIX+Data','Sur-QMIX+Physics'];seeds=[11,29,47]
        cls.methods=methods
        write(cls.root/'formal-plan.json',dict(methods=methods,seeds=seeds))
        design=read(ROOT/'learning_configs/sciml_round1_design_20261025.json')
        write(cls.root/'evaluation-plan.json',dict(policy_episodes=108,rule_episodes=27,forecast_rows=9720,gate=design['hypothesis_gate']))
        entries=[dict(id=f'event-{c}',split='holdout',family=f'parent-{c//3}',forcing_family=f'forcing-{c%3}') for c in range(9)]
        write(cls.root/'index.json',entries)
        for method in methods:
            for seed in seeds:
                write(cls.root/'formal'/(method.replace('+','-')+f'-s{seed}')/'validation.json',
                    [dict(episode=episode,mean_return=.2+episode/100000) for episode in range(200,2001,200)])
        for c,entry in enumerate(entries):
            for method in methods:
                for s,seed in enumerate(seeds):
                    quality=.3+c*.0001+s*.001+(0.01 if method==methods[3] else 0)
                    write(cls.root/'evaluation/policies'/f'{c}-{method.replace("+","-")}-{seed}.json',dict(case=entry['id'],family=entry['family'],forcing_family=entry['forcing_family'],method=method,seed=seed,
                        metrics=dict(time_average_correct_coverage=quality,collision_events=0,motion_cost=.02,return_=quality-.02)))
            for route in ('R0','R1','R2'):
                write(cls.root/'evaluation/rules'/f'{c}-{route}.json',dict(case=entry['id'],method=route,metrics=dict(time_average_correct_coverage=.5)))
            for method in methods[2:]:
                for s,seed in enumerate(seeds):
                    for r,route in enumerate(('R0','R1','R2')):
                        rows=[]
                        for t in range(60):
                            value=.1+c*.002+s*.001+r*.1+t*.0001
                            unknown=(r==0 or (r==1 and t==0));stale=(r==2 and t>50)
                            rows.append(dict(case=entry['id'],method=method,seed=seed,route=route,time_s=t*10,
                                unknown_cells=int(unknown),stale_cells=int(stale),future_h_mae_m=value,
                                future_h_unknown_mae_m=value if unknown else None,future_h_stale_mae_m=value if stale else None,
                                future_wet_classbalanced_error=value,future_momentum_mae_m2_s=value,current_h_mae_m=value,
                                supervised_loss=value,physical_rate_loss=value,absolute_volume_rate_residual_m3_s=value))
                        path=cls.root/'evaluation/forecasts'/f'{c}-{method.replace("+","-")}-{seed}-{route}.jsonl'
                        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        with patch('builtins.print'):summarize(cls.root)
        cls.summary=read(cls.root/'confirmation-summary.json')
        (cls.root/'docs/evidence/20261025-sciml-development').mkdir(parents=True)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def tearDown(self):write(self.root/'confirmation-summary.json',self.summary)

    def audit(self):
        with patch.object(statistics,'ROOT',self.root),patch.object(sys,'argv',['audit',str(self.root)]),patch('builtins.print'):
            statistics.main()

    def test_exact_domain_and_hierarchical_na_aggregation(self):
        self.audit()
        # 60 valid R0 frames and one valid R1 frame get equal route weight.
        row=next(r for r in self.summary['predictions'] if r['case']=='event-0' and r['method']==self.methods[2])
        expected=((.1+59*.0001/2)+.2)/2+.001
        self.assertAlmostEqual(row['future_h_unknown_mae_m'],expected)

    def test_rejects_corrupted_prediction_aggregate(self):
        broken=deepcopy(self.summary);broken['predictions'][0]['future_h_unknown_mae_m']+=.001
        write(self.root/'confirmation-summary.json',broken)
        with self.assertRaises(AssertionError):self.audit()

    def test_rejects_false_gate_or_missing_drift(self):
        broken=deepcopy(self.summary);broken['gate']['passed']=not broken['gate']['passed']
        write(self.root/'confirmation-summary.json',broken)
        with self.assertRaises(AssertionError):self.audit()
        broken=deepcopy(self.summary);broken['late_training_drift'].pop()
        write(self.root/'confirmation-summary.json',broken)
        with self.assertRaises(AssertionError):self.audit()


if __name__=='__main__':unittest.main()
