"""Synthetic checks of preregistered event-level decision criteria."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from evaluate_sciml import gate_from_events,average
from hyra_learning.sciml_data import read


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.criteria=read(ROOT/'learning_configs/sciml_round1_design_20261025.json')['hypothesis_gate']
        self.events=[]
        for case in range(9):
            for method in ('QMIX','Sur-QMIX','Sur-QMIX+Data','Sur-QMIX+Physics'):
                self.events.append(dict(case=str(case),family=str(case//3),forcing_family=str(case%3),method=method,
                    quality=.32 if method=='Sur-QMIX+Physics' else .30,collisions=0.,motion_cost=.02))

    def test_positive_gate_and_event_pairing(self):
        result=gate_from_events(self.events,self.criteria)
        self.assertTrue(result['passed']);self.assertEqual(result['positive_events'],9)
        self.assertAlmostEqual(result['mean_deltas']['Data'],.02)

    def test_forcing_failure_survives_positive_overall_mean(self):
        for row in self.events:
            if row['method']=='Sur-QMIX+Physics' and row['forcing_family']=='0':row['quality']=.299
        result=gate_from_events(self.events,self.criteria)
        self.assertGreater(result['mean_deltas']['Data'],.005)
        self.assertFalse(result['criteria']['each_forcing_positive'])
        self.assertEqual(result['positive_events'],6);self.assertFalse(result['passed'])

    def test_strict_positive_and_cost_checks(self):
        for row in self.events:
            if row['method']=='Sur-QMIX+Physics':row.update(quality=.30,collisions=1.01,motion_cost=.0401)
        result=gate_from_events(self.events,self.criteria)
        self.assertEqual(result['positive_events'],0)
        self.assertFalse(result['criteria']['collision_increase'])
        self.assertFalse(result['criteria']['motion_cost_increase'])

    def test_empty_prediction_subsets_remain_null(self):
        self.assertIsNone(average([dict(x=None),dict(x=None)],'x'))
        self.assertEqual(average([dict(x=None),dict(x=2.),dict(x=4.)],'x'),3.)


if __name__=='__main__':unittest.main()
