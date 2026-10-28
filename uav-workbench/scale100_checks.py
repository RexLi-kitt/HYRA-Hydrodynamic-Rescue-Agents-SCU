"""Behavioral and numerical gates, no performance claims from a smoke run."""
import json,unittest
from copy import deepcopy
from pathlib import Path
from dataclasses import replace
from scale100_train import *

ROOT=Path(__file__).parent
DATA=ROOT.parent/'terrain-workbench/terrain-lab/generated/scale100-900s-20261026/dataset.json'

class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.dataset=json.loads(DATA.read_text());cls.case=FloodCase(cls.dataset['cases'][0]['path'])
    def test_actual_frames_and_splits(self):
        rows=self.dataset['cases'];self.assertEqual(len({r['parent_terrain'] for r in rows}),6)
        for row in rows:
            case=FloodCase(row['path']);self.assertEqual(row['sha256'],case.digest);self.assertEqual(case.times.tolist(),list(range(0,901,5)))
            self.assertTrue(np.isfinite(case.probes['before']).all());self.assertTrue((case.probes['before'][:,0]>=0).all())
        with self.assertRaises(ValueError):self.case.index(2.5)
    def test_operator_matches_actual_internal_step(self):
        errors=[]
        for row in self.dataset['cases'][:4]:
            case=FloodCase(row['path'])
            for i in (0,19,20,79,80,139,140,180):
                pred=SWEOperator().step(torch.tensor(case.probes['before'][i])[None],torch.tensor(case.bed),float(case.probes['dt'][i]),source=torch.tensor(case.probes['source'][i]))
                errors.append(float((pred[0]-torch.tensor(case.probes['after'][i])).abs().max()))
        self.assertLess(max(errors),1e-10);self.operator_max_error=max(errors)
    def test_permissions_raw_gmrf_and_agent_count(self):
        env=ScaleEnv(self.case,config('E1','B3',11));v=env.reset();self.assertEqual(v.images.shape,(2,21,5,5));self.assertEqual(env._targets,[]);self.assertTrue((v.signal==0).all())
        saved=v.images.copy();original=self.case.values
        def future_changed(t):return original(t)+100 if t>0 else original(t)
        self.case.values=future_changed
        self.assertTrue(np.array_equal(saved,env.view().images));self.case.values=original
        self.assertTrue((env.water[:,~env.explored]==0).all());self.assertTrue((env.observed_s[~env.explored]==-1).all())
        raw=ScaleEnv(self.case,config('E1','B0',11));rv=raw.reset();self.assertTrue(np.array_equal(env._blocked,raw._blocked));self.assertTrue(np.array_equal(saved[:,9:],rv.images[:,9:]))
        four=ScaleEnv(self.case,config('E2','B3',11,4));self.assertEqual(four.reset().images.shape,(4,21,5,5));four.step([1,1,0,0]);self.assertEqual(len(four.busy),4)
    def test_service_counterfactual(self):
        # Artificial dry/wet substitute checks the interface; not a physical flood experiment.
        outputs=[]
        for depth in (0.,1.):
            case=deepcopy(self.case);env=ScaleEnv(case,config('E2','B3',11));old=case.values
            def constant(t,d=depth,old=old):
                value=old(t).copy();value[:3]=0;value[0]=d;return value
            case.values=constant;env.reset();p=tuple(env.positions[0]);env._targets=[p];env.target_status={p:'rescuing'};env.busy[0]=dict(cell=p,remaining=5.)
            _,reward,_=env.step([1,1]);outputs.append((reward,env.target_status[p],env.view().images.copy()))
        self.assertEqual(outputs[0][1],'rescued');self.assertEqual(outputs[1][1],'discovered');self.assertNotEqual(outputs[0][0],outputs[1][0]);self.assertFalse(np.array_equal(outputs[0][2],outputs[1][2]))
    def test_auxiliary_reaches_encoder_and_gru(self):
        env=ScaleEnv(self.case,config('E1','P',11));view=env.reset();torch.manual_seed(11);actor=Agent();actor.train()
        _,h,_=actor(torch.tensor(view.images.copy()),torch.zeros(2,8),0,actor.hidden(2));bed=torch.tensor(self.case.bed,dtype=torch.float32)[None].expand(2,-1,-1)
        pred=actor.predict(h,bed);step=SWEOperator().step(pred[:,0],bed,.5);loss=(pred[:,1]-step).square().mean()+pred.square().mean();loss.backward()
        for parameter in (actor.conv[0].weight,actor.gru.weight_ih,actor.aux[-1].weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all());self.assertGreater(float(parameter.grad.abs().sum()),0.)
    def test_frozen_sources_preserved(self):
        plan=json.loads((ROOT/'generated/sciml-round1/formal-plan.json').read_text(encoding='utf-8'))
        for path,expected in plan['sources'].items():self.assertEqual(digest(path),expected,path)
        for method in ('QMIX8','Sur-QMIX8'):
            meta=json.loads((ROOT/f'generated/sur-rescue-pilot/{method}-s11/actor.json').read_text(encoding='utf-8'))
            for name,expected in meta['sources'].items():self.assertEqual(digest(ROOT/'sur_rescue'/name),expected,name)

if __name__=='__main__':unittest.main(verbosity=2)
