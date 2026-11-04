import unittest
import numpy as np
import torch
from residual_env import ResidualEnv,DispatchEnv,initialize_planning_default
from test_dispatch import Config,scenario,field
from train import Learner,METHODS
from metrics import summary,quality

class Tests(unittest.TestCase):
    def test_people_primary_and19_person_boundary(self):
        a={'rows':[dict(supplied=n) for n in [20,18,18]]}
        b={'rows':[dict(supplied=n) for n in [19,19,19]]}
        self.assertGreater(quality(b),quality(a))
        self.assertEqual(sum(r['supplied']>=19 for r in b['rows']),3)
        self.assertEqual(sum(r['supplied']==20 for r in b['rows']),0)
    def test_default_matches_full_planning_trajectory_and_metrics(self):
        for nearby in (False,True):
            cfg=Config(duration=60,window=30);s=scenario(cfg,nearby)
            a=DispatchEnv(field(.1),s);b=ResidualEnv(field(.1),s)
            for _ in range(12):
                a.step(a.choices()[1]);b.step(np.full(4,8,dtype=np.int64))
                np.testing.assert_array_equal(a.pos,b.pos);np.testing.assert_array_equal(a.person,b.person)
                np.testing.assert_array_equal(a.status,b.status);self.assertEqual(a.metrics(),b.metrics())
    def test_default_wait_becomes_real_patrol_choice(self):
        e=ResidualEnv(field(),scenario(Config()));before=e.pos.copy();e.step(np.full(4,8,dtype=np.int64))
        self.assertTrue((np.linalg.norm(e.pos-before,axis=1)>0).all())
    def test_all_three_initial_policies_choose_exact_default(self):
        torch.set_num_threads(1)
        e=ResidualEnv(field(),scenario(Config()));obs=e.observe();mask=e.choices()[0]
        for m in METHODS:
            model=Learner(m,760);initialize_planning_default(model);model.eval()
            action,_,_=model.act(obs,mask,True);np.testing.assert_array_equal(action,[8]*4)
    def test_hidden_truth_cannot_change_residual_candidates(self):
        e=ResidualEnv(field(),scenario(Config()));obs=e.observe().copy();mask=e.choices()[0].copy()
        e.person[:]=[600,600];e.deadline[:]=0;e.fall[:]=200;e.water=field(8)
        np.testing.assert_array_equal(obs,e.observe());np.testing.assert_array_equal(mask,e.choices()[0])
    def test_three_learners_update_finitely_on_actual_residual_trajectory(self):
        torch.set_num_threads(1)
        for m in METHODS:
            model=Learner(m,760);initialize_planning_default(model)
            e=ResidualEnv(field(.1),scenario(Config(duration=60,window=30),True));rows=[];done=False
            obs=e.observe();mask=e.choices()[0]
            while not done:
                action,logs,value=model.act(obs,mask,False);nxt,reward,done,_=e.step(action);nm=e.choices()[0]
                rows.append(dict(obs=obs,mask=mask,action=action,reward=reward,done=float(done),next_obs=nxt,next_mask=nm,logp=logs,value=value))
                obs=nxt;mask=nm
            for i,r in enumerate(rows):r['next_action']=rows[i+1]['action'] if i+1<len(rows) else np.full(4,8,dtype=np.int64)
            if m=='maac':model.update_off_policy(rows)
            else:model.update_on_policy(rows)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
if __name__=='__main__':unittest.main()
