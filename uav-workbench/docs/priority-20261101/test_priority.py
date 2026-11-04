import copy
import unittest
import numpy as np
import torch
import priority_train as p
from retry_guard import RetryGuard


def public_obs(pos=None):
    obs=np.zeros((4,1468),np.float32)
    s=np.zeros((4,5),np.float32)
    s[:,:2]=.5 if pos is None else np.asarray(pos)/1000
    obs[:,1326:1346]=s.ravel()
    return obs


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.ones((4,9),bool)
        self.phases=['learned']*4
        self.obs=public_obs()

    def test_repeated_failed_move_activates_after_three_attempts(self):
        g=RetryGuard()
        for _ in range(3):
            m,_=g.mask(self.obs,self.mask,self.phases)
            self.assertTrue(m[0,3]);g.remember([3,8,8,8])
        m,_=g.mask(self.obs,self.mask,self.phases)
        self.assertFalse(m[0,3]);self.assertFalse(m[0,8])
        np.testing.assert_array_equal(m[1:],self.mask[1:])

    def test_deliberate_hover_never_triggers_guard(self):
        g=RetryGuard()
        for _ in range(20):
            m,_=g.mask(self.obs,self.mask,self.phases);g.remember([8]*4)
            np.testing.assert_array_equal(m,self.mask)

    def test_movement_clears_history(self):
        g=RetryGuard()
        for _ in range(4):g.mask(self.obs,self.mask,self.phases);g.remember([3]*4)
        moved=public_obs([[550,500]]*4)
        m,_=g.mask(moved,self.mask,self.phases)
        np.testing.assert_array_equal(m,self.mask)

    def test_service_mask_has_priority(self):
        g=RetryGuard()
        for _ in range(4):g.mask(self.obs,self.mask,self.phases);g.remember([3]*4)
        service=self.mask.copy();service[0]=False;service[0,8]=True
        m,_=g.mask(self.obs,service,['hold','learned','learned','learned'])
        np.testing.assert_array_equal(m[0],service[0])
        self.assertEqual(g.streak[0],0)

    def test_mask_never_expands_information_or_mutates_input(self):
        g=RetryGuard();original=self.mask.copy();self.mask[0,:8]=False
        for _ in range(10):
            m,_=g.mask(self.obs,self.mask,self.phases);g.remember([8]*4)
            self.assertFalse((m&~self.mask).any());self.assertTrue(m.any(1).all())
        np.testing.assert_array_equal(self.mask[1:],original[1:])


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);p.seed_all(401)
        cls.water=np.zeros((301,3,100,100),np.float32)

    def test_actual_transition_masks_and_update_logging(self):
        p.seed_all(401);model=p.LoggedLearner('maac',1468);rows=[]
        p.rollout(model,self.water,9001,True,consume=rows.append)
        self.assertGreater(len(rows),2)
        for i,r in enumerate(rows):
            self.assertTrue(r['mask'][np.arange(4),r['action']].all())
            self.assertTrue(r['next_mask'][np.arange(4),r['next_action']].all())
            if i+1<len(rows):
                np.testing.assert_array_equal(r['next_action'],rows[i+1]['action'])
                np.testing.assert_array_equal(r['next_mask'],rows[i+1]['mask'])
        self.assertEqual(rows[-1]['done'],1.)
        plain=p.Learner('maac',1468);plain.restore(copy.deepcopy(model.checkpoint()))
        saved=p.rng_state();plain.update_off_policy(rows[:16]);p.restore_rng(saved)
        loss=model.measured_update(rows[:16],True)
        self.assertEqual(len(loss['optimizer_steps']),5)
        for k,v in plain.state_dict().items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)

    def test_mat_forced_service_and_real_update(self):
        p.seed_all(402);model=p.LoggedLearner('mat',1468);rows=[]
        p.rollout(model,self.water,9001,True,consume=rows.append)
        controlled=[r for r in rows if (r['mask'].sum(1)==1).any()]
        self.assertTrue(controlled)
        for r in controlled:self.assertTrue((np.abs(r['logp'][r['mask'].sum(1)==1])<1e-6).all())
        model.measured_update(rows,False);model.check_finite()

    def test_selection_rejects_regression(self):
        b=dict(summary=dict(RCR=.8,coverage=.9,ACC=.01),by_action_seed={'1':dict(RCR=.8)})
        c=copy.deepcopy(b);c['summary']['RCR']=.9;c['summary']['coverage']=.8
        self.assertFalse(p.guard_admissible(b,c));self.assertFalse(p.significant_improvement(c,b))
        c=copy.deepcopy(b);c['summary']['coverage']=.91
        self.assertTrue(p.guard_admissible(b,c));self.assertTrue(p.significant_improvement(c,b))


if __name__=='__main__':unittest.main(verbosity=2)
