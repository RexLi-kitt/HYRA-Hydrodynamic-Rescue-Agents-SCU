import copy,unittest
import numpy as np
from dispatch_env import DispatchEnv,Config,anonymize_roster

def field(u=0):
    a=np.zeros((3,100,100));a[0]=.12;a[1]=u
    return np.broadcast_to(a,(301,3,100,100))
def scenario(cfg,near=False):
    return dict(config=cfg.__dict__,traps=[],events=[dict(fall=0,deadline=cfg.window,xy=[5,5] if near else [500,500])]+
        [dict(fall=cfg.duration-cfg.window,deadline=cfg.duration,xy=[400+i,500]) for i in range(19)])

class Tests(unittest.TestCase):
    def test_roster_shuffle_preserves_events_and_no_time_order_ids(self):
        s=scenario(Config());s['events']=[dict(fall=i,deadline=i+360,xy=[i,0]) for i in range(20)]
        a=anonymize_roster(s,47);self.assertEqual(sorted(e['fall'] for e in a['events']),list(range(20)))
        self.assertNotEqual([e['fall'] for e in a['events']],list(range(20)));self.assertEqual(s['events'][0]['fall'],0)
    def test_distinct_patrol_decisions_change_destination(self):
        cfg=Config();a=DispatchEnv(field(),scenario(cfg));b=DispatchEnv(field(),scenario(cfg))
        aa=a.choices()[1];bb=aa.copy();bb[0]=7 if aa[0]!=7 else 4
        a.step(aa);b.step(bb);self.assertFalse(np.array_equal(a.pos[0],b.pos[0]))
    def test_hidden_person_deadline_and_future_water_not_policy_input(self):
        e=DispatchEnv(field(),scenario(Config()));obs=e.observe().copy();m,x,_,f=e.choices()
        e.person[:]=[700,700];e.deadline[:]=0;e.fall[:]=1000;e.water=field(10)
        np.testing.assert_array_equal(obs,e.observe());mm,xx,_,ff=e.choices()
        np.testing.assert_array_equal(m,mm);np.testing.assert_array_equal(x,xx);np.testing.assert_array_equal(f,ff)
    def test_locked_follow_delivers_in_ten_seconds(self):
        cfg=Config(duration=30,window=10);e=DispatchEnv(field(.5),scenario(cfg,True))
        for _ in range(2):
            mask,expert,_,_=e.choices();self.assertEqual(expert[0],8);self.assertEqual(mask[0].sum(),1);e.step(expert)
        self.assertEqual(e.status[0],2);self.assertEqual(e.delivered_at[0],10)
    def test_model_dimensions_and_valid_choices(self):
        e=DispatchEnv(field(),scenario(Config()));self.assertEqual(e.observe().shape,(4,760))
        for _ in range(5):
            mask,expert,_,_=e.choices();self.assertTrue(mask[np.arange(4),expert].all());e.step(expert)
    def test_invalid_choices_fail_without_clamping(self):
        e=DispatchEnv(field(),scenario(Config()))
        for a in [[9]*4,[-1]*4,[.5]*4,[1,2]]:
            with self.assertRaises(ValueError):e.step(a)
if __name__=='__main__':unittest.main()
