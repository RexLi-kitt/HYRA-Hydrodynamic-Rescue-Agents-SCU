import copy,sys,unittest
from pathlib import Path
from dataclasses import replace,asdict
import numpy as np
from variable_environment import Config,Env,make_scenario
from variable_dispatch import DispatchEnv
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'flood-drop-dispatch-v1'))
from dispatch_env import DispatchEnv as LegacyDispatch
from environment import Env as LegacyEnv,Config as LegacyConfig

def water(u=0):
    a=np.zeros((3,100,100));a[0]=.12;a[1]=u
    return np.broadcast_to(a,(301,3,100,100))
def case(cfg,first=0,xy=(5,5)):
    return dict(config=asdict(cfg),traps=[],events=[dict(fall=first,deadline=first+cfg.window,xy=list(xy))]+
                [dict(fall=cfg.duration-cfg.window,deadline=cfg.duration,xy=[400+i%20*10,400+i//20*10]) for i in range(cfg.people-1)])
def complete(env,dispatch=False):
    while env.t<env.cfg.duration:
        a=env.choices()[1] if dispatch else env.control()[1]
        _,_,done,_=env.step(a)
        if done:break
    return env.metrics()

class Tests(unittest.TestCase):
    def test_legacy_full_trajectory_equivalence(self):
        cfg=LegacyConfig(duration=120,window=35);s=case(Config(duration=120,window=35))
        s['config']=asdict(cfg)
        for old_cls,new_cls,dispatch in [(LegacyEnv,Env,False),(LegacyDispatch,DispatchEnv,True)]:
            a=old_cls(water(.1),copy.deepcopy(s));b=new_cls(water(.1),copy.deepcopy(s))
            while a.t<cfg.duration:
                aa=a.choices()[1] if dispatch else a.control()[1];bb=b.choices()[1] if dispatch else b.control()[1]
                np.testing.assert_array_equal(a.observe(),b.observe());np.testing.assert_array_equal(aa,bb)
                oa,ra,da,_=a.step(aa);ob,rb,db,_=b.step(bb)
                self.assertEqual(ra,rb);self.assertEqual(da,db)
                for k in ('pos','person','status','progress','delivered_at'):np.testing.assert_array_equal(getattr(a,k),getattr(b,k))
                if da:break
    def test_nonmultiple_duration_and_fall_activation(self):
        cfg=Config(duration=23,window=7,people=1,drop_seconds=7,step=5)
        e=DispatchEnv(water(1),case(cfg,first=3,xy=(500,500)))
        e.step(e.choices()[1]);self.assertEqual(e.t,5);self.assertEqual(e.person[0,0],502)
        while e.t<23:e.step(e.choices()[1])
        self.assertEqual(e.t,23)
    def test_one_second_delivery_boundary(self):
        cfg=Config(duration=21,window=11,people=1,step=1,drop_seconds=11)
        e=DispatchEnv(water(.2),case(cfg))
        for _ in range(10):e.step(e.choices()[1])
        self.assertEqual(e.status[0],1)
        e.step(e.choices()[1]);self.assertEqual(e.status[0],2);self.assertEqual(e.delivered_at[0],11)
    def test_populations_teams_shapes_and_conservation(self):
        for people,uavs in [(1,1),(21,2),(41,3),(80,6),(81,8)]:
            cfg=Config(people=people,uavs=uavs,duration=23,window=11,drop_seconds=10)
            e=DispatchEnv(water(),case(cfg));self.assertEqual(e.observe().shape,(uavs,500+8*people+2*uavs+2+90))
            self.assertEqual(e.choices()[0].shape,(uavs,9));m=complete(e,True)
            self.assertEqual(m['supplied']+m['expired']+m['outside'],people)
    def test_95_percent_integer_boundary(self):
        for n,k in [(20,19),(21,20),(40,38),(41,39),(60,57),(80,76)]:
            cfg=Config(people=n);e=Env(water(),case(cfg));e.status[:]=3;e.status[:k-1]=2
            self.assertEqual(e.metrics()['threshold95'],k);self.assertFalse(e.metrics()['scene95_success'])
            e.status[k-1]=2;self.assertTrue(e.metrics()['scene95_success'])
    def test_nested_event_pool_and_integer_times(self):
        c=Config(duration=73,window=11,drop_seconds=10,people=20)
        a=make_scenario(water(),271,c,pool_size=80);b=make_scenario(water(),271,replace(c,people=41),pool_size=80)
        aa={e['exposure_id']:e for e in a['events']};bb={e['exposure_id']:e for e in b['events']}
        self.assertTrue(aa.keys()<=bb.keys());self.assertTrue(all(e==bb[k] for k,e in aa.items()))
        self.assertEqual(a['traps'],b['traps']);self.assertTrue(any(e['fall']%5 for e in b['events']))
        self.assertTrue(all(0<=e['fall']<e['deadline']<=73 for e in b['events']))
    def test_no_hidden_person_or_future_observation(self):
        cfg=Config(people=41,uavs=3);e=DispatchEnv(water(),case(cfg,first=100,xy=(500,500)))
        obs=e.observe().copy();m,x,_,f=e.choices();e.person[:]=700;e.deadline[:]=0;e.fall[:]=0;e.water=water(20)
        np.testing.assert_array_equal(obs,e.observe());mm,xx,_,ff=e.choices()
        for a,b in [(m,mm),(x,xx),(f,ff)]:np.testing.assert_array_equal(a,b)
    def test_sensor_interpolates_current_second(self):
        w=np.zeros((301,3,100,100));w[:,0]=.12;w[1,1]=5
        cfg=Config(people=1,step=1);e=Env(w,case(cfg,first=100));e.step(e.control()[1])
        self.assertTrue(np.all(e.measured[1,e.last==1]==1))
    def test_invalid_time_and_geometry_rejected(self):
        for kw in [dict(duration=1501),dict(people=0),dict(uavs=0),dict(speed=float('nan')),dict(window=361.0),dict(step=0),dict(drop_seconds=361)]:
            with self.assertRaises(ValueError):Config(**kw)
        cfg=Config(people=1);s=case(cfg);s['events'][0]['deadline']=1501
        with self.assertRaises(ValueError):Env(water(),s)
    def test_registered_profiles_and_changing_denominators(self):
        from screen import profiles,summarize
        ps=profiles();self.assertEqual(len(ps),len({p['id'] for p in ps}))
        self.assertEqual({p['config']['people'] for p in ps},{20,40,41,60,80})
        for p in ps:Config(**p['config'])
        rows=[dict(people_total=20,supplied=19,scene_success=False,scene95_success=True,expired_unseen=1,conflicts=0),
              dict(people_total=40,supplied=38,scene_success=False,scene95_success=True,expired_unseen=2,conflicts=0)]
        s=summarize(rows);self.assertEqual(s['people_supplied'],57);self.assertEqual(s['people_total'],60)
        self.assertEqual(s['people_completion'],.95);self.assertEqual(s['at_least95_successes'],2)
if __name__=='__main__':unittest.main()
