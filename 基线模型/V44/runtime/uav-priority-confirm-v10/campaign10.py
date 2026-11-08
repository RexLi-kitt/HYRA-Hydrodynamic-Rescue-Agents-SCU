"""Pre-registered development and single locked holdout. Each worker has exactly one radius."""
from scenario10 import *
import hashlib
from collections import defaultdict
ARMS=("planning","learning")
def runtime(radius,name):
    from delivery import DeliveryMission
    from extra import WaitingNavigator
    from planner import H,V,Planner
    from predictor import PredictSearch
    import judge,navigation,extra,local_search,local_tracker,approach_policy,delivery
    assert radius==int(os.environ["HYRA_STUDY_RADIUS"])
    assert all(float(x)==radius for x in [judge.DELIVERY_RADIUS_M,navigation.RADIUS,extra.RADIUS,local_search.DELIVERY_RADIUS_M,
        local_tracker.DELIVERY_RADIUS_M,approach_policy.DELIVERY_RADIUS_M,delivery.DELIVERY_RADIUS_M])
    assert judge.INNER_SQ==local_search.INNER_SQ==(radius/2)**2
    if name=="planning":
        a=WaitingNavigator(np.zeros(49),np.ones((20,20)),"wait")
        a.lookup=np.repeat((H>=.01).astype(float),len(V))
    elif name=="learning":
        with np.load(V8/"discovery-head.npz") as w:a=PredictSearch("net_early",dict(w))
    else:raise ValueError(name)
    a.legacy.original=Planner("water",radius)
    return DeliveryMission,a
def regression(radius):
    env_class,_=runtime(radius,"planning")
    runtime(radius,"learning")
    field=np.broadcast_to(np.zeros((481,3,1,1),np.float32),(481,3,100,100))
    for distance,excursion,success in [(radius-.01,False,True),(radius+.01,False,False),(radius-.01,True,False)]:
        points=np.zeros((2401,1,2))+np.array([500+distance,500.])
        if excursion:points[7,0,0]=500+radius+.01
        pop=dict(fall=np.array([2401]),points=points,available=np.ones((2401,1),bool),velocity=np.zeros((481,1,2)))
        env=env_class(field,pop,(1,1140,2400));env.pos=np.array([[500.,500.],[100.,100.],[900.,100.],[100.,900.]])
        env.heading=np.tile([1.,0.],(4,1));env.sample()
        for _ in range(2):env.step(np.full(4,8),np.array([0,-1,-1,-1]))
        assert (env.supplied[0]==10)==success and env.radius==radius,(radius,distance,excursion,env.supplied)
    if radius==3:
        from base import reference
        f,p=scenes.load(("val",72110001,"ordinary",0),0)
        for name in ARMS:
            env=env_class(f,p,(48,1440,2400));a=runtime(3,name)[1]
            if name=="planning":old=reference()
            else:
                from finish8 import make
                old=make("net_early")
            for _ in range(480):
                o=env.observation();first=a.act(o);second=old.act(o)
                assert np.array_equal(first,second)
                env.step(*first)
    save(D/f"regression-r{radius}.json",dict(radius=radius,physical_boundary_and_interruption=True,ordinary_replay_steps_per_arm=480 if radius==3 else 0))
def key(setting):return "-".join(map(str,setting))
def result_path(c,rep,s,arm):
    return D/"rows"/f"{c[0]}-{c[1]}-{c[2]}-{rep}-{key(s)}-{arm}.json"
def evaluate(job):
    c,rep,settings,arms=job
    m=read(inputs(c)/"input.json");field=water(m["water"])
    with np.load(history(c,rep)) as archive:pop={k:archive[k] for k in ("points","available","velocity","fall")}
    for setting in settings:
        w,n,r=setting
        for arm in arms:
            out=result_path(c,rep,setting,arm)
            if out.exists():continue
            env_class,agent=runtime(r,arm)
            sub={k:v[:n] if k=="fall" else v[:,:n] for k,v in pop.items()}
            env=env_class(field,sub,(n,w,2400))
            while env.t<2400:
                env.step(*agent.act(env.observation()))
                if (env.supplied>=0).all() and (env.last>=0).all():break
            met=env.metrics();seen=env.seen_at>=0
            assert met["total"]==n and met["delivered"]==int((env.supplied>=0).sum()) and ((env.supplied<0)|seen).all()
            save(out,dict(panel=c[0],seed=c[1],family=c[2],roster=rep,setting=setting,method=arm,
                found=seen.tolist(),natural_falls=int((sub["fall"]<=2400).sum()),**met))
def schedule(panels,settings,arms):
    (D/"rows").mkdir(exist_ok=True)
    for r in sorted({s[2] for s in settings},reverse=True):
        group=[s for s in settings if s[2]==r]
        jobs=[(c,i,group,arms) for panel in panels for c in cases(panel) for i in range(3 if panel=="test" else 2)
              if any(not result_path(c,i,s,m).is_file() for s in group for m in arms)]
        print("start",panels,r,len(group),len(jobs),flush=True)
        if jobs:pool(evaluate,jobs,r)
def rows(panels,setting,arm):
    records=[]
    for panel in panels:
        for c in cases(panel):
            for i in range(3 if panel=="test" else 2):
                row=read(result_path(c,i,setting,arm))
                assert (row["panel"],row["seed"],row["family"],row["roster"])==(panel,c[1],c[2],i)
                assert row["setting"]==list(setting) and row["method"]==arm and len(row["delivery_s"])==setting[1]
                records.append(row)
    return records
def draws(panels,B,seed):
    rng=np.random.default_rng(seed);res={}
    for panel in panels:
        ordn=sum(c[2]=="ordinary" for c in cases(panel));stress=sum(c[2]=="continuous" for c in cases(panel))
        res[panel]=[rng.integers(ordn,size=(B,ordn)),rng.integers(stress,size=(B,stress))]
    return res
def block_summary(data,random_draws):
    stats=[];means=[]
    for panel,(od,pd) in random_draws.items():
        bs=[]
        for fam in FAMILIES:
            rr=sorted([r for r in data if r["panel"]==panel and r["family"]==fam],key=lambda r:(r["seed"],r["roster"]))
            size=3 if panel=="test" else 2
            values=[[r["RCR"],r["all"],r["delivered"],sum(t>=0 for t in r["delivery_s"][:48])] for r in rr]
            bs.append(np.array(values).reshape(-1,size,4).mean(1))
        means.append(.6*bs[0].mean(0)+.2*(bs[1]+bs[2]).mean(0))
        stats.append(.6*bs[0][od].mean(1)+.2*(bs[1]+bs[2])[pd].mean(1))
    return np.mean(stats,axis=0),np.array(means)
def estimate(setting,random_draws):
    data=rows(("a","b"),setting,"planning")
    dist,means=block_summary(data,random_draws);lo,hi=np.quantile(dist[:,0],[.025,.975])
    rate=sum(r["delivered"] for r in data)/sum(r["total"] for r in data)
    panelrates=means[:,0]
    passed=bool(rate>=.83 and min(panelrates)>=.8 and lo>=.8)
    return dict(setting=setting,people=sum(r["delivered"] for r in data),total=sum(r["total"] for r in data),
        rate=rate,panelrates=panelrates.tolist(),interval=[lo,hi],eligible=passed,all=sum(r["all"] for r in data))
def hashset():
    files=[D/n for n in ("design.json","scenario10.py","campaign10.py")]+[V8/n for n in ("predictor.py","discovery-head.npz","judge.py","delivery.py","risk-lock.json","local_route.py","approach_policy.py","local_search.py","local_tracker.py","navigation.py","extra.py","scenes.py","search8.py","motion.py")]
    from common import Drift
    from mission_env import MissionSupply
    import inspect,planner,supply
    files.extend(Path(inspect.getsourcefile(x)) for x in (Drift,MissionSupply,MissionSupply.__mro__[1],planner,supply))
    return {str(f):sha(f) for f in files}
def dev():
    quota()
    for r in (3,2,1):pool(regression,[r],r,workers=1)
    hashes=hashset()
    save(D/"registered-hashes.json",hashes)
    grid=[(w,n,3) for w in WINDOWS for n in COUNTS]
    schedule(("a","b"),grid,["planning"])
    dd=draws(("a","b"),4000,115710)
    summary=[estimate(s,dd) for s in grid]
    eligible=sorted([e["setting"] for e in summary if e["eligible"]],key=lambda s:(s[0],-s[1]))
    target=eligible[0] if eligible else ORIGINAL
    secondary=[(target[0],target[1],r) for r in (2,1)]
    schedule(("a","b"),secondary,["planning"])
    extra=[estimate(s,dd) for s in secondary]
    choices=[s["setting"] for s in extra if s["eligible"]]+[target]
    primary=min(choices,key=lambda s:s[2]) if eligible else ORIGINAL
    assert hashes==hashset()
    save(D/"lock.json",dict(selected=primary,original=ORIGINAL,grid=summary+extra,no_eligible_3m=not bool(eligible),hashes=hashes))
    print("LOCKED",primary,[(e["setting"],round(e["rate"],4)) for e in summary+extra if e["setting"][:2]==target[:2]],flush=True)
def test():
    lock=read(D/"lock.json");assert lock["hashes"]==hashset()
    prepare("test")
    schedule(("test",),list(dict.fromkeys([ORIGINAL,tuple(lock["selected"])])),list(ARMS))
if __name__=="__main__":
    if sys.argv[1]=="dev":dev()
    elif sys.argv[1]=="test":test()
