"""Nested rosters; water, fall hazard, drift, obstacles and per-person randomness inherited from V8."""
from pathlib import Path
import os,sys
from concurrent.futures import ProcessPoolExecutor
D=Path(__file__).resolve().parent
ROOT=D.parent
V8=ROOT/"uav-pressure-search-v8"
OLD=ROOT/"uav-pressure-search-v9"/"time-confirm"
sys.path.insert(0,str(V8))
import scenes
from common import read,save,sha,water,quota,np,Config,Drift,sample_water,SimpleNamespace
sys.path.insert(0,str(V8))
from itertools import product
FAMILIES=("ordinary","continuous","rewet")
WINDOWS=(480,720,960,1200,1440)
COUNTS=(48,56,64,72,80)
ORIGINAL=(1440,48,3)
def cases(panel):
    a,b,n,m={"a":(72110001,88520001,12,4),"b":(95851001,95861001,12,4),"test":(97151001,97161001,60,20)}[panel]
    return [(panel,a+i,"ordinary",i) for i in range(n)]+[(panel,b+i,f,i) for i in range(m) for f in FAMILIES[1:]]
def inputs(c):
    panel,seed,f,i=c
    root={"a":ROOT/"uav-revised-delivery-training-v5","b":OLD,"test":D}[panel]
    return root/"scenes"/f"{seed}-{f}"
def history(c,rep):
    return D/"population"/f"{c[1]}-{c[2]}-{rep}.npz"
def water_job(c):
    scenes.D=D
    scenes.make_inputs(c)
def draw(c,rep):
    path=inputs(c)
    with np.load(path/"eligible.npz") as a:ids=a["ids"]
    assert len(ids)>=80,("eligible population smaller than required",c,len(ids))
    pos,clock=map(np.random.default_rng,np.random.SeedSequence([91821,c[1],rep]).spawn(2))
    original=pos.choice(ids,48,replace=False)
    selected=np.r_[original,pos.choice(np.setdiff1d(ids,original),32,replace=False)]
    clocks=clock.exponential(size=80)
    return selected,clocks
def simulate(field,ids,fall,seed,rep,traps):
    count=len(ids)
    yy,xx=np.unravel_index(ids,(100,100))
    pos=np.column_stack([xx,yy])*10.+5
    cfg=Config(duration=2400,window=1320,people=count,sensor=20.,trigger_depth=2.,trigger_speed=1.)
    salt=int(np.random.SeedSequence([7812,seed,rep]).generate_state(1)[0])
    motion=SimpleNamespace(cfg=cfg,water=field,person=pos.copy(),traps=traps,status=np.zeros(count,int),
        stuck=np.zeros(count,bool),stop_kind=np.zeros(count,int),refloat_counts=np.zeros(count,int),
        scenario={"seed":salt,"events":[{} for _ in range(count)]},refloat_enabled=True,events=[])
    motion.refloat_draw=lambda j,now:Drift.refloat_draw(motion,j,now)
    points=np.empty((2401,count,2));avail=np.empty((2401,count),bool);velocity=np.zeros((481,count,2))
    for t in range(2401):
        motion.status[(motion.status==0)&(fall<=t)]=1
        points[t]=motion.person;avail[t]=motion.status!=4
        if t%5==0:
            active=(motion.status==1)&~motion.stuck
            if active.any():
                values=sample_water(field,t,motion.person[active])
                velocity[t//5,active]=np.where(values[:,:1]>=.01,values[:,1:],0.)
        if t<2400:Drift.move_people(motion,t)
    assert np.isfinite(points).all() and np.isfinite(velocity).all() and avail[0].all()
    return dict(points=points,available=avail,velocity=velocity,fall=fall)
def make_history(job):
    c,rep=job;out=history(c,rep)
    if out.exists():return
    m=read(inputs(c)/"input.json")
    f=water(m["water"]);selected,clocks=draw(c,rep)
    yy,xx=np.unravel_index(selected,(100,100))
    z=f[:,:,yy,xx].astype(float);delta=z[1:]-z[:-1]
    k=read(V8/"risk-lock.json")["k"]
    hazard=np.empty((480,5,80))
    for i in range(5):
        h,u,v=np.moveaxis(z[:-1]+((i+.5)/5)*delta,1,0)
        hazard[:,i]=k*scenes.shape(h,np.hypot(u,v))
    hit=hazard.reshape(2400,80).cumsum(0)>=clocks
    fall=np.where(hit.any(0),hit.argmax(0)+1,2401)
    pop=simulate(f,selected,fall,c[1],rep,np.array(m["traps"]))
    if c[0]!="test":
        old=read(inputs(c)/"manifest.json")["layouts"][rep]
        assert np.array_equal(selected[:48],old["xy"]) and np.array_equal(clocks[:48],old["clock"])
        with np.load(inputs(c)/f"hist-{rep}.npz") as a:
            for key,v in pop.items():
                slice48=v[:48] if key=="fall" else v[:,:48]
                assert np.array_equal(slice48,a[key]),(c,rep,key)
    temp=out.with_suffix(".tmp.npz")
    np.savez_compressed(temp,**pop,ids=selected,clock=clocks)
    temp.replace(out)
def pool(fn,jobs,radius=3,workers=8):
    os.environ["HYRA_STUDY_RADIUS"]=str(radius)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for start in range(0,len(jobs),workers):
            quota();list(executor.map(fn,jobs[start:start+workers]))
            if start%(workers*4)==0 or start+workers>=len(jobs):
                print(fn.__name__,min(start+workers,len(jobs)),len(jobs),flush=True)
def prepare(panel):
    cs=cases(panel);nrep=3 if panel=="test" else 2
    (D/"population").mkdir(exist_ok=True)
    if panel=="test":
        assert (D/"lock.json").is_file()
        parents={c[1] for c in cs}
        for root in [OLD,OLD.parent,V8,ROOT/"uav-delivery-radius-study-v7",ROOT/"uav-delivery-radius1-v6",ROOT/"uav-revised-delivery-training-v5"]:
            assert not parents.intersection(int(path.name.split("-")[0]) for path in (root/"scenes").iterdir() if path.is_dir())
        pool(water_job,[c for c in cs if not (inputs(c)/"input.json").exists()])
    pool(make_history,[(c,rep) for c in cs for rep in range(nrep) if not history(c,rep).exists()])
    records=[]
    for c in cs:
        m=read(inputs(c)/"input.json");assert m["water_sha"]==sha(m["water"])
        if c[2]=="continuous":
            other=(panel,c[1],"rewet",c[3]);partner=read(inputs(other)/"input.json")
            assert m["traps"]==partner["traps"] and m["direction"]==partner["direction"]==90*(c[3]%4)
            for rep in range(nrep):
                with np.load(history(c,rep)) as a,np.load(history(other,rep)) as b:
                    assert np.array_equal(a["ids"],b["ids"]) and np.array_equal(a["clock"],b["clock"])
        records.append(dict(case=c[1:],water=m["water_sha"],histories=[sha(history(c,i)) for i in range(nrep)]))
    save(D/f"inputs-{panel}.json",records)
