"""Fit short-horizon delivery advantages and freeze the selector before full-map tests."""
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import HERE,ROOT,material,load_scene,cloned_mission,finish,NoveltyBeam,read,save,sha,quota,s
from model import load_model
from early_delivery44 import QuickDispatch,QuickSearch
from boundary_trial44 import stop_at_edge
from study_occupancy44 import world

SETTINGS=((30,12),(75,6),(120,24))
TIMES=(60,180,300,420,660,900,1260,1620)
NEW=HERE/"speed-fresh"


def configured(fit=None):
    agent=load_model()
    parts=[]
    for key in ("sparse","deep"):
        parts.append(None if fit is None else {k:np.array(v) for k,v in fit[key].items()})
    agent.sparse=QuickDispatch(parts[0])
    agent.sparse.cohorts.personal_window_seconds=720
    agent.deep=QuickSearch(parts[1],read(HERE/"occupancy-fit.json")["weights"])
    agent.deep.belief.personal_window_seconds=720
    return agent


def cases(indices):
    return [(str(ROOT/d),start+off+i,f,i) for d,start in (
        ("uav-priority-confirm-v10",97151001),("uav-short-deadline-learning-v11",98751001))
        for i in indices for f,off in (("ordinary",0),("continuous",10000),("rewet",10000))]


def run_to(env,agent,t):
    while env.t<t:
        env.step(*agent.act(env.observation()))


def training_case(c):
    quota()
    field,pop=material(c,0)
    env=world(field,stop_at_edge(pop)[0])
    agent=configured()
    rows=[]
    for t in TIMES:
        run_to(env,agent,t)
        xs,ds=[],[]
        for arm in (0,1,2):
            fork,policy=cloned_mission(env),copy.deepcopy(agent)
            policy.branch.force=arm
            run_to(fork,policy,t+120)
            decision=policy.branch.log[-1]
            assert decision["arm"]==arm and decision["t"]==t
            xs.append(decision["x"])
            policy.branch.force=0
            run_to(fork,policy,t+180)
            ds.append((fork.supplied[(fork.supplied>t)&(fork.supplied<=t+180)]-t).tolist())
        assert np.allclose(xs,xs[0])
        rows.append(dict(t=t,x=xs[0],delivery_delays=ds))
    return dict(seed=c[1],family=c[2],branch="deep" if agent.branch is agent.deep else "sparse",rows=rows)


def train():
    with ProcessPoolExecutor(4) as pool:
        series=list(pool.map(training_case,cases(range(4))))
    variants=[]
    for tau,reg in SETTINGS:
        fitted={}
        for branch in ("deep","sparse"):
            samples=[v for r in series if r["branch"]==branch for v in r["rows"]]
            x=np.array([v["x"] for v in samples])
            y=np.array([[sum(1+np.exp(-np.asarray(a,dtype=float)/tau)) for a in v["delivery_delays"]] for v in samples])
            y=y[:,1:]-y[:,:1]
            mu,sd=x.mean(0),np.maximum(.05,x.std(0))
            z=np.clip((x-mu)/sd,-4,4)
            zm=z.mean(0)
            coef=np.linalg.solve((z-zm).T@(z-zm)+reg*np.eye(z.shape[1]),(z-zm).T@(y-y.mean(0)))
            fitted[branch]=dict(mean=mu.tolist(),scale=sd.tolist(),bias=(y.mean(0)-zm@coef).tolist(),coef=coef.tolist())
        variants.append(fitted)
    save(HERE/"speed-fit.json",dict(settings=SETTINGS,variants=variants,series=series,code=hashes()))
    print("trained",len(series),sum(len(r["rows"]) for r in series),flush=True)


def hashes():
    return {f:sha(HERE/f) for f in ("early_delivery44.py","study_speed44.py","speed-policy-protocol.md","occupancy-fit.json")}


def fly(job):
    panel,c,r,selected=job
    quota()
    if panel=="fresh":
        s.D=NEW
        s.scenes.D=NEW
        s.scenes.make_inputs(c)
        s.make_history((c,r))
    field,pop=(material if panel=="select" else load_scene)(c,r)
    env=world(field,stop_at_edge(pop)[0])
    fitted=read(HERE/"speed-fit.json")
    assert fitted["code"]==hashes()
    modes=["base",*range(3)] if panel=="select" else ["base","pure","density",selected]
    arms={}
    for name in modes:
        if isinstance(name,int):
            a=configured(fitted["variants"][name])
        elif name=="pure":
            a=NoveltyBeam("service")
        else:
            a=load_model(edge_search="occupancy" if name=="density" else None)
        out=finish(cloned_mission(env),a,np.full(64,-1))
        d,f=map(np.asarray,(out["delivery"],out["first"]))
        assert ((d<0)|((f>=0)&(d>=f+10)&(d<=env.deadline))).all()
        out["counts"]=[int(((d>=0)&(d<=t)).sum()) for t in (600,1200,2400)]
        if isinstance(name,int):
            out["choices"]=[entry["arm"] for entry in a.branch.log]
            out["dispatch_changes"]=a.sparse.dispatch_changes
        arms[str(name)]=out
    return dict(seed=c[1],family=c[2],index=c[3],roster=r,arms=arms)


def totals(rows):
    return {arm:np.sum([r["arms"][arm]["counts"] for r in rows],axis=0).tolist() for arm in rows[0]["arms"]}


def evaluate(panel):
    fit=read(HERE/"speed-fit.json")
    if panel=="select":
        work=[(panel,c,0,None) for c in cases(range(4,8))]
    else:
        best=read(HERE/"speed-select.json")
        assert best["fit"]==sha(HERE/"speed-fit.json")
        winner=best["winner"]
        if panel=="fresh":
            (NEW/"population").mkdir(parents=True,exist_ok=True)
            work=[(panel,("test",2026111100+i,f,i),0,winner) for i in range(8) for f in ("continuous","rewet")]
        else:
            assert panel=="mixed"
            work=[(panel,c,r,winner) for side,r in (("a",0),("b",1)) for c in s.cases(side)]
        assert not {m[1][1] for m in work}&{r["seed"] for r in best["rows"]}
    assert not {m[1][1] for m in work}&{r["seed"] for r in fit["series"]}
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(fly,work))
    groups=totals(rows)
    result=dict(rows=rows,groups=groups,fit=sha(HERE/"speed-fit.json"))
    if panel=="select":
        delta=np.array([groups[str(i)] for i in range(3)])-groups["base"]
        result["winner"]=max(range(3),key=lambda i:(bool(np.all(delta[i]>=0)),sum(delta[i,:2]),delta[i,2],-i))
    if panel=="mixed":
        old=read(HERE/"occupancy-mixed-result.json")
        refs={(r["seed"],r["family"],r["roster"]):r for r in old["rows"]}
        for r in rows:
            ref=refs[r["seed"],r["family"],r["roster"]]
            assert all(r["arms"][a][k]==ref["arms"][b][k] for a,b in (("base","base"),("pure","pure"),("density","learned")) for k in ("delivery","first","conflicts"))
    save(HERE/f"speed-{panel}.json",result)
    print(panel,groups,result.get("winner"),flush=True)


if __name__=="__main__":
    for stage in sys.argv[1:]:
        train() if stage=="train" else evaluate(stage)
