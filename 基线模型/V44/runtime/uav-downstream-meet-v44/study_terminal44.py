"""Label real target-change opportunities with paired full-mission rollouts."""
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
from experiment44 import HERE,ROOT,material,load_scene,cloned_mission,finish,NoveltyBeam,read,save,sha,quota,s
from study_occupancy44 import world
from boundary_trial44 import stop_at_edge
from terminal_dispatch44 import build_terminal
from model import load_model

FRESH=HERE/"terminal-ordinary"
PARAMS=((8,0.),(24,.25),(64,.5))
SOURCES=(("uav-priority-confirm-v10",97151001),("uav-short-deadline-learning-v11",98751001))


def sources():
    return {f:sha(HERE/f) for f in ("terminal_dispatch44.py","study_terminal44.py","early_delivery44.py",
                "terminal-dispatch-protocol.md","boundary_trial44.py")}


def cases(indices,rosters):
    return [((str(ROOT/name),seed+i,"ordinary",i),r) for name,seed in SOURCES for i in indices for r in rosters]


def mission(c,r,panel):
    if panel=="fresh":
        s.D=FRESH
        s.scenes.D=FRESH
        s.scenes.make_inputs(c)
        s.make_history((c,r))
    field,pop=(load_scene if panel in ("mixed","fresh") else material)(c,r)
    return world(field,stop_at_edge(pop)[0])


def run(env,agent,initial_first=None):
    result=finish(env,agent,np.full(64,-1) if initial_first is None else initial_first.copy())
    d,f=np.asarray(result["delivery"]),np.asarray(result["first"])
    assert ((d<0)|((f>=0)&(d>=f+10)&(d<=env.deadline))).all()
    result["counts"]=[int(((d>=0)&(d<=t)).sum()) for t in (600,1200,2400)]
    return result


def sample(job):
    quota()
    c,r=job
    initial=mission(c,r,"train")
    observer=build_terminal()
    baseline=run(cloned_mission(initial),observer)
    events=observer.sparse.events
    # Replay the common prefix once, then clone only at logged opportunities.
    agent=build_terminal()
    env=cloned_mission(initial)
    first=np.full(64,-1)
    labels=[]
    for event in events:
        t=event["t"]
        while env.t<t:
            o=env.observation()
            first[(first<0)&o.visible]=env.t
            env.step(*agent.act(o))
        import copy
        alt=copy.deepcopy(agent)
        alt.sparse.force_at=t
        outcome=run(cloned_mission(env),alt,first)
        chosen=alt.sparse.events[-1]
        assert chosen["taken"] and alt.sparse.accepted==t
        assert all(chosen[k]==event[k] for k in ("x","base","proposed","t"))
        old,new=(np.asarray(row["delivery"]) for row in (baseline,outcome))
        assert np.array_equal(old[(old>=0)&(old<=t)],new[(old>=0)&(old<=t)])
        y=[int((new>=0).sum()-(old>=0).sum()),
           float((np.where(new>=0,2400-new,0)-np.where(old>=0,2400-old,0)).sum()/2400)]
        labels.append(dict(event=event,label=y,alternate=outcome))
    return dict(seed=c[1],roster=r,base=baseline,labels=labels)


def fit():
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(sample,cases(range(8),(0,1))))
    examples=[l for r in rows for l in r["labels"]]
    X=np.array([e["event"]["x"] for e in examples])
    Y=np.array([e["label"] for e in examples])
    # Equal weighting of each contributing roster, regardless of opportunity count.
    sw=np.array([1/len(r["labels"]) for r in rows for _ in r["labels"]])
    sw*=len(sw)/sum(sw)
    mean=np.average(X,axis=0,weights=sw)
    scale=np.maximum(.05,np.sqrt(np.average((X-mean)**2,axis=0,weights=sw)))
    Z=np.clip((X-mean)/scale,-4.,4.)
    zm=np.average(Z,axis=0,weights=sw)
    ym=np.average(Y,axis=0,weights=sw)
    A=Z-zm
    variants=[]
    for reg,margin in PARAMS:
        coef=np.linalg.solve(A.T@(A*sw[:,None])+reg*np.eye(A.shape[1]),A.T@((Y-ym)*sw[:,None]))
        variants.append(dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist(),bias=(ym-zm@coef).tolist(),min_gain=margin))
    save(HERE/"terminal-fit.json",dict(rows=rows,variants=variants,params=PARAMS,source=sources()))
    print("fit",len(rows),len(examples),np.sign(Y[:,0]).tolist().count(1),flush=True)


def unpack(fit):
    return {k:np.asarray(v) if k!="min_gain" else v for k,v in fit.items()}


def evaluate(job):
    panel,c,r,winner=job
    quota()
    if panel=="fresh":
        try:
            env=mission(c,r,panel)
        except (AssertionError,ValueError) as exc:
            return dict(seed=c[1],roster=r,family=c[2],input_failure=repr(exc))
    else:
        env=mission(c,r,panel)
    fit=read(HERE/"terminal-fit.json")
    assert fit["source"]==sources()
    opts=["base",0,1,2] if panel=="select" else ["base","previous","pure",winner]
    arms={}
    for opt in opts:
        if isinstance(opt,int):
            agent=build_terminal(unpack(fit["variants"][opt]))
        elif opt=="pure":
            agent=NoveltyBeam("service")
        elif opt=="previous":
            agent=load_model(fleet_policy="early_delivery")
        else:
            agent=load_model()
        arms[str(opt)]=run(cloned_mission(env),agent)
        if isinstance(opt,int):
            arms[str(opt)]["events"]=agent.sparse.events
            arms[str(opt)]["accepted"]=agent.sparse.accepted
            if agent.branch is agent.deep:
                assert all(arms[str(opt)][k]==arms["base"][k] for k in ("delivery","first","conflicts"))
    item=dict(seed=c[1],roster=r,family=c[2],arms=arms)
    if panel=="fresh":
        item.update(water=read(s.inputs(c)/"input.json")["water_sha"],population=sha(s.history(c,r)))
    return item


def panel(stage):
    fitfile=HERE/"terminal-fit.json"
    winner=None
    if stage=="select":
        jobs=cases(range(8,16),(0,))
    else:
        selection=read(HERE/"terminal-select.json")
        assert selection["fit"]==sha(fitfile)
        winner=selection["winner"]
        if stage=="repeat":
            jobs=cases(range(24,36),(1,))
        elif stage=="mixed":
            jobs=[(c,r) for side,r in (("a",0),("b",1)) for c in s.cases(side)]
        else:
            assert stage=="fresh"
            (FRESH/"population").mkdir(parents=True,exist_ok=True)
            jobs=[(("test",2026112200+i,"ordinary",i),0) for i in range(8)]
        assert not {c[1] for c,_ in jobs}&{r["seed"] for r in selection["rows"]}
    assert not {c[1] for c,_ in jobs}&{r["seed"] for r in read(fitfile)["rows"]}
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(evaluate,[(stage,c,r,winner) for c,r in jobs]))
    valid=[r for r in rows if "arms" in r]
    totals={m:np.sum([r["arms"][m]["counts"] for r in valid],axis=0).tolist() for m in valid[0]["arms"]} if valid else {}
    if stage=="select":
        winner=max(range(3),key=lambda i:(totals[str(i)][2],sum(totals[str(i)][:2]),-i))
    if stage in ("repeat","mixed"):
        archive=read(HERE/("speed-mixed.json" if stage=="mixed" else "speed-ordinary-repeat.json"))
        refs={(r["seed"],r.get("family","ordinary"),r["roster"]):r for r in archive["rows"]}
        for row in valid:
            ref=refs[row["seed"],row["family"],row["roster"]]["arms"]
            for name,old in (("base","base"),("pure","pure"),("previous","0" if stage=="mixed" else "new")):
                assert all(row["arms"][name][k]==ref[old][k] for k in ("first","delivery","conflicts"))
    save(HERE/f"terminal-{stage}.json",dict(rows=rows,totals=totals,winner=winner,fit=sha(fitfile)))
    print(stage,len(valid),totals,"winner",winner,flush=True)


if __name__=="__main__":
    for stage in sys.argv[1:]:
        fit() if stage=="fit" else panel(stage)
