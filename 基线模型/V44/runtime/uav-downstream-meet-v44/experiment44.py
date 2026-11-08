from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import copy
import sys
import numpy as np
from rendezvous44 import HERE,ROOT,Routed,NoveltyBeam,runtime,finished,s,material,load_scene,cloned_mission,read,save,sha,quota
from tracker44 import Downstream

V39=ROOT/"uav-visibility-audit-v39";V43=ROOT/"uav-prefall-zones-v43"
MODES=("base","ordinary_meet","extended","pure")


def code_key():
    return {p.name:sha(p) for p in (HERE/"protocol.md",HERE/"experiment44.py",HERE/"tracker44.py",HERE/"rendezvous44.py",
        ROOT/"uav-combined-close-v25"/"combined.py",ROOT/"uav-moving-intercept-v21"/"intercept.py",
        ROOT/"uav-pressure-search-v8"/"delivery.py",ROOT/"uav-window-rollout-v19"/"weights.npz")}


def new_agent(mode="continuous_search"):
    """Build the current V44 policy. Explicit historical arms keep their original search."""
    agent = Routed()
    if mode in ("continuous_search", "dry_search"):
        from continuous44 import Continuous
        agent.deep = Continuous(dry_service=mode == "dry_search")
    elif mode == "learned_route":
        from route_value44 import RouteValue
        agent.deep = RouteValue()
    elif mode in ("base", "ordinary_meet", "extended"):
        agent.deep = Downstream(mode)
    else:
        raise ValueError(f"Unknown V44 agent mode: {mode}")
    return agent


def historical_record_matches(record_key, current_key):
    # The frozen experiments explicitly pass their arm names. Old records also
    # have to match every other input, including the judge-facing tracker and weights.
    archived = dict(current_key)
    archived["experiment44.py"] = "6323f7cef6f78321260264ff315e1c9e3ad952fd3766b3e10a7dd2e0ddf4ad14"
    return record_key == current_key or record_key == archived


def jobs(phase):
    if phase=="screen":
        return [((str(ROOT/d),start+k,f,k),0) for d,start in (("uav-priority-confirm-v10",97161001),
             ("uav-short-deadline-learning-v11",98761001)) for k in range(4,8) for f in ("continuous","rewet")]
    return [(c,r) for c in s.cases("a")+s.cases("b") if phase=="panel" or c[2]!="ordinary" for r in (0,1)]


def finish(env,agent,first):
    while not finished(env):
        o=env.observation();first[(first<0)&o.visible]=env.t
        env.step(*agent.act(o))
    first[(first<0)&env.visible]=env.t
    assert env.t==2400 and env.metrics()["discovered"]==int((first>=0).sum())
    return dict(delivery=env.supplied.tolist(),first=first.tolist(),conflicts=int(env.conflicts))


def matched(c,r,name):
    field,pop=load_scene(c,r);cls,_=runtime(1,"planning")
    env=cls(field,pop,(64,720,2400));agent=new_agent("base")
    with np.load(V39/"traces"/(name+".npz")) as z:
        first=z["first"];delivery=z["supplied"];moves=z["actions"];claims=z["claims"];visible=z["visible"]
        assert sha(s.history(c,r))==str(z["history_sha"])
    old_first=np.full(64,-1);records=[]
    for k in range(480):
        o=env.observation();old_first[(old_first<0)&o.visible]=o.t
        assert np.array_equal(o.visible,visible[k])
        focal=[]
        for j in np.flatnonzero(first==o.t):
            assert o.visible[j]
            v=o.seen_v[j];speed=np.linalg.norm(v)
            if speed<=.05:
                continue
            delta=o.seen_xy[j]-o.pos
            see=((delta*delta).sum(1)<=400+1e-9)&((delta*o.heading).sum(1)>=-1e-9)
            cross=np.abs(delta[:,0]*v[1]-delta[:,1]*v[0])/speed
            if np.min(cross[see])>1+1e-9:
                focal.append(int(j))
        before=copy.deepcopy(agent) if focal else None
        orig_action,orig_claim=agent.act(o)
        assert np.array_equal(orig_action,moves[k]) and np.array_equal(orig_claim,claims[k])
        for j in focal:
            arms={};cache={}
            for mode in MODES[1:3]:
                a=copy.deepcopy(before);a.deep.method44=mode;a.deep.only44=j
                m,cl=a.act(env.observation())
                events=a.deep.events44;assert len(events)<=1
                launched=bool(events and events[0]["used"])
                if not launched:
                    assert np.array_equal(m,orig_action) and np.array_equal(cl,orig_claim)
                    result=dict(delivery=delivery.tolist(),first=first.tolist(),shortcut="identical_control")
                else:
                    key=(tuple(m),tuple(cl),events[0]["i"],tuple(events[0]["codes"]))
                    if key in cache:
                        result,prior_event=cache[key]
                        events[0].update(end=prior_event["end"],commands=prior_event["commands"])
                    else:
                        fork=cloned_mission(env);fork.step(m,cl)
                        result=finish(fork,a,old_first.copy());cache[key]=(result,events[0])
                arms[mode]=dict(**result,event=events[0] if events else None)
            records.append(dict(j=j,t=o.t,arms=arms))
        env.step(orig_action,orig_claim)
    old_first[(old_first<0)&env.visible]=env.t
    assert np.array_equal(env.supplied,delivery) and np.array_equal(old_first,first)
    return dict(records=records,base=dict(delivery=delivery.tolist(),first=first.tolist()))


def full(c,r,name,phase):
    field,pop=material(c,r) if phase=="screen" else load_scene(c,r)
    cls,_=runtime(1,"planning");arms={}
    if phase=="screen":
        archived=read(ROOT/"uav-search-detour-v40"/"screen"/f"{c[1]}-{c[2]}.json")["groups"]
        reference=archived["base"]["delivery"];pure=archived["pure"]["delivery"]
    else:
        reference=read(ROOT/"uav-combined-close-v25"/"dev"/(name+".json"))["metrics"]["delivery_s"]
        pure=read(ROOT/"uav-rescue-gate-v16"/"dev"/(name+".json"))["models"]["service"]["delivery_s"]
    for mode in MODES:
        a=NoveltyBeam("service") if mode=="pure" else new_agent(mode)
        env=cls(field,pop,(64,720,2400));assert env.radius==1
        row=finish(env,a,np.full(64,-1));row["events"]=[] if mode=="pure" else a.deep.events44
        if mode=="base":
            assert row["delivery"]==reference
        if mode=="pure":
            assert row["delivery"]==pure
        arms[mode]=row
    return dict(arms=arms)


def run(job):
    phase,c,r=job;name=f"{c[1]}-{c[2]}-{r}";path=HERE/phase/(name+".json");key=code_key()
    if path.is_file():
        assert historical_record_matches(read(path)["code"],key)
        return
    payload=matched(c,r,name) if phase=="branch" else full(c,r,name,phase)
    save(path,dict(name=name,seed=c[1],family=c[2],roster=r,code=key,**payload))


def branch_summary(rows):
    counts={}
    for mode in ("base",*MODES[1:3]):
        g=Counter();parents={}
        for case in rows:
            baseline=np.asarray(case["base"]["delivery"])
            for row in case["records"]:
                j,t=row["j"],row["t"];g["states"]+=1
                arm=case["base"] if mode=="base" else row["arms"][mode];event=arm.get("event")
                outcome=np.asarray(arm["delivery"])
                g["assigned_eligible"]+=int(event is not None)
                g["offered"]+=int(bool(event and event["offered"]))
                g["launched"]+=int(bool(event and event["used"]))
                g["target_60s"]+=int(t<=outcome[j]<=t+60);g["target_final"]+=int(outcome[j]>=0)
                g["new_other"]+=int(((outcome>=0)&(baseline<0)).sum())-int(outcome[j]>=0 and baseline[j]<0)
                g["lost_other"]+=int(((outcome<0)&(baseline>=0)).sum())-int(outcome[j]<0 and baseline[j]>=0)
                change=int((outcome>=0).sum()-(baseline>=0).sum())
                g["sum_fork_team_deltas"]+=change
                key=str(case["seed"]);parents[key]=parents.get(key,0)+change
        assert g["states"]==190
        counts[mode]=dict(**g,by_parent=parents)
    assert counts["base"]["target_60s"]==28 and counts["base"]["target_final"]==75
    return dict(summary=counts,unit="separate single-target forks; never pool their delivery counts as a mission rate")


def full_summary(rows):
    groups={};pairs={};qualified=[]
    for f in ["all",*sorted({r["family"] for r in rows})]:
        rs=[r for r in rows if f=="all" or r["family"]==f];groups[f]={}
        for m in MODES:
            g=Counter()
            for r in rs:
                arm=r["arms"][m];d=np.array(arm["delivery"]);events=arm["events"]
                g.update(people=len(d),delivered=int((d>=0).sum()),found=sum(t>=0 for t in arm["first"]),
                         launched=sum(e["used"] for e in events),offered=sum(e["offered"] for e in events),
                         conflicts=arm["conflicts"],eligible=len(events))
            groups[f][m]=dict(g)
    for m in MODES[1:3]:
        pairs[m]={}
        for control in (k for k in MODES if k!=m):
            parents={};gained=lost=0
            for r in rows:
                d=np.array(r["arms"][m]["delivery"])>=0;old=np.array(r["arms"][control]["delivery"])>=0
                gained+=int((d&~old).sum());lost+=int((~d&old).sum());key=(r["seed"],r["family"]=="ordinary")
                parents[key]=parents.get(key,0)+int(d.sum()-old.sum())
            rng=np.random.default_rng(4406)
            strata=[[v for (s,o),v in parents.items() if o==b] for b in (True,False)]
            diff=sum(rng.choice(g,(20000,len(g))).sum(1) for g in strata if g)
            pairs[m][control]=dict(gained=gained,lost=lost,ci_pp=np.quantile(100*diff/(64*len(rows)),[.025,.975]).tolist(),
                                     parents={str(k):v for (k,o),v in parents.items()})
        pressure={str(r["seed"]) for r in rows if r["family"]!="ordinary"}
        if (all(groups[f][m]["delivered"]>groups[f]["base"]["delivered"] for f in ("all","continuous","rewet"))
            and sum(pairs[m]["base"]["parents"][k]>0 for k in pressure)*2>=len(pressure)
            and (m!="extended" or groups["all"][m]["delivered"]>groups["all"]["ordinary_meet"]["delivered"])):
            qualified.append(m)
    return dict(groups=groups,pairs=pairs,qualified=qualified,selected=max(qualified,key=lambda m:groups["all"][m]["delivered"]) if qualified else None)


if __name__=="__main__":
    phase=sys.argv[1];assert phase in ("branch","screen","panel")
    if phase=="screen":
        assert read(HERE/"branch-result.json")["summary"]["extended"]["launched"] or read(HERE/"branch-result.json")["summary"]["ordinary_meet"]["launched"]
    if phase=="panel":
        assert read(HERE/"screen-result.json")["selected"]
    dest=HERE/phase;dest.mkdir(exist_ok=True);items=jobs(phase)
    for k in range(0,len(items),4):
        quota()
        with ProcessPoolExecutor(max_workers=4) as pool:
            list(pool.map(run,[(phase,c,r) for c,r in items[k:k+4]]))
        print(phase,k+4,len(items),flush=True)
    docs=[read(p) for p in sorted(dest.glob("*.json"))]
    key = code_key()
    assert len(docs)==len(items) and all(historical_record_matches(d["code"],key) for d in docs)
    z=branch_summary(docs) if phase=="branch" else full_summary(docs)
    save(HERE/(phase+"-result.json"),z)
    print(z["summary"] if phase=="branch" else (z["groups"]["all"],z["selected"]),flush=True)
