"""Collect full-mission counterfactuals, fit one model and replay disjoint development maps."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
import sys
import numpy as np
from experiment44 import HERE, ROOT, Routed, jobs, material, load_scene, finish, runtime, read, save, sha, quota
from route_value44 import FEATURES, SLOTS, WEIGHTS, RouteValue


def mission(field, people, mode, slot=None):
    judge, _ = runtime(1,"planning")
    env = judge(field,people,(64,720,2400))
    assert env.radius == 1
    agent = Routed()
    agent.deep = RouteValue(mode, slot)
    row = finish(env,agent,np.full(64,-1))
    row["delivered"] = sum(t >= 0 for t in row["delivery"])
    row["decisions"] = agent.deep.decisions
    return row


def trial(job):
    stage, case, roster = job
    quota()
    _, seed, family, _ = case
    f,p = (load_scene if stage=="panel" else material)(case,roster)
    base = mission(f,p,"base")
    result = dict(seed=seed,family=family,roster=roster,base=base)
    if stage=="fit":
        result["forks"] = []
        for slot, event in base["decisions"].items():
            fork = mission(f,p,"fork",slot)
            action = fork["decisions"][slot]
            assert action["take"] and sum(e["take"] for e in fork["decisions"].values()) == 1
            assert all(event[k]==action[k] for k in ("t","drone","x","first"))
            result["forks"].append(dict(slot=slot, gain=fork["delivered"]-base["delivered"], row=fork))
    else:
        if stage=="screen":
            archived=read(HERE/"screen"/f"{seed}-{family}-{roster}.json")["arms"]["pure"]
            old=next(r["revised"] for r in read(HERE/"search-integration.json")["rows"]
                     if (r["seed"],r["family"],r["roster"])==(seed,family,roster))
            assert all(old[k]==base[k] for k in ("first","delivery","conflicts"))
            pure=archived["delivery"]
        else:
            pure=read(ROOT/"uav-rescue-gate-v16"/"dev"/f"{seed}-{family}-{roster}.json")["models"]["service"]["delivery_s"]
        result.update(learned=mission(f,p,"learned"),always=mission(f,p,"always"),
                      pure=dict(delivery=pure,delivered=sum(t>=0 for t in pure)))
    return result


def training_cases():
    return [((str(ROOT/name),start+k,f,k),0)
            for name,start in (("uav-priority-confirm-v10",97161001),("uav-short-deadline-learning-v11",98761001))
            for k in range(4) for f in ("continuous","rewet")]


def train(rows):
    x,y=[],[]
    for r in rows:
        for fork in r["forks"]:
            x.append(r["base"]["decisions"][fork["slot"]]["x"])
            y.append(fork["gain"])
    design = np.array(x)
    penalty = np.diag([0]+[4]*(len(FEATURES)-1))
    w = np.linalg.solve(design.T@design+penalty,design.T@np.array(y))
    assert np.isfinite(w).all()
    return dict(features=list(FEATURES),weights=w.tolist(),n=len(y),gains=dict(Counter(y)),
                training_seeds=sorted({r["seed"] for r in rows}))


def summarize(rows):
    groups = {}
    parents = defaultdict(int)
    for family in sorted({r["family"] for r in rows} | {"all"}):
        subset = [r for r in rows if family=="all" or r["family"]==family]
        groups[family] = {m:sum(r[m]["delivered"] for r in subset) for m in ("base","learned","always","pure")}
    for r in rows:
        parents[r["seed"]] += r["learned"]["delivered"]-r["base"]["delivered"]
    n = 64*len(rows)
    samples=np.random.default_rng(4406).choice(list(parents.values()),size=(20000,len(parents))).sum(axis=1)*100/n
    ci=np.quantile(samples,[.025,.975]).tolist()
    g=groups["all"]
    advance=(g["learned"]>max(g["base"],g["always"],g["pure"]) and ci[0]>0 and
             all(groups[f]["learned"]>groups[f]["base"] for f in ("continuous","rewet")))
    return dict(groups=groups,people=n,parents=parents,interval_pp=ci,advance=advance)


def main(stage):
    assert stage in ("fit","screen","panel")
    if stage=="panel":
        assert read(HERE/"route-value-screen.json")["summary"]["advance"]
    items = training_cases() if stage=="fit" else jobs(stage)
    if stage!="fit":
        assert not set(read(WEIGHTS)["training_seeds"]) & {c[1] for c,_ in items}
    version={n:sha(HERE/n) for n in ("route-value-protocol.md","fit_route_value44.py","route_value44.py",
             "continuous44.py","experiment44.py","tracker44.py","shortlist44.py","fraction.dll")}
    if stage!="fit":
        version["weights"]=sha(WEIGHTS)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(trial,[(stage,c,r) for c,r in items]):
            rows.append(row)
            if len(rows)%4==0:
                print(stage,len(rows),len(items),flush=True)
    summary=train(rows) if stage=="fit" else summarize(rows)
    save(HERE/f"route-value-{stage}.json",dict(version=version,summary=summary,rows=rows))
    if stage=="fit":
        save(WEIGHTS,summary)
    print(summary,flush=True)


if __name__=="__main__":
    main(sys.argv[1])
