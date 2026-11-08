"""Replay one deployable, observation-routed policy; validate pairing before attribution."""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import csv

from destinations import OUT,np,s,runtime,DestinationSearch,read,save,sha
from support import load_scene,quota
from beam16 import NoveltyBeam


class Routed:
    def __init__(self):
        self.sparse=DestinationSearch(use_templates=True)
        self.deep=NoveltyBeam("service")
        self.branch=None
        self.choice_s=None

    def act(self,o):
        if self.branch is None:
            observed=o.last>=0
            if observed.sum()>=40 or o.t>=60:
                self.choice_s=int(o.t)
                wet_share=np.mean(o.measured[0,observed]>=.01) if observed.any() else 0.
                self.branch=self.deep if wet_share>=.8 else self.sparse
            else:
                actions=self.sparse.act(o)
                control=self.deep.act(o)
                if not all(np.array_equal(a,b) for a,b in zip(actions,control)):
                    raise ValueError("Divergent actions before observable regime selection")
                return actions
        return self.branch.act(o)


def run(job):
    c,rep=job
    filename=f"{c[1]}-{c[2]}-{rep}.json"
    destination=OUT/"combined"/filename
    if destination.exists():return
    field,pop=load_scene(c,rep)
    cls,_=runtime(1,"planning")
    env=cls(field,pop,(64,720,2400))
    a=Routed()
    while env.t<2400:
        env.step(*a.act(env.observation()))
        if np.all(env.supplied>=0) and np.all(env.last>=0):break
    m=env.metrics()
    assert len(m["delivery_s"])==64 and m["delivered"]==sum(t>=0 for t in m["delivery_s"])
    save(destination,dict(seed=c[1],family=c[2],panel=c[0],rep=rep,method="causal_routed",choice=("template" if a.branch is a.sparse else "beam"),choice_s=a.choice_s,metrics=m,found=(env.seen_at>=0).tolist(),source=sha(OUT/"deploy.py")))


def summarize(jobs):
    groups=defaultdict(lambda:defaultdict(int))
    per_parent=defaultdict(lambda:defaultdict(int))
    decision=defaultdict(int)
    same_person=0
    for c,rep in jobs:
        name=f"{c[1]}-{c[2]}-{rep}.json"
        cur=read(OUT/"combined"/name)
        assert cur["source"]==sha(OUT/"deploy.py")
        prior=read(OUT.parent/"uav-rescue-gate-v16"/"dev"/name)["models"]["service"]
        n=cur["metrics"]["delivered"]
        decision[c[2]+"/"+cur["choice"]]+=1
        # Class is used only here to audit against archived single-branch rollouts.
        if c[2]=="ordinary":
            arm=read(OUT/"dev"/name)["models"]
            matched=arm["templates"];abl=arm["geometry"]["delivered"]
        else:
            matched=prior;abl=prior["delivered"]
        equal=cur["metrics"]["delivery_s"]==matched["delivery_s"]
        same_person+=int(equal)
        for k,v in (("routed",n),("v16",prior["delivered"])):
            groups[c[2]][k]+=v
            groups[c[0]][k]+=v
            groups["all"][k]+=v
        per_parent[c[2] if c[2]=="ordinary" else "paired"][c[1]]+=n-prior["delivered"]
        if equal: groups[c[2]]["matched_ablation"]+=abl
    rng=np.random.default_rng(28114)
    diff=sum(rng.choice(list(v.values()),(20000,len(v))).sum(1) for v in per_parent.values())
    h=groups["all"]
    qualify=h["routed"]>=4352 or h["routed"]>=1.05*h["v16"] and all(groups[p]["routed"]>groups[p]["v16"] for p in ("a","b"))
    # Retain historical comparators under these exact rosters and conditions.
    history=defaultdict(int)
    with (OUT.parent/"uav-fleet-paths-v15"/"完整逐场.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):history[row["方法"]]+=int(row["送达"])
    save(OUT/"deployment-summary.json",dict(counts=groups,decisions=decision,matches= same_person,missions=len(jobs),total_people=len(jobs)*64,paired_difference_pp_interval=np.quantile(diff*100/(len(jobs)*64),[.025,.975]).tolist(),qualify=bool(qualify),v15_history=history))
    print(dict(h),decision,same_person,qualify,flush=True)


if __name__=="__main__":
    (OUT/"combined").mkdir(exist_ok=True)
    jobs=[(c,i) for c in s.cases("a")+s.cases("b") for i in (0,1)]
    for start in range(0,len(jobs),8):
        quota()
        with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(run,jobs[start:start+8]))
        print("paired replay",min(start+8,len(jobs)),len(jobs),flush=True)
    summarize(jobs)
