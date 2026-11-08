from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import sys
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"uav-moving-intercept-v21"))
from intercept import Intercept
sys.path.insert(0,str(OUT.parent/"uav-window-rollout-v19"))
from windows import choose_profile,features,TIMES,OPTIONS,runtime,finished,quota,read,save,sha
sys.path.insert(0,str(OUT.parent/"uav-belief-routes-v17"))
from deploy import Routed
from support import s,load_scene


class Combined(Intercept):
    def __init__(self):
        super().__init__(4)
        with np.load(OUT.parent/"uav-window-rollout-v19"/"weights.npz") as z:self.weights=dict(z)
        self.restore=None;self.chosen=[]

    def act(self,o):
        if o.t==self.restore:
            choose_profile(self,"service");self.restore=None
        if o.t in TIMES and self.wet:
            w=self.weights
            scores=w["bias"]+(np.array(features(o))-w["mean"])/w["std"]@w["coef"]
            k=int(np.argmax(np.r_[.5,scores]))
            self.chosen.append([int(o.t),k])
            if k:
                choose_profile(self,OPTIONS[k]);self.restore=o.t+120
        return super().act(o)


def run(job):
    c,rep=job
    name=f"{c[1]}-{c[2]}-{rep}.json"
    out=OUT/"dev"/name
    if out.exists():
        assert read(out)["source"]==sha(Path(__file__))
        return
    field,pop=load_scene(c,rep);cls,_=runtime(1,"planning")
    env=cls(field,pop,(64,720,2400))
    agent=Routed();agent.deep=Combined()
    while not finished(env):env.step(*agent.act(env.observation()))
    m=env.metrics()
    assert m["total"]==len(m["delivery_s"])==64 and m["delivered"]==sum(t>=0 for t in m["delivery_s"])
    save(out,dict(seed=c[1],family=c[2],rep=rep,metrics=m,found=(env.seen_at>=0).tolist(),chosen=agent.deep.chosen,
                 meeting_plans=agent.deep.proposed,source=sha(Path(__file__)),weights=sha(OUT.parent/"uav-window-rollout-v19"/"weights.npz")))


def summarize(jobs):
    groups=defaultdict(lambda:defaultdict(int));blocks=defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    for c,rep in jobs:
        name=f"{c[1]}-{c[2]}-{rep}.json"
        cur=read(OUT/"dev"/name)
        learn=read(OUT.parent/"uav-window-rollout-v19"/"online"/name)["arms"]["learned"]
        wait=read(OUT.parent/"uav-moving-intercept-v21"/"dev"/name)["arms"]
        pure=read(OUT.parent/"uav-rescue-gate-v16"/"dev"/name)["models"]["service"]
        assert cur["source"]==sha(Path(__file__)) and cur["weights"]==sha(OUT.parent/"uav-window-rollout-v19"/"weights.npz")
        if c[2]=="ordinary":assert cur["metrics"]["delivery_s"]==learn["delivery_s"]==wait["wait4"]["delivery_s"]==wait["service"]["delivery_s"]
        for k,m in (("both",cur["metrics"]),("learn",learn),("wait",wait["wait4"]),("neither",wait["service"]),("pure",pure)):
            for g in ("all",c[0],c[2]):groups[g][k]+=m["delivered"]
            blocks["ordinary" if c[2]=="ordinary" else "paired"][c[1]][k]+=m["delivered"]
    rng=np.random.default_rng(25006);cis={}
    for other in ("pure","learn","wait","neither","interaction"):
        vals=[np.array([a["both"]-a[other] if other!="interaction" else a["both"]-a["learn"]-a["wait"]+a["neither"] for a in block.values()]) for block in blocks.values()]
        diff=sum(rng.choice(v,(20000,len(v))).sum(1) for v in vals)
        cis[other]=np.quantile(100*diff/5120,[.025,.975]).tolist()
    a=groups["all"]
    keep=a["both"]>=4352 or a["both"]>=1.05*a["pure"] and all(groups[p]["both"]>groups[p]["pure"] for p in ("a","b"))
    save(OUT/"summary.json",dict(counts=groups,total=5120,paired_ci_pp=cis,interaction=a["both"]-a["learn"]-a["wait"]+a["neither"],keep=bool(keep)))
    print(dict(a),cis,keep,flush=True)

if __name__=="__main__":
    (OUT/"dev").mkdir(exist_ok=True)
    jobs=[(c,rep) for c in s.cases("a")+s.cases("b") for rep in (0,1)]
    for i in range(0,len(jobs),8):
        quota()
        with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(run,jobs[i:i+8]))
        print(min(i+8,len(jobs)),len(jobs),flush=True)
    summarize(jobs)
