from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
from experiment44 import HERE, s, finish, jobs, material, load_scene, runtime, NoveltyBeam, quota, read, save, sha
from boundary_trial44 import stop_at_edge
from fit_route_value44 import training_cases
from model import load_model


def run(job):
    panel,c,rep,modes=job
    quota()
    field,pop=(material if panel in ("fit","pressure") else load_scene)(c,rep)
    pop,contact=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    arms={}
    for mode in modes:
        policy=NoveltyBeam("service") if mode=="pure" else load_model(edge_search=None if mode=="base" else mode)
        env=judge(field,pop,(64,720,2400))
        assert env.radius==1
        row=finish(env,policy,np.full(64,-1))
        delivery=np.array(row["delivery"])
        row["sent"]=int((delivery>=0).sum())
        row["found"]=sum(t>=0 for t in row["first"])
        row["at_edge"]=int(((contact>=0)&(delivery>=contact)).sum())
        arms[mode]=row
    if panel in ("pressure","A"):
        file="boundary-factorial.json" if panel=="pressure" else "boundary-mixed-720-stop.json"
        ref=next(r for r in read(HERE/file)["rows"] if (r["seed"],r["family"],r["roster"],r["count"],r["window"],r["wall"])==(c[1],c[2],rep,64,720,True))
        for old,new in (("v44","base"),("pure","pure")):
            assert all(ref[old][k]==arms[new][k] for k in ("delivery","first","conflicts"))
    return dict(panel=panel,seed=c[1],family=c[2],roster=rep,contact=contact.tolist(),arms=arms)


def main(stage):
    version={n:sha(HERE/n) for n in ("edge-search-protocol.md","edge_search44.py","model.py","evaluate_edge44.py","continuous44.py","boundary_trial44.py")}
    if stage=="fit":
        work=[("fit",c,r,("base","one","all")) for c,r in training_cases()]
    else:
        assert stage=="evaluate"
        fitted=read(HERE/"edge-search-fit.json")
        assert fitted["version"]==version
        modes=("base","pure",fitted["winner"])
        work=[("pressure",c,r,modes) for c,r in jobs("screen")]
        for panel,rep in (("a",0),("b",1)):
            work.extend((panel.upper(),c,rep,modes) for c in s.cases(panel))
        assert not {row["seed"] for row in fitted["rows"]}.intersection(c[1] for _,c,_,_ in work)
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(run,work):
            rows.append(row)
            if len(rows)%8==0:
                print(stage,len(rows),len(work),flush=True)
    data=dict(rows=rows,version=version)
    if stage=="fit":
        totals={m:sum(r["arms"][m]["sent"] for r in rows) for m in ("base","one","all")}
        data["winner"]=max(("one","all"),key=lambda m:totals[m])
        print(totals,data["winner"],flush=True)
    save(HERE/f"edge-search-{stage}.json",data)


if __name__=="__main__":
    main(sys.argv[1])
