"""Audit full-mission effects, clustered uncertainty, guards and individual outcomes."""
import numpy as np
from experiment44 import HERE,read,save,sha

def summarize(stage,chosen):
    d=read(HERE/f"terminal-{stage}.json")
    valid=[r for r in d["rows"] if "arms" in r]
    if not valid:
        return dict(input_failures=d["rows"])
    changes=[]
    for r in valid:
        b=r["arms"]["base"]
        a=r["arms"][chosen]
        assert sum(e["taken"] for e in a["events"])<=1
        t=a["accepted"]
        assert (t is None)==(not any(e["taken"] for e in a["events"]))
        if t is None:
            assert all(a[k]==b[k] for k in ("first","delivery","conflicts"))
        else:
            old,new=map(np.asarray,(b["delivery"],a["delivery"]))
            mask=(old>=0)&(old<=t)
            assert np.array_equal(old[mask],new[mask])
            changes.append(dict(seed=r["seed"],roster=r["roster"],t=t,delta=(np.array(a["counts"])-b["counts"]).tolist()))
    comparisons={}
    for other in ("base","previous","pure"):
        if other not in valid[0]["arms"]:
            continue
        rng=np.random.default_rng(4481)
        boot=np.zeros((20000,3))
        gained=lost=0
        for ordinary in (True,False):
            parents={}
            for r in valid:
                if (r["family"]=="ordinary")!=ordinary:
                    continue
                a,b=(r["arms"][m] for m in (chosen,other))
                diff=np.subtract(a["counts"],b["counts"])
                parents[r["seed"]]=parents.get(r["seed"],np.zeros(3))+diff
                ja,jb=(np.array(x["delivery"])>=0 for x in (a,b))
                gained+=int((ja&~jb).sum())
                lost+=int((jb&~ja).sum())
            if parents:
                arr=np.array(list(parents.values()))
                boot+=arr[rng.integers(0,len(arr),(len(boot),len(arr)))].sum(1)
        comparisons[other]=dict(gained=gained,lost=lost,ci_pp=np.quantile(boot*100/(64*len(valid)),[.025,.975],axis=0).tolist())
    return dict(accepted=len(changes),changes=changes,counts=d["totals"],comparisons=comparisons,
                input_failures=[r for r in d["rows"] if "arms" not in r],
                conflicts={k:sum(r["arms"][k]["conflicts"] for r in valid) for k in valid[0]["arms"]})

if __name__=="__main__":
    chosen=str(read(HERE/"terminal-select.json")["winner"])
    fit=read(HERE/"terminal-fit.json")
    labels=np.array([l["label"] for r in fit["rows"] for l in r["labels"]])
    result=dict(winner=chosen,labels=len(labels),final_signs=[int((np.sign(labels[:,0])==x).sum()) for x in (-1,0,1)])
    for stage in ("select","repeat","mixed","fresh"):
        result[stage]=summarize(stage,chosen)
        print(stage,result[stage].get("counts"),"accept",result[stage].get("accepted"),flush=True)
    result["hashes"]={f:sha(HERE/f) for f in ("model.py","terminal-loss-audit.json","terminal-fit.json","terminal-select.json","terminal-mixed.json","terminal-fresh.json","terminal-repeat.json","terminal_dispatch44.py","study_terminal44.py","terminal-dispatch-protocol.md")}
    save(HERE/"terminal-summary.json",result)
