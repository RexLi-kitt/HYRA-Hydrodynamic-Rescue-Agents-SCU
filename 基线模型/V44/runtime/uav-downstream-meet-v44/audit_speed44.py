"""Compare cumulative paired deliveries at predeclared horizons."""
import numpy as np
from experiment44 import HERE,read,save,sha
from study_speed44 import NEW

def analyze(panel,winner):
    doc=read(HERE/f"speed-{panel}.json")
    rows=doc["rows"]
    assert len({(r["seed"],r["family"],r["roster"]) for r in rows})==len(rows)
    groups={}
    w=str(winner)
    for name in ["all",*sorted({r["family"] for r in rows})]:
        sub=[r for r in rows if name=="all" or name==r["family"]]
        groups[name]=dict(n=len(sub)*64)
        for k in sub[0]["arms"]:
            arms=[r["arms"][k] for r in sub]
            groups[name][k]=dict(counts=np.sum([a["counts"] for a in arms],0).tolist(),
                found=sum(sum(f>=0 for f in a["first"]) for a in arms),
                conflicts=sum(a["conflicts"] for a in arms))
        groups[name]["choices"]=np.bincount([i for r in sub for i in r["arms"][w]["choices"]],minlength=3).tolist()
        groups[name]["dispatch_changes"]=sum(r["arms"][w]["dispatch_changes"] for r in sub)
    ci={}
    for k in ("base","pure","density"):
        rng=np.random.default_rng(4413)
        dist=np.zeros((20000,3))
        for regular in (False,True):
            parents={}
            for r in rows:
                if (r["family"]=="ordinary")!=regular:
                    continue
                diff=np.array(r["arms"][w]["counts"])-r["arms"][k]["counts"]
                parents[r["seed"]]=parents.get(r["seed"],np.zeros(3))+diff
            if parents:
                vals=np.array(list(parents.values()))
                dist+=vals[rng.integers(len(vals),size=(len(dist),len(vals)))].sum(1)
        ci[k]=np.quantile(dist*100/(64*len(rows)),[.025,.975],axis=0).tolist()
    return dict(groups=groups,paired_ci_pp=ci)

if __name__=="__main__":
    winner=read(HERE/"speed-select.json")["winner"]
    for i in range(8):
        seed=2026111100+i
        a,b=[read(NEW/"scenes"/f"{seed}-{f}"/"input.json") for f in ("rewet","continuous")]
        assert a["traps"]==b["traps"] and a["direction"]==b["direction"]==90*(i%4)
        with np.load(NEW/"population"/f"{seed}-continuous-0.npz") as x,np.load(NEW/"population"/f"{seed}-rewet-0.npz") as y:
            assert np.array_equal(x["ids"],y["ids"]) and np.array_equal(x["clock"],y["clock"])
    result={p:analyze(p,winner) for p in ("mixed","fresh")}
    result["winner"]=winner
    result["hashes"]={f:sha(HERE/f) for f in ("speed-fit.json","speed-select.json","speed-mixed.json","speed-fresh.json","speed-policy-protocol.md","study_speed44.py","early_delivery44.py","model.py","boundary_trial44.py")}
    save(HERE/"speed-summary.json",result)
    for p in ("mixed","fresh"):
        print(p,result[p],flush=True)
