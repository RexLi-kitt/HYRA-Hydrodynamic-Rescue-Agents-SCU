"""Report paired effects and verify population draws and output provenance."""
import numpy as np
from experiment44 import HERE, ROOT, read, save, sha
from occupancy44 import regions, XY, heat_from_counts
from study_occupancy44 import FRESH, WEIGHTS


def analyze(panel):
    doc=read(HERE/f"occupancy-{panel}-result.json")
    assert doc["fit"]==sha(WEIGHTS)
    rows=doc["rows"]
    keys={(r["seed"],r["family"],r["roster"]) for r in rows}
    assert len(keys)==len(rows)==(16 if panel=="fresh" else 40)
    grouped={}
    for fam in ["all",*sorted({r["family"] for r in rows})]:
        subset=[r for r in rows if fam=="all" or fam==r["family"]]
        grouped[fam]={m:{k:sum(r["arms"][m][k] for r in subset) for k in ("sent","found","conflicts")}
                      for m in ("base","pure","learned")}
        grouped[fam]["people"]=len(subset)*64
    pairs={}
    for mode in ("base","pure"):
        rng=np.random.default_rng(44771)
        dist=np.zeros(20000)
        gained=lost=0
        for regular in (True,False):
            parents={}
            for r in rows:
                if regular!=(r["family"]=="ordinary"):
                    continue
                a,b=(np.asarray(r["arms"][m]["delivery"])>=0 for m in ("learned",mode))
                gained+=int((a&~b).sum())
                lost+=int((b&~a).sum())
                parents[r["seed"]]=parents.get(r["seed"],0)+int(a.sum()-b.sum())
            if parents:
                vals=list(parents.values())
                dist+=rng.choice(vals,(len(dist),len(vals))).sum(1)
        pairs[mode]=dict(gained=gained,lost=lost,ci_pp=np.quantile(dist*100/(64*len(rows)),[.025,.975]).tolist())
    return dict(groups=grouped,paired=pairs)


if __name__=="__main__":
    assert np.array_equal(regions(XY),np.arange(480))
    q=np.arange(480,dtype=float)
    assert np.isclose(heat_from_counts(q).sum(),q.sum())
    for seed in range(2026110890,2026110898):
        with np.load(FRESH/"population"/f"{seed}-continuous-0.npz") as a,np.load(FRESH/"population"/f"{seed}-rewet-0.npz") as b:
            for k in ("ids","clock"):
                assert np.array_equal(a[k],b[k])
        ab=[read(FRESH/"scenes"/f"{seed}-{f}"/"input.json") for f in ("continuous","rewet")]
        assert ab[0]["traps"]==ab[1]["traps"] and ab[0]["direction"]==ab[1]["direction"]==90*((seed-2026110890)%4)
    data={p:analyze(p) for p in ("fresh","mixed")}
    data["hashes"]={p.name:sha(p) for p in [HERE/n for n in ("occupancy44.py","study_occupancy44.py","model.py","boundary_trial44.py","occupancy-protocol.md","occupancy-fit.json","occupancy-fresh-result.json","occupancy-mixed-result.json")]+[ROOT/"uav-pressure-search-v8"/"risk-lock.json",ROOT/"uav-priority-confirm-v10"/"scenario10.py"]}
    save(HERE/"occupancy-summary.json",data)
    for p in ("fresh","mixed"):
        print(p,data[p],flush=True)
