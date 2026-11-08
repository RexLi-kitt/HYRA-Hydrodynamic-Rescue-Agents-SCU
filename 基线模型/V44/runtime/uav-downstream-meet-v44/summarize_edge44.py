"""Summarize actual deliveries, with uncertainty clustered by paired water-map parent."""
from collections import defaultdict
import numpy as np
from experiment44 import HERE,read,save,sha


def describe(rows,mode):
    totals={}
    for m in ("base","pure",mode):
        arms=[r["arms"][m] for r in rows]
        times=np.array([t for a in arms for t in a["delivery"]])
        totals[m]=dict(sent=int((times>=0).sum()),found=sum(a["found"] for a in arms),
                       by_1200=int(((times>=0)&(times<=1200)).sum()),
                       by_1800=int(((times>=0)&(times<=1800)).sum()),
                       after_contact=sum(a["at_edge"] for a in arms))
    intervals={}
    for other in ("base","pure"):
        blocks=defaultdict(lambda:defaultdict(int))
        for r in rows:
            blocks[r["panel"],r["family"]=="ordinary"][r["seed"]]+=r["arms"][mode]["sent"]-r["arms"][other]["sent"]
        rng=np.random.default_rng(4406)
        draws=sum(rng.choice(list(b.values()),(20000,len(b))).sum(1) for b in blocks.values())
        intervals[other]=np.quantile(draws*100/(64*len(rows)),[.025,.975]).tolist()
    return dict(n=64*len(rows),totals=totals,paired_interval_pp=intervals)


if __name__=="__main__":
    result=read(HERE/"edge-search-evaluate.json")
    fit=read(HERE/"edge-search-fit.json")
    assert fit["version"]==result["version"]
    mode=fit["winner"]
    groups={}
    for label in ("pressure","A","B","mixed"):
        rows=[r for r in result["rows"] if (r["panel"] in ("A","B") if label=="mixed" else r["panel"]==label)]
        assert len(rows)=={"pressure":16,"A":20,"B":20,"mixed":40}[label]
        groups[label]=describe(rows,mode)
        print(label,groups[label],flush=True)
    mixed=[r for r in result["rows"] if r["panel"] in ("A","B")]
    for f in ("ordinary","continuous","rewet"):
        groups["mixed_"+f]=describe([r for r in mixed if r["family"]==f],mode)
        print(f,groups["mixed_"+f]["totals"],flush=True)
    gate=all(groups[k]["totals"][mode]["sent"]>groups[k]["totals"]["base"]["sent"] for k in ("pressure","A","B"))
    g=groups["mixed"]["totals"]
    save(HERE/"edge-search-summary.json",dict(source=sha(HERE/"edge-search-evaluate.json"),winner=mode,
        recommend=gate and g[mode]["sent"]>g["pure"]["sent"],groups=groups))
