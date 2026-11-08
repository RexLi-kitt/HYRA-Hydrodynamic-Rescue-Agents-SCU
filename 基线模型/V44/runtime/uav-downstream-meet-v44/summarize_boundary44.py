"""Aggregate the paired scenario tests; retain denominators when comparing roster sizes."""
from collections import defaultdict
import numpy as np
from boundary_trial44 import HERE, read


def summarize():
    rows=read(HERE/"boundary-factorial.json")["rows"]
    small=HERE/"boundary-smaller.json"
    if small.exists():
        rows+=read(small)["rows"]
    groups=defaultdict(list)
    for r in rows:
        groups[(r["count"],r["window"],r["wall"])].append(r)
    for config,group in groups.items():
        assert len(group)==len({(r['seed'],r['family']) for r in group})==16
        print(config,"denominator",sum(r["count"] for r in group))
        for family in ("continuous","rewet","all"):
            subset=[r for r in group if family=="all" or r["family"]==family]
            print(family,{m:tuple(sum(r[m][k] for r in subset) for k in ("sent","found","sent_after_contact"))
                          for m in ("v44","pure")})
    joint={(r["seed"],r["family"]):r for r in groups[64,900,True]}
    for cfg in ((64,720,False),(64,900,False),(64,720,True),(56,900,True)):
        if cfg not in groups:
            continue
        pairs=defaultdict(int)
        gains=losses=0
        for r in groups[cfg]:
            partner=joint[r["seed"],r["family"]]
            a=np.array(r["v44"]["delivery"])>=0
            b=np.array(partner["v44"]["delivery"][:len(a)])>=0
            gains+=int((b&~a).sum())
            losses+=int((a&~b).sum())
            pairs[r["seed"]]+=int(b.sum()-a.sum())
        samples=np.random.default_rng(4406).choice(list(pairs.values()), (20000,len(pairs))).sum(1)
        print("joint 64 (matched IDs) minus",cfg,"gained/lost",gains,losses,"by parent",dict(pairs),
              "CI pp",np.quantile(samples*100/(len(groups[cfg])*cfg[0]),[.025,.975]))


if __name__=="__main__":
    summarize()
