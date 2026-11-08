from collections import defaultdict,Counter
import numpy as np
from rendezvous44 import HERE,ROOT,DIR,read,save,sha,projection


def main():
    ref=ROOT/"uav-free-window-v42"/"rows";arc=ROOT/"uav-visibility-audit-v39"/"traces"
    out=[];totals=Counter()
    for p in sorted(ref.glob("*.json")):
        q=read(p);groups=defaultdict(list)
        for r in q["rows"]:
            groups[(r["k"],r["i"],r["j"])].append(r["a"])
        with np.load(arc/(p.stem+".npz")) as tr:
            pos=tr["pos"];heading=tr["heading"]
        count=Counter();multi=set();distinct=set()
        for (k,i,j),actions in groups.items():
            assert len(actions)==len(set(actions))
            views=set();full=0;hold=False
            for a in actions:
                end,h=projection(pos[k,i],heading[k,i],np.zeros(2),np.zeros(2),False,a)
                full+=int(np.linalg.norm(end-pos[k,i])>=50-1e-7)
                hold|=(a==8)
                views.add(tuple(np.round(np.r_[end,h],7)))
            assert full<=1 and not (full and hold)
            count["states"]+=1;count["attempts"]+=len(actions)
            if len(actions)>1:
                multi.add(j);count["multiple_indices"]+=1
            if len(views)>1:
                count["different_pose"]+=1;distinct.add(j)
        count["distinct_people_with_choice"]=len(distinct)
        count["people_with_multiple_indices"]=len(multi)
        out.append(dict(name=p.stem,counts=dict(count),ledger_sha=sha(p)));totals.update(count)
    assert len(out)==32 and totals["attempts"]==2604
    theta=np.arccos(np.clip(DIR@DIR.T,-1,1))
    angle=theta[theta>.01].min();rmin=50*np.tan(angle/2)
    assert 20.71<rmin<20.72
    save(HERE/"search-geometry.json",dict(summary=dict(totals),rows=out,min_radius_m=float(rmin),
               source=sha(HERE/"search_geometry.py"),protocol=sha(HERE/"search-geometry-protocol.md")))
    print(totals,rmin)


if __name__=="__main__":
    main()
