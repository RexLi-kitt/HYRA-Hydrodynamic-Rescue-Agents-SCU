"""Supplemental ordinary-only replay; keep the original selection and main table intact."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study_speed44 import HERE,ROOT,material,world,configured,load_model,NoveltyBeam,stop_at_edge,finish,cloned_mission,read,save,sha,quota

def replay(case):
    quota()
    field,pop=material(case,1)
    env=world(field,stop_at_edge(pop)[0])
    arms={}
    for key in ("new","base","pure"):
        agent=load_model(fleet_policy="early_delivery") if key=="new" else load_model() if key=="base" else NoveltyBeam("service")
        outcome=finish(cloned_mission(env),agent,np.full(64,-1))
        d,f=map(np.asarray,(outcome["delivery"],outcome["first"]))
        assert ((d<0)|((d>=f+10)&(f>=0)&(d<=env.deadline))).all()
        arms[key]=dict(**outcome,counts=[int(((d>=0)&(d<=t)).sum()) for t in (600,1200,2400)])
    return dict(seed=case[1],roster=1,arms=arms)

if __name__=="__main__":
    cases=[(str(ROOT/d),seed+i,"ordinary",i) for d,seed in (("uav-priority-confirm-v10",97151001),("uav-short-deadline-learning-v11",98751001)) for i in range(24,36)]
    old=set(r["seed"] for f,k in (("speed-fit.json","series"),("speed-select.json","rows"),("speed-mixed.json","rows"),("speed-fresh.json","rows")) for r in read(HERE/f)[k])
    assert not old.intersection(c[1] for c in cases)
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(replay,cases))
    totals={k:np.sum([r["arms"][k]["counts"] for r in rows],0).tolist() for k in ("new","base","pure")}
    save(HERE/"speed-ordinary-repeat.json",dict(rows=rows,totals=totals,fit=sha(HERE/"speed-fit.json"),selection=sha(HERE/"speed-select.json"),design="supplemental ordinary only; historical maps outside this round's training and selection; no refit"))
    print(totals,flush=True)
