"""Offline diagnosis on the unchanged gated-policy flights. Truth never enters the policy."""
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
from experiment44 import HERE,s,load_scene,quota,read,save,runtime
from boundary_trial44 import stop_at_edge
from model import load_model


def audit(job):
    case,rep,expected=job
    quota()
    water,pop=load_scene(case,rep)
    pop,_=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    env=judge(water,pop,(64,720,2400))
    agent=load_model(edge_search="detour_gate")
    first=np.full(64,-1)
    nearby=np.full(64,-1)
    while env.t<2400:
        obs=env.observation()
        first[(first<0)&obs.visible]=env.t
        delta=pop["points"][env.t,:,None]-obs.pos
        in_circle=((delta*delta).sum(axis=2)<=400+1e-9).any(axis=1)
        nearby[(nearby<0)&in_circle&env.valid(env.t)&(env.t<env.deadline)]=env.t
        env.step(*agent.act(obs))
    assert expected["first"]==first.tolist() and expected["delivery"]==env.supplied.tolist()
    missed=first<0
    return dict(seed=case[1],family=case[2],roster=rep,first_near=nearby.tolist(),
                unseen=int(missed.sum()),never_near=int((missed&(nearby<0)).sum()),
                near_only_late=int((missed&(nearby>=0)&(nearby+10>env.deadline)).sum()),
                near_with_time=int((missed&(nearby>=0)&(nearby+10<=env.deadline)).sum()))


if __name__=="__main__":
    prior=read(HERE/"detour-gate-result.json")
    ref={(r["seed"],r["family"],r["roster"]):r["arms"]["detour_gate"] for r in prior["rows"]}
    work=[(c,rep,ref[c[1],c[2],rep]) for group,rep in (("a",0),("b",1)) for c in s.cases(group)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(audit,work))
    summary={}
    for band in ("mixed","pressure","ordinary"):
        rs=[r for r in rows if band=="mixed" or (r["family"]=="ordinary")== (band=="ordinary")]
        summary[band]={k:sum(r[k] for r in rs) for k in ("unseen","never_near","near_only_late","near_with_time")}
    save(HERE/"detour-range-audit.json",dict(rows=rows,summary=summary))
    print(summary)
