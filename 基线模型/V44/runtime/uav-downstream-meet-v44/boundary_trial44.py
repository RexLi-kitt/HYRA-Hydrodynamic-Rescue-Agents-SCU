"""Replay V44 pressure scenarios with an absorbing perimeter that retains recipients."""
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np
from experiment44 import HERE, ROOT, NoveltyBeam, finish, jobs, material, quota, read, runtime, save, sha
from model import load_model


def first_contact(start, end):
    delta = end - start
    limits = np.full(2, np.inf)
    np.divide(1000-start, delta, out=limits, where=delta>0)
    np.divide(-start, delta, out=limits, where=delta<0)
    fraction = float(limits.min())
    assert 0 <= fraction <= 1+1e-10
    # The positive edge is exclusive in the original geometry and cell index.
    return np.clip(start+min(1.,fraction)*delta, 0, np.nextafter(1000.,0.))


def stop_at_edge(pop):
    out={k:v.copy() for k,v in pop.items()}
    xy=out["points"]
    assert np.isfinite(xy).all() and ((xy[0]>0)&(xy[0]<1000)).all()
    assert np.array_equal(pop["available"], ((xy>=0)&(xy<1000)).all(axis=2))
    hit=((xy<=0)|(xy>=1000)).any(axis=2)
    seconds=np.where(hit.any(axis=0),hit.argmax(axis=0),-1)
    for j in np.flatnonzero(seconds>=0):
        t=seconds[j]
        xy[t:,j]=first_contact(xy[t-1,j],xy[t,j])
        out["velocity"][(t+4)//5:,j]=0.
        out["available"][t:,j]=True
    assert out["available"].all() and ((xy>=0)&(xy<1000)).all()
    return out,seconds


def replay(item):
    case,roster,count,window,wall=item
    _,seed,family,_=case
    quota()
    field,pop=material(case,roster)
    pop={k:(v[:count] if k=="fall" else v[:,:count]) for k,v in pop.items()}
    contact=np.full(count,-1)
    if wall:
        pop,contact=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    pure=NoveltyBeam("service")
    pure.belief.personal_window_seconds=window
    rows={}
    for name,agent in (("v44",load_model(count,window)),("pure",pure)):
        env=judge(field,pop,(count,window,2400))
        assert env.radius==1 and np.array_equal(env.deadline,np.minimum(2400,pop["fall"]+window))
        row=finish(env,agent,np.full(count,-1))
        delivered=np.array(row["delivery"])
        first=np.array(row["first"])
        assert np.all((delivered<0)|((first>=0)&(first+10<=delivered)&(delivered<=env.deadline)))
        if count==64 and window==720 and not wall:
            if name=="v44":
                ref=next(r["revised"] for r in read(HERE/"search-integration.json")["rows"]
                         if r["seed"]==seed and r["family"]==family and r["roster"]==roster)
            else:
                ref=read(HERE/"screen"/f"{seed}-{family}-{roster}.json")["arms"]["pure"]
            assert all(row[k]==ref[k] for k in ("first","delivery","conflicts"))
        row.update(sent=int((delivered>=0).sum()),found=int((first>=0).sum()),
                   sent_after_contact=int(((contact>=0)&(delivered>=contact)).sum()))
        rows[name]=row
    return dict(seed=seed,family=family,roster=roster,count=count,window=window,wall=wall,
                contact=contact.tolist(),**rows)


def main(stage):
    configs=[(64,w,wall) for w in (720,900) for wall in (False,True)]
    if stage=="smaller":
        baseline=read(HERE/"boundary-factorial.json")["rows"]
        joint=[r for r in baseline if r["wall"] and r["window"]==900]
        assert len(joint)==16 and sum(r["v44"]["sent"] for r in joint)<.85*16*64
        configs=[(56,900,True)]
    else:
        assert stage=="factorial"
    cases=jobs("screen")
    version={p.name:sha(p) for p in (HERE/"boundary_trial44.py",HERE/"model.py",
             HERE/"boundary-trial-protocol.md",HERE/"continuous44.py",ROOT/"uav-rescue-gate-v16"/"beam16.py")}
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(replay,[(c,r,*config) for config in configs for c,r in cases]):
            rows.append(row)
            if len(rows)%16==0:
                print(stage,len(rows),flush=True)
    save(HERE/f"boundary-{stage}.json",dict(version=version,rows=rows))


if __name__=="__main__":
    main(sys.argv[1])
