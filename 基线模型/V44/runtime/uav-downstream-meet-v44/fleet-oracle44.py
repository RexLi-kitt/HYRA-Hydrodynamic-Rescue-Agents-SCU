"""Privileged development diagnostic. The online fleet policy must never import this file."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import HERE,material,runtime,save,quota
from fit_route_value44 import training_cases
from boundary_trial44 import stop_at_edge
from rendezvous44 import DIR,projection,sight,HOLD,FOLLOW
from lattice import near,directions,route
from local_route import endpoints


def approach(p,head,t,j,pop,deadline):
    # Enumerate feasible full-step displacements and one visible short approach.
    points=pop["points"][:,j]
    for n in range(min(8,(deadline-t-10)//5)+1):
        at=t+5*n
        x=points[at]
        if np.linalg.norm(x-p)>n*50+20.000001:
            continue
        for offset,counts in near(x-p,n):
            stage=p+offset
            gap=x-stage
            if (stage<0).any() or (stage>1000).any() or gap@gap>400+1e-9:
                continue
            moves=directions(counts)
            for last in sorted(set(moves)) if n else [-1]:
                h=DIR[last] if n else head
                if h@gap < -1e-9:
                    continue
                along=DIR@gap
                shorts=np.flatnonzero((along>=0)&(gap@gap-along*along<=.25+1e-10))
                for short in [None,*shorts]:
                    start=at+(5 if short is not None else 0)
                    if start+10>deadline:
                        continue
                    pos,h2=stage,h
                    if short is not None:
                        pos,h2=projection(stage,h,x,pop["velocity"][at//5,j],True,9+int(short))
                    ok=True
                    for dt in (0,5):
                        k=start+dt
                        if not sight(pos,h2,points[k]) or np.linalg.norm(pos-points[k])>1+1e-9:
                            ok=False
                            break
                        vel=pop["velocity"][k//5,j]
                        track=np.clip(pos+np.arange(6)[:,None]*vel,0,1000)
                        if np.linalg.norm(track-points[k:k+6],axis=1).max()>1+1e-9:
                            ok=False
                            break
                        pos=track[-1]
                    if not ok:
                        continue
                    prefix=route(p,moves,last) if n else []
                    if prefix is not None:
                        return [a for a,_ in prefix]+([] if short is None else [9+int(short)])+[FOLLOW,FOLLOW]
    return []


def run(item):
    case,rep=item
    quota()
    field,pop=material(case,rep)
    pop,_=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    env=judge(field,pop,(64,720,2400))
    plans={}; log=[]
    while env.t<2400:
        t=env.t;o=env.observation()
        for i,(j,codes,pose) in list(plans.items()):
            if not codes or env.supplied[j]>=0 or np.linalg.norm(pose-o.pos[i])>.01:
                del plans[i]
        actions=np.full(4,HOLD);claims=np.full(4,-1)
        used={j for j,_,_ in plans.values()}
        for i in np.roll(np.arange(4),t//5%4):
            if i not in plans:
                ids=[j for j in range(64) if j not in used and env.supplied[j]<0 and env.deadline[j]>=t+10]
                if not ids:
                    continue
                pos=o.pos[i]
                distances=np.linalg.norm(pop["points"][t,ids]-pos,axis=1)
                candidates=np.asarray(ids)[np.argsort(distances)[:8]]
                options=[]
                for j in candidates:
                    cmd=approach(pos,o.heading[i],t,j,pop,int(env.deadline[j]))
                    if cmd:
                        options.append((len(cmd),int(env.deadline[j]),int(j),cmd))
                        if len(cmd)<=3:
                            break
                if options:
                    _,_,j,cmd=min(options)
                else:
                    j=int(candidates[0])
                    dest=pop["points"][min(t+15,2400),j]
                    ends,_=endpoints(pos)
                    cmd=[int(np.linalg.norm(ends-dest,axis=1).argmin())]
                plans[i]=(j,cmd,pos);used.add(j)
            j,codes,_=plans[i];code=codes[0]
            if code>=9 and not o.visible[j]:
                del plans[i]
                continue
            nxt,_=projection(o.pos[i],o.heading[i],pop["points"][t,j],pop["velocity"][t//5,j],o.visible[j],code)
            actions[i]=code%9;claims[i]=j if code>=9 else -1
            plans[i]=(j,codes[1:],nxt)
        log.append([actions.tolist(),claims.tolist()])
        env.step(actions,claims)
    # Reproduce the schedule without access to privileged information during replay.
    replay=judge(field,pop,(64,720,2400))
    for actions,claims in log:
        assert all(j<0 or replay.visible[j] for j in claims)
        replay.step(actions,claims)
    assert np.array_equal(replay.supplied,env.supplied)
    return dict(seed=case[1],family=case[2],sent=int((env.supplied>=0).sum()),
                delivery=env.supplied.tolist(),commands=log)


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,training_cases()[:4]))
    save(HERE/"fleet-oracle-schedule.json",dict(rows=rows))
    print([(r["family"],r["sent"]) for r in rows])
