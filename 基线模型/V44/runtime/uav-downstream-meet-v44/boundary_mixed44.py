from concurrent.futures import ProcessPoolExecutor
import numpy as np
from experiment44 import HERE, ROOT, s, read, save, sha, load_scene, quota, runtime, NoveltyBeam, finish
from boundary_trial44 import stop_at_edge, load_model


def replay(job):
    case,rep,count,window,wall=job
    quota()
    field,full=load_scene(case,rep)
    pop={k:(v[:count] if k=="fall" else v[:,:count]) for k,v in full.items()}
    if wall:
        pop,_=stop_at_edge(pop)
    judge,_=runtime(1,"planning")
    planner=NoveltyBeam("service")
    planner.belief.personal_window_seconds=window
    result=dict(seed=case[1],family=case[2],roster=rep,count=count,window=window,wall=wall)
    for name,agent in (("v44",load_model(count,window)),("pure",planner)):
        env=judge(field,pop,(count,window,2400))
        assert env.radius==1
        row=finish(env,agent,np.full(count,-1))
        if name=="pure" and not wall:
            prior=read(ROOT/"uav-rescue-gate-v16"/"dev"/f"{case[1]}-{case[2]}-{rep}.json")
            assert prior["models"]["service"]["delivery_s"]==row["delivery"]
        row["sent"]=sum(t>=0 for t in row["delivery"])
        result[name]=row
    return result


if __name__=="__main__":
    configs=[(64,720,False),(64,900,True)]
    if (HERE/"boundary-smaller.json").is_file():
        configs.append((56,900,True))
    cases=s.cases("a")
    assert len(cases)==20
    screen={r["seed"] for r in read(HERE/"boundary-factorial.json")["rows"]}
    assert not screen.intersection(c[1] for c in cases)
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=[]
        for row in pool.map(replay,[(c,0,*cfg) for cfg in configs for c in cases]):
            rows.append(row)
            if len(rows)%20==0:
                print("mixed",len(rows),flush=True)
    save(HERE/"boundary-mixed.json",dict(protocol_sha=sha(HERE/"boundary-mixed-protocol.md"),
         runner_sha=sha(HERE/"boundary_trial44.py"),rows=rows))
