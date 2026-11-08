"""Replay the previous ordinary panel to locate which observable service stages were reached."""
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study_speed44 import HERE,ROOT,material,world,stop_at_edge,load_model,read,save,quota

def replay(row):
    quota()
    seed=row["seed"]
    name,start=("uav-priority-confirm-v10",97151001) if seed<98000000 else ("uav-short-deadline-learning-v11",98751001)
    c=(str(ROOT/name),seed,"ordinary",seed-start)
    field,pop=material(c,1)
    env=world(field,stop_at_edge(pop)[0])
    agent=load_model(fleet_policy="early_delivery")
    tracked=np.zeros(64,bool)
    credited=np.zeros(64,bool)
    for _ in range(480):
        actions,claims=agent.act(env.observation())
        if agent.branch is not None:
            targets=agent.branch.track
            tracked[targets[targets>=0]]=True
        tracked[claims[claims>=0]]=True
        env.step(actions,claims)
        credited|=env.progress>0
    old=np.array(row["arms"]["base"]["delivery"])
    new=np.array(row["arms"]["new"]["delivery"])
    assert np.array_equal(new,env.supplied)
    found=np.array(row["arms"]["new"]["first"])>=0
    assert np.array_equal(found,env.seen_at>=0)
    lost=(old>=0)&(new<0)
    stage=np.where(~found,"unseen",np.where(~tracked,"seen_no_assignment",np.where(~credited,"assigned_no_5s","credited_5s")))
    return dict(seed=seed,gained=np.flatnonzero((old<0)&(new>=0)).tolist(),
                losses=[dict(person=int(j),stage=str(stage[j])) for j in np.flatnonzero(lost)])

if __name__=="__main__":
    from collections import Counter
    with ProcessPoolExecutor(4) as pool:
        rows=list(pool.map(replay,read(HERE/"speed-ordinary-repeat.json")["rows"]))
    summary=dict(Counter(x["stage"] for r in rows for x in r["losses"]))
    save(HERE/"terminal-loss-audit.json",dict(rows=rows,summary=summary))
    print(summary,flush=True)
