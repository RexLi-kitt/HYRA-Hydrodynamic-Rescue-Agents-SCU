"""Counterfactual training labels for changing the approach to free search for 120 s."""
from pathlib import Path
import copy
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/"uav-rescue-gate-v16"))
from beam16 import NoveltyBeam,PROFILES,material,runtime,read,save,sha
from support import quota

TIMES=(180,600,1020)
OPTIONS=("service","cross","upstream")


def features(o):
    known=o.seen_at>=0
    water=o.measured.reshape(3,-1)[:,o.observed_cells]
    vector=np.median(water[1:],axis=1) if water.shape[1] else np.zeros(2)
    speed=np.linalg.norm(vector)
    axis=vector/max(speed,.01)
    lateral=np.array([-axis[1],axis[0]])
    return [o.t/2400,known.mean(),(known&o.active).mean(),o.visible.mean(),
            (o.last>=0).mean(),((o.last>=0)&(o.t-o.last<=150)).mean(),
            (o.pos@lateral).std()/500,(o.pos@axis).std()/500,
            np.mean((o.pos-500)@axis)/500,speed,
            np.sum(known&(np.linalg.norm(o.seen_v,axis=1)>.02))/max(1,known.sum())]


def choose_profile(agent,name):
    (agent.horizon,agent.width,agent.easy,agent.tilt,agent.cross,agent.commit)=PROFILES[name]
    agent.plans_search.clear()


def cloned_mission(env):
    # During step only dynamic state is written. The underlying field and replayed
    # population trajectories are fixed read-only material for every paired fork.
    memo={id(getattr(env,k)):getattr(env,k) for k in ("truth","water") if hasattr(env,k)}
    return copy.deepcopy(env,memo)


def advance(env,agent):
    env.step(*agent.act(env.observation()))


def finished(env):
    return env.t>=2400 or (np.all(env.supplied>=0) and np.all(env.last>=0))


def fork(env,agent,name):
    trial=cloned_mission(env)
    policy=copy.deepcopy(agent)
    choose_profile(policy,name)
    end=trial.t+120
    while not finished(trial):
        if trial.t==end:choose_profile(policy,"service")
        advance(trial,policy)
    return trial.metrics()["delivery_s"]


def trial_case(c):
    p=OUT/"labels"/f"{c[1]}-{c[2]}.json"
    if p.exists():
        x=read(p)
        assert x["source"]==sha(Path(__file__)) and x["protocol"]==sha(OUT/"protocol.md")
        return
    field,pop=material(c,0)
    cls,_=runtime(1,"planning");env=cls(field,pop,(64,720,2400))
    agent=NoveltyBeam("service");rows=[]
    while not finished(env):
        if env.t in TIMES:
            state=features(env.observation())
            previous_supplied=env.supplied.copy()
            arms={k:fork(env,agent,k) for k in OPTIONS[1:]}
            assert np.array_equal(previous_supplied,env.supplied)
            rows.append(dict(t=env.t,features=state,other=arms))
        advance(env,agent)
    times=env.metrics()["delivery_s"]
    assert len(rows)==3 and len(times)==64
    save(p,dict(seed=c[1],family=c[2],baseline=times,rows=rows,source=sha(Path(__file__)),protocol=sha(OUT/"protocol.md")))


def score(rows):
    data=[]
    for r in rows:
        base=sum(v>=0 for v in r["baseline"])
        for row in r["rows"]:
            truth=[sum(v>=0 for v in row["other"][k])-base for k in OPTIONS[1:]]
            data.append((r["seed"],r["family"],row["t"],row["features"],truth))
    return data


def fit(samples):
    X=np.array([r[3] for r in samples]);Y=np.array([r[4] for r in samples])
    mean=X.mean(0);std=np.maximum(X.std(0),.08)
    Z=(X-mean)/std
    bias=Y.mean(0)
    coef=np.linalg.solve(Z.T@Z+24*np.eye(Z.shape[1]),Z.T@(Y-bias))
    return dict(mean=mean,std=std,coef=coef,bias=bias)


def summarize(cases):
    rows=[read(OUT/"labels"/f"{c[1]}-{c[2]}.json") for c in cases]
    train=score(rows[:8]);test=score(rows[8:])
    assert len(train)==len(test)==24 and set(r[0] for r in train).isdisjoint(r[0] for r in test)
    weights=fit(train);np.savez(OUT/"weights.npz",**weights)
    mean,std,coef,bias=(weights[k] for k in ("mean","std","coef","bias"))
    stat={}
    for family in ("continuous","rewet","all"):
        part=[r for r in test if family=="all" or r[1]==family]
        x=np.array([r[3] for r in part]);actual=np.c_[np.zeros(len(part),int),[r[4] for r in part]]
        prediction=(x-mean)/std@coef+bias
        choices=np.c_[np.full(len(part),.5),prediction].argmax(1)
        marginal=np.r_[.5,bias].argmax()
        stat[family]=dict(forks=len(part),selected_margins=int(actual[np.arange(len(part)),choices].sum()),
                          choice_counts=np.bincount(choices,minlength=3).tolist(),
                          base_margin=0,constant_training_mean_margin=int(actual[:,marginal].sum()),
                          fixed_option_margins=actual.sum(0).tolist(),oracle_margin=int(actual.max(1).sum()))
    keep=all(stat[f]["selected_margins"]>0 for f in ("continuous","rewet"))
    save(OUT/"screen.json",dict(train_snapshots=len(train),val_snapshots=len(test),summary=stat,keep=keep,source=sha(Path(__file__)),weights=sha(OUT/"weights.npz")))
    print(stat,keep,flush=True)


if __name__=="__main__":
    cases=[(str(OUT.parent/root),seed+i,f,i) for root,seed in (("uav-priority-confirm-v10",97161001),("uav-short-deadline-learning-v11",98761001))
           for i in range(8,12) for f in ("continuous","rewet")]
    (OUT/"labels").mkdir(exist_ok=True)
    for start in range(0,len(cases),4):
        quota()
        with ProcessPoolExecutor(max_workers=4) as ex:list(ex.map(trial_case,cases[start:start+4]))
        print("four-way paired cases",min(start+4,len(cases)),len(cases),flush=True)
    summarize(cases)
