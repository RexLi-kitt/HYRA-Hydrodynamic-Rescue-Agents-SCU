"""Train using simulated labels only offline; compare policies on paired full missions."""
from concurrent.futures import ProcessPoolExecutor
import sys
import numpy as np

from experiment44 import HERE,s,material,load_scene,cloned_mission,runtime,finish,NoveltyBeam,read,save,sha,quota
from boundary_trial44 import stop_at_edge
from fit_route_value44 import training_cases
from model import load_model
from occupancy44 import predictors, regions, heat_from_counts

FRESH = HERE/"occupancy-fresh"
WEIGHTS = HERE/"occupancy-fit.json"


def world(field,pop):
    cls,_ = runtime(1,"planning")
    e=cls(field,pop,(64,720,2400))
    assert (e.count,e.window,e.radius,e.duration)==(64,720,1,2400)
    assert np.array_equal(e.deadline,np.minimum(2400,pop["fall"]+720))
    return e


def sample(job):
    quota()
    c,r=job
    field,pop=material(c,r)
    pop,contact=stop_at_edge(pop)
    env=world(field,pop)
    agent=load_model()
    xs,ys=[],[]
    while env.t<2400:
        o=env.observation()
        acts=agent.act(o)
        if 60<=o.t<=2340 and o.t%60==0:
            x=predictors(o,agent.deep.belief)
            y=np.zeros(480)
            for dt in (30,45,60):
                end=o.t+dt
                ids=(o.seen_at<0)&(env.deadline>=end+10)&pop["available"][end]
                y+=np.bincount(regions(pop["points"][end,ids]),minlength=480)/3
            xs.append(x)
            ys.append(y)
        env.step(*acts)
    missed=env.seen_at<0
    # Last moment allowing a 10-second delivery, evaluated independently per person.
    last=np.clip(env.deadline-10,0,2400).astype(int)
    loc=pop["points"][last,np.arange(64)]
    return np.vstack(xs),np.concatenate(ys),dict(seed=c[1],family=c[2],never_seen=int(missed.sum()),
         at_edge=int((missed&(regions(loc)>=400)).sum()),
         deadlines=[int((missed&(env.deadline>a)&(env.deadline<=b)).sum()) for a,b in ((0,1200),(1200,1800),(1800,2400))])


def train():
    with ProcessPoolExecutor(4) as workers:
        records=list(workers.map(sample,training_cases()))
    X=np.vstack([r[0] for r in records])
    y=np.concatenate([r[1] for r in records])
    w=np.zeros(X.shape[1])
    w[0]=np.log(y.mean())
    penalty=np.diag([0.]+[8.]*(len(w)-1))
    def loss(a):
        z=X@a
        return np.exp(np.clip(z,-80,60)).sum()-y@z+.5*a@penalty@a
    for it in range(50):
        mu=np.exp(np.clip(X@w,-80,60))
        hess=X.T@(X*mu[:,None])+penalty
        step=np.linalg.solve(hess,X.T@(mu-y)+penalty@w)
        if abs(step).max()<1e-7:
            break
        scale=1.
        before=loss(w)
        while scale>1e-8 and loss(w-scale*step)>before:
            scale*=.5
        w-=scale*step
    assert np.isfinite(w).all()
    save(WEIGHTS,dict(weights=w.tolist(),iterations=it+1,rows=len(y),snapshots=len(y)//480,
          training=[r[2] for r in records],protocol=sha(HERE/"occupancy-protocol.md"),
          source=sha(HERE/"occupancy44.py"),pipeline=sha(HERE/"study_occupancy44.py")))
    print("fit",len(y),it+1,flush=True)


def evaluate(job):
    quota()
    panel,c,r=job
    if panel=="fresh":
        s.D=FRESH
        s.scenes.D=FRESH
        s.scenes.make_inputs(c)
        s.make_history((c,r))
    field,pop=load_scene(c,r)
    pop,_=stop_at_edge(pop)
    env=world(field,pop)
    arms={}
    for name in ("base","learned","pure"):
        agent=NoveltyBeam("service") if name=="pure" else load_model(edge_search="occupancy" if name=="learned" else None)
        a=finish(cloned_mission(env),agent,np.full(64,-1))
        d,f=np.array(a["delivery"]),np.array(a["first"])
        assert ((d<0)|((f>=0)&(d>=f+10)&(d<=env.deadline))).all()
        arms[name]=dict(**a,found=int((f>=0).sum()),sent=int((d>=0).sum()))
    if c[2]=="ordinary" and agent is not None:
        # Ordinary shared branch must keep identical participant outcomes.
        for k in ("delivery","first","conflicts"):
            assert arms["learned"][k]==arms["base"][k]
    return dict(panel=panel,seed=c[1],family=c[2],index=c[3],roster=r,arms=arms,
                water=read(s.inputs(c)/"input.json")["water_sha"],population=sha(s.history(c,r)))


def main():
    if sys.argv[1]=="fit":
        train()
        return
    fitted=read(WEIGHTS)
    assert all(fitted[k]==sha(HERE/n) for k,n in (("source","occupancy44.py"),("pipeline","study_occupancy44.py"),
               ("protocol","occupancy-protocol.md")))
    (FRESH/"population").mkdir(parents=True,exist_ok=True)
    fresh=[("fresh",("test",2026110890+i,f,i),0) for i in range(8) for f in ("continuous","rewet")]
    old=[(p,c,r) for p,r in (("a",0),("b",1)) for c in s.cases(p)]
    assert not {c[1] for _,c,_ in fresh}.intersection({r["seed"] for r in fitted["training"]}|{c[1] for _,c,_ in old})
    historical=read(HERE/"fleet-options-evaluate.json")["rows"]
    refs={(r["seed"],r["family"],r["roster"]):r["arms"] for r in historical}
    rows=[]
    # Separate pools so fresh generator paths cannot escape into the A/B reader.
    for panel,work in (("fresh",fresh),("mixed",old)):
        with ProcessPoolExecutor(4) as workers:
            for r in workers.map(evaluate,work):
                if panel=="mixed":
                    ref=refs[r["seed"],r["family"],r["roster"]]
                    assert all(r["arms"][a][k]==ref[a][k] for a in ("base","pure") for k in ("delivery","first","conflicts"))
                rows.append(r)
        subset=[r for r in rows if (r["panel"]=="fresh")==(panel=="fresh")]
        save(HERE/f"occupancy-{panel}-result.json",dict(rows=subset,fit=sha(WEIGHTS)))
        totals={m:[sum(r["arms"][m][k] for r in subset) for k in ("found","sent")] for m in ("base","learned","pure")}
        print(panel,totals,flush=True)


if __name__=="__main__":
    main()
