from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from train15 import *
from support import load_scene

LOW=np.array([.12,0.,.05,0.,-1.2,.12])
HIGH=np.array([1.,1.6,3.,1.4,1.2,1.])
def candidates():
    rng=np.random.default_rng(50617)
    theta=LOW+(HIGH-LOW)*rng.random((8,6))
    theta=np.vstack([theta,FIXED,[.8,.12,1.7,.25,.0,.65]])
    return {f"initial{i}":dict(theta=row.tolist(),scope="all" if i%3==0 else "pressure",joint=True,weights=True) for i,row in enumerate(theta)}
def make_agent(k,config):
    with np.load(OUT/"routes.npz") as z:weights=dict(z) if config["weights"] else None
    return FleetPaths(np.array(config["theta"]),weights,config["scope"],config["joint"])
def path_for(stage,c,i):return OUT/stage/f"{c[1]}-{c[2]}-{i}.json"
def run_one(job):
    stage,c,i,configs=job
    path=path_for(stage,c,i)
    if path.exists():
        assert list(read(path)["models"])==list(configs)
        return
    f,p=load_scene(c,i) if stage=="dev" else material(c,i)
    cls,_=runtime(1,"planning")
    rows={}
    for k,config in configs.items():
        a=make_agent(k,config);env=cls(f,p,(64,720,2400))
        while env.t<2400:
            env.step(*a.act(env.observation()))
            if (env.supplied>=0).all() and (env.last>=0).all():break
        m=env.metrics()
        assert m["total"]==64 and sum(t>=0 for t in m["delivery_s"])==m["delivered"]
        rows[k]=dict(**m,found=(env.seen_at>=0).tolist(),replans=a.replans,route_actions=a.flown)
    save(path,dict(location=c[0],seed=c[1],family=c[2],rep=i,models=rows))
def run(stage,panel,reps,configs):
    (OUT/stage).mkdir(exist_ok=True)
    jobs=[(stage,c,i,configs) for c in panel for i in reps]
    batch(run_one,[j for j in jobs if not path_for(*j[:3]).exists()],6)
    rows=[read(path_for(stage,c,i)) for _,c,i,_ in jobs]
    assert all(list(r["models"])==list(configs) for r in rows)
    return rows
def rank(rows,configs):
    scores={}
    for k in configs:
        by_family=[np.mean([r["models"][k]["delivered"] for r in rows if r["family"]==fam]) for fam in s.FAMILIES]
        scores[k]=.6*by_family[0]+.2*(by_family[1]+by_family[2])
    return sorted(scores,key=scores.get,reverse=True),scores
def tune():
    baseline=candidates()
    fit=[c for c in cases() if c[3]<3]
    assert len(fit)==18
    rows=run("initial-fit",fit,[0],baseline)
    ranked,values=rank(rows,baseline)
    print("FIRST",ranked[:3],values,flush=True)
    # One population update: breed around the best three settings on complete delivered counts.
    elite=ranked[:3];second={k:baseline[k] for k in elite}
    center=np.mean([baseline[k]["theta"] for k in elite],axis=0)
    rng=np.random.default_rng(70021)
    for j in range(5):
        parent=baseline[elite[j%3]]
        mix=.55*np.array(parent["theta"])+.45*center+rng.normal(0,.12,6)*(HIGH-LOW)
        second[f"updated{j}"]=dict(theta=np.clip(mix,LOW,HIGH).tolist(),scope=parent["scope"],joint=True,weights=True)
    next_rows=run("second-fit",fit,[1],second)
    ranked2,values2=rank(next_rows,second)
    keep=ranked2[:2]
    other=next((k for k in ranked2 if second[k]["scope"]!=second[keep[0]]["scope"]),None)
    if other and other not in keep:keep.append(other)
    formal={k:second[k] for k in keep}
    for k in keep:formal["off_"+k]={**second[k],"weights":False}
    formal["solo_"+keep[0]]={**second[keep[0]],"joint":False}
    formal["fixed_joint"]=dict(theta=FIXED.tolist(),scope="pressure",joint=True,weights=False)
    formal["fixed_all"]={**formal["fixed_joint"],"scope":"all"}
    formal["fixed_solo"]={**formal["fixed_joint"],"joint":False}
    save(OUT/"tuning.json",dict(initial=baseline,first_values=values,second_population=second,second_values=values2,final=formal))
    print("NOMINATED",keep,flush=True)
    dev=run("dev",s.cases("a")+s.cases("b"),[0,1],formal)
    counts={k:sum(r["models"][k]["delivered"] for r in dev) for k in formal}
    previous=read(V14/"final-dev.json")
    for r in dev:
        f=f"{r['seed']}-{r['family']}-{r['rep']}.json"
        ref=read(V14/"dev"/f)
        assert all(ref[n]==r[n] for n in ("seed","family","rep"))
        for g in (ref,read(V14/"gate-dev"/f)):
            assert all(z["total"]==64 for z in g["methods"].values())
    oldpure=read(V14/"development.json")["purenames"]+["geometry_gate"]
    pure={**{k:previous["counts"][k] for k in oldpure},**{k:counts[k] for k,v in formal.items() if not v["weights"]}}
    strongest=max(pure,key=pure.get)
    chosen=max(keep,key=counts.get)
    panel=[]
    for g in ("a","b"):
        subset=[r for r in dev if r["location"]==g];assert len(subset)==40
        def refvalue(r):
            if strongest in formal:return r["models"][strongest]["delivered"]
            f=f"{r['seed']}-{r['family']}-{r['rep']}.json"
            return read(V14/("gate-dev" if strongest=="geometry_gate" else "dev")/f)["methods"][strongest]["delivered"]
        panel.append([sum(r["models"][chosen]["delivered"] for r in subset),sum(refvalue(r) for r in subset)])
    assert sum(p[1] for p in panel)==pure[strongest]
    rel=counts[chosen]/pure[strongest]-1
    result=dict(selected=chosen,strongest=strongest,pure=pure,counts=counts,panels=panel,total=5120,rate=counts[chosen]/5120,gain=rel,
        gate=(counts[chosen]/5120>=.85 or rel>.15 or (rel>=.05 and all(a>b for a,b in panel))),
        hashes={n:sha(OUT/n) for n in ("routes.npz","protocol.json","paths15.py","train15.py","tune15.py","fit.json","tuning.json")})
    save(OUT/"development.json",result);print("DEV",result,flush=True)
if __name__=="__main__":tune()
