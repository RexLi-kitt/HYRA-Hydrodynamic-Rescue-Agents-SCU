"""Run paired training, development selection, and sealed holdout evaluation."""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from common import *
from planner import H,V,record,train,smooth
def gaussian_filter(a,sigma):
    x=np.arange(-3,4)
    weights=np.exp(-.5*(x/sigma)**2)
    weights/=weights.sum()
    for axis in (0,1):
        a=np.apply_along_axis(lambda v:np.convolve(np.pad(v,3,mode="symmetric"),weights,mode="valid"),axis,a)
    return a

FROZEN=ROOT/"uav-unified-supply-20261105/run02/occupancy-mlp.npy"
WEIGHT=OUT/"fit-weights.npz"
TIMES=[1680,1800,1920,2040,2160,2280,2400]
WINDOWS=[600,780,960,1140,1320]
CONTROLS=["water","endpoint","frozen"]
CANDIDATES=["refit","dispatch","spatial","concentrated"]
def policy(method):
    if method in ("endpoint","water"):return Planner(method,10)
    if method=="frozen":return Lookahead(np.load(FROZEN),[1,4,35],[.5,0,2])
    with np.load(WEIGHT) as data:
        if method=="refit":return Lookahead(data["theta"],[1,4,35],[.5,0,2])
        if method in ("tracker","urgent","spread"):
            from tracker import Tracking
            return Tracking(data["theta"],data["prior"],method)
        from search import Cooperative
        return Cooperative(data["theta"],data["prior"],method)

def study_cells(o,geo):
    ids=o.new_cells
    if not len(ids):return
    coarse=(ids//100//5)*20+(ids%100//5)
    geo[0]+=np.bincount(coarse,minlength=400)
    xy=(o.seen_xy[o.visible]//10).astype(int).clip(0,99)
    hits=100*xy[:,1]+xy[:,0]
    at=np.searchsorted(ids,hits)
    overlap=(at<len(ids))&(ids[at.clip(0,len(ids)-1)]==hits)
    geo[1]+=np.bincount(coarse[at[overlap]],minlength=400)
def scenario(group):
    s,rep,stage,methods,windows,duration=group
    path=OUT/stage;path.mkdir(exist_ok=True)
    file=path/f"{s}-{rep}.json"
    if file.exists():return
    water,pop,meta=load_scene(s,rep)
    rows=[]
    for method in methods:
        for window in windows:
            agent=policy(method);env=MissionSupply(water,pop,(48,window,duration))
            counts=np.zeros((2,len(H)*len(V)));geo=np.zeros((2,400))
            collect=stage=="base"
            while env.t<duration:
                obs=env.observation()
                if collect:
                    record(obs,counts);study_cells(obs,geo)
                env.step(*agent.act(obs))
                if (env.supplied>=0).all() and np.all(env.last>=0):break
            result=env.metrics()
            result.update(seed=s,rep=rep,family=meta["family"],method=method,window=window,duration=duration,
                natural_fall_time_s=pop["fall"].tolist(),first_seen_available_ever=(env.seen_at>=0).tolist())
            rows.append(result)
            if collect:
                np.savez(path/f"{s}-{rep}-{method}.npz",counts=counts,geo=geo)
    save(file,rows)

def tasks(fold,stage,methods,windows,duration=2400,reps=None):
    reps=range(2 if fold=="fit" else 3) if reps is None else reps
    return [(s,i,stage,methods,windows,duration) for s,(f,*_) in manifest().items() if f==fold for i in reps]
def run(jobs):
    quota()
    todo=[j for j in jobs if not (OUT/j[2]/f"{j[0]}-{j[1]}.json").exists()]
    if not todo:return
    print("Starting",todo[0][2],len(todo),"paired layouts",flush=True)
    started=time.perf_counter()
    # A bounded single local campaign. At most one unit per worker can be pending at a quota stop.
    from concurrent.futures import wait,FIRST_COMPLETED
    with ProcessPoolExecutor(max_workers=12) as pool:
        queue=iter(todo); pending={}
        def submit():
            item=next(queue,None)
            if item:pending[pool.submit(scenario,item)]=item
        for _ in range(min(12,len(todo))):submit()
        n=0
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for job in done:
                del pending[job];job.result();n+=1
                if n%10==0 or n==len(todo):
                    print(todo[0][2],n,"/",len(todo),round(time.perf_counter()-started,1),"s",flush=True)
                    quota()
                submit()
def rows(stage):
    return [row for p in sorted((OUT/stage).glob("*.json")) for row in read(p)]
def learn():
    counts=np.zeros((2,len(H)*len(V)));geo=np.zeros((2,400))
    files=list((OUT/"base").glob("*.npz"));assert len(files)==180
    for f in files:
        with np.load(f) as data:
            counts+=data["counts"];geo+=data["geo"]
    theta,trace=train(counts,20261105)
    hits,samples=geo[1].reshape(20,20),geo[0].reshape(20,20)
    mean=hits.sum()/samples.sum()
    prior=(gaussian_filter(hits,.85)+25*mean)/(gaussian_filter(samples,.85)+25)
    np.savez(WEIGHT,theta=theta,prior=prior,counts=counts,geo=geo)
    save(OUT/"training.json",dict(steps=400,seed=20261105,trace=trace,fit_maps=30,paired_layouts=60,behavior_episodes=len(files),
       first_exposures=int(counts[0].sum()),first_exposure_positive_observations=int(counts[1].sum()),weights_sha=sha(WEIGHT),control_sha=sha(FROZEN)))
def score(rows_,m,w,t):
    values=[sum(0<=x<=t for x in row["delivery_s"])/48 for row in rows_ if row["method"]==m and row["window"]==w]
    if not values:raise ValueError((m,w,t))
    return float(np.mean(values))
def main():
    phase=sys.argv[1]
    if phase=="base":
        run(tasks("fit","base",CONTROLS,[1320]));learn()
    elif phase=="fit":
        run(tasks("fit","fit",CANDIDATES,[960]))
        data=rows("fit")
        ranking=sorted([(score(data,m,960,2040),m) for m in CANDIDATES],reverse=True)
        save(OUT/"screen.json",dict(screen_ranking=ranking,retain=list(dict.fromkeys(["frozen",ranking[0][1],ranking[1][1]]))))
    elif phase=="fit2":
        extra=["tracker","urgent","spread"]
        run(tasks("fit","fit2",extra,[960]))
        data=rows("fit")+rows("fit2")
        ranking=sorted([(score(data,m,960,2040),m) for m in CANDIDATES+extra],reverse=True)
        save(OUT/"screen-all.json",dict(screen_ranking=ranking,retain=["frozen",ranking[0][1],ranking[1][1]]))
    elif phase=="val":
        methods=read(OUT/"screen-all.json")["retain"]
        run(tasks("val","val",methods,WINDOWS))
        data=rows("val");grid=[]
        for w in WINDOWS:
            for t in TIMES:
                scores={m:score(data,m,w,t) for m in methods}
                best=max(methods,key=scores.get)
                grid.append(dict(w=w,t=t,best=best,score=scores[best],scores=scores))
        short=[x for x in grid if x["w"]<1320 and x["t"]<2400]
        choice=min(short,key=lambda x:(abs(x["score"]-.85),x["w"],x["t"]))
        save(OUT/"selection.json",dict(choice=choice,grid=grid,weight_sha=sha(WEIGHT),source_hashes={n:sha(OUT/n) for n in ["common.py","campaign.py","search.py","tracker.py","protocol.json","fit-extension.json"]}))
        # Direct shorter replay, independent of the prefix implementation.
        run(tasks("val","prefix-check",[choice["best"]],[choice["w"]],choice["t"],[0]))
        for direct in rows("prefix-check"):
            source=next(row for row in data if all(row[k]==direct[k] for k in ("method","seed","rep","window")))
            expected=[x if 0<=x<=choice["t"] else -1 for x in source["delivery_s"]]
            assert expected==direct["delivery_s"],direct["seed"]
        save(OUT/"prefix-verified.json",dict(maps=len(rows("prefix-check")),all_delivery_times_match=True))
    elif phase=="final":
        assert (OUT/"prefix-verified.json").exists()
        selected=read(OUT/"selection.json");choice=selected["choice"]
        assert sha(WEIGHT)==selected["weight_sha"] and all(sha(OUT/n)==digest for n,digest in selected["source_hashes"].items())
        run(tasks("final","final",list(dict.fromkeys([choice["best"]]+CONTROLS)),[choice["w"]],choice["t"]))
        run(tasks("final","anchor",[choice["best"]],[1320]))
if __name__=="__main__":main()

