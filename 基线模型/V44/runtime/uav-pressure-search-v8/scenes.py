"""Versioned environment data. Split IDs and calibration are fixed in protocol.json."""
import sys
from pathlib import Path
D=Path(__file__).resolve().parent
R=D.parent
OLD=R/"uav-short-window-training-20261105"
sys.path[:0]=[str(OLD),str(R/"uav-protocol-audit-paired-20261105")]
from common import np,read,save,sha,quota,trap_layout,response,U,water,V3,V4
import paired
from motion import simulate
sys.path.insert(0,str(V3))
import calibrate as original
from evaluate import pools
from concurrent.futures import ProcessPoolExecutor

NREP={"fit":2,"val":3,"test":3}
def cases(fold):
    ordinary,stress,n,m={"fit":(64110001,88510001,24,8),"val":(72110001,88520001,12,4),"test":(93251001,93261001,36,12)}[fold]
    return [(fold,ordinary+i,"ordinary",i) for i in range(n)]+[(fold,stress+i,f,i) for i in range(m) for f in ["continuous","rewet"]]
def folder(c):
    root = D if c[0] == "test" else R/"uav-revised-delivery-training-v5"
    return root/"scenes"/f"{c[1]}-{c[2]}"
def shape(h,v):
    return h/(h+.05)*response(h,v)+.02*(h/(h+1.))**4
def cumulative(a):
    a=np.asarray(a,dtype=float)
    first=a[:-1]
    delta=a[1:]-first
    rate=0.
    for i in range(5):
        h,u,v=np.moveaxis(first+((i+.5)/5)*delta,1,0)
        rate=rate+shape(h,np.hypot(u,v))
    return rate.sum(axis=0)
def make_inputs(c):
    fold,seed,kind,idx=c
    store=folder(c);store.mkdir(parents=True,exist_ok=True)
    if (store/"input.json").exists():return
    if kind=="ordinary":
        if fold=="test":
            (D/"hydraulics").mkdir(exist_ok=True)
            original.OUT=D/"hydraulics"
            field,audit=original.get_ordinary(seed,1)
            path=original.OUT/f"ord-{seed}-p1"/"mission.npy"
        else:
            path=(V3 if fold=="fit" else V4)/f"ord-{seed}-p1"/"mission.npy"
            audit=read(path.parent/"water.json")
            field=water(path)
        assert audit["valid"] and audit["sha256"]==sha(path)
        _,mask,speed=pools(field[0],audit["cells"])
        cutoff=min(.55,np.sort(speed[mask])[-min(mask.sum(),96)])
        ids=np.flatnonzero(mask & (speed>=cutoff))
        yy,xx=np.unravel_index(ids,(100,100))
        frame=field[:457,:,yy,xx]
        dose=cumulative(frame)
        traps=trap_layout(seed,dict(family="ordinary"),idx)
    else:
        base=paired.base(seed)
        lo,hi=(200,800) if (idx//4+idx%2)%2==0 else (50,950)
        traps=np.random.default_rng(np.random.SeedSequence([7811,seed])).uniform(lo,hi,(12,2))
        frames=paired.fields(base,90*(idx%4))[kind=="rewet"]
        path=store/"field.npy";np.save(path,frames)
        ids=np.setdiff1d(np.arange(10000),[0,99,9900,9999])
        dose=np.atleast_1d(cumulative(frames[:457]))
    assert len(ids)>=48 and np.isfinite(dose).all()
    np.savez(store/"eligible.npz",ids=ids,dose=dose)
    save(store/"input.json",dict(fold=fold,seed=seed,family=kind,water=str(path),water_sha=sha(path),traps=traps.tolist(),direction=90*(idx%4) if kind!="ordinary" else None))
def calibrate_k():
    doses=[]
    for c in cases("fit"):
        with np.load(folder(c)/"eligible.npz") as a:doses.append(a["dose"].copy())
    lo,hi=0.,.1
    for _ in range(65):
        mid=(lo+hi)/2
        if np.mean([np.exp(-mid*v).mean() for v in doses])>.15:lo=mid
        else:hi=mid
    save(D/"risk-lock.json",dict(k=(lo+hi)/2,protocol=sha(D/"protocol.json"),shape_source=sha(Path(__file__)),fit_maps=len(doses),target=.15))
def generate(c):
    fold,seed,kind,idx=c;store=folder(c)
    k=read(D/"risk-lock.json")["k"];inp=read(store/"input.json")
    f=water(inp["water"])
    with np.load(store/"eligible.npz") as a:ids,dose=a["ids"],a["dose"]
    records=[]
    for rep in range(NREP[fold]):
        posrng,clockrng=map(np.random.default_rng,np.random.SeedSequence([91821,seed,rep]).spawn(2))
        selected=posrng.choice(ids,48,replace=False);clocks=clockrng.exponential(size=48)
        yy,xx=np.unravel_index(selected,(100,100))
        a=f[:,:,yy,xx].astype(float);delta=a[1:]-a[:-1]
        hazard=np.empty((480,5,48))
        for i in range(5):
            h,u,v=np.moveaxis(a[:-1]+(i+.5)/5*delta,1,0)
            hazard[:,i]=k*shape(h,np.hypot(u,v))
        threshold=hazard.reshape(2400,48).cumsum(axis=0)>=clocks
        fall=np.where(threshold.any(axis=0),1+threshold.argmax(axis=0),2401)
        pop=simulate(f,selected,fall,seed,rep,np.array(inp["traps"]))
        path=store/f"hist-{rep}.npz"
        np.savez_compressed(path,**pop)
        records.append(dict(rep=rep,xy=selected.tolist(),clock=clocks.tolist(),fall_s=fall.tolist(),natural_falls=int((fall<=2280).sum()),history_sha=sha(path)))
    save(store/"manifest.json",dict(**inp,never_expected=float(np.exp(-k*dose).mean()),eligible=int(len(ids)),layouts=records))
def load(c,rep):
    store=folder(c);rec=read(store/"manifest.json")
    with np.load(store/f"hist-{rep}.npz") as a:pop=dict(a)
    return water(rec["water"]),pop
def parallel(fn,items):
    if not items:return
    for i in range(0,len(items),8):
        quota()
        with ProcessPoolExecutor(max_workers=min(8,len(items)-i)) as pool:list(pool.map(fn,items[i:i+8]))
        print(fn.__name__,min(i+8,len(items)),len(items),flush=True)
if __name__=="__main__":
    fold=sys.argv[1];jobs=cases(fold)
    if fold=="test":assert (D/"model-lock.json").exists()
    parallel(make_inputs,[c for c in jobs if not (folder(c)/"input.json").exists()])
    if fold=="fit" and not (D/"risk-lock.json").exists():calibrate_k()
    parallel(generate,[c for c in jobs if not (folder(c)/"manifest.json").exists()])

