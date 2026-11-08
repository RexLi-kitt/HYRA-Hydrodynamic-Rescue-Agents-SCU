"""Frozen three-family exposure calibration, independent of rescue policy outcomes."""
from pathlib import Path
import itertools
import json
import sys
import time
import hashlib
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / "uav-ordinary-exposure-v2-20261105"))
from evaluate import geometry, FloodSolver, risk, response, pools, K, save, read, sha

THRESHOLDS = (.25, .35, .45, .55, .65, .75)
SPEED_LOWS = (.44, .48, .52, .56, .60)
FIT_ORD = list(range(1104001,1104007))
FIT_STRESS = list(range(64100001,64100005))
MAIN_ORD = list(range(64110001,64110031))
MAIN_STRESS = list(range(64120001,64120021))


def get_ordinary(seed, scale):
    folder = OUT / f"ord-{seed}-p{scale}"
    folder.mkdir(exist_ok=True)
    path = folder / "mission.npy"
    audit_path = folder / "water.json"
    if audit_path.exists():
        audit=read(audit_path)
        assert audit["sha256"] == sha(path)
        return np.load(path, mmap_mode="r"), audit
    bed,h,base_peak,sources = geometry(seed)
    peak = scale * base_peak
    graph = [[0.,3.],[300.,peak],[2700.,peak*.5],[3000.,0.]]
    solver = FloodSolver(bed,h,10., boundary="closed",max_timestep=2.5,
                         spatial_order=1,time_integrator="euler",manning=.035)
    solver.set_source_schedule([dict(cells=sources,hydrograph=graph)])
    field = np.lib.format.open_memmap(path,mode="w+",dtype="<f4",shape=(481,3,100,100))
    st=time.monotonic()
    initial_volume = 0.
    for t in range(0,3001,5):
        solver.advance(float(t))
        if t<600:
            continue
        u,v=solver.velocity()
        frame=np.stack((solver.h,u,v))
        if not np.isfinite(frame).all() or frame[0].min()<0:
            raise ValueError(f"Invalid hydraulic state {seed} {t}")
        if t==600:
            initial_volume=float(100*solver.h.sum())
        field[(t-600)//5]=frame
    field.flush()
    speed=np.hypot(field[:,1],field[:,2])[field[:,0]>=.01]
    wanted=sum(q*(b-a) for (a,q),(b,_) in zip(graph,graph[1:]))
    added=solver.balance["scheduled_inflow_m3"]
    residual=float(100*(solver.h.sum()-h.sum())-added)
    mission_residual=float(100*solver.h.sum()-initial_volume-peak*(2100+150))
    p99,top=float(np.quantile(speed,.99)),float(speed.max())
    valid=(max(abs(added-wanted),abs(residual),abs(mission_residual))<=max(1e-5,wanted*1e-9)
           and p99<=2 and top<=4)
    audit=dict(seed=seed,scale=scale,graph=graph,cells=sources,hash_dtype="5s-f32-huv",
               sha256=sha(path),whole_residual_m3=residual,window_residual_m3=mission_residual,
               source_integrated_m3=added,source_expected_m3=wanted,finite_nonnegative=True,
               wet_p99=p99,wet_max=top,valid=bool(valid),runtime_s=time.monotonic()-st)
    save(audit_path,audit)
    return field,audit


def test_people(seed, ids, dose):
    # dose vector is indexed by the eligible site list; an independent stream draws clocks.
    if len(ids)<48:
        return dict(valid=False,sites=len(ids))
    layouts=[]
    for rep in range(3):
        sr,cr=np.random.SeedSequence([91821,seed,rep]).spawn(2)
        at=np.random.default_rng(sr).choice(len(ids),48,replace=False)
        clocks=np.random.default_rng(cr).exponential(size=48)
        falls=dose[at]>=clocks
        layouts.append(dict(count=int(falls.sum()),xy_flat=ids[at].tolist(),clock=clocks.tolist(),
                            first_fall_by_end=falls.tolist()))
    return dict(valid=True,sites=len(ids),standing=float(np.exp(-dose).mean()),layouts=layouts)


def ordinary(seed,scale,thresholds):
    field,audit=get_ordinary(seed,scale)
    path=OUT/f"ord-{seed}-p{scale}"/"exposures.json"
    if path.exists():
        rows=read(path)
        if all(str(v) in rows for v in thresholds):
            return rows
    dose=risk(field).ravel()
    _,wet,speed=pools(field[0],audit["cells"])
    vals=speed[wet]
    cap=np.sort(vals)[-min(96,len(vals))] if len(vals) else 0.
    rows={}
    for threshold in thresholds:
        actual=min(float(cap),threshold)
        ids=np.flatnonzero(wet & (speed>=actual))
        row=test_people(seed,ids,dose[ids])
        row.update(seed=seed,family="ordinary",cutoff=actual,nominal=threshold,
                   water_hash=audit["sha256"],valid=audit["valid"] and row["valid"])
        rows[str(threshold)]=row
    save(path,rows)
    return rows


def stress(seed,idx,lo):
    rng=np.random.default_rng(seed)
    h0=float(rng.uniform(2.05,2.8)); speed=float(rng.uniform(lo,lo+.1))
    direction=(idx%4)*np.pi/2
    levels=np.full(481,h0,dtype=np.float32)
    dry=[]
    if idx%2:
        for center in (450,1030,1600):
            begin=center+int(rng.integers(-60,61))
            dry.append([begin,begin+40])
            levels[begin//5:(begin+40)//5+1]=0.
    uv=np.column_stack((levels,np.where(levels>0,speed*np.cos(direction),0.),
                        np.where(levels>0,speed*np.sin(direction),0.))).astype(np.float32)
    assert np.isfinite(uv).all() and (uv[:,0]>=0).all()
    pairs=uv.astype(float)
    dose=0.
    for start,end in zip(pairs[:-1],pairs[1:]):
        mid=start+((np.arange(5)+.5)/5)[:,None]*(end-start)
        dose+=float(K*response(mid[:,0],np.hypot(mid[:,1],mid[:,2])).sum())
    ids=np.setdiff1d(np.arange(10000),[0,99,9900,9999])
    folder=OUT/f"stress-{seed}"
    folder.mkdir(exist_ok=True)
    path=folder/f"fields-{lo:g}.npy"
    np.save(path,uv)
    row=test_people(seed,ids,np.full(len(ids),dose))
    row.update(seed=seed,family="rewet" if dry else "continuous",depth=h0,speed=speed,dry=dry,
               direction_rad=direction,water_hash=sha(path),uniform_field=True)
    save(folder/f"exposure-{lo:g}.json",row)
    return row


def stats(rows):
    valid=[r for r in rows if r["valid"]]
    if not valid:
        return dict(n=len(rows),valid=0)
    ps=[r["standing"] for r in valid]
    counts=[k["count"] for r in valid for k in r["layouts"]]
    main=[r["layouts"][0]["count"] for r in valid]
    return dict(n=len(rows),valid=len(valid),mean=float(np.mean(ps)),min_fall=1-max(ps),max_fall=1-min(ps),
                expected_floor=sum(1-x>=.50 for x in ps),actual_main_floor=sum(n>=24 for n in main),
                actual_all_floor=sum(n>=24 for n in counts),main_falls=sum(main),
                all_falls=sum(counts),main_min=min(main),main_max=max(main),robustness_min=min(counts))


def fit():
    decisions=[]
    stress_table={lo:[stress(seed,i,lo) for i,seed in enumerate(FIT_STRESS)] for lo in SPEED_LOWS}
    for scale in (1,2,3):
        hydro=[ordinary(s,scale,THRESHOLDS) for s in FIT_ORD]
        for th,lo in itertools.product(THRESHOLDS,SPEED_LOWS):
            ordinary_rows=[v[str(th)] for v in hydro]
            stress_rows=stress_table[lo]
            results=stats(ordinary_rows+stress_rows)
            if results["valid"]!=10:
                continue
            # Six ordinary, two continuous and two rewet = main mixture.
            passed=(.08<=results["mean"]<=.12 and results["expected_floor"]==10
                    and results["actual_all_floor"]==30)
            decisions.append(dict(scale=scale,th=th,lo=lo,passed=passed,**results))
        save(OUT/"fit-candidates.json",decisions)
        winners=[a for a in decisions if a["passed"]]
        print("FIT",scale,"mean standing range",min(a["mean"] for a in decisions),
              max(a["mean"] for a in decisions),"pass",len(winners),flush=True)
        if winners:
            return min(winners,key=lambda a:(abs(a["mean"]-.10),a["th"],a["lo"]))


def main():
    decision_file=OUT/"fit-lock.json"
    if decision_file.exists():
        record=read(decision_file)
        assert record["protocol"]==sha(OUT/"protocol.json") and record["amendment"]==sha(OUT/"pre-run-amendment.json")
        winner=record["selected"]
    else:
        winner=fit()
        save(decision_file,dict(selected=winner,protocol=sha(OUT/"protocol.json"),
             amendment=sha(OUT/"pre-run-amendment.json"),source=sha(Path(__file__))))
    if not winner:
        print("no candidate accepted; no validation started",flush=True)
        return
    rows=[]
    for seed in MAIN_ORD:
        row=ordinary(seed,winner["scale"],[winner["th"]])[str(winner["th"])]
        rows.append(row)
        print("validated",seed,row.get("standing"),flush=True)
    rows.extend(stress(s,i,winner["lo"]) for i,s in enumerate(MAIN_STRESS))
    groups={g:stats([q for q in rows if q["family"]==g]) for g in ("ordinary","continuous","rewet")}
    save(OUT/"main.json",dict(groups=groups,overall=stats(rows),rows=rows,decision_hash=sha(decision_file)))
    print(json.dumps(dict(groups=groups,overall=stats(rows)),indent=2),flush=True)


if __name__=="__main__":
    main()

