"""Calibrate a synthetic inflow event and observable initial deployment on FIT only."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
EXP = ROOT / "hyra-uav-pr-20261101/uav-workbench/experiments/flood-drop-sector5-v1"
sys.path.insert(0, str(EXP / "speed_first"))
from hydraulics import FloodSolver, generate_seed
from fall_risk import response

PLAN = json.loads((OUT / "protocol.json").read_text())
BASE = ROOT / "uav-task85-20261105"
K = json.loads((EXP / "unified_supply/protocol.json").read_text())["hazard_k"]
WATER_LOCK = json.loads((BASE / "water-lock.json").read_text())
N = 48
DESIGNS = [(0., 0.)] + [(threshold, f) for threshold in (.15, .25) for f in (.25, .5, .75, 1.)]


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def geometry(seed):
    # Same RNG draw order and floating expressions as hydraulics.generate_seed.
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((100, 100), dtype=float)
    x, y = (xx + .5) * 10, (yy + .5) * 10
    center = rng.uniform(300,700) + rng.uniform(20,65) * np.sin(y / rng.uniform(140,250))
    bed = rng.uniform(.0001,.0003)*(1000-y) + rng.uniform(.000001,.000003)*(x-center)**2
    for _ in range(4):
        cx, cy = rng.uniform(50,950,2)
        bed += rng.uniform(-.1,.2)*np.exp(-((x-cx)**2+(y-cy)**2)/rng.uniform(15000,60000))
    bed -= bed.min()
    h = np.maximum(0, rng.uniform(.03,.08)-bed)
    peak = 3 * float(rng.uniform(8,18))
    c = int(center[15,0]/10)
    return bed, h, peak, [[15, c+i] for i in range(-2,3)]


def old_water(seed):
    if seed in PLAN["splits"]["fresh"]:
        path = OUT / "legacy" / f"{seed}-q3.npy"
        if not path.exists():
            generate_seed(seed, path.parent, 2400, 3.)
        assert sha(path) == read(path.with_suffix(".json"))["hash"]
    else:
        record = next(v for v in WATER_LOCK.values() if v["parent"] == f"synthetic-swe-{seed}")
        path = Path(record["path"])
        assert sha(path) == record["sha256"]
    return path


def fields(folder, seed, legacy, sites):
    path = folder / "mission.npy"
    config = folder / "water.json"
    if config.exists():
        audit = read(config)
        assert sha(path) == audit["sha256"]
        return np.load(path, mmap_mode="r"), audit
    start_time = time.monotonic()
    bed, h, peak, sources = geometry(seed)
    assert np.array_equal(h.astype("f4"), legacy[0,0]) and sites == sources
    solver = FloodSolver(bed, h, 10., boundary="closed", max_timestep=2.5,
                         spatial_order=1, time_integrator="euler", manning=.035)
    graph = [[0.,3.],[300.,peak],[1500.,peak/2],[2100.,1.5],[2400.,0.],[3000.,0.]]
    solver.set_source_schedule([dict(cells=sources, hydrograph=graph)])
    field = np.lib.format.open_memmap(path, mode="w+", dtype="<f4", shape=(481,3,100,100))
    pre_volume = None
    for physical in range(0, 3001, 5):
        solver.advance(float(physical))
        if physical < 600:
            continue
        u, v = solver.velocity()
        frame = np.stack((solver.h, u, v))
        assert np.isfinite(frame).all() and frame[0].min() >= 0
        if physical == 600:
            assert np.array_equal(frame.astype("f4"), legacy[120])
            pre_volume = float(solver.h.sum()*100)
        field[(physical-600)//5] = frame
    field.flush()
    budget = solver.summary()["balance"] if "balance" in solver.summary() else solver.balance
    added = solver.balance["scheduled_inflow_m3"]
    expected = sum(q*(b-a) for (a,q),(b,_) in zip(graph,graph[1:]))
    residual = float(solver.h.sum()*100 - h.sum()*100 - added)
    post_added = (1500-600)*peak + 600*peak/2 + 300*1.5
    post_residual = float(solver.h.sum()*100 - pre_volume - post_added)
    assert max(abs(added-expected),abs(residual),abs(post_residual)) < max(1.e-5,expected*1.e-9)
    speed = np.hypot(field[:,1],field[:,2])[field[:,0]>=.01]
    p99, max_speed = float(np.quantile(speed,.99)), float(speed.max())
    audit = dict(sha256=sha(path), graph=graph, cells=sources, elapsed_s=time.monotonic()-start_time,
                 residual_m3=residual, mission_residual_m3=post_residual, added_m3=added,
                 expected_added_m3=expected, p99=p99, max_speed=max_speed, valid=(p99<=2 and max_speed<=4))
    save(config, audit)
    return field, audit


def risk(field):
    dose = np.zeros((100,100))
    for start, end in zip(field[:-1], field[1:]):
        start = start.astype(float)
        delta = end - start
        for sub in range(5):
            h, u, v = start + ((sub+.5)/5)*delta
            dose += K*response(h,np.hypot(u,v))
    return dose


def pools(frame, sites):
    wet = frame[0]>=.01
    wet[[0,0,99,99],[0,99,0,99]]=False
    clean = wet.copy()
    for y,x in sites:
        clean[y,x] = False
    return wet, clean, np.hypot(frame[1], frame[2])


def evaluate(dose, wet, speed, threshold, share, seed):
    mobil = wet & (speed>=threshold)
    if threshold == 0:
        pools_sizes = [(np.flatnonzero(wet), N)]
    else:
        high = int(N*share)
        pools_sizes = [(np.flatnonzero(mobil), high), (np.flatnonzero(wet & ~mobil), N-high)]
    if any(len(ids)<n for ids,n in pools_sizes):
        return dict(valid=False, sites=[len(ids) for ids,n in pools_sizes])
    p = -np.expm1(-dose.ravel())
    theory = sum(n/N*float(p[ids].mean()) for ids,n in pools_sizes if n)
    layouts=[]
    for rep in range(3):
        loc_seed, fall_seed = np.random.SeedSequence([73418,seed,rep]).spawn(2)
        loc, clock = np.random.default_rng(loc_seed), np.random.default_rng(fall_seed)
        chosen = np.concatenate([loc.choice(ids,n,replace=False) for ids,n in pools_sizes if n])
        loc.shuffle(chosen)
        threshold_person = clock.exponential(size=N)
        fell = dose.ravel()[chosen]>=threshold_person
        assert len(chosen)==len(np.unique(chosen))==N
        layouts.append(dict(sites=chosen.tolist(), thresholds=threshold_person.tolist(), fell=fell.tolist(),
                            number=int(fell.sum())))
    return dict(valid=True, eligible=[len(ids) for ids,n in pools_sizes], p=theory, layouts=layouts)


def label(threshold, share):
    return "wet" if not threshold else f"v{threshold:g}-share{share:g}"


def process(seed, split, selected=None):
    folder=OUT/str(seed)
    folder.mkdir(exist_ok=True)
    path=folder/"scores.json"
    if path.exists():
        return read(path)
    legacy_path=old_water(seed)
    legacy=np.load(legacy_path,mmap_mode="r")
    sites=read(legacy_path.with_suffix(".json"))["source_cells"]
    old_dose=risk(legacy)
    prob=-np.expm1(-old_dose)
    oldwet, _, oldspeed=pools(legacy[0],sites)
    old=evaluate(old_dose,oldwet,oldspeed,0.,0.,seed)
    diagnostics=dict(top48_retrospective=float(np.sort(prob.ravel())[-48:].mean()),
                     initial_moving=int((oldwet & (oldspeed>=.15)).sum()))
    for t in (0,300,600):
        wet, clean, spd=pools(legacy[t//5],sites)
        diagnostics[str(t)]=dict(initial_wet=int(wet.sum()), current_moving=int((clean&(spd>=.15)).sum()),
                                 initial_wet_mean_risk_full_horizon=float(prob[wet].mean()))
    current,audit=fields(folder,seed,legacy,sites)
    dose=risk(current)
    wet,clean,speed=pools(current[0],sites)
    designs=DESIGNS if selected is None else [selected]
    scores={label(th,share):evaluate(dose,clean,speed,th,share,seed) for th,share in designs}
    allwet=evaluate(dose,clean,speed,0.,0.,seed)
    paired=evaluate(dose,oldwet,oldspeed,0.,0.,seed)
    result=dict(seed=seed,split=split,old=old,old_diagnostics=diagnostics,
                legacy_hash=sha(legacy_path),hydraulic=audit,extended_old_starts=paired,
                current_wet_unstratified=allwet,designs=scores)
    save(path,result)
    print(split,seed,round(old["p"],3),"->",[(key,round(v.get("p",0),3)) for key,v in scores.items()],flush=True)
    return result


def main():
    fit=[process(seed,"fit") for seed in PLAN["splits"]["fit"]]
    candidates=[]
    for th,share in DESIGNS:
        key=label(th,share)
        trials=[row["designs"][key] for row in fit]
        if not all(row["hydraulic"]["valid"] and trial["valid"] for row,trial in zip(fit,trials)):
            continue
        prob=[r["p"] for r in trials]
        misses=sum(lay["number"]<15 for r in trials for lay in r["layouts"])
        passes=min(prob)>=.30 and .40<=np.median(prob)<=.60
        candidates.append(dict(th=th,share=share,key=key,p=prob,misses=misses,passes=bool(passes)))
    valid=[c for c in candidates if c["passes"]]
    if not valid:
        save(OUT/"fit-decision.json",dict(selected=None,candidates=candidates))
        print("NO ELIGIBLE DESIGN",flush=True)
        return
    choice=min(valid,key=lambda c:(int(c["th"]!=0),c["misses"],abs(np.mean(c["p"])-.5),c["th"],c["share"]))
    decision=dict(selected=choice,candidates=candidates,protocol_hash=sha(OUT/"protocol.json"))
    decision_path=OUT/"fit-decision.json"
    if decision_path.exists():
        assert read(decision_path)==decision
    else:
        save(decision_path,decision)
    for split in ("check","fresh"):
        for seed in PLAN["splits"][split]:
            process(seed,split,(choice["th"],choice["share"]))


if __name__ == "__main__":
    main()

