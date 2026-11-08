"""Materialize paired mission histories from the frozen plus5 exposure records."""
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import json
import hashlib
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
V4=ROOT/"uav-exposure-plus5-20261105"
V3=ROOT/"uav-exposure-mixed50-v3-20261105"
EXP=ROOT/"hyra-uav-pr-20261101/uav-workbench/experiments"
U=EXP/"flood-drop-sector5-v1/unified_supply"
for p in (U,U/"mission85",U/"stress_search",U/"stress_search/lookahead",ROOT/"uav-ordinary-exposure-v2-20261105"):
    sys.path.insert(0,str(p))
from evaluate import geometry, FloodSolver, response, K
from population import sample_water, Drift, Config
from mission_env import MissionSupply
from planner import Planner
from route import Lookahead

def read(path): return json.loads(Path(path).read_text(encoding="utf-8-sig"))
def sha(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def save(path,obj):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False),encoding="utf8")
    temp.replace(path)
def quota():
    return
@lru_cache(maxsize=1)
def manifest():
    ans={}
    for split,filename in (("development","regression.json"),("final","independent.json")):
        rows=read(V4/filename)["updated"]
        for i,r in enumerate(rows):
            seed=r["seed"]
            assert seed not in ans
            fold="final" if split=="final" else ("fit" if (i<18 or 30<=i<42) else "val")
            path=(V4 if split=="final" else V3)/f"ord-{seed}-p1"/"mission.npy" if i<30 else V4/f"stress-{seed}"/"fields-0.35.npy"
            ans[seed]=(fold,r,path,i)
    assert len(ans)==100
    return ans
def water(path):
    a=np.load(path,mmap_mode="r")
    return np.broadcast_to(a[:,:,None,None],(481,3,100,100)) if a.ndim==2 else a
def trap_layout(seed,r,i):
    if r["family"]!="ordinary":
        lo,hi=(200,800) if ((i-30)//2)%2==0 else (50,950)
        return np.random.default_rng(np.random.SeedSequence([7811,seed])).uniform(lo,hi,(12,2))
    bed,h,peak,cells=geometry(seed)
    sim=FloodSolver(bed,h,10.,boundary="closed",max_timestep=2.5,spatial_order=1,time_integrator="euler",manning=.035)
    sim.set_source_schedule([dict(cells=cells,hydrograph=[[0.,1.],[300.,peak/3],[750.,peak/6],[1200.,.5],[1500.,0.]])])
    # Original make_scenario selects the wet reference frame at (2400-660)//2.
    for t in range(0,871,5):sim.advance(float(t))
    wet=np.argwhere(sim.h.astype("f4")>=.01)
    rng=np.random.default_rng(np.random.SeedSequence(seed+10000000).spawn(4)[1])
    return wet[rng.permutation(len(wet))[:12],::-1]*10.+5
def history(water,layout,seed,rep,traps):
    ids=np.array(layout["xy_flat"])
    yy,xx=np.unravel_index(ids,(100,100))
    pos=np.column_stack([xx,yy])*10.+5
    at=water[:,:,yy,xx].astype(float)
    thresholds=np.asarray(layout["clock"])
    dose=np.zeros(48);fall=np.full(48,2401,int)
    for k in range(480):
        delta=at[k+1]-at[k]
        for sub in range(5):
            h,u,v=at[k]+((sub+.5)/5)*delta
            dose+=K*response(h,np.hypot(u,v))
            fall[(fall>2400)&(dose>=thresholds)]=5*k+sub+1
    assert np.array_equal(fall<=2400,layout["first_fall_by_end"])
    cfg=Config(duration=2400,window=1320,people=48,sensor=20.,trigger_depth=2.,trigger_speed=1.)
    salt=int(np.random.SeedSequence([7812,seed,rep]).generate_state(1)[0])
    motion=SimpleNamespace(cfg=cfg,water=water,person=pos.copy(),traps=traps,status=np.zeros(48,int),
             stuck=np.zeros(48,bool),stop_kind=np.zeros(48,int),refloat_counts=np.zeros(48,int),
             scenario={"seed":salt,"events":[{} for _ in range(48)]},refloat_enabled=True,events=[])
    motion.refloat_draw=lambda j,now: Drift.refloat_draw(motion,j,now)
    points=np.empty((2401,48,2));avail=np.empty((2401,48),bool);velocity=np.zeros((481,48,2))
    for t in range(2401):
        motion.status[(motion.status==0)&(fall<=t)]=1
        points[t]=motion.person;avail[t]=motion.status!=4
        if t%5==0:
            active=(motion.status==1)&~motion.stuck
            if active.any():
                value=sample_water(water,t,motion.person[active])
                velocity[t//5,active]=np.where(value[:,:1]>=.01,value[:,1:],0.)
        if t<2400:Drift.move_people(motion,t)
    assert np.isfinite(points).all() and np.isfinite(velocity).all() and avail[0].all()
    return dict(points=points,available=avail,velocity=velocity,fall=fall)
def make(seed):
    folder=OUT/"histories"/str(seed);folder.mkdir(exist_ok=True,parents=True)
    end=folder/"manifest.json"
    if end.exists():return
    fold,r,path,i=manifest()[seed];assert sha(path)==r["water_hash"]
    w=water(path);traps=trap_layout(seed,r,i)
    assert traps.shape==(12,2)
    hashes=[]
    for rep,layout in enumerate(r["layouts"]):
        result=history(w,layout,seed,rep,traps)
        p=folder/f"{rep}.npz";np.savez_compressed(p,**result);hashes.append(sha(p))
    save(end,dict(seed=seed,fold=fold,family=r["family"],water=str(path),water_hash=r["water_hash"],traps=traps.tolist(),histories=hashes))
def load_scene(seed,rep):
    folder=OUT/"histories"/str(seed);rec=read(folder/"manifest.json")
    with np.load(folder/f"{rep}.npz") as a:pop=dict(a)
    return water(rec["water"]),pop,rec
def main():
    split=sys.argv[1] if len(sys.argv)>1 else "dev"
    ids=[s for s,(f,*_) in manifest().items() if (f!="final")== (split=="dev")]
    for start in range(0,len(ids),10):
        quota()
        with ProcessPoolExecutor(max_workers=8) as pool:list(pool.map(make,ids[start:start+10]))
        print("prepared",split,min(len(ids),start+10),flush=True)
if __name__=="__main__":main()

