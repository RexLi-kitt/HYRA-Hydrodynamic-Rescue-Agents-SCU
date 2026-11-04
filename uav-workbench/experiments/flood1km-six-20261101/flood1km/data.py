"""Generate independently seeded synthetic SWE parents; keep provenance and budgets."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def generate(folder,terrain,seed,split):
    os.environ['HYRA_FLOOD_BACKEND']='numpy'; sys.path.insert(0,str(terrain))
    from flood_solver import FloodSolver
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{split}-{seed}.npy'; meta=path.with_suffix('.json')
    if meta.exists():
        row=json.loads(meta.read_text()); assert row['sha256']==sha(path); return row
    if path.exists(): raise RuntimeError('Incomplete field exists; retain it and investigate')
    rng=np.random.default_rng(seed); yy,xx=np.indices((100,100),dtype=float); x=(xx+.5)*10;y=(yy+.5)*10
    center=rng.uniform(300,700)+rng.uniform(20,65)*np.sin(y/rng.uniform(140,250))
    bed=rng.uniform(.0001,.0003)*(1000-y)+rng.uniform(.000001,.000003)*(x-center)**2
    for _ in range(4):
        cx,cy=rng.uniform(50,950,2)
        bed+=rng.uniform(-.1,.2)*np.exp(-((x-cx)**2+(y-cy)**2)/rng.uniform(15000,60000))
    bed-=bed.min(); h=np.maximum(0,rng.uniform(.03,.08)-bed)
    solver=FloodSolver(bed,h,10.,boundary='closed',max_timestep=2.5,spatial_order=1,time_integrator='euler',manning=.035)
    c=int(center[15,0]/10); cells=[[15,c+k] for k in range(-2,3)]
    peak=float(rng.uniform(8,18)); graph=[[0.,1.],[300.,peak],[750.,peak*.5],[1200.,.5],[1500.,0.]]
    solver.set_source_schedule([dict(cells=cells,hydrograph=graph)])
    array=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(301,3,100,100))
    started=time.perf_counter(); masses=[]; max_residual=0.
    for k,t in enumerate(range(0,1501,5)):
        solver.advance(float(t)); u,v=solver.velocity(); f=np.stack((solver.h,u,v))
        if not np.isfinite(f).all() or (f[0]<0).any(): raise RuntimeError('Invalid hydraulic field')
        array[k]=f; masses.append(float(solver.h.sum()*100))
        if k%60==0: print(json.dumps(dict(seed=seed,time_s=t,elapsed_s=time.perf_counter()-started)),flush=True)
    array.flush(); del array
    summary=solver.summary()
    row=dict(seed=seed,split=split,parent=f'synthetic-swe-{seed}',scope='uncalibrated synthetic flood, no field accuracy claim',
             shape=[301,3,100,100],dx_m=10.,duration_s=1500,frame_dt_s=5,fields=['h','u','v'],
             forcing_m3_s=graph,source_cells=cells,numerics=solver.numerical_config(),summary=summary,
             finite_nonnegative=True,water_volume_initial_m3=masses[0],water_volume_final_m3=masses[-1],
             elapsed_s=time.perf_counter()-started,path=str(path.resolve()),sha256=sha(path),
             source_hashes={p.name:sha(p) for p in terrain.glob('flood_*.py')})
    # Closed, no sinks: the solver's accumulated source ledger must match storage.
    source=float(solver.balance['scheduled_inflow_m3'])
    residual=masses[-1]-masses[0]-source
    row.update(source_volume_m3=source,mass_residual_m3=residual)
    if abs(residual)>max(1e-6,abs(source)*1e-9): raise RuntimeError(f'Water budget failed: {residual}')
    meta.write_text(json.dumps(row,indent=2),encoding='utf-8')
    return row

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--terrain',type=Path,required=True);a=p.parse_args()
    rows=[generate(a.output,a.terrain,s,split) for split,seeds in [('train',[1101,1201,1301,1401]),('validation',[2101,2201])] for s in seeds]
    (a.output/'dataset.json').write_text(json.dumps(dict(cases=rows,phase='pilot',test_generated=False),indent=2),encoding='utf-8')
