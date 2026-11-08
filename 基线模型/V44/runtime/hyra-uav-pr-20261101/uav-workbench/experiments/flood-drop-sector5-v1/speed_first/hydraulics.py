"""Paired synthetic seed generator. Increase prescribed discharge, never rescale velocities."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
TERRAIN = HERE.parents[3] / 'terrain-workbench/terrain-lab'
os.environ['HYRA_FLOOD_BACKEND'] = 'numpy'
sys.path.insert(0, str(TERRAIN))
from flood_solver import FloodSolver


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate(case, folder, horizon=1500, factor=3.):
    meta = case['field']
    return generate_seed(meta['seed'],folder,horizon,factor,baseline=meta)


def generate_seed(seed, folder, horizon=1800, factor=3., baseline=None):
    """Opt-in research profile, accepting new seeds without any personnel data."""
    if not 1 <= factor <= 3 or horizon not in (1500,1800,2100,2400):
        raise ValueError('Outside the registered research profile')
    folder = Path(folder)
    folder.mkdir(exist_ok=True,parents=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((100,100), dtype=float)
    x, y = (xx+.5)*10, (yy+.5)*10
    center = rng.uniform(300,700) + rng.uniform(20,65)*np.sin(y/rng.uniform(140,250))
    bed = rng.uniform(.0001,.0003)*(1000-y) + rng.uniform(.000001,.000003)*(x-center)**2
    for _ in range(4):
        cx, cy = rng.uniform(50,950,2)
        bed += rng.uniform(-.1,.2)*np.exp(-((x-cx)**2+(y-cy)**2)/rng.uniform(15000,60000))
    bed -= bed.min()
    h = np.maximum(0,rng.uniform(.03,.08)-bed)
    peak=float(rng.uniform(8,18))
    c=int(center[15,0]/10)
    cells=[[15,c+k] for k in range(-2,3)]
    graph=[[0.,1.],[300.,peak],[750.,peak*.5],[1200.,.5],[1500.,0.]]
    if baseline is not None:
        ref=np.load(baseline['path'],mmap_mode='r')
        if not np.array_equal(h.astype(np.float32),ref[0,0]) or cells!=baseline['source_cells'] or graph!=baseline['forcing_m3_s'][:5]:
            raise ValueError(f'Initial field or forcing differs on seed {seed}')
    solver=FloodSolver(bed,h,10.,boundary='closed',max_timestep=2.5,spatial_order=1,time_integrator='euler',manning=.035)
    if baseline is not None and solver.numerical_config()!=baseline['numerics']:
        raise ValueError('Solver configuration differs from archived seed')
    graph=[[t,q*factor] for t,q in graph]
    if horizon>1500: graph.append([float(horizon),0.])
    solver.set_source_schedule([dict(cells=cells,hydrograph=graph)])
    path=folder/f'{seed}-q{factor:g}.npy'
    if path.exists():
        raise FileExistsError(path)
    arr=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(horizon//5+1,3,100,100))
    for k,t in enumerate(range(0,horizon+1,5)):
        solver.advance(float(t))
        u,v=solver.velocity()
        field=np.stack((solver.h,u,v))
        if not np.isfinite(field).all() or field[0].min()<0:
            raise ValueError(f'Invalid state at {t}')
        arr[k]=field
    arr.flush()
    del arr
    added=solver.balance['scheduled_inflow_m3']
    residual=float((solver.h.sum()-h.sum())*100-added)
    forcing_volume=sum(q*max(0,min(stop,horizon)-start) for (start,q),(stop,_) in zip(graph,graph[1:]))
    if abs(residual)>max(1e-6,added*1e-9) or abs(added-forcing_volume)>max(1e-5,added*1e-8):
        raise ValueError(f'Storage or source volume mismatch: {residual}, {added-forcing_volume}')
    wet_speed=np.load(path,mmap_mode='r')
    speeds=np.hypot(wet_speed[:,1],wet_speed[:,2])[wet_speed[:,0]>=.01]
    p99=float(np.quantile(speeds,.99))
    top=float(speeds.max())
    audit=dict(seed=seed,source_cells=cells,forcing_m3_s=graph,hash=sha(path),
               wet_speed_p99=p99,wet_speed_max=top,within_study_speed_envelope=p99<=2 and top<=4,
               forcing_volume_m3=forcing_volume,actual_source_m3=added,storage_residual_m3=residual,
               summary=solver.summary(),original_hash=baseline['sha256'] if baseline else None)
    path.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    return path
