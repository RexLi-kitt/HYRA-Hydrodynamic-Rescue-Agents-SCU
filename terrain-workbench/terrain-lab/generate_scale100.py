"""Generate actual 900s SWE cases; run from terrain-lab, no resampled time frames."""
import os
os.environ['HYRA_FLOOD_BACKEND']='numpy'
import argparse, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
from flood_solver import FloodSolver

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state(s):return np.stack((s.h,s.mx,s.mz)).copy()
def npbytes(a):
    f=io.BytesIO();np.save(f,a,allow_pickle=False);return f.getvalue()
def generate(root,seed,split):
    path=root/f'{split}-{seed}.zip'
    if path.exists():raise FileExistsError(path)
    rng=np.random.default_rng(seed);z,x=np.indices((20,20));bed=np.zeros((20,20))
    # Independent sampled hills, slope and source locations per parent terrain.
    bed+=rng.uniform(-.001,.001)*x*5+rng.uniform(-.001,.001)*z*5
    for _ in range(5):
        r,c=rng.uniform(0,20,2);bed+=rng.uniform(-.12,.18)*np.exp(-((z-r)**2+(x-c)**2)/rng.uniform(10,60))
    bed-=bed.min();h=np.maximum(0,.06-bed)
    r,c=map(int,rng.integers(3,17,2));h+=.12*np.exp(-((z-r)**2+(x-c)**2)/5)
    solver=FloodSolver(bed,h,5.,boundary='closed',max_timestep=.5)
    source_cells=[[r,c]];q=float(rng.uniform(.12,.3))
    graph=[[0.,0.],[100.,q],[400.,q*.4],[700.,0.]]
    solver.set_source_schedule([dict(cells=source_cells,hydrograph=graph)])
    frames=[];before=[];after=[];dts=[];rates=[];speeds=[];wet=[]
    start=time.perf_counter()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for t in range(0,901,5):
            solver.advance(float(t));u,v=solver.velocity();name=f'frame-{t:04d}.bin'
            archive.writestr(name,np.stack((solver.h,u,v,solver.maximum)).astype('<f4').tobytes())
            frames.append(dict(time_s=float(t),url=name));speeds.append(np.hypot(u,v)[solver.h>=.05]);wet.append(int((solver.h>=.05).sum()))
            before.append(state(solver));dt=solver.timestep(.5);rate=solver._scheduled_rate(solver.time)
            # This is the NEXT accepted internal solver step, not a 5s Euler approximation.
            solver.step(dt);after.append(state(solver));dts.append(dt);rates.append(rate)
        process=dict(grid=dict(shape=[20,20],dx_m=5.,axis_order=['z','x']),frames=frames,frame_fields=['depth','u','v','maximum_depth'],frame_dtype='<f4')
        archive.writestr('process.json',json.dumps(process))
        for name,array in [('bed',bed),('probe_before',before),('probe_after',after),('probe_dt',dts),('probe_source',rates)]:archive.writestr(name+'.npy',npbytes(np.asarray(array)))
        manifest=dict(schema='hyra-scale100-hydraulics-v1',seed=seed,split=split,parent_terrain=f'procedural-parent-{seed}',scope='synthetic procedural SWE event; not a calibrated real flood',duration_s=900.,grid=process['grid'],source_schedule=graph,source_cells=source_cells,numerics=solver.numerical_config(),solver_sha256=digest(Path(__file__).with_name('flood_solver.py')),frame_count=len(frames),actual_internal_probes=True,wet_cells_initial=wet[0],wet_cells_final=wet[-1],wet_cells_changed_range=max(wet)-min(wet),speed_p50_p90_p95=np.quantile(np.concatenate(speeds),[.5,.9,.95]).tolist(),summary=solver.summary())
        archive.writestr('case.json',json.dumps(manifest))
    return dict(path=str(path.resolve()),sha256=digest(path),**manifest,elapsed_s=time.perf_counter()-start)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for split,seed in [('fit',101),('fit',211),('fit',307),('development',401),('holdout',503),('holdout',601)]:
        row=generate(a.output,seed,split);rows.append(row);print(json.dumps(dict(split=split,seed=seed,elapsed_s=row['elapsed_s'])),flush=True)
    (a.output/'dataset.json').write_text(json.dumps(dict(schema='hyra-scale100-dataset-v1',cases=rows,split_unit='independently sampled procedural parent terrain and forcing; not real-event generalization'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
