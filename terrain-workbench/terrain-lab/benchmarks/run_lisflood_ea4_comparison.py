"""Full 5-hour EA Test 4 comparison: current HYRA vs official CPU DG2 8.0."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[1];WORK=LAB/'.runtime/lisflood-comparison'
sys.path.insert(0,str(WORK/'python-packages'));sys.path.insert(0,str(LAB))
import numpy as np
from flood_solver import FloodSolver

def read_ascii(path):
    with Path(path).open() as f:
        h={}
        for _ in range(6):
            k,v=f.readline().split();h[k.lower()]=float(v)
        a=np.loadtxt(f).reshape(int(h['nrows']),int(h['ncols']))
    return a,h

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--forcing',choices=['stepwise','linear'],default='stepwise');parser.add_argument('--reuse-reference',action='store_true');args=parser.parse_args()
    source=WORK/'source/ea4/ea4';folder=WORK/'ea4-10m';folder.mkdir(exist_ok=True)
    for name in ['ea4.par','ea4.bci','ea4.bdy','ea4.stage']:shutil.copyfile(source/name,folder/name)
    data=gzip.open(source/'ea4-10m.dem.gz','rb').read()
    for name in ['ea4.dem','ea4.dem1x','ea4.dem1y']:(folder/name).write_bytes(data)
    bed,header=read_ascii(folder/'ea4.dem');assert np.count_nonzero(bed)==0
    exe=WORK/'build-manual/lisflood.exe';env=dict(os.environ,OMP_NUM_THREADS='4',OMP_DYNAMIC='FALSE')
    if args.reuse_reference:
        assert (folder/'dg2/ea4-0005.wd').exists()
        ref_wall=json.loads((folder/'result-stepwise.json').read_text())['original_dg2_wall_s']
    else:
        start=time.perf_counter()
        with (folder/'dg2.log').open('wb') as log:
            r=subprocess.run([str(exe),'-dirroot','dg2','-dg2','ea4.par'],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        ref_wall=time.perf_counter()-start
        if r.returncode:raise RuntimeError(f'original DG2 failed {r.returncode}; see dg2.log')
    print('Official DG2 complete',ref_wall,flush=True)
    # Warm production kernels outside timed integration.
    warm=FloodSolver(np.zeros((16,16)),np.zeros((16,16)),10,manning=.05);warm.step(.01)
    class LinearInputSolver(FloodSolver):
        @staticmethod
        def external_discharge(entry,time):
            curve=entry['hydrograph']
            return float(np.interp(time,[p[0] for p in curve],[p[1] for p in curve]))
    Solver=LinearInputSolver if args.forcing=='linear' else FloodSolver
    s=Solver(bed,np.zeros_like(bed),10,manning=.05,boundary='open',flux_method='rusanov',
        spatial_order=2,time_integrator='rk2',low_froude=0,cfl=.4,max_timestep=1,max_steps=500000)
    nz,nx=bed.shape;wx=np.zeros((nz,nx+1),bool);wy=np.zeros((nz+1,nx),bool);wx[:,0]=True
    rows=[99,100];wx[rows,0]=False;s.set_environment(wall_x=wx,wall_y=wy)
    curve=[[0,0],[300,0],[3600,20],[14400,20],[18000,0]]
    s.set_external_inflows([dict(id='ea4-west',enabled=True,delivery='boundary_discharge_with_momentum',
        inlet_velocity_m_s=0,hydrograph=curve,boundary_faces=[dict(edge='west',offset=z,z=z,x=0,width_m=10.,weight=.5,momentum_direction_xz=[1.,0.]) for z in rows])])
    coords=np.loadtxt(source/'ea4.stage',skiprows=1)
    iz=((2000-coords[:,1])/10).astype(int);ix=(coords[:,0]/10).astype(int)
    snapshots={};records=[];start=time.perf_counter()
    for end in range(100,18001,100):
        if time.perf_counter()-start>300:raise TimeoutError('HYRA wall budget 300s reached')
        s.advance(end,cancelled=lambda:time.perf_counter()-start>300)
        u,v=s.velocity();records.append([s.time,*s.h[iz,ix],*np.hypot(u[iz,ix],v[iz,ix])])
        if end%3600==0:
            snapshots[str(end)]=s.h.copy();print('HYRA',end,'wall',time.perf_counter()-start,flush=True)
    hyra_wall=time.perf_counter()-start
    np.savez_compressed(folder/f'hyra-{args.forcing}.npz',depth=s.h,mx=s.mx,mz=s.mz,maximum=s.maximum,records=records,**{'h_'+k:v for k,v in snapshots.items()})
    comparisons=[]
    for index,end in enumerate(range(3600,18001,3600),start=1):
        ref,h=read_ascii(folder/'dg2'/f'ea4-{index:04d}.wd');local=snapshots[str(end)]
        e=local-ref
        comparisons.append(dict(time_s=end,depth_rmse_m=float(np.sqrt(np.mean(e*e))),depth_mae_m=float(np.mean(abs(e))),max_abs_depth_difference_m=float(abs(e).max()),
            hyra_volume_m3=float(local.sum()*100),dg2_volume_m3=float(ref.sum()*100),hyra_max_depth_m=float(local.max()),dg2_max_depth_m=float(ref.max())))
    result=dict(case='EA Test 4, paper benchmark, flat 1000x2000m floodplain',complete=True,duration_s=18000,
        shape=list(bed.shape),dx_m=10,original_dg2_wall_s=ref_wall,hyra_wall_s=hyra_wall,
        scope='model agreement; no independent observed truth; native numerical settings and boundary closures differ',
        hyra_settings=dict(flux='rusanov',spatial_order=2,time='rk2',cfl=.4,max_timestep=1,manning=.05,acceleration=bool(s.acceleration)),
        original_settings=dict(solver='CPU DG2 8.0.3',threads=4,local_limiter=False,DepthThresh=.001,netcdf=False,numeric_precision='float32'),
        forcing_interpretation=args.forcing,reference_reused=args.reuse_reference,
        input_sha256={n:hashlib.sha256((folder/n).read_bytes()).hexdigest() for n in ['ea4.dem','ea4.par','ea4.bci','ea4.bdy','ea4.stage']},
        solver_sha256=hashlib.sha256((LAB/'flood_solver.py').read_bytes()).hexdigest(),
        hyra_steps=s.steps,hyra_balance=s.summary()['balance'],hyra_finite=bool(np.isfinite(s.h).all()),
        hyra_nonnegative=bool((s.h>=0).all()),comparisons=comparisons)
    (folder/f'result-{args.forcing}.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
