"""Prespecified time-step/output-event sensitivity; no production edits."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--solver-dir',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.solver_dir));from flood_solver import FloodSolver
a.output.mkdir(parents=True,exist_ok=True)
cases=[('baseline',.05,.01),('cap_0p0005',.0005,.01),('cap_0p00025',.00025,.01),('output_0p005',.05,.005)]
plan=dict(cases=cases,fixed=dict(dx_m=.01,flux='rusanov',wall='reflective',velocity_depth=1e-6,manning=.03,end_s=3),
 hypothesis='Return-flow mismatch may be dominated by temporal integration or output-event clipping.',
 criteria=dict(mass_residual_m3=1e-9,repeatability_N=1e-10,
 screening='Compare curve differences and both fixed-window metrics; do not tune to minimize measured RMSE. No observational pass threshold asserted.'),
 solver_sha256=hashlib.sha256((a.solver_dir/'flood_solver.py').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((a.data/'for_exp.txt').read_bytes()).hexdigest())
(a.output/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
x=(np.arange(160)+.5)*.01;y=(np.arange(61)+.5)*.01
solid=(x[None,:]>=.9)&(x[None,:]<1.02)&(y[:,None]>=.25)&(y[:,None]<.37)
depth=np.broadcast_to(np.where(x<=.4,.3,.01),(61,160)).copy();depth[solid]=0
rows=np.flatnonzero((y>=.25)&(y<.37))
obs=np.array([[float(v.split()[0]),float(v.split()[-1])] for v in (a.data/'for_exp.txt').read_text().splitlines()])
results=[];baseline=None;common=np.linspace(0,3,301)
for name,cap,interval in cases:
 s=FloodSolver(np.zeros_like(depth),depth,.01,boundary='closed',manning=.03,wall_flux_mode='reflective',max_timestep=cap)
 s.set_environment(solid=solid);start=time.perf_counter();curve=[]
 for t in np.linspace(0,3,round(3/interval)+1):
  s.advance(float(t));curve.append([t,.5*1024*9.8*.01*np.sum(s.h[rows,89]**2-s.h[rows,102]**2)])
 curve=np.asarray(curve);shared=np.interp(common,curve[:,0],curve[:,1])
 if baseline is None:baseline=shared.copy()
 metrics={}
 for shift in (0.,.1):
  for label,lo,hi in [('whole',shift,3+shift),('return_flow',1.5,1.8)]:
   valid=(obs[:,0]>=lo)&(obs[:,0]<=hi);e=np.interp(obs[valid,0],curve[:,0]+shift,curve[:,1])-obs[valid,1]
   # Shared output grid separates reporting interpolation from solver changes.
   ec=np.interp(obs[valid,0],common+shift,shared)-obs[valid,1]
   metrics[f'{label}_shift_{shift}']=dict(n=int(valid.sum()),rmse_N=float(np.sqrt(np.mean(e**2))),mae_N=float(np.mean(abs(e))),common_sampling_rmse_N=float(np.sqrt(np.mean(ec**2))))
 result=dict(name=name,max_timestep=cap,output_interval=interval,metrics=metrics,
  max_difference_on_common_times_N=float(np.max(abs(shared-baseline))),rms_difference_on_common_times_N=float(np.sqrt(np.mean((shared-baseline)**2))),
  peak_force_N=float(curve[:,1].max()),summary=s.summary(),elapsed_s=time.perf_counter()-start)
 assert abs(result['summary']['balance']['residual_m3'])<1e-9
 np.savetxt(a.output/(name+'.csv'),curve,delimiter=',',header='time_s,force_N',comments='')
 results.append(result);(a.output/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
