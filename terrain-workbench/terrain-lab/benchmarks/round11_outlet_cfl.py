import os,sys,json,hashlib,time
from pathlib import Path
import numpy as np
os.environ['HYRA_FLOOD_BACKEND']='numpy'
root=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab')
sys.path.insert(0,str(root))
from flood_solver import FloodSolver
out=Path('outputs/round11');out.mkdir(exist_ok=True)
plan=dict(category='same-solver extended-domain comparison, not experimental truth',depth_m=1,amplitude_m=.001,center_m=6,sigma_m=.5,short_length_m=10,reference_length_m=30,width_m=1,dx_m=[.125],dt_caps_s=[.02],cfl_values=[.4,.2,.1],end_s=4,output_s=.05,comparison_region_m=[7,9],window_s=[2,4],mass_tolerance_m3=1e-9,source_sha256=hashlib.sha256((root/'flood_solver.py').read_bytes()).hexdigest())
(out/'plan.json').write_text(json.dumps(plan,indent=2));results=[];start=time.perf_counter()
for dx in plan['dx_m']:
 for cfl in plan['cfl_values']:
  cap=.02
  for sign in (-1,):
   traces={}
   for mode,length in [('reference',30),('outflow_only',10),('extrapolated',10)]:
    x=(np.arange(round(length/dx))+.5)*dx
    eta=sign*.001*np.exp(-.5*((x-6)/.5)**2)
    h=np.tile(1+eta,(round(1/dx),1))
    s=FloodSolver(np.zeros_like(h),h,dx,boundary='open',manning=0,max_timestep=cap,cfl=cfl,open_boundary_mode='extrapolated' if mode=='reference' else mode)
    s.mx[:]=s.h*(2*(np.sqrt(9.81*s.h)-np.sqrt(9.81)))
    records=[];mask=(x>=7)&(x<=9)
    for t in np.arange(.05,4.0001,.05):
     s.advance(float(t));records.append(s.h[:,mask].mean(axis=0).tolist())
    traces[mode]=np.array(records)
    balance=s.summary()['balance'];assert abs(balance['corrected_residual_m3'])<1e-9
    np.savez(out/f'dx{dx}_cfl{cfl}_sign{sign}_{mode}.npz',depth=traces[mode],x=x[mask],times=np.arange(.05,4.0001,.05))
    if mode!='reference':
     difference=traces[mode][39:]-traces['reference'][39:]
     results.append(dict(dx=dx,cap=cap,cfl=cfl,steps=s.steps,dt_max=s.diagnostics['max_dt_s'],dt_min=s.diagnostics['min_dt_s'],sign=sign,mode=mode,max_depth_difference_m=float(np.max(abs(difference))),rms_depth_difference_m=float(np.sqrt(np.mean(difference**2))),relative_to_initial_amplitude=float(np.max(abs(difference))/.001),mass_residual_m3=balance['corrected_residual_m3']))
    (out/'results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(dict(runs=9,seconds=time.perf_counter()-start,results=results)))
