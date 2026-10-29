import os,sys,json,hashlib,time
from pathlib import Path
import numpy as np
os.environ['HYRA_FLOOD_BACKEND']='numpy'
ROOT=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab')
sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
out=Path('outputs/round9');out.mkdir(exist_ok=True)
plan=dict(category='synthetic conservation/stability; no observed accuracy claim',domain_m=[4,4],inlet_width_m=1,hydrograph=[[0,.1],[.4,.2],[.8,0]],end_s=1.2,expected_input_m3=.12,mass_tolerance_m3=1e-10,grids_m=[.5,.25],dt_caps_s=[.02,.01],initial_depths_m=[0,1e-7],velocity_modes_m_s=[0,.8],fluxes=['rusanov','knp'],boundary='open',manning=0,source_sha256=hashlib.sha256((ROOT/'flood_solver.py').read_bytes()).hexdigest())
(out/'plan.json').write_text(json.dumps(plan,indent=2))
rows=[];start=time.perf_counter()
for dx in plan['grids_m']:
 for cap in plan['dt_caps_s']:
  for initial in plan['initial_depths_m']:
   for velocity in plan['velocity_modes_m_s']:
    for flux in plan['fluxes']:
     n=round(4/dx);s=FloodSolver(np.zeros((n,n)),np.full((n,n),initial),dx,boundary='open',manning=0,max_timestep=cap,flux_method=flux)
     indices=range(round(1.5/dx),round(2.5/dx))
     faces=[dict(edge='west',offset=z,z=z,x=0,width_m=dx,weight=dx) for z in indices]
     s.set_external_inflows([dict(id='in',enabled=True,delivery='boundary_discharge_with_momentum',inlet_velocity_m_s=velocity,hydrograph=plan['hydrograph'],boundary_faces=faces)])
     s.advance(1.2)
     b=s.balance;volume=float(s.h.sum()*dx*dx)
     residual=volume-16*initial-b['external_inflow_m3']+b['outflow_m3']-b['numerical_adjustment_m3']
     row=dict(dx=dx,cap=cap,initial=initial,velocity=velocity,flux=flux,volume=volume,residual=residual,input=b['external_inflow_m3'],outflow=b['outflow_m3'],adjustment=b['numerical_adjustment_m3'],min_depth=float(s.h.min()),max_depth=float(s.h.max()),momentum_x=float(s.mx.sum()*dx*dx),steps=s.steps)
     rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2))
     assert abs(residual)<1e-10 and abs(row['input']-.12)<1e-10 and row['min_depth']>=0,row
print(json.dumps(dict(cases=len(rows),seconds=time.perf_counter()-start,max_residual=max(abs(r['residual']) for r in rows),max_input_error=max(abs(r['input']-.12) for r in rows),max_adjustment=max(abs(r['adjustment']) for r in rows))))
