import os, sys, json, hashlib
from pathlib import Path
import numpy as np
os.environ['HYRA_FLOOD_BACKEND']='numpy'
ROOT=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab')
sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
OUT=Path('outputs/round8'); OUT.mkdir(exist_ok=True)
plan=dict(category='analytic conservation identities, not observed flood accuracy',
    volume_expected_m3=.1*.13+.3*.14,volume_tolerance_m3=1e-10,
    momentum_tolerance_m4_s=1e-12,source_sha256=hashlib.sha256((ROOT/'flood_solver.py').read_bytes()).hexdigest(),
    hydrograph=[[0,.1],[.13,.3],[.27,0]],end_s=.4,dx_m=1,grid=[5,5],depth_m=1,manning=0)
(OUT/'boundary-plan.json').write_text(json.dumps(plan,indent=2))
results=[]
for edge,normal,z,x in [('west',(1,0),2,0),('east',(-1,0),2,4),('north',(0,1),0,2),('south',(0,-1),4,2)]:
 for angle in (0,np.pi/6):
  direction=[normal[0]*np.cos(angle)-normal[1]*np.sin(angle),normal[0]*np.sin(angle)+normal[1]*np.cos(angle)]
  for width in (.25,1):
   for flux in ('rusanov','knp'):
    for integrator in ('euler','rk2'):
     s=FloodSolver(np.zeros((5,5)),np.ones((5,5)),1,boundary='closed',manning=0,flux_method=flux,time_integrator=integrator,max_timestep=.07)
     e=dict(id='test',enabled=True,delivery='boundary_discharge_with_momentum',hydrograph=plan['hydrograph'],inlet_velocity_m_s=.8,boundary_faces=[dict(edge=edge,offset=2,z=z,x=x,width_m=width,weight=1,momentum_direction_xz=direction)])
     s.set_external_inflows([e]); dt=1e-4;s.advance(dt)
     mom_error=None
     if integrator=='euler':
      mom_error=float(np.max(np.abs(np.array([s.mx.sum(),s.mz.sum()])-.1*.8*dt*np.array(direction))))
      assert mom_error<plan['momentum_tolerance_m4_s']
     s.advance(.4)
     err=float(s.h.sum()-25-plan['volume_expected_m3']);ledger=float(s.external_volume_by_id['test']-plan['volume_expected_m3'])
     assert abs(err)<1e-10 and abs(ledger)<1e-10,(edge,err,ledger)
     results.append(dict(edge=edge,angle_deg=float(np.degrees(angle)),width_m=width,flux=flux,integrator=integrator,volume_error_m3=err,ledger_error_m3=ledger,first_step_momentum_error_m4_s=mom_error,steps=s.steps))
(OUT/'boundary-results.json').write_text(json.dumps(results,indent=2))
print(json.dumps(dict(cases=len(results),max_volume_error=max(abs(r['volume_error_m3']) for r in results),max_ledger_error=max(abs(r['ledger_error_m3']) for r in results),max_momentum_error=max(r['first_step_momentum_error_m4_s'] or 0 for r in results))))
