import sys,os,json
from pathlib import Path
import numpy as np
os.environ['HYRA_FLOOD_BACKEND']='numpy'
root=Path('C:/Users/Lenovo/Desktop/无人机大创/HYRA/terrain-workbench/terrain-lab');sys.path.insert(0,str(root))
from flood_solver import FloodSolver
out=Path('outputs/round15');out.mkdir(exist_ok=True)
(out/'plan.json').write_text(json.dumps(dict(category='synthetic sampling sensitivity, not observed accuracy',grid=[8,8],dx_m=.5,depth_m=.01,window_s=[0,1],dense_sample_s=.01,sparse_sample_s=.5,hydrograph=[[0,0],[.1,.2],[.2,0]],inlet_speed_m_s=2,manning=[.03,.3]),indent=2))
results=[]
for n in (.03,.3):
 s=FloodSolver(np.zeros((8,8)),np.full((8,8),.01),.5,boundary='closed',manning=n,max_timestep=.01)
 s.set_external_inflows([dict(id='pulse',enabled=True,delivery='boundary_discharge_with_momentum',inlet_velocity_m_s=2,hydrograph=[[0,0],[.1,.2],[.2,0]],boundary_faces=[dict(edge='west',offset=4,z=4,x=0,width_m=.5,weight=1)])])
 rows=[]
 for t in np.linspace(0,1,101):
  s.advance(float(t));rows.append([float(t),s.summary()['max_speed_m_s']])
 a=np.array(rows);np.save(out/f'trace-{n}.npy',a)
 results.append(dict(manning=n,step_peak=s.peak_speed_m_s,step_peak_time=s.peak_speed_time_s,dense_sampled_peak=float(a[:,1].max()),sparse_sampled_peak=float(a[[0,50,100],1].max()),dense_peak_time=float(a[a[:,1].argmax(),0]),final_instantaneous_speed=float(a[-1,1])))
(out/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
