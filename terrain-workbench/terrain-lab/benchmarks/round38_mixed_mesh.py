"""Mixed mesh Thacker verification, with leaf-area weighted metrics."""
import argparse,json,time
from pathlib import Path
import numpy as np
from round33_two_dimensional import FloodSolver,RefinedFloodSolver,reference,PERIOD

def run(dx,cfl,out):
 started=time.perf_counter();pdx=2*dx;n=round(4/pdx);z,x=np.indices((n,n));mask=x>=n//2
 s=RefinedFloodSolver.from_solver(FloodSolver(np.zeros((n,n)),np.zeros((n,n)),pdx,boundary='closed',manning=0,max_timestep=.02,cfl=cfl),mask)
 m=s.mesh;x=m.x-2;z=m.z-2;width=m.width;area=m.area
 assert abs(area.sum()-16)<1e-12
 s.leaf['bed'][:]=.1*(x*x+z*z+width*width/6-1)
 for key,value in zip(('h','mx','mz'),reference(x,z,width,0)):s.leaf[key][:]=value
 s.bed=m.project(s.leaf['bed']);s.start_scenario();rows=[]
 for phase in (.25,.5,.75,1.):
  s.advance(PERIOD*phase);assert abs(s.time-PERIOD*phase)<1e-12
  h=s.leaf['h'];mx=s.leaf['mx'];mz=s.leaf['mz'];exact=reference(x,z,width,s.time);q8=reference(x,z,width,s.time,8)[0]
  assert all(np.isfinite(a).all() for a in (h,mx,mz)) and h.min()>=0
  residual=s.summary()['balance']['residual_m3'];assert abs(residual)<1e-9
  error=abs(h-exact[0]);vol=h*area
  rows.append(dict(phase=phase,time=s.time,steps=s.steps,depth_L1_m=float((error*area).sum()/16),unweighted_depth_mean_m=float(error.mean()),momentum_L1_m2_s=float(((abs(mx-exact[1])+abs(mz-exact[2]))*area).sum()/16),quadrature_L1_m=float((abs(exact[0]-q8)*area).sum()/16),mass_residual_m3=residual,max_dt_s=s.diagnostics['max_dt_s'],centre_x_m=float((vol*x).sum()/vol.sum()),centre_z_m=float((vol*z).sum()/vol.sum())))
 np.savez_compressed(out/f'mixed-{dx}-{cfl}.npz',x=x,z=z,width=width,area=area,bed=s.leaf['bed'],h=h,mx=mx,mz=mz,exact_h=exact[0])
 return dict(fine_dx=dx,coarse_dx=pdx,cfl=cfl,leaf_count=m.count,records=rows,elapsed_s=time.perf_counter()-started)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();rows=[]
 for dx in (.2,.1):
  for cfl in (.4,.2):
   r=run(dx,cfl,a.out);rows.append(r);(a.out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(dx,cfl,r['records'][-1],flush=True)
