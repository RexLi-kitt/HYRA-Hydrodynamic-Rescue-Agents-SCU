"""Two-dimensional Thacker rotating planar surface, SWASHES 4.2.2.
Analytic reference integrated with tensor Gauss quadrature; no observational data.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver,G
from refined_solver import RefinedFloodSolver
from round32_axis_dry_candidate import AxisDryCandidate
from analytic_cases import dam_reference
OMEGA=np.sqrt(2*G*.1); PERIOD=2*np.pi/OMEGA

def reference(x,z,dx,t,n=16):
 nodes,weights=np.polynomial.legendre.leggauss(n);h=np.zeros_like(x)
 cx=.5*np.cos(OMEGA*t);cz=.5*np.sin(OMEGA*t)
 for a,wa in zip(nodes,weights):
  for b,wb in zip(nodes,weights):
   h+=wa*wb*.25*np.maximum(0,.1*(1-(x+a*dx/2-cx)**2-(z+b*dx/2-cz)**2))
 return h,-.5*OMEGA*np.sin(OMEGA*t)*h,.5*OMEGA*np.cos(OMEGA*t)*h

def bowl(kind,dx,max_dt,output,*,cycles=1,cfl=.4):
 start=time.perf_counter();pdx=dx if kind=='native' else 2*dx;n=round(4/pdx)
 s=FloodSolver(np.zeros((n,n)),np.zeros((n,n)),pdx,boundary='closed',manning=0,max_timestep=max_dt,cfl=cfl)
 if kind!='native':s=(AxisDryCandidate if kind=='candidate' else RefinedFloodSolver).from_solver(s,np.ones((n,n),bool))
 if kind=='native':z,x=np.indices((n,n));x=(x+.5)*dx-2;z=(z+.5)*dx-2
 else:x=s.mesh.x-2;z=s.mesh.z-2
 bed=.1*(x*x+z*z+dx*dx/6-1);initial=reference(x,z,dx,0)
 if kind=='native':s.bed[:]=bed;s.h[:],s.mx[:],s.mz[:]=initial
 else:
  s.leaf['bed'][:]=bed
  for key,value in zip(('h','mx','mz'),initial):s.leaf[key][:]=value
  s.bed=s.mesh.project(bed)
 s.start_scenario();records=[]
 for phase in np.arange(1,4*cycles+1)/4:
  s.advance(PERIOD*phase)
  assert abs(s.time-PERIOD*phase)<1e-12
  actual=(s.h,s.mx,s.mz) if kind=='native' else tuple(s.leaf[k] for k in ('h','mx','mz'))
  exact=reference(x,z,dx,s.time);q8=reference(x,z,dx,s.time,8)[0];h=actual[0]
  assert all(np.isfinite(a).all() for a in actual) and h.min()>=0
  residual=s.summary()['balance']['residual_m3'];assert abs(residual)<1e-9
  records.append(dict(phase=float(phase),time=s.time,steps=s.steps,min_dt_s=s.diagnostics['min_dt_s'],max_dt_s=s.diagnostics['max_dt_s'],depth_L1_m=float(abs(h-exact[0]).mean()),momentum_L1_m2_s=float((abs(actual[1]-exact[1])+abs(actual[2]-exact[2])).mean()),quadrature_L1_m=float(abs(exact[0]-q8).mean()),mass_residual_m3=residual,centre_x_m=float((h*x).sum()/h.sum()),centre_z_m=float((h*z).sum()/h.sum())))
 np.savez_compressed(output/f'bowl-{kind}-{dx}-{max_dt}-cycles{cycles}-cfl{cfl}.npz',x=x,z=z,bed=bed,h=actual[0],mx=actual[1],mz=actual[2],exact_h=exact[0])
 return dict(case='bowl',kind=kind,dx=dx,max_dt=max_dt,cycles=cycles,cfl=cfl,records=records,elapsed_s=time.perf_counter()-start)

def ritter(kind,dx):
 pdx=dx if kind=='native' else 2*dx;shape=(round(2/pdx),round(20/pdx));h=np.zeros(shape);h[:,:shape[1]//2]=1
 s=FloodSolver(np.zeros_like(h),h,pdx,boundary='closed',manning=0,max_timestep=.005)
 if kind!='native':s=(AxisDryCandidate if kind=='candidate' else RefinedFloodSolver).from_solver(s,np.ones_like(h,bool))
 s.advance(.2)
 if kind=='native':x=(np.indices(s.h.shape)[1]+.5)*dx;h=s.h
 else:x=s.mesh.x;h=s.leaf['h']
 eh,_=dam_reference(np.arange(round(20/dx)+1)*dx,.2,centre=10)
 exact=eh[np.rint(x/dx-.5).astype(int)];region=abs(x-10)<2
 return dict(case='ritter_cell_average',kind=kind,dx=dx,depth_L1_m=float(abs(h[region]-exact[region]).mean()),mass_residual_m3=s.summary()['balance']['residual_m3'])

def main(out):
 rows=[]
 for dx in (.2,.1):
  for kind in ('native','baseline','candidate'):
   r=bowl(kind,dx,.02,out);rows.append(r);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(kind,dx,r['records'][-1],flush=True)
 for dx in (.25,.125):
  for kind in ('native','baseline','candidate'):rows.append(ritter(kind,dx))
 for kind in ('native','baseline','candidate'):rows.append(bowl(kind,.1,.01,out))
 (out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output)
