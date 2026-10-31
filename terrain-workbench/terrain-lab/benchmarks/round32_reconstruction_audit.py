import argparse,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import round30_fine_terrain as channel
from flood_solver import FloodSolver,G
from refined_solver import RefinedFloodSolver
from round32_axis_dry_candidate import AxisDryCandidate

def ritter(kind,dx,rotate):
 parent_dx=dx if kind=='native' else 2*dx
 shape=(8,round(20/parent_dx));h=np.zeros(shape);h[:,:shape[1]//2]=1
 if rotate:h=h.T
 s=FloodSolver(np.zeros_like(h),h,parent_dx,boundary='closed',manning=0,max_timestep=.005)
 if kind!='native':s=(AxisDryCandidate if kind=='candidate' else RefinedFloodSolver).from_solver(s,np.ones_like(h,bool))
 s.advance(.2)
 if kind=='native':
  coordinate=(np.indices(s.h.shape)[0 if rotate else 1]+.5)*dx;depth=s.h;u=s.velocity()[1 if rotate else 0];area=dx*dx
 else:
  coordinate=s.mesh.z if rotate else s.mesh.x;depth=s.leaf['h'];u=s.leaf_velocity()[1 if rotate else 0];area=s.mesh.area
 c=np.sqrt(G);xi=(coordinate-10)/.2
 exact=np.where(xi<-c,1,np.where(xi>2*c,0,(2*c-xi)**2/(9*G)))
 region=abs(coordinate-10)<2
 return dict(case='ritter',kind=kind,dx=dx,rotate=rotate,depth_L1_m=float(np.mean(abs(depth[region]-exact[region]))),mass_residual=s.summary()['balance']['residual_m3'],finite=bool(np.isfinite(depth).all() and np.isfinite(u).all()),minimum=float(depth.min()))

def main(output):
 rows=[]
 for closed in (False,True):
  for rotate in (False,True):
   for kind in ('baseline','candidate'):
    channel.RefinedFloodSolver=AxisDryCandidate if kind=='candidate' else RefinedFloodSolver
    row=channel.run('native',rotate,closed,bounds=(0,12,0,16),max_dt=.01)
    row.update(case='channel',variant=kind);rows.append(row)
 for dx in (.25,.125):
  for rotate in (False,True):
   for kind in ('native','baseline','candidate'):rows.append(ritter(kind,dx,rotate))
 output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
 for row in rows:print(json.dumps(row),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
