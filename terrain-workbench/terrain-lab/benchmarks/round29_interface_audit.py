"""Read-only local reconstructed-interface audit; clipping is not itself a defect."""
import sys,json,argparse
from pathlib import Path
import numpy as np
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
from flood_solver import FloodSolver

def audit(folder):
 r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
 with np.load(folder/'final.npz') as f:
  s=FloodSolver(f['bed'],f['depth'],r['dx_m'],manning=f['n'])
  s.set_environment(solid=f['solid'],wall_x=f['wall_x'],wall_y=f['wall_y']);s.mx[:]=f['mx'];s.mz[:]=f['mz']
 recon=s._reconstruction();axes={}
 for axis in (0,1):
  sh,sb,_,_=recon[axis];hl,hr=s._faces(s.h,sh,axis);bl,br=s._faces(s.bed,np.zeros_like(s.h),axis);ol,orr=s._faces(np.zeros_like(s.h),sb,axis)
  jump=(br-bl)+(orr-ol)
  axes[axis]=(hl,hr,jump,s._blocked_faces(axis))
 rows=[]
 for p in r['points']:
  if p['id'] not in ('16-K','19-J'):continue
  z,x=p['grid_row'],p['grid_col'];faces=[]
  for name,axis,zz,xx in [('west',1,z,x),('east',1,z,x+1),('north',0,z,x),('south',0,z+1,x)]:
   hl,hr,jump,blocked=axes[axis];j=float(jump[zz,xx]);l=float(hl[zz,xx]);rr=float(hr[zz,xx])
   faces.append(dict(side=name,hl_m=l,hr_m=rr,bed_jump_m=j,clipped_left=l>0 and j>=l,clipped_right=rr>0 and -j>=rr,blocked=bool(blocked[zz,xx])))
  fallback={str(axis):bool(s.h[z,x]<=s.velocity_depth or np.roll(s.h,1,axis)[z,x]<=s.velocity_depth or np.roll(s.h,-1,axis)[z,x]<=s.velocity_depth) for axis in (0,1)}
  rows.append(dict(id=p['id'],depth=float(s.h[z,x]),near_dry_fallback=fallback,faces=faces))
 return dict(dx_m=s.dx,points=rows)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folders',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 a.output.write_text(json.dumps([audit(f) for f in a.folders],indent=2),encoding='utf-8')
