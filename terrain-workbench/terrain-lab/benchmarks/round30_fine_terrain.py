"""Synthetic narrow terrain passage: inherited, native-leaf and uniform-fine."""
import argparse,json,sys
from pathlib import Path
import numpy as np
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver

def run(kind,transpose=False,closed_gap=False,*,bounds=(3,9,4,10),max_dt=.02,spatial_order=2):
 shape=(12,16);bed=np.zeros(shape);bed[:,6:8]=1
 fine=np.repeat(np.repeat(bed,2,axis=0),2,axis=1)
 if not closed_gap:fine[11,12:16]=0
 mask=np.zeros(shape,bool);z0,z1,x0,x1=bounds;mask[z0:z1,x0:x1]=True
 source=[dict(z=5,x=4,weight=1.)]
 if kind=='uniform':
  source=[dict(z=z,x=x,weight=.25) for z in (10,11) for x in (8,9)]
 if transpose:
  bed=bed.T;fine=fine.T;mask=mask.T
  source=[dict(z=c['x'],x=c['z'],weight=c['weight']) for c in source]
 s=FloodSolver(fine if kind=='uniform' else bed,np.zeros_like(fine if kind=='uniform' else bed),.5 if kind=='uniform' else 1,boundary='closed',manning=.02,max_timestep=max_dt,spatial_order=spatial_order)
 s.set_external_inflows([dict(id='source',enabled=True,cells=source,hydrograph=[[0,.2],[3,0]])])
 if kind!='uniform':s=RefinedFloodSolver.from_solver(s,mask,**(dict(fine_bed=fine,fine_dx=.5) if kind=='native' else {}))
 s.advance(6)
 if kind=='uniform':
  wet=s.h;area=.25;downstream=wet[16:,:] if transpose else wet[:,16:]
  downstream_volume=float(downstream.sum()*area)
 else:
  downstream=(s.mesh.z>=8) if transpose else (s.mesh.x>=8)
  downstream_volume=float((s.leaf['h'][downstream]*s.mesh.area[downstream]).sum())
 return dict(kind=kind,transpose=transpose,closed_gap=closed_gap,downstream_volume_m3=downstream_volume,
             balance=s.summary()['balance'],finite=bool(np.isfinite(s.h).all()),minimum_depth=float(s.h.min()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 rows=[run(kind,transpose,gap) for gap in (False,True) for transpose in (False,True) for kind in ('inherited','native','uniform')]
 a.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
 for r in rows:print(r['kind'],r['transpose'],r['closed_gap'],r['downstream_volume_m3'],r['balance']['residual_m3'])
