"""Interior instantaneous flux audit from a saved native Merewether state.
Not an accepted-step budget, streamline reconstruction, or measured discharge.
Outer forcing is not restored; diagnostics reject boxes near the domain edge.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
from flood_solver import FloodSolver

def audit(folder):
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    c=config['scenario']['flood']
    with np.load(folder/'final.npz') as f:
        solver=FloodSolver(f['bed'],f['depth'],result['dx_m'],boundary='open',manning=f['n'],
            flux_method=c['flux_method'],spatial_order=c['spatial_order'],time_integrator=c['time_integrator'],low_froude=c['low_froude'])
        solver.set_environment(solid=f['solid'],wall_x=f['wall_x'],wall_y=f['wall_y'])
        solver.mx[:]=f['mx'];solver.mz[:]=f['mz']
    fx=solver._flux(1,details=True)['mass'];fz=solver._flux(0,details=True)['mass']
    rows=[]
    for p in result['points']:
        if p['id'] not in ('16-K','19-J'):continue
        z,x=p['grid_row'],p['grid_col'];dx=solver.dx
        boxes=[]
        for radius in (0,5,10):
            cells=int(np.ceil(radius/dx));z0,z1=z-cells,z+cells+1;x0,x1=x-cells,x+cells+1
            if z0<3 or x0<3 or z1>solver.h.shape[0]-3 or x1>solver.h.shape[1]-3:
                raise ValueError('Audit must be strictly interior; outer forcing not restored')
            faces={'west':fx[z0:z1,x0]*dx,'east':-fx[z0:z1,x1]*dx,
                   'north':fz[z0,x0:x1]*dx,'south':-fz[z1,x0:x1]*dx}
            incoming={k:float(np.maximum(v,0).sum()) for k,v in faces.items()}
            outgoing={k:float(np.maximum(-v,0).sum()) for k,v in faces.items()}
            boxes.append(dict(requested_radius_m=radius,rows=[z0,z1],cols=[x0,x1],
                width_m=(x1-x0)*dx,incoming_m3_s=incoming,outgoing_m3_s=outgoing,
                net_incoming_m3_s=sum(incoming.values())-sum(outgoing.values()),
                volume_m3=float(solver.h[z0:z1,x0:x1].sum()*dx**2)))
        h=solver.h[z,x]
        rows.append(dict(id=p['id'],bed_m=float(solver.bed[z,x]),final_depth_m=float(h),
            raw_final_speed_m_s=float(np.hypot(solver.mx[z,x],solver.mz[z,x])/h) if h>solver.velocity_depth else None,
            observed_max_speed_m_s=p['speed_m_s'],boxes=boxes))
    return dict(folder=str(folder),dx_m=solver.dx,flux_method=solver.flux_method,points=rows,
        limitations='Instantaneous reconstructed flux before positivity limiting and RK stage averaging; not a time-integrated budget or true flow path. Interior only.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.write_text(json.dumps([audit(f) for f in a.folders],indent=2,allow_nan=False),encoding='utf-8')
