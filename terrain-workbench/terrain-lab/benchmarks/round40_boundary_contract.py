"""Isolated Okushiri boundary contract audit, not a simulation benchmark."""
import os,sys,json
from pathlib import Path
os.environ['HYRA_FLOOD_BACKEND']='numpy'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
from flood_numerics import face_flux

def run():
    rows=[]
    for momentum in [0.,.01,-.01]:
        bed=np.full((4,8),-.135);h=np.full_like(bed,.135)
        s=FloodSolver(bed,h,.1,boundary='ocean',sea_level=0.,sea_edges={'west':np.ones(4,dtype=bool)},manning=0,spatial_order=1)
        s.mx[:]=momentum
        actual=s._flux(1,details=True)
        # Source BC: stage specified; normal conserved momentum copied;
        # tangential conserved momentum zero. Uniform, wet, flat-bed oracle.
        depth=np.full(4,.135);u=np.full(4,momentum/.135);zero=np.zeros(4)
        expected=face_flux(depth,depth,u,zero,u,zero,normal=(1.,0.),method='rusanov',low_froude=0)
        rows.append(dict(interior_normal_momentum_m2_s=momentum,existing_ocean_west_flux_m2_s=float(actual['mass'][0,0]),required_west_flux_m2_s=float(expected[0][0]),maximum_difference=float(abs(actual['mass'][:,0]-expected[0]).max())))
    assert rows[0]['maximum_difference']==0
    assert np.isclose(rows[1]['existing_ocean_west_flux_m2_s'],.005)
    assert np.isclose(rows[2]['existing_ocean_west_flux_m2_s'],-.005)
    assert np.isclose(rows[1]['required_west_flux_m2_s'],.01)
    assert np.isclose(rows[2]['required_west_flux_m2_s'],-.01)
    return dict(evidence='discrete boundary contract, not laboratory fit',cases=rows)
if __name__=='__main__':
    result=run();Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
