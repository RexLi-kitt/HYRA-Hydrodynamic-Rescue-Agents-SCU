import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver

def experiment():
 base=FloodSolver(np.zeros((4,4)),np.ones((4,4)),1,boundary='closed',manning=0,max_timestep=.01)
 base.set_external_inflows([dict(id='inlet',enabled=True,cells=[dict(z=1,x=1,weight=1.)],hydrograph=[[0,.2],[.1,0]])])
 mask=np.ones((4,4),bool)
 a=RefinedFloodSolver.from_solver(base,mask);b=RefinedFloodSolver.from_solver(base,mask)
 a.advance(.1)
 result=dict(base_time=base.time,base_external=base.balance['external_inflow_m3'],base_inlet=base.external_volume_by_id['inlet'],sibling_before_external=b.balance['external_inflow_m3'],a_external=a.balance['external_inflow_m3'])
 b.advance(.1)
 result.update(b_external=b.balance['external_inflow_m3'],a_external_after_b=a.balance['external_inflow_m3'],a_residual=a.summary()['balance']['residual_m3'],b_residual=b.summary()['balance']['residual_m3'],max_state_difference=float(abs(a.h-b.h).max()))
 return result
if __name__=='__main__':
 result=experiment();Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)
