import unittest,sys,tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from grid_coarsening import coarsen_water_state
from flood_solver import FloodSolver
import flood_run,map_sessions
class ShorelineTests(unittest.TestCase):
 def test_mixed_shore_is_rejected_and_native_lake_stays_still(self):
  b=np.zeros((8,16));b[:,7:]=2;h=np.maximum(1-b,0)
  for k in (2,4):
   with self.assertRaisesRegex(ValueError,'原始格宽'):coarsen_water_state(b,h,k)
  b,h=coarsen_water_state(b,h,1);s=FloodSolver(b,h,1,boundary='closed',manning=0);s.advance(2)
  np.testing.assert_array_equal(s.h,h);np.testing.assert_array_equal(s.velocity(),np.zeros((2,8,16)))
 def test_fully_wet_lake_conserves_stage_and_volume(self):
  b=np.tile(np.linspace(0,.5,16),(8,1));h=1-b
  for k in (1,2,4):
   z,d=coarsen_water_state(b,h,k)
   np.testing.assert_allclose(z+d,1);self.assertAlmostEqual(d.sum()*k*k,h.sum())
 def test_dry_map_still_allowed(self):
  b=np.arange(128).reshape(8,16)
  _,d=coarsen_water_state(b,np.zeros_like(b),4);self.assertEqual(d.sum(),0)
 def test_adapter_and_user_supplied_depth_use_guard(self):
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);b=np.zeros((8,16));b[:,7:]=2;fields={}
   for name,a in [('ground_height',b),('terrain',np.zeros_like(b)),('water_surface',np.ones_like(b))]:
    a.astype('<f4').tofile(root/name);fields[name]=dict(url='/'+name,dtype='<f4')
   terrain=dict(id='test',shape=[8,16],fields=fields,full=True,scale=1,step=1,sea_level=-1,terrain_ids={'0':'land'},origin=[0,0],config={'seed':'1'})
   config=dict(grid_factor=2,baseflow=False,boundary='closed',roughness_mode='uniform')
   with self.assertRaisesRegex(ValueError,'原始格宽'):flood_run.prepare(root,terrain,flood_run.settings(config,terrain['shape']))
   data=dict(mode='current',terrain_id='test',flood=config,initial_depth=np.maximum(1-b,0).tolist())
   with self.assertRaisesRegex(ValueError,'原始格宽'):map_sessions.create(root,terrain,data)
if __name__=='__main__':unittest.main()
