import sys,unittest,warnings
from pathlib import Path
import numpy as np
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from round30_fine_terrain import run

class FineTerrainTests(unittest.TestCase):
    def base(self):
        return FloodSolver(np.zeros((6,8)),np.zeros((6,8)),1,boundary='closed')
    def test_mapping_ownership_and_unchanged_coarse_cells(self):
        s=self.base();mask=np.zeros((6,8),bool);mask[2,3]=True
        f=np.zeros((12,16));f[4:6,6:8]=[[1,2],[3,4]]
        a=RefinedFloodSolver.from_solver(s,mask,fine_bed=f,fine_dx=.5)
        i=a.mesh.start[2*8+3];np.testing.assert_array_equal(a.leaf['bed'][i:i+4],[1,2,3,4])
        self.assertEqual(a.bed[2,3],2.5);self.assertEqual(s.bed[2,3],0)
        f[:]=99;self.assertEqual(a.leaf['bed'][i],1)
        np.testing.assert_array_equal(a.leaf['bed'][a.mesh.child<0],0)
    def test_invalid_and_wet_inputs_rejected(self):
        mask=np.ones((6,8),bool)
        for field,dx in [(np.zeros((6,8)),.5),(np.full((12,16),np.nan),.5),(np.zeros((12,16)),1),(np.zeros((12,16)),'bad')]:
            with self.assertRaises(ValueError):RefinedFloodSolver.from_solver(self.base(),mask,fine_bed=field,fine_dx=dx)
        f=np.zeros((12,16));f[0,0]=1
        with self.assertRaisesRegex(ValueError,'boundary'):RefinedFloodSolver.from_solver(self.base(),mask,fine_bed=f,fine_dx=.5)
        s=self.base();s.h[2,3]=.1
        with self.assertRaisesRegex(ValueError,'dry'):RefinedFloodSolver.from_solver(s,mask,fine_bed=np.zeros((12,16)),fine_dx=.5)
    def test_fine_bed_lake_at_rest(self):
        s=self.base();mask=np.zeros((6,8),bool);mask[1:-1,1:-1]=True
        f=np.zeros((12,16));f[3:9,3:13]=.1
        a=RefinedFloodSolver.from_solver(s,mask,fine_bed=f,fine_dx=.5)
        a.leaf['h'][:]=.3-a.leaf['bed'];a.start_scenario();before=a.leaf['h'].copy();a.advance(.5)
        np.testing.assert_allclose(a.leaf['h'],before,atol=1e-12,rtol=0)
        np.testing.assert_allclose(a.leaf_velocity(),0,atol=1e-11,rtol=0)
        self.assertLess(abs(a.summary()['balance']['residual_m3']),1e-10)
    def test_channel_and_real_barrier_with_rotation(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            for closed in (False,True):
                rows=[run('native',rotate,closed) for rotate in (False,True)]
                self.assertAlmostEqual(rows[0]['downstream_volume_m3'],rows[1]['downstream_volume_m3'],places=12)
                for r in rows:
                    self.assertLess(abs(r['balance']['residual_m3']),1e-9)
                    if closed:self.assertLess(r['downstream_volume_m3'],1e-10)
                    else:self.assertGreater(r['downstream_volume_m3'],0)
