"""Uniform larger cells preserve water volume, source flow and authored layers."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import flood_run
import map_sessions


class CoarseGridTests(unittest.TestCase):
    def fixture(self,root):
        shape=(16,24);z,x=np.indices(shape);bed=(.01*x+.02*z).astype('<f4')
        arrays=dict(height=bed,terrain=np.ones(shape,dtype='<i4'),riverWaterLevel=(bed+.5).astype('<f4'))
        fields={}
        for name,a in arrays.items():
            a.tofile(root/name);fields[name]=dict(url='/'+name,dtype=a.dtype.str)
        return dict(shape=list(shape),fields=fields,full=False,scale=1,sea_level=-1,
            terrain_ids={'1':'river'},step=1,origin=[7,-4],id='test-native',config=dict(seed='1'))

    def test_volume_extent_and_total_baseflow_preserved_at_two_and_four(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root=Path(tmp);terrain=self.fixture(root);native=None
            for factor in (1,2,4):
                c=flood_run.settings(dict(grid_factor=factor,metres_per_block=2,baseflow_m3_s=3),terrain['shape'])
                s,meta=flood_run.prepare(root,terrain,c)
                if factor==1:native=s
                self.assertEqual(s.h.shape,(16//factor,24//factor))
                self.assertEqual(s.dx,2*factor)
                self.assertAlmostEqual(s.balance['initial_m3'],native.balance['initial_m3'],places=10)
                self.assertAlmostEqual(s.baseflow_m3_s,3)
                self.assertEqual(meta['origin_blocks'],[7+(factor-1)/2,-4+(factor-1)/2])

    def test_authored_building_drain_and_manual_source_survive_grid_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root=Path(tmp);terrain=self.fixture(root)
            data=dict(mode='current',terrain_id=terrain['id'],flood=dict(grid_factor=4,
                baseflow=False,boundary='closed',rain_mm_h=0,rain_duration_s=0),
                layers=dict(patches=[dict(kind='building',x=12,z=8,width=2,height=2),
                    dict(kind='drain',x=0,z=0,width=4,height=4,value=.5)]),
                external_inflows=[dict(id='manual',kind='manual',x=8,z=4,hydrograph=[[0,7]])])
            w=map_sessions.create(root,terrain,data)
            self.assertEqual(w.solver.h.shape,(4,6))
            self.assertTrue(w.base.solid[2,3])
            self.assertEqual(w.base.drain_capacity.sum(),8)
            self.assertEqual(w.external_inflows[0]['cells'],[dict(z=1,x=2,weight=1.)])
            w.advance_to(.2)
            self.assertAlmostEqual(w.solver.balance['external_inflow_m3'],1.4,places=12)
            self.assertLess(abs(w.solver.summary()['balance']['residual_m3']),1e-9)

    def test_invalid_factors_and_misaligned_sizes_are_rejected(self):
        for factor in (True,0,3,4.,'4'):
            with self.subTest(factor=factor),self.assertRaises(ValueError):
                flood_run.settings(dict(grid_factor=factor),(16,24))
        with self.assertRaises(ValueError):flood_run.settings(dict(grid_factor=4),(15,24))


if __name__=='__main__':unittest.main()
