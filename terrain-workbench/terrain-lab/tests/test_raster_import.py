"""Imported elevations and partial-width physical boundary conserve SI units."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import raster_import
import map_sessions
import river_inlets
from flood_solver import FloodSolver
from multilayer import MapWorld
import flood_run


class RasterImportTests(unittest.TestCase):
    def test_ascii_orientation_origin_and_fractional_elevation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'map.asc'
            p.write_text('ncols 2\nnrows 2\nxllcenter 100.5\nyllcenter 200.5\ncellsize 1\nNODATA_value -9999\n1.123 2.456\n3.789 -9999\n')
            bed,valid,h=raster_import.read_ascii(p)
            self.assertEqual(h['xllcorner'],100);self.assertEqual(h['yllcorner'],200)
            self.assertAlmostEqual(bed[0,0],1.123);self.assertFalse(valid[-1,-1])

    def test_coarsening_keeps_south_origin_and_conservative_obstacles(self):
        bed=np.arange(15,dtype=float).reshape(5,3);valid=np.ones_like(bed,bool)
        coarse,solid,meta=raster_import.coarsen(bed,valid,2,'max')
        self.assertEqual(coarse.shape,(3,2));self.assertEqual(meta['padded_north_cells'],1)
        self.assertEqual(coarse[-1,0],13);self.assertFalse(solid.any())
        valid[4,0]=False
        _,solid,_=raster_import.coarsen(bed,valid,2,'mean')
        self.assertTrue(solid[-1,0])

    def test_imported_defaults_reach_solver_without_integer_height_or_landcover_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);shape=(4,6);bed=np.full(shape,3.14159);wx=np.zeros((4,7),bool);wx[:,0]=True
            scene=dict(flood=dict(manning=.042,roughness_mode='uniform',baseflow=False,warmup_s=0,
                initial_water=False,rain_mm_h=0,rain_duration_s=0,boundary='open'))
            tid=raster_import.write(root,'test',bed,.7,dict(source='unit fixture'),wall_x=wx,scenario=scene)
            terrain=raster_import.describe(root,tid)
            world=map_sessions.create(root,terrain,dict(mode='current',terrain_id=tid))
            np.testing.assert_array_equal(world.base.z,bed);np.testing.assert_allclose(world.base.n,.042)
            np.testing.assert_array_equal(world.base.wall_x,wx)
            self.assertEqual(world.solver.dx,.7)
            self.assertEqual(world.meta['river_network_grid_check']['status'],'not_applicable')
            self.assertEqual(world.meta['terrain_source'],'imported_experiment_raster')

    def test_partial_boundary_width_and_discharge_are_not_rounded_to_cell_count(self):
        shape=(4,6);solver=FloodSolver(np.zeros(shape),np.ones(shape),1,boundary='open')
        config=flood_run.settings(dict(baseflow=False,warmup_s=0,rain_mm_h=0,rain_duration_s=0),shape)
        world=MapWorld(solver,dict(config=config,coarsening_factor=1))
        p=dict(id='partial',kind='river',native_cells=[[3,1],[3,2]],
               faces=[dict(edge='south',offset=1,fraction=.25),dict(edge='south',offset=2,fraction=.75)])
        river_inlets.configure(world,[p],[dict(id='partial',kind='river',hydrograph=[[0,2]])],list(shape))
        faces=world.external_inflows[0]['boundary_faces']
        self.assertAlmostEqual(sum(f['width_m'] for f in faces),1)
        self.assertAlmostEqual(faces[0]['weight'],.25)
        world.advance_to(.1)
        self.assertAlmostEqual(solver.summary()['balance']['external_inflow_m3'],.2,places=12)
        p['faces'][0]['fraction']=1.1
        with self.assertRaises(ValueError):river_inlets.configure(world,[p],[dict(id='partial',kind='river',hydrograph=[[0,2]])],list(shape))

    def test_import_field_cannot_escape_data_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);tid=raster_import.write(root,'safe',np.zeros((2,2)),1,{})
            p=root/tid/'imported.json';meta=json.loads(p.read_text())
            meta['fields']['height']['file']='../secret.bin';p.write_text(json.dumps(meta))
            with self.assertRaises(ValueError):raster_import.describe(root,tid)


if __name__=='__main__':unittest.main()
