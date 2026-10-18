"""Physical invariants and terrain adapter checks (python -m unittest discover -s tests)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flood_solver import FloodSolver
import flood_run
import server


class PhysicsTests(unittest.TestCase):
    def test_sea_at_rest_with_sloping_bed_and_dry_coast(self):
        z, x = np.mgrid[:10, :14]
        bed = 10000 + .1*x + .03*z
        depth = np.maximum(10001-bed, 0)
        edges = dict(west=np.ones(10, bool), north=np.ones(14, bool))
        sim = FloodSolver(bed, depth, 2, boundary='ocean', sea_level=10001, sea_edges=edges)
        sim.advance(30)
        np.testing.assert_allclose(sim.h, depth, atol=1e-12)
        np.testing.assert_allclose(sim.velocity(), 0, atol=1e-11)
        self.assertAlmostEqual(sim.balance['sea_inflow_m3'], 0, places=9)
        self.assertAlmostEqual(sim.balance['sea_outflow_m3'], 0, places=9)

    def test_sea_can_supply_dry_map_and_receive_excess_water(self):
        bed = np.full((6, 8), -2.0)
        edges = dict(west=np.ones(6, bool), east=np.ones(6, bool),
                     north=np.ones(8, bool), south=np.ones(8, bool))
        for initial, direction in ((0, 'sea_inflow_m3'), (3, 'sea_outflow_m3')):
            sim = FloodSolver(bed, np.full_like(bed, initial), 1, boundary='ocean',
                              sea_level=0, sea_edges=edges)
            self.assertLess(sim.timestep(), .06)
            sim.advance(20)
            self.assertGreater(sim.balance[direction], 0)
            self.assertLess(abs(sim.h.mean()-2), abs(initial-2)*.05)
            self.assertGreaterEqual(sim.h.min(), 0)
            self.assertLess(abs(sim.summary()['balance']['residual_m3']), 1e-9)

    def test_sea_does_not_reset_inland_lake_or_open_unmarked_faces(self):
        bed = np.full((8, 12), 5.0)
        bed[:, :4] = -2; bed[2:6, 7:10] = 1
        depth = np.maximum(-bed, 0); depth[2:6, 7:10] = 2
        sim = FloodSolver(bed, depth, 2, boundary='ocean', sea_level=0,
                          sea_edges=dict(west=np.ones(8, bool)))
        sim.advance(30)
        np.testing.assert_allclose(sim.h, depth, atol=1e-12)
        # With no marked sea faces the mode is exactly the old outflow boundary.
        a = FloodSolver(np.zeros((6, 8)), np.ones((6, 8)), 1, boundary='open')
        b = FloodSolver(a.bed, a.h, 1, boundary='ocean', sea_level=10, sea_edges={})
        a.mx.fill(1); b.mx.fill(1)
        a.advance(3); b.advance(3)
        np.testing.assert_allclose(a.h, b.h, atol=1e-12)
        self.assertEqual(b.balance['sea_inflow_m3'], 0)

    def test_sea_exchange_and_forcing_have_a_closed_mass_budget(self):
        bed = np.zeros((8, 10)); depth = np.ones_like(bed)
        sim = FloodSolver(bed, depth, 2, boundary='ocean', sea_level=1,
                          sea_edges=dict(east=np.ones(8, bool)))
        sim.advance(15, rain=.001, rain_until=4, infiltration=.0005, inflow=2, source=(4, 5))
        self.assertGreater(sim.balance['sea_outflow_m3'], 0)
        self.assertGreater(sim.balance['outflow_m3'], 0)
        self.assertAlmostEqual(sim.balance['inflow_m3'], 30)
        self.assertLess(abs(sim.summary()['balance']['residual_m3']), 1e-9)

    def test_lake_at_rest_with_dry_shore_and_large_elevation_offset(self):
        z, x = np.mgrid[:12, :16]
        bed = 10000 + .08*x + .1*z
        depth = np.maximum(10001.4-bed, 0)
        for boundary in ('open', 'closed'):
            sim = FloodSolver(bed, depth, 2, boundary=boundary)
            sim.advance(20)
            np.testing.assert_allclose(sim.h, depth, atol=1e-12)
            np.testing.assert_allclose(sim.velocity(), 0, atol=1e-11)
            self.assertAlmostEqual(sim.summary()['balance']['residual_m3'], 0, places=10)

    def test_rain_cutoff_and_point_inflow_mass_budget(self):
        sim = FloodSolver(np.zeros((8, 10)), np.zeros((8, 10)), 4, boundary='closed')
        sim.advance(30, rain=.001, rain_until=7.3, inflow=2, source=(3, 4))
        expected = .001*7.3*80*16 + 2*30
        self.assertAlmostEqual(sim.h.sum()*16, expected, places=9)
        self.assertGreater(sim.h[3, 5], 0)
        self.assertGreater(np.hypot(*sim.velocity()).max(), 0)
        self.assertGreaterEqual(sim.h.min(), 0)

    def test_infiltration_cannot_remove_more_than_available(self):
        sim = FloodSolver(np.zeros((6, 7)), np.full((6, 7), .002), 1, boundary='closed')
        sim.advance(10, infiltration=.1)
        np.testing.assert_array_equal(sim.h, 0)
        self.assertAlmostEqual(sim.balance['infiltration_m3'], 42*.002)
        self.assertAlmostEqual(sim.summary()['balance']['residual_m3'], 0)

    def test_dry_bed_dam_break_and_rotational_symmetry(self):
        bed = np.zeros((12, 30)); depth = bed.copy(); depth[:, :8] = 1
        a = FloodSolver(bed, depth, 1, boundary='closed', manning=0)
        b = FloodSolver(bed.T, depth.T, 1, boundary='closed', manning=0)
        a.advance(2); b.advance(2)
        self.assertGreater(a.h[:, 10].max(), 0)
        self.assertGreater(a.velocity()[0][:, 8].mean(), 0)
        self.assertGreaterEqual(a.h.min(), 0)
        self.assertAlmostEqual(a.h.sum(), depth.sum(), places=9)
        np.testing.assert_allclose(a.h, b.h.T, atol=1e-12)
        np.testing.assert_allclose(a.velocity()[0], b.velocity()[1].T, atol=1e-12)

    def test_steep_random_bed_does_not_create_or_destroy_water(self):
        rng = np.random.default_rng(42)
        sim = FloodSolver(rng.uniform(-10, 50, (16, 16)), rng.uniform(0, 3, (16, 16)),
                          4, boundary='closed')
        sim.advance(10)
        self.assertGreaterEqual(sim.h.min(), 0)
        self.assertLess(abs(sim.summary()['balance']['residual_m3']), 1e-8)

    def test_open_boundary_outflow_is_accounted_and_cannot_add_water(self):
        sim = FloodSolver(np.zeros((8, 16)), np.ones((8, 16)), 1, boundary='open', manning=0)
        sim.mx.fill(1)
        sim.advance(2)
        self.assertGreater(sim.balance['outflow_m3'], 0)
        self.assertLess(sim.h.sum(), 128)
        self.assertAlmostEqual(sim.summary()['balance']['residual_m3'], 0, places=9)


class AdapterTests(unittest.TestCase):
    def test_native_sea_faces_use_actual_map_edge_and_convert_stage(self):
        # Ocean inside a coarse corner must not open its north face; a lake
        # below sea level must not be mistaken for an external ocean either.
        with tempfile.TemporaryDirectory(dir=ROOT / '.runtime') as folder:
            root = Path(folder); shape = [128, 128]
            types = np.ones(shape, dtype='<i4')
            types[3, 0] = 0; types[3, 3] = 0; types[20:24, 20:24] = 0
            arrays = dict(height=np.full(shape, -2, dtype='<f4'), terrain=types,
                          riverWaterLevel=np.zeros(shape, dtype='<f4'))
            fields = {}
            for name, a in arrays.items():
                a.tofile(root / name)
                fields[name] = dict(url='/'+name, dtype=a.dtype.str)
            terrain = dict(shape=shape, fields=fields, full=False, scale=1, sea_level=0,
                           terrain_ids={'0':'ocean', '1':'lake'}, step=1, origin=[0,0],
                           id='test', config=dict(seed='test'))
            config = flood_run.settings(dict(resolution=32, metres_per_block=2,
                                            source_x=0, source_z=0), shape)
            sim, meta = flood_run.prepare(root, terrain, config)
            self.assertEqual(meta['boundary_conditions']['sea_face_count'], 1)
            self.assertTrue(sim.sea_edges['west'][3])
            self.assertFalse(sim.sea_edges['north'].any())
            self.assertEqual(sim.sea_level, 0)
            self.assertEqual(sim.h[0,0], 4)
            self.assertEqual(sim.dx, 2)
            self.assertEqual(sim.h.shape,tuple(shape))
            self.assertEqual(meta['coarsening_factor'],1)
            # Full export uses actual ground and water but keeps ocean identity.
            terrain['full'] = True; terrain['sea_level'] = 3
            fields['ground_height'] = fields['height']
            fields['water_surface'] = fields['riverWaterLevel']
            sim, meta = flood_run.prepare(root, terrain, config)
            self.assertEqual(sim.sea_level, 6)
            self.assertEqual(meta['boundary_conditions']['sea_face_count'], 1)

    def test_legacy_resolution_does_not_resample_and_si_conversion_is_preserved(self):
        terrain = server.describe('example/raw')
        full, _ = flood_run.prepare(ROOT, terrain, flood_run.settings(dict(resolution=256), terrain['shape']))
        native, meta = flood_run.prepare(ROOT, terrain, flood_run.settings(dict(resolution=64, metres_per_block=2), terrain['shape']))
        self.assertEqual(meta['shape'], terrain['shape'])
        self.assertEqual(meta['dx_m'], terrain['step']*2)
        self.assertAlmostEqual(native.balance['initial_m3'], 8*full.balance['initial_m3'])
        np.testing.assert_array_equal(native.bed, full.bed*2)

    def test_rectangular_non_power_of_two_native_grid_is_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as tmp:
            root=Path(tmp);shape=[33,47];bed=np.arange(np.prod(shape),dtype='<f4').reshape(shape)/100
            arrays=dict(height=bed,terrain=np.ones(shape,dtype='<i4'),riverWaterLevel=bed)
            fields={}
            for name,a in arrays.items():
                a.tofile(root/name);fields[name]=dict(url='/'+name,dtype=a.dtype.str)
            terrain=dict(shape=shape,fields=fields,full=False,scale=1,sea_level=-1,
                terrain_ids={'1':'plains'},step=1,origin=[7,-4],id='native',config=dict(seed='1'))
            config=flood_run.settings(dict(resolution=32,source_x=0,source_z=0,baseflow=False),shape)
            sim,meta=flood_run.prepare(root,terrain,config)
            np.testing.assert_array_equal(sim.bed,bed)
            self.assertEqual(sim.dx,1);self.assertEqual(meta['origin_blocks'],[7,-4])

    def test_invalid_scenarios_are_rejected(self):
        for values in (dict(rain_mm_h=float('nan')), dict(source_x=256), dict(source_x=1.2),
                       dict(duration_s=2, rain_duration_s=3), dict(boundary='bad'),
                       dict(initial_water='false'), dict(metres_per_block=0)):
            with self.assertRaises(ValueError): flood_run.settings(values, [256, 256])


if __name__ == '__main__': unittest.main()
