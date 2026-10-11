"""Physical invariants and terrain adapter checks (python -m unittest discover -s tests)."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flood_solver import FloodSolver
import flood_run
import server


class PhysicsTests(unittest.TestCase):
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
    def test_example_coarsening_preserves_volume_and_si_conversion(self):
        terrain = server.describe('example/raw')
        full, _ = flood_run.prepare(ROOT, terrain, flood_run.settings(dict(resolution=256), terrain['shape']))
        coarse, meta = flood_run.prepare(ROOT, terrain, flood_run.settings(dict(resolution=64, metres_per_block=2), terrain['shape']))
        self.assertEqual(meta['shape'], [64, 64])
        self.assertEqual(meta['dx_m'], 32)
        self.assertAlmostEqual(coarse.balance['initial_m3'], 8*full.balance['initial_m3'])
        np.testing.assert_allclose(coarse.bed, full.bed.reshape(64,4,64,4).mean(axis=(1,3))*2)

    def test_invalid_scenarios_are_rejected(self):
        for values in (dict(rain_mm_h=float('nan')), dict(source_x=256), dict(source_x=1.2),
                       dict(duration_s=2, rain_duration_s=3), dict(boundary='bad'),
                       dict(initial_water='false'), dict(metres_per_block=0)):
            with self.assertRaises(ValueError): flood_run.settings(values, [256, 256])


if __name__ == '__main__': unittest.main()
