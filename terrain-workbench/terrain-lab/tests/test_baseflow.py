"""Daily source conservation, steady channel flow, spin-up and terrain detection."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flood_solver import FloodSolver
from river_baseflow import plan, spin_up
import flood_run
import server


class BaseflowTests(unittest.TestCase):
    def test_distributed_baseflow_and_flood_sources_close_the_budget(self):
        sim = FloodSolver(np.zeros((6, 8)), np.zeros((6, 8)), 2, boundary='closed')
        sim.set_baseflow([dict(z=2,x=2,discharge_m3_s=.3),dict(z=2,x=3,discharge_m3_s=.7)])
        sim.advance(20,rain=.001,rain_until=5,inflow=.5,source=(4,5))
        self.assertAlmostEqual(sim.balance['baseflow_m3'],20,places=10)
        self.assertAlmostEqual(sim.balance['inflow_m3'],10,places=10)
        self.assertAlmostEqual(sim.h.sum()*4,30+.001*5*48*4,places=10)
        self.assertLess(abs(sim.summary()['balance']['residual_m3']),1e-9)

    def test_no_rain_channel_reaches_steady_flow_and_keeps_receiving_water(self):
        bed = np.full((8,24),4.0); bed[3:5,1:] = .02*np.arange(22,-1,-1)
        depth = np.zeros_like(bed); depth[3:5,1:] = .3
        sim = FloodSolver(bed,depth,1)
        sim.set_baseflow([dict(z=3,x=1,discharge_m3_s=.15),dict(z=4,x=1,discharge_m3_s=.15)])
        report = spin_up(sim,600,0,lambda **kw:None,lambda:False)
        self.assertTrue(report['converged'])
        initial, momentum = sim.h.copy(), sim.mx.copy()
        sim.start_scenario()
        np.testing.assert_array_equal(sim.mx,momentum)
        self.assertEqual(sim.time,0)
        self.assertEqual(sim.balance['baseflow_m3'],0)
        sim.advance(300)
        # Finite-step source transport can redistribute a fraction of a mm at
        # the head when the scenario clock starts with a new end-time sequence.
        np.testing.assert_allclose(sim.h,initial,atol=1e-3)
        self.assertGreater(sim.h[3:5,2:].min(),.1)
        self.assertGreater(sim.velocity()[0][3:5,2:].min(),.1)
        # The warm-up criterion allows a small storage drift. Test actual
        # conservation instead of assuming exactly zero storage change after
        # changing the integration scheme or the output-time step sequence.
        storage_change = float((sim.h-initial).sum()*sim.dx**2)
        self.assertAlmostEqual(sim.balance['outflow_m3']+storage_change,90,places=8)
        storage_tolerance = int((initial>0).sum())*sim.dx**2*1e-3
        self.assertLess(abs(sim.balance['outflow_m3']-90),storage_tolerance)
        self.assertAlmostEqual(sim.balance['baseflow_m3'],90,places=8)
        self.assertLess(abs(sim.summary()['balance']['residual_m3']),1e-9)
        # Flood accounting starts from the warmed river, not the generator pool.
        sim.advance(310,rain=.001,inflow=.2,source=(3,8))
        self.assertAlmostEqual(sim.balance['rainfall_m3'],.001*10*bed.size,places=9)
        self.assertAlmostEqual(sim.balance['inflow_m3'],2,places=9)

    def test_short_or_blocked_warmup_is_not_reported_as_stable_and_can_cancel(self):
        sim = FloodSolver(np.zeros((4,4)),np.zeros((4,4)),2,boundary='closed')
        sim.set_baseflow([dict(z=1,x=1,discharge_m3_s=2)])
        report = spin_up(sim,30,0,lambda **kw:None,lambda:False)
        self.assertFalse(report['converged'])
        self.assertEqual(report['duration_s'],30)
        self.assertAlmostEqual(report['balance']['baseflow_m3'],60,places=9)
        with self.assertRaises(InterruptedError):
            spin_up(sim,120,0,lambda **kw:None,lambda:True)

    def test_head_detection_and_coarsening_preserve_requested_discharge(self):
        river = np.zeros((12,16),bool); river[2:4,1:7]=True;river[8:10,9:15]=True
        bed = np.zeros(river.shape);water = np.ones(river.shape)
        water[2:4,1:7] = np.linspace(4,1,6)
        water[8:10,9:15] = np.linspace(1,3,6)
        for factor in (1,2,4):
            groups,sources = plan(river,water,bed,factor,6)
            self.assertEqual(len(groups),2)
            self.assertEqual(groups[0]['native_head'],[2,1])
            self.assertEqual(groups[1]['native_head'],[8,14])
            self.assertAlmostEqual(sum(s['discharge_m3_s'] for s in sources),6)
            self.assertTrue(all(river[s['z']*factor:min(12,(s['z']+1)*factor),
                                          s['x']*factor:min(16,(s['x']+1)*factor)].any() for s in sources))
        self.assertEqual(plan(river,water,bed,2,0)[1],[])
        self.assertEqual(plan(np.zeros_like(river),water,bed,2,6),( [], [] ))

    def test_adapter_excludes_lakes_and_no_river_map_receives_no_baseflow(self):
        # Current seed is not required: the bundled map provides river/lake data.
        terrain = server.describe('example/raw')
        sim,meta = flood_run.prepare(ROOT,terrain,flood_run.settings(dict(resolution=64),terrain['shape']))
        self.assertGreater(meta['river_baseflow']['river_native_cells'],0)
        self.assertGreater(meta['river_baseflow']['reach_count'],0)
        self.assertAlmostEqual(sim.baseflow_m3_s,5)
        terrain = dict(terrain,terrain_ids={code:'lake' if name=='river' else name
                                          for code,name in terrain['terrain_ids'].items()})
        sim,meta = flood_run.prepare(ROOT,terrain,flood_run.settings(dict(resolution=64),terrain['shape']))
        self.assertEqual(sim.baseflow_m3_s,0)
        self.assertEqual(meta['river_baseflow']['sources'],[])

    def test_invalid_baseflow_scenarios_and_sources_are_rejected(self):
        for data in (dict(baseflow='yes'),dict(baseflow_m3_s=-1),dict(warmup_s=float('nan'))):
            with self.assertRaises(ValueError): flood_run.settings(data,[256,256])
        sim = FloodSolver(np.zeros((4,4)),np.zeros((4,4)),1)
        for source in (dict(z=4,x=1,discharge_m3_s=1),dict(z=1,x=1,discharge_m3_s=-1)):
            with self.assertRaises(ValueError): sim.set_baseflow([source])


if __name__ == '__main__': unittest.main()
