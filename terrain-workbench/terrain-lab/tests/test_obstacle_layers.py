"""Physical/dataflow checks for independent fluid ground and obstacles."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import obstacle_layers
import raster_import
import server
import map_sessions
from flood_solver import FloodSolver


class ObstacleLayerTests(unittest.TestCase):
    def test_known_fluid_ground_survives_obstacle_height_and_aggregation(self):
        surface=np.full((12,12),20.);mask=np.zeros_like(surface,bool);mask[4:6,4:6]=True
        surface[mask]=30
        layers=obstacle_layers.separate(surface,mask)
        np.testing.assert_array_equal(layers['ground'][~mask],surface[~mask])
        coarse=obstacle_layers.aggregate(layers,4,native_dx=.2)
        np.testing.assert_array_equal(coarse['ground'],20)
        self.assertTrue(coarse['solid'][1,1]);self.assertEqual(coarse['coverage'][1,1],.25)
        self.assertFalse(coarse['audit']['partial_coverage_used_by_solver'])
        self.assertEqual(coarse['audit']['minimum_component_cells_added'],1)

    def test_native_footprint_identity_and_high_natural_slope_not_thresholded(self):
        z,x=np.indices((40,40));surface=33+.01*x+.02*z
        surface[15:24,15:24]=30
        layers=obstacle_layers.merewether_layers(surface)
        # High natural ground is not classified merely for exceeding 30 m.
        self.assertFalse(layers['obstacle'][2:8,2:8].any())
        native=obstacle_layers.aggregate(layers,1)
        np.testing.assert_array_equal(native['solid'],layers['obstacle'])
        self.assertEqual(native['audit']['extra_blocked_native_area_m2'],0)
        self.assertEqual(native['audit']['lost_obstacle_native_area_m2'],0)
        with self.assertRaises(ValueError):obstacle_layers.separate(surface,np.ones_like(surface,bool))

    def test_solid_and_thin_wall_exclude_storage_and_prevent_crossing(self):
        bed=np.zeros((12,16));h=np.zeros_like(bed);h[:,:5]=.3
        solid=np.zeros_like(bed,bool);solid[3:7,3:5]=True
        wall=np.zeros((12,17),bool);wall[:,8]=True
        solver=FloodSolver(bed,h,.175,boundary='closed',manning=.042)
        solver.set_environment(solid=solid,wall_x=wall)
        solver.advance(1)
        self.assertEqual(float(solver.h[solid].sum()),0)
        self.assertEqual(float(solver.h[:,8:].sum()),0)
        self.assertLess(abs(solver.summary()['balance']['residual_m3']),1e-10)
        self.assertGreater(float(solver.h[:,:8].sum()),0)

    def test_imported_layers_reach_actual_solver_and_forbid_silent_recoarsening(self):
        bed=np.full((8,8),19.25);solid=np.zeros_like(bed,bool);solid[3:5,3:5]=True
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            tid=raster_import.write(root,'layered-test',bed,.175,{'obstacle_layers':{'method':'explicit-test'}},
                solid=solid,obstacle_top=np.where(solid,30,bed),obstacle_fraction=solid.astype(float),
                scenario={'flood':dict(initial_water=False,baseflow=False,boundary='closed',rain_mm_h=0)})
            terrain=raster_import.describe(root,tid)
            world=map_sessions.create(root,terrain,dict(terrain_id=tid))
            np.testing.assert_array_equal(world.solver.bed,bed)
            np.testing.assert_array_equal(world.solver.solid,solid)
            self.assertEqual(world.solver.dx,.175)
            with self.assertRaisesRegex(ValueError,'目标格宽'):
                map_sessions.create(root,terrain,dict(terrain_id=tid,flood={'grid_factor':2}))


if __name__=='__main__':unittest.main()
