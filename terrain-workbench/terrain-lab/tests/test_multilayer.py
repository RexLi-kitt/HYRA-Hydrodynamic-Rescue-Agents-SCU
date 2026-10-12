"""PPT acceptance: shared faces, buildings, drainage, observation isolation and export."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
from multilayer import BaseMap,Knowledge,MapWorld
import flood_run
import map_sessions
import server


def world(shape=(8,12),layers=None,depth=None):
    b=np.zeros(shape);h=np.zeros(shape) if depth is None else depth
    c=flood_run.settings(dict(resolution=32,source_x=0,source_z=0,rain_mm_h=0,
        rain_duration_s=0,baseflow=False,warmup_s=0),shape)
    s=FloodSolver(b,h,2,boundary='closed')
    return MapWorld(s,dict(config=c,shape=list(shape),coarsening_factor=1,origin_blocks=[0,0]),layers,
        [dict(id='uav-1',x=1,z=2,radius=1),dict(id='uav-2',x=9,z=5,radius=1)])


class HydraulicLayersTests(unittest.TestCase):
    def test_internal_wall_blocks_water_and_preserves_stationary_pressure(self):
        bed=np.zeros((8,12));h=np.zeros_like(bed);h[:,:6]=1
        wall=np.zeros((8,13),bool);wall[:,6]=True
        sim=FloodSolver(bed,h,2,boundary='closed');sim.set_environment(wall_x=wall)
        sim.advance(20)
        np.testing.assert_allclose(sim.h,h,atol=1e-12)
        np.testing.assert_allclose(sim.velocity(),0,atol=1e-12)
        np.testing.assert_array_equal(sim.qx[:,6],0)
        self.assertLess(abs(sim.summary()['balance']['residual_m3']),1e-10)

    def test_solid_building_is_impermeable_and_rain_falls_only_on_fluid_cells(self):
        s=FloodSolver(np.zeros((8,12)),np.ones((8,12))*.2,2,boundary='closed')
        solid=np.zeros((8,12),bool);solid[2:6,4:8]=True;s.set_environment(solid=solid)
        s.advance(10,rain=.001)
        self.assertAlmostEqual(s.balance['rainfall_m3'],.001*10*80*4,places=10)
        np.testing.assert_array_equal(s.h[solid],0)
        self.assertAlmostEqual(s.h.sum()*4,80*.2*4+3.2,places=9)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)

    def test_uniform_water_surface_with_variable_bed_and_walls_stays_at_rest(self):
        zz,xx=np.indices((8,12));bed=.07*xx+.03*zz;h=np.maximum(.8-bed,0)
        wall=np.zeros((8,13),bool);wall[1:6,5]=True
        sim=FloodSolver(bed,h,2,boundary='closed');sim.set_environment(wall_x=wall)
        sim.advance(20)
        np.testing.assert_allclose(sim.h,h,atol=1e-12)
        np.testing.assert_allclose(sim.velocity(),0,atol=1e-11)

    def test_hydrograph_event_alignment_and_bounded_drainage_close_budget(self):
        layers=dict(patches=[dict(kind='source',x=1,z=2,value=3),dict(kind='drain',x=1,z=2,value=100)],
                    hydrographs={'3':[[0,10],[.5,0],[1,4]]})
        w=world(layers=layers);w.advance(2)
        b=w.solver.summary()['balance']
        self.assertAlmostEqual(b['scheduled_inflow_m3'],9,places=9)
        self.assertAlmostEqual(b['drainage_m3'],9,places=9)
        self.assertGreaterEqual(w.solver.h.min(),0)
        self.assertLess(abs(b['residual_m3']),1e-9)

    def test_spatial_roughness_changes_drag(self):
        n=np.full((8,12),.02);n[:,6:]=.2
        s=FloodSolver(np.zeros(n.shape),np.ones(n.shape),2,boundary='closed',manning=n)
        s.mx.fill(.5);s.step(.01)
        self.assertGreater(s.velocity()[0][4,3],s.velocity()[0][4,9])

    def test_invalid_fixed_layer_and_source_configuration_is_rejected(self):
        for config in (dict(n=np.zeros((8,12))),dict(solid=[[0]]),dict(patches=[dict(kind='drain',x=0,z=0,value=-1)]),
            dict(patches=[dict(kind='source',x=0,z=0,value=0)]),
            dict(patches=[dict(kind='source',x=0,z=0,value=0)],hydrographs={'0':[[0,-1]]})):
            with self.assertRaises(ValueError):BaseMap(np.zeros((8,12)),2,config)

    def test_imported_elevation_signal_and_initial_water_are_used(self):
        data=dict(mode='demo',initial_depth=np.full((32,32),.3).tolist(),
            layers=dict(z=np.ones((32,32)).tolist(),signal_field=np.zeros((32,32)).tolist()))
        w=map_sessions.create(ROOT,server.describe('example/raw'),data)
        np.testing.assert_array_equal(w.solver.bed,1)
        self.assertAlmostEqual(w.solver.h[2,7],.3)
        w.observe('uav-1');k=w.knowledge['uav-1']
        self.assertTrue(k.signal_valid[16,7]);self.assertEqual(k.signal_field[16,7],0)
        self.assertFalse(k.signal_valid[0,0])

    def test_invalid_inflow_into_building_does_not_mutate_the_state(self):
        s=FloodSolver(np.zeros((4,4)),np.zeros((4,4)),2,boundary='closed')
        solid=np.zeros((4,4),bool);solid[1,1]=True;s.set_environment(solid=solid)
        before=s.summary()
        with self.assertRaises(ValueError):s.step(.1,rain=.1,inflow=1,source=(1,1))
        self.assertEqual(s.summary(),before)


class ObservationTests(unittest.TestCase):
    def test_truth_advances_without_updating_known_water_until_reobserved(self):
        w=world(depth=np.full((8,12),.2));w.config['rain_mm_h']=360;w.config['rain_duration_s']=100
        self.assertEqual(w.observe('uav-1'),5)
        k=w.knowledge['uav-1'];old=k.water_last.copy();before=w.solver.h.copy();w.advance(10)
        self.assertGreater(w.solver.h[2,1],before[2,1])
        np.testing.assert_equal(k.water_last,old)
        self.assertEqual(k.age(10)[2,1],10)
        w.observe('uav-1');self.assertEqual(k.t_water[2,1],10)
        self.assertEqual(k.age(10)[2,1],0)
        self.assertEqual(k.water_last[2,1],w.solver.h[2,1])

    def test_known_dry_and_unknown_are_distinct_and_policy_has_no_truth(self):
        w=world(layers=dict(patches=[dict(kind='person',x=11,z=7,value=1)]));w.observe('uav-1')
        view=w.policy_view('uav-1');obs=view['M_obs']
        self.assertEqual(obs['water_last'][2][1],0)
        self.assertTrue(obs['water_valid'][2][1])
        self.assertIsNone(obs['water_last'][7][11])
        self.assertFalse(obs['water_valid'][7][11])
        self.assertNotIn('M_water',view);self.assertNotIn('M_ref',view);self.assertNotIn('M_base',view)
        self.assertEqual(view['channels']['person_seen'][7][11],0)
        self.assertFalse(view['channels']['person_valid'][7][11])
        self.assertTrue(all(np.isfinite(a).all() for a in view['channels'].values()))

    def test_water_and_person_can_coexist_and_positions_only_include_discovered_people(self):
        w=world(layers=dict(patches=[dict(kind='person',x=1,z=2,value=1)]),depth=np.full((8,12),.5))
        self.assertEqual(w.snapshot()['M_pos']['discovered_persons'],[])
        w.observe('uav-1');self.assertEqual(w.knowledge['uav-1'].water_last[2,1],.5)
        self.assertEqual(w.knowledge['uav-1'].person_last[2,1],1)
        self.assertEqual(len(w.policy_view('uav-1')['M_pos']['discovered_persons']),1)
        self.assertEqual(w.policy_view('uav-2')['M_pos']['discovered_persons'],[])
        w.share('uav-1','uav-2');self.assertEqual(len(w.policy_view('uav-2')['M_pos']['discovered_persons']),1)

    def test_invalid_or_occluded_observation_retains_old_record(self):
        w=world();w.observe('uav-1');k=w.knowledge['uav-1'];before=k.water_last.copy()
        w.advance(10);w.observe('uav-1',False)
        np.testing.assert_equal(k.water_last,before);self.assertEqual(k.t_water[2,1],0)
        w.base.solid[2,1]=True;w.observe('uav-1')
        self.assertEqual(k.t_water[2,1],0);self.assertEqual(k.t_terrain[2,1],10)

    def test_old_shared_data_cannot_overwrite_new_and_future_data_is_rejected(self):
        w=world(depth=np.ones((8,12))*.2);w.observe('uav-1');old=copy.deepcopy(w.knowledge['uav-1'])
        w.advance(10);w.observe('uav-1');w.share('uav-1');new=w.team.water_last.copy()
        self.assertEqual(w.team.merge(old,10),0);np.testing.assert_equal(w.team.water_last,new)
        future=copy.deepcopy(old);future.t_water[future.water_valid]=20
        with self.assertRaises(ValueError):w.team.merge(future,10)
        np.testing.assert_equal(w.team.water_last,new)

    def test_no_fly_path_and_time_reversal_are_rejected(self):
        w=world(layers=dict(patches=[dict(kind='no_fly',x=4,z=2)]))
        with self.assertRaises(ValueError):w.move('uav-1',7,2)
        self.assertEqual(w.agents['uav-1']['x'],1)
        for dt in (-1,0,float('nan')):
            with self.assertRaises(ValueError):w.advance(dt)
        corner=world(layers=dict(patches=[dict(kind='no_fly',x=2,z=2)]))
        with self.assertRaises(ValueError):corner.move('uav-1',4,5)

    def test_pooling_excludes_unknown_and_exploration_history_is_retained(self):
        k=Knowledge((2,2));k.water_last[0,:]=[.4,1];k.water_valid[0,:]=True;k.t_water[0,:]=0
        pooled=k.pool_water(2,10);self.assertAlmostEqual(pooled['water_last'][0,0],.7)
        self.assertEqual(pooled['known_fraction'][0,0],.5)
        original=np.zeros((9,8,12));original[0,3,3]=1;original[4]=.3
        w=world();w.team=Knowledge((8,12),original);w.observe('uav-1');w.share('uav-1')
        self.assertTrue(w.team.explored_ever[3,3]);np.testing.assert_array_equal(w.team.legacy_channels,original)

    def test_original_terrain_and_signal_estimates_can_be_imported_without_fabricating_water(self):
        terrain=np.ones((8,12))*7;signal=np.zeros((8,12));legacy=np.zeros((9,8,12));legacy[1,4,4]=1
        k=Knowledge((8,12),dict(legacy_channels=legacy,terrain_field=terrain,signal_field=signal))
        self.assertTrue(k.explored_ever[4,4]);self.assertTrue(k.terrain_valid.all());self.assertTrue(k.signal_valid.all())
        np.testing.assert_array_equal(k.terrain_field,terrain);np.testing.assert_array_equal(k.signal_field,signal)
        self.assertFalse(k.water_valid.any());self.assertTrue(np.isnan(k.water_last).all())

    def test_archive_has_all_map_groups_shared_faces_and_unknown_nan_masks(self):
        w=world();w.observe('uav-1');w.advance(10);w.share('uav-1')
        with zipfile.ZipFile(io.BytesIO(w.export_zip())) as z:
            self.assertIsNone(z.testzip());state=json.loads(z.read('map_model.json'))
            self.assertEqual(state['team']['information_age'][2][1],10)
            self.assertIsNone(state['team']['M_obs']['water_last'][7][11])
            arrays=np.load(io.BytesIO(z.read('layers.npz')),allow_pickle=False)
            for prefix in ('M_base','M_ref','M_water','team','agents.uav-1'):
                self.assertTrue(any(n.startswith(prefix+'.') for n in arrays.files))
            self.assertEqual(arrays['M_water.qx'].shape,(8,13))
            self.assertEqual(arrays['M_water.qy'].shape,(9,12))
            self.assertTrue(np.isnan(arrays['team.M_obs.water_last'][7,11]))
            self.assertIn('source_hydrographs',state)

    def test_current_generated_map_uses_common_coarse_grid_for_every_layer(self):
        terrain=server.describe('example/raw')
        w=map_sessions.create(ROOT,terrain,dict(mode='current',terrain_id=terrain['id'],
            flood=dict(resolution=32,baseflow=False,warmup_s=0)))
        self.assertEqual(w.base.shape,(32,32));self.assertEqual(w.base.dx,32)
        self.assertEqual(w.base.terrain_class.shape,(32,32))
        self.assertEqual(w.knowledge['uav-1'].water_valid.shape,(32,32))


if __name__=='__main__':unittest.main()
