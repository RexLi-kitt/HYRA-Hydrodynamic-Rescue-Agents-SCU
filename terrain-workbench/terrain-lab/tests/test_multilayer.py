"""PPT acceptance: physical maps, continuous process, warm-up and export."""
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
from multilayer import BaseMap,MapWorld
import flood_run
import map_sessions
import server


def world(shape=(8,12),layers=None,depth=None):
    b=np.zeros(shape);h=np.zeros(shape) if depth is None else depth
    c=flood_run.settings(dict(resolution=32,source_x=0,source_z=0,rain_mm_h=0,
        rain_duration_s=0,baseflow=False,warmup_s=0),shape)
    s=FloodSolver(b,h,2,boundary='closed')
    return MapWorld(s,dict(config=c,shape=list(shape),coarsening_factor=1,origin_blocks=[0,0]),layers)


class HydraulicLayersTests(unittest.TestCase):
    def test_frictionless_solver_keeps_zero_roughness_in_map_and_export(self):
        shape=(8,12)
        config=flood_run.settings(dict(baseflow=False,warmup_s=0),shape)
        solver=FloodSolver(np.zeros(shape),np.ones(shape),2,manning=0)
        scene=MapWorld(solver,dict(config=config,coarsening_factor=1))
        np.testing.assert_array_equal(scene.base.n,0)
        np.testing.assert_array_equal(scene.solver.manning,0)
        np.testing.assert_array_equal(scene.base.snapshot()['n'],0)
        with self.assertRaises(ValueError):
            BaseMap(np.zeros(shape),2,dict(n=np.full(shape,-.001)))

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
        w=world(layers=layers);w.advance_to(2)
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
        for config in (dict(n=np.full((8,12),-.001)),dict(solid=[[0]]),dict(patches=[dict(kind='drain',x=0,z=0,value=-1)]),
            dict(patches=[dict(kind='source',x=0,z=0,value=0)]),
            dict(patches=[dict(kind='source',x=0,z=0,value=0)],hydrographs={'0':[[0,-1]]})):
            with self.assertRaises(ValueError):BaseMap(np.zeros((8,12)),2,config)

    def test_imported_elevation_and_initial_water_are_used(self):
        data=dict(mode='demo',initial_depth=np.full((32,32),.3).tolist(),
            layers=dict(z=np.ones((32,32)).tolist()))
        w=map_sessions.create(ROOT,server.describe('example/raw'),data)
        np.testing.assert_array_equal(w.solver.bed,1)
        self.assertAlmostEqual(w.solver.h[2,7],.3)

    def test_invalid_inflow_into_building_does_not_mutate_the_state(self):
        s=FloodSolver(np.zeros((4,4)),np.zeros((4,4)),2,boundary='closed')
        solid=np.zeros((4,4),bool);solid[1,1]=True;s.set_environment(solid=solid)
        before=s.summary()
        with self.assertRaises(ValueError):s.step(.1,rain=.1,inflow=1,source=(1,1))
        self.assertEqual(s.summary(),before)


class MapProcessTests(unittest.TestCase):
    def test_map_snapshot_and_archive_contain_only_physical_layers(self):
        w=world();w.advance_to(75)
        state=w.snapshot()
        self.assertEqual(state['time_s'],75)
        for key in ('agents','team','M_obs','M_exp','M_pos'):self.assertNotIn(key,state)
        self.assertFalse(hasattr(w,'observe'));self.assertFalse(hasattr(w,'move'))
        with zipfile.ZipFile(io.BytesIO(w.export_zip())) as z:
            self.assertIsNone(z.testzip());arrays=np.load(io.BytesIO(z.read('layers.npz')))
            self.assertEqual(arrays['M_water.qx'].shape,(8,13))
            self.assertEqual(arrays['M_water.qy'].shape,(9,12))
            self.assertTrue(all(k.startswith(('M_base.','M_water.','M_forcing.','M_people.')) for k in arrays.files))

    def test_current_generated_map_has_common_grid_and_no_agent_defaults(self):
        terrain=server.describe('example/raw')
        w=map_sessions.create(ROOT,terrain,dict(mode='current',terrain_id=terrain['id'],
            flood=dict(resolution=32,baseflow=False,warmup_s=0)))
        self.assertEqual(w.base.shape,tuple(terrain['shape']));self.assertEqual(w.base.dx,terrain['step'])
        self.assertEqual(w.base.terrain_class.shape,tuple(terrain['shape']))
        self.assertEqual(w.meta['coarsening_factor'],1)
        raw=terrain['fields']['terrain']
        np.testing.assert_array_equal(w.base.terrain_class,np.fromfile(ROOT/raw['url'].lstrip('/'),dtype=raw['dtype']).reshape(terrain['shape']))
        self.assertNotIn('agents',w.snapshot())

    def test_binary_scene_and_streamed_archive_preserve_all_grid_values(self):
        import tempfile
        w=world(depth=np.full((8,12),.25));w.scenario_config=dict(mode='demo')
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/test-temp') as tmp:
            root=Path(tmp)
            scene=map_sessions.browser_scene(root,'packed-test',w,pack_threshold=0)
            for group,fields in [('M_base',w.base.snapshot()),('M_water',w.water(compact=True))]:
                for name,a in fields.items():
                    if not isinstance(a,np.ndarray):continue
                    field=scene[group][name]
                    actual=np.fromfile(root/field['array_url'].lstrip('/'),dtype=field['dtype']).reshape(field['shape'])
                    np.testing.assert_array_equal(actual,a)
            out=io.StringIO();map_sessions.write_snapshot(out,w)
            self.assertEqual(json.loads(out.getvalue()),w.snapshot())
            url=map_sessions.save_process(root,'packed-test',w,[])
            with zipfile.ZipFile(root/url.lstrip('/')) as archive:
                arrays=np.load(io.BytesIO(archive.read('layers.npz')))
                np.testing.assert_array_equal(arrays['M_base.z'],w.base.z)
                self.assertEqual(json.loads(archive.read('map_model.json')),w.snapshot())

    def test_removed_agent_config_and_reversed_time_are_rejected(self):
        terrain=server.describe('example/raw')
        with self.assertRaises(ValueError):map_sessions.create(ROOT,terrain,dict(mode='demo',agents=[]))
        w=world();w.advance_to(2)
        for end in (-1,0,float('nan')):
            with self.assertRaises(ValueError):w.advance_to(end)

    def test_full_process_frames_use_edited_physics_and_have_complete_archive(self):
        import tempfile
        w=world(layers=dict(patches=[dict(kind='source',x=1,z=2,value=3)],hydrographs={'3':[[0,2],[1,0]]}))
        w.config.update(duration_s=2,frames=5)
        updates=[]
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/test-temp') as tmp:
            result=map_sessions.run(Path(tmp),'test',w,lambda **v:updates.append(v),lambda:False)
            self.assertEqual([f['time_s'] for f in result['frames']],[0,.5,1,1.5,2])
            self.assertAlmostEqual(w.solver.summary()['balance']['scheduled_inflow_m3'],2)
            for f in result['frames']:
                data=np.fromfile(Path(tmp)/f['url'].lstrip('/'),dtype='<f4').reshape(4,8,12)
                self.assertTrue(np.isfinite(data).all());self.assertGreaterEqual(data[0].min(),0)
            self.assertTrue(np.all(data[3]>=data[0]-1e-6))
            with zipfile.ZipFile(Path(tmp)/result['archive_url'].lstrip('/')) as z:
                self.assertIsNone(z.testzip());self.assertIn('process.json',z.namelist())
                self.assertEqual(len(json.loads(z.read('process.json'))['frames']),5)
                self.assertEqual(len([n for n in z.namelist() if n.endswith('.f32')]),5)
                self.assertNotIn('agents',json.loads(z.read('map_model.json')))

    def test_cancelled_process_keeps_completed_frames_and_can_export(self):
        import tempfile
        w=world();w.config.update(duration_s=4,frames=5)
        updates=[];cancelled=False
        def update(**values):
            nonlocal cancelled
            updates.append(values)
            if len(values.get('frames',[]))>=2:cancelled=True
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/test-temp') as tmp:
            with self.assertRaises(InterruptedError):
                map_sessions.run(Path(tmp),'cancel-test',w,update,lambda:cancelled)
            frames=updates[-1]['frames'];self.assertEqual(len(frames),2)
            url=map_sessions.save_process(Path(tmp),'cancel-test',w,frames)
            with zipfile.ZipFile(Path(tmp)/url.lstrip('/')) as z:
                self.assertEqual(len(json.loads(z.read('process.json'))['frames']),2)
                self.assertIsNone(z.testzip())

    def test_warmup_excludes_flood_hydrographs_and_resets_scenario_clock(self):
        import tempfile
        w=world(layers=dict(patches=[dict(kind='source',x=1,z=2,value=0)],hydrographs={'0':[[0,2]]}))
        w.solver.set_baseflow([dict(x=1,z=2,discharge_m3_s=.1)])
        w.config.update(warmup_s=1,duration_s=1,frames=3)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/test-temp') as tmp:
            result=map_sessions.run(Path(tmp),'warm-test',w,lambda **v:None,lambda:False)
        initial=w.meta['initial_conditions']
        self.assertAlmostEqual(initial['balance']['baseflow_m3'],.1)
        self.assertEqual(initial['balance']['scheduled_inflow_m3'],0)
        self.assertEqual(result['frames'][0]['time_s'],0)
        self.assertAlmostEqual(result['frames'][0]['balance']['initial_m3'],.1)
        self.assertAlmostEqual(result['frames'][-1]['balance']['scheduled_inflow_m3'],2)
        self.assertAlmostEqual(result['frames'][-1]['balance']['baseflow_m3'],.1)

    def test_concurrent_run_returns_existing_process_for_ui_recovery(self):
        import threading
        from unittest.mock import patch
        active=dict(process=dict(state='running'),lock=threading.Lock())
        ready=dict(process=dict(state='ready'),lock=threading.Lock())
        h=object.__new__(server.Handler);h.path='/api/maps/second/run'
        h.headers={'Content-Length':'2'};h.rfile=io.BytesIO(b'{}');reply=[]
        h.json=lambda data,status=200:reply.append((data,status))
        with patch.dict(server.MAP_SESSIONS,{'first':active,'second':ready},clear=True):
            h.do_POST()
        self.assertEqual(reply[0][1],409)
        self.assertEqual(reply[0][0]['process_id'],'first')
        self.assertEqual(ready['process']['state'],'ready')

    def test_no_initial_water_and_frame_validation(self):
        t=server.describe('example/raw')
        w=map_sessions.create(ROOT,t,dict(mode='demo',flood=dict(initial_water=False)))
        self.assertEqual(w.solver.h.sum(),0)
        for frames in (1,242,True,4.5):
            with self.assertRaises(ValueError):map_sessions.create(ROOT,t,dict(mode='demo',flood=dict(frames=frames)))


if __name__=='__main__':
    (ROOT/'.runtime/test-temp').mkdir(parents=True,exist_ok=True)
    unittest.main()
