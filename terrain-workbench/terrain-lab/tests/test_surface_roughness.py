"""Classification evidence, native-grid aggregation and manual-override precedence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import surface_roughness as roughness
import flood_run
import map_sessions
from grid_coarsening import blocks


class ClassificationTests(unittest.TestCase):
    def test_biome_proxy_distinguishes_forest_grass_and_wetland(self):
        types=np.zeros((2,3),int)
        biomes=np.array([[0,1,2],[3,4,99]])
        n,meta=roughness.classify(types,{'0':'unknown'},.042,biomes=biomes,
            biome_ids={'minecraft:forest':0,'minecraft:plains':1,'minecraft:desert':2,
                       'minecraft:swamp':3,'minecraft:meadow':4})
        np.testing.assert_allclose(n,[[.1,.035,.025],[.12,.035,.042]])
        self.assertEqual(meta['fallback_cells'],1)

    def test_observed_tree_and_grass_cover_precede_biome_proxy(self):
        n,_=roughness.classify(np.zeros((2,2)),{'0':'unknown'},.035,
            biomes=np.zeros((2,2)),biome_ids={'minecraft:forest':0},
            materials=np.zeros((2,2)),block_ids={'minecraft:grass_block':0},
            cover=np.array([[0,1],[2,0]]))
        np.testing.assert_allclose(n,[[.035,.1],[.035,.035]])

    def test_channel_and_water_identity_precede_cover(self):
        n,_=roughness.classify(np.array([[1,2],[3,0]]),
            {'0':'unknown','1':'river','2':'lake','3':'deep_ocean'},.041,cover=np.ones((2,2)))
        np.testing.assert_allclose(n,[[.035,.03],[.03,.1]])

    def test_exposed_material_and_unknown_fallback(self):
        n,_=roughness.classify(np.zeros((2,2)),{'0':'mountains'},.041,
            materials=np.array([[0,1],[2,99]]),block_ids={'minecraft:stone':0,'minecraft:sand':1,'minecraft:gravel':2})
        np.testing.assert_allclose(n,[[.03,.025],[.035,.041]])

    def test_invalid_cover_is_rejected(self):
        with self.assertRaises(ValueError):
            roughness.classify(np.zeros((2,2)),{'0':'plains'},.035,cover=np.full((2,2),16))


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name);shape=(4,4)
        arrays={'height':np.zeros(shape,dtype='<f4'),'terrain':np.zeros(shape,dtype='<i4'),
                'riverWaterLevel':np.zeros(shape,dtype='<f4'),
                'ground_height':np.zeros(shape,dtype='<f4'),'water_surface':np.zeros(shape,dtype='<f4'),
                'biome_id':np.zeros(shape,dtype='<i4'),'ground_block_id':np.zeros(shape,dtype='<i4'),
                'cover_flags':np.tile(np.array([0,1,2,0],dtype='<i4'),(4,1))}
        fields={}
        for key,array in arrays.items():
            array.tofile(self.root/key);fields[key]=dict(url='/'+key,dtype=array.dtype.str)
        self.terrain=dict(id='fixture',shape=list(shape),step=1,scale=1,origin=[0,0],full=True,
            sea_level=-1,fields=fields,terrain_ids={'0':'plains'},config={'seed':'1'},
            block_ids={'minecraft:grass_block':0},biome_ids={'minecraft:forest':0})

    def tearDown(self):self.temp.cleanup()

    def world(self,**kwargs):
        flood=dict(roughness_mode='landcover',baseflow=False,warmup_s=0,rain_mm_h=0,boundary='closed')
        flood.update(kwargs.pop('flood',{}))
        data=dict(mode='current',terrain_id='fixture',flood=flood,**kwargs)
        return map_sessions.create(self.root,self.terrain,data)

    def test_solver_receives_native_classification_and_export_keeps_it(self):
        world=self.world()
        expected=np.tile([.035,.1,.035,.035],(4,1))
        np.testing.assert_allclose(world.solver.manning,expected)
        np.testing.assert_allclose(world.snapshot()['M_base']['n'],expected)
        self.assertEqual(world.meta['roughness']['source_kind'],'actual_surface')
        self.assertEqual(world.meta['roughness']['effective_max_n'],.1)

    def test_native_values_are_averaged_before_classification_is_coarsened(self):
        fine=self.world();coarse=self.world(flood=dict(grid_factor=2))
        np.testing.assert_allclose(coarse.base.n,blocks(fine.base.n,2))
        np.testing.assert_allclose(coarse.solver.manning,coarse.base.n)

    def test_patch_is_applied_once_before_coarsening(self):
        patch=dict(kind='roughness',x=0,z=0,value=.12)
        fine=self.world(layers=dict(patches=[patch]))
        coarse=self.world(flood=dict(grid_factor=2),layers=dict(patches=[patch]))
        np.testing.assert_allclose(coarse.base.n,blocks(fine.base.n,2))
        self.assertEqual(coarse.meta['roughness']['manual_patch_count'],1)

    def test_explicit_array_then_patch_override_automatic_n(self):
        world=self.world(layers=dict(n=np.full((4,4),.04),patches=[dict(kind='roughness',x=1,z=1,value=.09)]))
        self.assertAlmostEqual(world.base.n[0,0],.04)
        self.assertAlmostEqual(world.base.n[1,1],.09)
        self.assertEqual(world.meta['roughness']['base_source'],'explicit_n_layer')

    def test_return_to_fine_grid_restores_auto_detail_outside_coarse_patch(self):
        original=self.world()
        restored=self.world(layers=dict(grid_factor=2,patches=[]))
        np.testing.assert_allclose(restored.base.n,original.base.n)
        patched=self.world(layers=dict(grid_factor=2,patches=[dict(kind='roughness',x=0,z=0,value=.12)]))
        expected=original.base.n.copy();expected[:2,:2]=.12
        np.testing.assert_allclose(patched.base.n,expected)

    def test_uniform_mode_is_available_for_comparisons(self):
        world=self.world(flood=dict(roughness_mode='uniform',manning=.045))
        np.testing.assert_allclose(world.base.n,.045)

    def test_default_mode_is_saved_for_config_round_trip(self):
        data=dict(mode='current',terrain_id='fixture',flood=dict(baseflow=False,warmup_s=0))
        world=map_sessions.create(self.root,self.terrain,data)
        exported=json.loads(json.dumps(world.scenario_config))
        self.assertEqual(exported['flood']['roughness_mode'],'landcover')
        restored=map_sessions.create(self.root,self.terrain,exported)
        np.testing.assert_array_equal(restored.base.n,world.base.n)

    def test_quick_preview_works_without_biome_or_material(self):
        self.terrain['full']=False
        for name in ['biome_id','ground_block_id','cover_flags']:self.terrain['fields'].pop(name)
        world=self.world()
        np.testing.assert_allclose(world.base.n,.035)
        self.assertEqual(world.meta['roughness']['source_kind'],'terrain_preview_proxy')

    def test_unknown_mode_and_truncated_source_fail_explicitly(self):
        with self.assertRaises(ValueError):self.world(flood=dict(roughness_mode='random'))
        (self.root/'cover_flags').write_bytes(b'bad')
        with self.assertRaises(ValueError):self.world()


if __name__=='__main__':unittest.main()
