import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server
import terrain_modes as modes
import terrain_cache


class TerrainModeTests(unittest.TestCase):
    def test_profiles_control_all_terrain_families_and_keep_seed(self):
        for name in modes.PROFILES:
            cfg=server.settings(dict(terrain_mode=name,seed='123',mountains_weight=3))
            self.assertEqual(cfg['seed'],'123')
            self.assertEqual(cfg['mountains_weight'],modes.PROFILES[name]['parameters']['mountains_weight'])
            preset=server.preset_for(cfg)
            self.assertEqual(preset['world']['continent']['continentType'],
                             'UPLIFT' if name=='mountains' else 'MULTI')
            if name!='mountains':
                self.assertLess(preset['terrain']['mountains']['verticalScale'],.1)
                self.assertEqual(preset['terrain']['plains']['baseScale'],preset['terrain']['mountains']['baseScale'])
            for family in ('plateau','torridonian','dales','steppe','badlands','volcano'):
                self.assertEqual(preset['terrain'][family]['weight'],0)
        self.assertEqual(server.settings({})['terrain_mode'],'custom')
        self.assertGreater(server.preset_for(server.settings({}))['terrain']['plateau']['weight'],0)
        with self.assertRaises(ValueError): server.settings(dict(terrain_mode='desert'))

    def test_search_is_bounded_deterministic_and_valid_near_world_limit(self):
        cfg=server.settings(dict(terrain_mode='plains',seed='7'))
        self.assertEqual(modes.candidate_origins(cfg),modes.candidate_origins(cfg.copy()))
        self.assertEqual(len(modes.candidate_origins(cfg)),9)
        cfg.update(origin_x=30000000,origin_z=-30000000)
        for x,z in modes.candidate_origins(cfg): self.assertTrue(abs(x)<=30000000 and abs(z)<=30000000)

    def test_sea_and_isolated_lake_do_not_count_as_network(self):
        h=np.full((32,32),60.); t=np.zeros(h.shape,int);w=np.full(h.shape,63.)
        ids={'0':'ocean','1':'river','2':'plains','3':'mountains','4':'lake'}
        for kind in (0,4):
            t[:]=kind
            m=modes.assess_arrays(h,t,w,ids,63,4)
            self.assertEqual(m['wet_river_fraction'],0)
            self.assertFalse(modes.verdict('water_network',m))

    def test_flat_and_mountain_outputs_have_distinct_verdicts(self):
        ids={'0':'plains','1':'mountains'}
        h=np.full((32,32),80.);t=np.zeros(h.shape,int)
        flat=modes.assess_arrays(h,t,h,ids,63,4)
        self.assertTrue(modes.verdict('plains',flat))
        self.assertFalse(modes.verdict('mountains',flat))
        h=80+np.arange(32)[:,None]*np.ones((32,32))*5;t[:]=1
        mountain=modes.assess_arrays(h,t,h,ids,63,4)
        self.assertTrue(modes.verdict('mountains',mountain))
        self.assertFalse(modes.verdict('plains',mountain))

    def test_distributed_wet_channels_required(self):
        ids={'0':'plains','1':'river'}
        h=np.full((32,32),80.);t=np.zeros(h.shape,int);w=h.copy()
        t[::8,:]=1;t[:,::8]=1;h[t==1]=75;w[t==1]=80
        network=modes.assess_arrays(h,t,w,ids,63,4)
        self.assertTrue(modes.verdict('water_network',network))
        w[:]=h
        self.assertFalse(modes.verdict('water_network',modes.assess_arrays(h,t,w,ids,63,4)))

    def test_network_density_is_stable_across_sampling_steps(self):
        densities=[]
        for step in (4,8,16):
            coords=np.arange(0,2048,step)
            channels=np.zeros((len(coords),len(coords)),bool)
            for center in (256,768,1280,1792):
                hit=np.abs(coords-center)<=16
                channels[hit,:]=True;channels[:,hit]=True
            h=np.where(channels,75.,80.);types=channels.astype(int);water=np.full(h.shape,80.)
            m=modes.assess_arrays(h,types,water,{'0':'plains','1':'river'},63,step)
            self.assertTrue(modes.verdict('water_network',m))
            densities.append(m['river_length_density_km_per_km2'])
        self.assertLess(max(densities)/min(densities),1.1)

    def test_three_crossings_accepts_broad_wet_network_but_other_gates_still_apply(self):
        metrics=dict(land_fraction=.834278,river_length_density_km_per_km2=6.000988,
            river_tile_coverage=.625,river_boundary_crossings=3,land_relief_m=10.86485)
        self.assertTrue(modes.verdict('water_network',metrics))
        for key,value in [('river_boundary_crossings',2),('river_tile_coverage',.5),
                          ('river_length_density_km_per_km2',.79),('land_fraction',.54),
                          ('land_relief_m',81)]:
            with self.subTest(key=key):
                changed=dict(metrics,**{key:value})
                self.assertFalse(modes.verdict('water_network',changed))
                self.assertEqual(len(modes.rejection_reasons('water_network',changed)),1)

    def test_channel_length_is_not_a_water_area_threshold(self):
        mask=np.zeros((160,160),bool);mask[80,10:150]=True
        thin=modes.river_length(mask,8)
        wide=np.zeros_like(mask);wide[74:87,10:150]=True
        thick=modes.river_length(wide,8)
        self.assertGreater(thick/thin,.85);self.assertLessEqual(thick/thin,1.05)
        self.assertEqual(modes.river_length(np.zeros_like(mask),8),0)

    def test_cache_requires_matching_parameters_and_complete_native_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job=root/'generated/old';raw=job/'candidate-1/raw';raw.mkdir(parents=True)
            preset={'parameters':1};(job/'preset.json').write_text(json.dumps(preset))
            (job/'config.json').write_text('{"seed":"42"}')
            fields={name:dict(file=name+'.f32',dtype='<f4') for name in ('height','terrain','riverWaterLevel','riverZone')}
            for field in fields.values():np.ones((4,4),dtype='<f4').tofile(raw/field['file'])
            grid=dict(seed='42',shape=[4,4],step_blocks=16,origin_blocks=[512,1536],
                upstream_revision=terrain_cache.REVISION,height_policy='simulation_without_game_ceiling',fields=fields)
            (raw/'grid.json').write_text(json.dumps(grid))
            request=dict(seed='42',nx=4,nz=4,step_blocks=16)
            match=terrain_cache.previews(root,root/'generated/new',preset,request)
            self.assertEqual(match,{(512,1536):raw})
            self.assertEqual(terrain_cache.previews(root,root/'generated/new',{'parameters':2},request),{})
            self.assertEqual(terrain_cache.previews(root,root/'generated/new',preset,dict(request,seed='43')),{})
            self.assertEqual(terrain_cache.previews(root,root/'generated/new',preset,dict(request,step_blocks=8)),{})
            (raw/'river-inlets.json').write_text('{"old_url":"/generated/old/candidate-1/raw"}')
            target=root/'generated/new/raw';terrain_cache.copy_preview(raw,target)
            self.assertTrue((target/'height.f32').is_file());self.assertFalse((target/'river-inlets.json').exists())
            (raw/'height.f32').write_bytes(b'broken')
            self.assertEqual(terrain_cache.previews(root,root/'generated/new',preset,request),{})

    def generate_fake(self,checks,*,mode='quick',cancel=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'current.json').write_text('{"directory":"previous/raw"}')
            cfg=server.settings(dict(seed='42',terrain_mode='plains',mode=mode))
            calls=[]
            def export(command,cwd,log,report,cancelled,**kwargs):
                request=json.loads(Path(command[-1]).read_text())
                calls.append(request)
                (Path(command[-1]).parent/request['output']).mkdir(parents=True,exist_ok=True)
                return 0
            def describe(relative):
                if cancel:server.JOBS['test']['cancel_requested']=True
                return dict(id=relative,full=mode=='surface')
            with patch.object(server,'ROOT',root),patch.object(server.generation_process,'run',side_effect=export),\
                 patch.object(modes,'assess',side_effect=[dict(accepted=value,label='平原') for value in checks]),\
                 patch.object(server,'describe',side_effect=describe):
                server.JOBS['test']={};server.generate('test',cfg)
            job=server.JOBS.pop('test')
            return job,calls,json.loads((root/'current.json').read_text()),\
                json.loads((root/'generated/test/config.json').read_text())

    def test_rejected_crop_then_accept_preserves_seed_and_records_actual_origin(self):
        job,calls,pointer,cfg=self.generate_fake([False,True])
        self.assertEqual(job['state'],'done');self.assertEqual(len(calls),2)
        self.assertEqual(pointer['directory'],'generated/test/raw')
        self.assertEqual(cfg['seed'],'42');self.assertEqual(cfg['origin_x'],calls[-1]['origin_x'])
        self.assertNotEqual(cfg['origin_x'],calls[0]['origin_x'])

    def test_no_match_preserves_previous_pointer(self):
        job,calls,pointer,_=self.generate_fake([False]*9)
        self.assertEqual(job['state'],'failed');self.assertEqual(len(calls),9)
        self.assertEqual(pointer['directory'],'previous/raw')

    def test_surface_is_only_built_after_preview_acceptance_and_checked_again(self):
        job,calls,pointer,_=self.generate_fake([False,True,False],mode='surface')
        self.assertEqual(job['state'],'failed');self.assertEqual(len(calls),3)
        self.assertTrue(all('surface' not in request for request in calls[:2]))
        self.assertEqual(calls[-1]['surface']['origin_x'],calls[1]['origin_x'])
        self.assertEqual(pointer['directory'],'previous/raw')

    def test_cancel_after_verification_still_preserves_pointer(self):
        job,_,pointer,_=self.generate_fake([True],cancel=True)
        self.assertEqual(job['state'],'cancelled');self.assertEqual(pointer['directory'],'previous/raw')


if __name__=='__main__':unittest.main()
