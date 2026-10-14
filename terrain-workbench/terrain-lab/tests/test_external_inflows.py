import io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
from multilayer import MapWorld
import flood_run,river_inlets,map_sessions,server


def world(factor=1):
    shape=(8,12);c=flood_run.settings(dict(resolution=32,source_x=0,source_z=0,
        rain_mm_h=0,rain_duration_s=0,baseflow=False,warmup_s=0),shape)
    return MapWorld(FloodSolver(np.zeros(shape),np.zeros(shape),2,boundary='closed'),
        dict(config=c,coarsening_factor=factor,origin_blocks=[0,0]))


class RiverDetectionTests(unittest.TestCase):
    def test_sea_connected_river_override_excludes_channel_banks_and_isolated_water(self):
        types=np.zeros((8,12),np.int32);types[:,0]=2;types[2:5,1:5]=5
        types[6,8:10]=5;types[0,8:10]=5
        bed=np.full(types.shape,60.);water=np.full(types.shape,62.);zones=np.full(types.shape,4)
        zones[2,3:5]=1 # Banks below sea level remain valid river cells.
        zones[0,8:10]=0 # Riverbed also remains river.
        corrected,changed=river_inlets.water_types(types,bed,water,zones,
            {'2':'ocean','5':'river'},{'0':'Riverbed','1':'Banks','4':'None'},63)
        self.assertEqual(corrected[3,3],2);self.assertEqual(corrected[2,3],5)
        self.assertEqual(corrected[6,8],5);self.assertEqual(corrected[0,8],5)
        self.assertEqual(types[3,3],5);self.assertGreater(changed.sum(),0)
        ports=river_inlets.detect(corrected==5,bed,water)
        self.assertEqual([(p['x'],p['z']) for p in ports],[(8,0)])

    def test_sea_override_requires_zone_provenance_and_sea_stage(self):
        types=np.array([[2,5],[2,5]]);bed=np.zeros((2,2));water=np.array([[2,5],[2,5]])
        args=({'2':'ocean','5':'river'},{'4':'None'},2)
        corrected,changed=river_inlets.water_types(types,bed,water,np.full((2,2),4),*args)
        self.assertFalse(changed.any());self.assertTrue(np.array_equal(corrected,types))
        corrected,changed=river_inlets.water_types(types,bed,water,None,*args)
        self.assertFalse(changed.any())

    def test_upstream_inlet_and_downstream_outlet_follow_native_slope(self):
        zz,xx=np.indices((12,16));river=np.zeros((12,16),bool);river[5:7,:]=True
        bed=8-.1*xx;ports=river_inlets.detect(river,bed,bed+.2)
        self.assertEqual(len(ports),2)
        self.assertEqual({p['edge']:p['role'] for p in ports},{'east':'outlet','west':'inlet'})
        self.assertTrue(all(p['direction_inferred'] and not p['enabled'] for p in ports))
        self.assertEqual([p['id'] for p in ports],[p['id'] for p in river_inlets.detect(river,bed,bed+.2)])

    def test_no_river_no_boundary_intersection_and_flat_ambiguous(self):
        bed=np.zeros((12,16));river=np.zeros_like(bed,bool)
        self.assertEqual(river_inlets.detect(river,bed,bed),[])
        river[5,4:12]=True;self.assertEqual(river_inlets.detect(river,bed,bed),[])
        river[5,:]=True;ports=river_inlets.detect(river,bed,bed)
        self.assertTrue(all(p['role']=='uncertain' for p in ports))

    def test_corner_crossing_is_grouped_once(self):
        bed=np.zeros((12,16));river=np.zeros_like(bed,bool);river[:4,:2]=True
        ports=river_inlets.detect(river,bed,bed)
        self.assertEqual(len(ports),1)
        self.assertEqual(set(p['edge'] for p in ports[0]['faces']),{'north','west'})
        self.assertEqual(len(ports[0]['native_cells']),len({tuple(p) for p in ports[0]['native_cells']}))

    def test_generation_description_persists_native_markers_without_pointer_change(self):
        before=(ROOT/'current.json').read_bytes() if (ROOT/'current.json').exists() else None
        t=server.describe('example/raw');ports=t['river_inlets']['crossings']
        self.assertGreater(len(ports),0);self.assertTrue((ROOT/'example/raw/river-inlets.json').exists())
        self.assertEqual(ports,server.describe('example/raw')['river_inlets']['crossings'])
        if before is not None:self.assertEqual(before,(ROOT/'current.json').read_bytes())


class ExternalFlowTests(unittest.TestCase):
    def test_two_uncertain_river_crossings_can_both_supply_water(self):
        w=world();ports=[dict(id='river-a',kind='river',role='uncertain',x=0,z=3,native_cells=[[3,0]]),
                        dict(id='river-b',kind='river',role='uncertain',x=11,z=5,native_cells=[[5,11]])]
        river_inlets.configure(w,ports,[dict(id=p['id'],kind='river',hydrograph=[[0,q],[1,0]])
            for p,q in zip(ports,[3,7])],[8,12])
        w.advance_to(1);summary=w.solver.summary()
        self.assertEqual(len(w.external_inflows),2)
        self.assertAlmostEqual(summary['external_inflow_by_id_m3']['river-a'],3)
        self.assertAlmostEqual(summary['external_inflow_by_id_m3']['river-b'],7)
        self.assertAlmostEqual(summary['balance']['external_inflow_m3'],10)

    def test_multiple_points_and_event_times_close_volume_budget(self):
        w=world();settings=[dict(id='manual-a',kind='manual',x=2,z=3,enabled=True,
            hydrograph=[[0,10],[.5,0],[1,4],[2,0]]),
            dict(id='manual-b',kind='manual',x=7,z=3,enabled=True,hydrograph=[[.5,2],[2,0]])]
        river_inlets.configure(w,[],settings,[8,12]);w.advance_to(3)
        summary=w.solver.summary();self.assertAlmostEqual(summary['balance']['external_inflow_m3'],12,places=9)
        self.assertAlmostEqual(summary['external_inflow_by_id_m3']['manual-a'],9,places=9)
        self.assertAlmostEqual(summary['external_inflow_by_id_m3']['manual-b'],3,places=9)
        self.assertAlmostEqual(w.solver.h.sum()*4,12,places=9)
        self.assertLess(abs(summary['balance']['residual_m3']),1e-9)
        self.assertGreaterEqual(w.solver.h.min(),0)

    def test_native_width_distribution_preserves_discharge_after_coarsening(self):
        w=world(factor=4);port=dict(id='river-test',kind='river',x=0,z=9,role='inlet',edge='west',
            native_cells=[[z,0] for z in range(6,14)])
        river_inlets.configure(w,[port],[dict(id=port['id'],kind='river',hydrograph=[[0,8]],enabled=True)],[32,48])
        cells=w.external_inflows[0]['cells'];self.assertEqual(len(cells),3)
        self.assertAlmostEqual(sum(c['weight'] for c in cells),1)
        self.assertEqual(w.solver.external_rate.sum(),0)
        self.assertAlmostEqual(w.solver.forcing_rate().sum()*w.solver.dx**2,8)
        w.advance_to(1);self.assertAlmostEqual(w.solver.summary()['balance']['external_inflow_m3'],8)
        self.assertEqual(w.external_inflows[0]['id'],port['id'])

    def test_no_river_manual_point_and_disabled_point_are_supported(self):
        w=world();river_inlets.configure(w,[],[
            dict(id='forced',kind='manual',x=1,z=1,hydrograph=[[0,3]]),
            dict(id='off',kind='manual',x=5,z=1,enabled=False,hydrograph=[[0,100]])],[8,12])
        w.advance_to(2);self.assertAlmostEqual(w.solver.summary()['balance']['external_inflow_m3'],6)
        self.assertEqual(w.solver.external_volume_by_id['off'],0)
        self.assertEqual(w.forcing()['inlet_count'].sum(),1)

    def test_default_detection_does_not_enable_extra_flow(self):
        t=server.describe('example/raw');w=map_sessions.create(ROOT,t,dict(mode='current',terrain_id=t['id'],
            flood=dict(resolution=32,baseflow=False,warmup_s=0)))
        self.assertTrue(w.river_crossings);self.assertEqual(w.solver.external_rate.sum(),0)
        self.assertEqual(w.external_inflows,[])

    def test_invalid_stale_duplicate_blocked_and_bad_hydrographs_are_rejected(self):
        samples=[[dict(id='missing',kind='river',hydrograph=[[0,1]])],
            [dict(id='x',kind='manual',x=-1,z=0)],
            [dict(id='x',kind='manual',x=1,z=1),dict(id='x',kind='manual',x=2,z=1)],
            [dict(id='x',kind='manual',x=1,z=1,hydrograph=[[1,2],[0,3]])],
            [dict(id='x',kind='manual',x=1,z=1,hydrograph=[[0,-1]])],
            [dict(id='x',kind='manual',x=1,z=1,enabled='yes')]]
        for entries in samples:
            with self.subTest(entries=entries),self.assertRaises(ValueError):river_inlets.configure(world(),[],entries,[8,12])
        w=world();w.base.solid[1,1]=True
        with self.assertRaises(ValueError):river_inlets.configure(w,[],[dict(id='x',kind='manual',x=1,z=1)],[8,12])

    def test_external_flood_is_excluded_from_daily_warmup(self):
        w=world();w.solver.set_baseflow([dict(x=1,z=2,discharge_m3_s=.1)])
        river_inlets.configure(w,[],[dict(id='upstream',kind='manual',x=1,z=2,hydrograph=[[0,10]])],[8,12])
        w.config.update(warmup_s=1,duration_s=1,frames=3)
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime/test-temp') as tmp:
            result=map_sessions.run(Path(tmp),'external',w,lambda **v:None,lambda:False)
        initial=w.meta['initial_conditions']['balance']
        self.assertEqual(initial['external_inflow_m3'],0)
        self.assertAlmostEqual(initial['baseflow_m3'],.1)
        self.assertAlmostEqual(result['frames'][-1]['balance']['external_inflow_m3'],10)
        self.assertAlmostEqual(result['frames'][-1]['external_inflow_by_id_m3']['upstream'],10)

    def test_snapshot_archive_and_roundtrip_preserve_forcing_for_training(self):
        t=server.describe('example/raw');port=next(p for p in t['river_inlets']['crossings'] if p['role']=='inlet')
        data=dict(mode='current',terrain_id=t['id'],external_inflows=[dict(id=port['id'],kind='river',enabled=True,
            hydrograph=[[0,5],[.5,0]])],flood=dict(resolution=32,baseflow=False,warmup_s=0))
        w=map_sessions.create(ROOT,t,json.loads(json.dumps(data)));w.advance_to(1)
        snapshot=w.snapshot();self.assertIn('M_forcing',snapshot)
        self.assertEqual(snapshot['M_forcing']['external_inflows'][0]['id'],port['id'])
        self.assertAlmostEqual(snapshot['M_forcing']['accumulated_m3'][port['id']],2.5)
        with zipfile.ZipFile(io.BytesIO(w.export_zip())) as z:
            state=json.loads(z.read('map_model.json'));self.assertEqual(state['M_forcing']['external_inflows'][0]['id'],port['id'])
            arrays=np.load(io.BytesIO(z.read('layers.npz')),allow_pickle=False)
            self.assertEqual(arrays['M_forcing.inlet_count'].shape,(32,32))
            self.assertTrue(np.isfinite(arrays['M_forcing.external_rate_m_s']).all())
        again=map_sessions.create(ROOT,t,data)
        self.assertEqual(again.external_inflows,w.external_inflows)

if __name__=='__main__':unittest.main()
