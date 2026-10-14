import io,json,sys,tempfile,unittest,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver,G
from multilayer import MapWorld
import flood_run,river_inlets,map_sessions

def world(depth=1,dx=2,factor=1,boundary='closed',sea=None):
    shape=(8,12);c=flood_run.settings(dict(resolution=32,source_x=0,source_z=0,
        rain_mm_h=0,rain_duration_s=0,baseflow=False,warmup_s=0),shape)
    return MapWorld(FloodSolver(np.zeros(shape),np.full(shape,float(depth)),dx,
        boundary=boundary,sea_level=depth,sea_edges=sea,manning=0),dict(config=c,coarsening_factor=factor))

def port(edge='west',sid='river-a',factor=1):
    z,x={'west':(3*factor,0),'east':(3*factor,12*factor-1),
         'north':(0,5*factor),'south':(8*factor-1,5*factor)}[edge]
    offset=z if edge in ('west','east') else x
    return dict(id=sid,kind='river',x=x,z=z,native_cells=[[z,x]],
        faces=[dict(edge=edge,offset=offset)],role='uncertain',edge=edge)

def configure(w,ports,curves=None,velocity=1.5):
    curves=curves or [[[0,4]]]*len(ports)
    river_inlets.configure(w,ports,[dict(id=p['id'],kind='river',hydrograph=q,
        inlet_velocity_m_s=velocity) for p,q in zip(ports,curves)],
        [8*w.meta['coarsening_factor'],12*w.meta['coarsening_factor']])

class BoundaryInflowTests(unittest.TestCase):
    def test_each_outer_edge_has_exact_discharge_and_inward_momentum(self):
        for edge,axis,sign in [('west',1,1),('east',1,-1),('north',0,1),('south',0,-1)]:
            with self.subTest(edge=edge):
                w=world();p=port(edge);configure(w,[p]);s=w.solver
                f=w.external_inflows[0]['boundary_faces'][0];i=f['offset'];j=0 if sign==1 else -1
                mass,fx,fz,*_=s._flux(axis);index=(i,j) if axis==1 else (j,i)
                self.assertAlmostEqual(sign*mass[index]*s.dx,4)
                self.assertAlmostEqual((fx if axis==1 else fz)[index],4/s.dx*1.5+.5*G)
                self.assertEqual(s.external_rate.sum(),0)
                before=s.h.sum()*s.dx**2;s.step(.01);u,v=s.velocity()
                self.assertGreater(sign*(u if axis==1 else v)[f['z'],f['x']],0)
                self.assertAlmostEqual(s.h.sum()*s.dx**2-before,.04)
                self.assertAlmostEqual(s.summary()['balance']['external_inflow_m3'],.04)

    def test_multiple_boundary_curves_and_manual_source_count_water_once(self):
        w=world();ports=[port(),port('south','river-b')]
        entries=[dict(id=ports[0]['id'],kind='river',hydrograph=[[0,4],[.5,0],[1,2],[2,0]]),
                 dict(id=ports[1]['id'],kind='river',hydrograph=[[.5,3],[2,0]]),
                 dict(id='manual',kind='manual',x=6,z=4,hydrograph=[[0,1],[2,0]])]
        river_inlets.configure(w,ports,entries,[8,12]);w.advance_to(3)
        b=w.solver.summary();self.assertAlmostEqual(b['balance']['external_inflow_m3'],10.5)
        self.assertAlmostEqual(b['external_inflow_by_id_m3']['river-a'],4)
        self.assertAlmostEqual(b['external_inflow_by_id_m3']['river-b'],4.5)
        self.assertAlmostEqual(b['external_inflow_by_id_m3']['manual'],2)
        self.assertLess(abs(b['balance']['residual_m3']),1e-9)

    def test_native_width_coarsening_preserves_total_flow_and_incoming_velocity(self):
        for factor in [1,2,4]:
            w=world(dx=2*factor,factor=factor);configure(w,[port(factor=factor)],velocity=2)
            e=w.external_inflows[0];f=e['boundary_faces'][0]
            self.assertEqual(f['width_m'],2)
            state=w.solver.boundary_forcing()[1][0]['faces'][0]
            self.assertEqual(state['inward_velocity_m_s'],2)
            self.assertAlmostEqual(state['reference_depth_m'],1)
            self.assertAlmostEqual(w.solver._flux(1)[0][:,0].sum()*w.solver.dx,4)
            w.advance_to(.5);self.assertAlmostEqual(w.solver.balance['external_inflow_m3'],2)

    def test_dry_start_auto_speed_and_explicit_high_speed_are_cfl_limited(self):
        for velocity in [0,30]:
            with self.subTest(velocity=velocity):
                w=world(depth=0,dx=1);configure(w,[port()],velocity=velocity)
                state=w.solver.boundary_forcing()[1][0]['faces'][0]
                self.assertGreater(state['inward_velocity_m_s'],0)
                self.assertLess(w.solver.timestep(),.1)
                w.advance_to(.5);s=w.solver
                self.assertTrue(np.isfinite(s.h).all());self.assertGreaterEqual(s.h.min(),0)
                self.assertAlmostEqual(s.balance['external_inflow_m3'],2)
                self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)

    def test_backwater_and_outward_cell_momentum_do_not_reverse_boundary_flow(self):
        w=world(depth=10);configure(w,[port()],velocity=2);s=w.solver;s.mx[:]=-5*s.h
        mass=s._flux(1)[0];self.assertAlmostEqual(mass[3,0]*s.dx,4)
        s.step(.0001);self.assertLess(s.velocity()[0][3,4],0) # Interior flow is not clamped.
        self.assertAlmostEqual(s.qx[3,0]*s.dx,4)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)

    def test_corner_crossing_splits_discharge_between_normals_once(self):
        w=world();p=dict(id='corner',kind='river',x=0,z=0,native_cells=[[0,0]],role='uncertain',
            faces=[dict(edge='west',offset=0),dict(edge='north',offset=0)])
        configure(w,[p]);s=w.solver
        self.assertAlmostEqual(s._flux(1)[0][0,0]*s.dx,2)
        self.assertAlmostEqual(s._flux(0)[0][0,0]*s.dx,2)
        self.assertEqual(w.external_inflows[0]['cells'],[dict(z=0,x=0,weight=1)])
        w.advance_to(.5);self.assertAlmostEqual(s.balance['external_inflow_m3'],2)

    def test_mixed_ocean_face_keeps_separate_sea_exchange_and_source_budget(self):
        w=world(depth=1,dx=8,factor=4,boundary='ocean',sea={'west':[True]*8})
        configure(w,[port(factor=4)],velocity=2);w.solver.h[:,0]=2;w.solver.start_scenario()
        w.advance_to(.5);b=w.solver.summary()['balance']
        self.assertAlmostEqual(b['external_inflow_m3'],2)
        self.assertGreater(b['sea_outflow_m3'],0);self.assertLess(abs(b['residual_m3']),1e-9)

    def test_zero_discharge_stops_only_selected_face_and_restores_on_disable(self):
        w=world(boundary='open');configure(w,[port()],curves=[[[0,0]]]);s=w.solver;s.mx[:]=-2*s.h
        self.assertEqual(s._flux(1)[0][3,0],0)
        self.assertLess(s._flux(1)[0][2,0],0)
        s.set_external_inflows([]);self.assertLess(s._flux(1)[0][3,0],0)

    def test_blocked_faces_and_invalid_velocity_are_rejected(self):
        w=world();w.base.wall_x[3,0]=True
        with self.assertRaises(ValueError):configure(w,[port()])
        for v in [-1,31,True,float('nan')]:
            with self.subTest(v=v),self.assertRaises(ValueError):configure(world(),[port()],velocity=v)

    def test_warmup_export_has_no_premature_flood_and_records_face_delivery(self):
        w=world();configure(w,[port()],curves=[[[0,4],[1,0]]]);w.solver.set_baseflow([dict(x=5,z=3,discharge_m3_s=.1)])
        w.config.update(warmup_s=1,duration_s=1,frames=3)
        with tempfile.TemporaryDirectory() as tmp:
            result=map_sessions.run(Path(tmp),'boundary',w,lambda **v:None,lambda:False)
            self.assertEqual(w.meta['initial_conditions']['balance']['external_inflow_m3'],0)
            self.assertAlmostEqual(result['frames'][-1]['balance']['external_inflow_m3'],4)
            state=result['frames'][-1]['boundary_inflows'][0]
            self.assertEqual(state['discharge_m3_s'],4)
            with zipfile.ZipFile(Path(tmp)/'generated/boundary/maps/map-result.zip') as z:
                model=json.loads(z.read('map_model.json'));entry=model['M_forcing']['external_inflows'][0]
                self.assertEqual(entry['delivery'],'boundary_discharge_with_momentum')
                self.assertIn('boundary_faces',entry)
                a=np.load(io.BytesIO(z.read('layers.npz')),allow_pickle=False)
                self.assertAlmostEqual(a['M_water.qx'][3,0]*w.solver.dx,4)

    def test_completed_process_saves_replay_without_runtime_session(self):
        w=map_sessions.create(ROOT,dict(shape=[32,32]),dict(mode='demo',
            flood=dict(duration_s=1,rain_duration_s=0,rain_mm_h=0,frames=3)))
        with tempfile.TemporaryDirectory() as tmp:
            result=map_sessions.run(Path(tmp),'maps-replay',w,lambda **v:None,lambda:False)
            folder=Path(tmp)/'generated/maps-replay/maps'
            scene=json.loads((folder/'replay.json').read_text(encoding='utf-8'))
            status=json.loads((folder/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(scene['scenario_config']['mode'],'demo')
            self.assertEqual(scene['id'],'maps-replay');self.assertEqual(status['state'],'done')
            self.assertEqual(status['frames'],result['frames'])
            self.assertEqual(scene['archive_status_url'],'/generated/maps-replay/maps/status.json')

    def test_partial_process_replay_remains_cancelled(self):
        w=map_sessions.create(ROOT,dict(shape=[32,32]),dict(mode='demo',
            flood=dict(duration_s=1,rain_duration_s=0,rain_mm_h=0,frames=3)))
        frames=[]
        def update(**v):
            if 'frames' in v:frames[:]=v['frames']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError):map_sessions.run(Path(tmp),'maps-partial',w,update,lambda:bool(frames))
            map_sessions.save_process(Path(tmp),'maps-partial',w,frames)
            status=json.loads((Path(tmp)/'generated/maps-partial/maps/status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['state'],'cancelled');self.assertEqual(len(status['frames']),1)
            self.assertEqual(status['frames'][0]['time_s'],0)

if __name__=='__main__':unittest.main()
