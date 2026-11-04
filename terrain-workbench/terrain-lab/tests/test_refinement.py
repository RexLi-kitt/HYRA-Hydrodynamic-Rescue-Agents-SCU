"""Physical invariants on a true mixed coarse/fine leaf mesh and its exports."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver
from refined_mesh import LeafMesh,inland_mask
from refined_solver import RefinedFloodSolver
import flood_run
import map_sessions


def refined(bed,h,mask,**options):
    return RefinedFloodSolver.from_solver(FloodSolver(bed,h,1,**options),mask)


class RefinementTests(unittest.TestCase):
    def test_class_selection_buffer_and_ocean_exclusion(self):
        types=np.zeros((8,12),int);types[:,8:]=4;types[2,7]=1;types[4,3]=2;types[6,1]=3
        names={'0':'land','1':'river','2':'lake','3':'wetland','4':'deep_ocean'}
        selected,mask=inland_mask(types,names,buffer=1)
        self.assertEqual(selected.sum(),3);self.assertGreater(mask.sum(),3)
        self.assertFalse(mask[:,8:].any())
        for code in (1,2,3):self.assertTrue(mask[types==code].all())
        solid=np.zeros_like(mask);solid[4,3]=True
        self.assertFalse(inland_mask(types,names,solid,0)[1][4,3])
        self.assertFalse(inland_mask(types,{'4':'ocean'},buffer=3)[1].any())

    def test_mixed_mesh_covers_area_and_all_cell_faces_close(self):
        z,x=np.indices((7,11));mask=(z+x)%3==0
        for boundary in ('closed','periodic'):
            m=LeafMesh(mask,2,boundary)
            self.assertEqual(m.count,mask.size+3*mask.sum())
            self.assertEqual(m.area.sum(),mask.size*4)
            np.testing.assert_array_equal(m.project(np.ones(m.count)),1)
            self.assertTrue(np.all(m.child[~mask.ravel()[m.parent]]==-1))
            for axis in (0,1):np.testing.assert_allclose(m.divergence((m.axis==axis).astype(float)),0,atol=0)
            self.assertTrue(np.all(m.length>0))
            internal=m.internal
            self.assertTrue(np.all(np.maximum(m.width[m.l[internal]],m.width[m.r[internal]])/
                np.minimum(m.width[m.l[internal]],m.width[m.r[internal]])<=2))

    def test_sloped_lake_at_rest_wet_dry_and_large_elevation_offset(self):
        z,x=np.indices((8,12));mask=(x>=3)&(x<8)&(z>=1)&(z<7)
        for order in (1,2):
            for offset in (0.,10000.):
                bed=offset+.08*x+.04*z;h=np.maximum(offset+.7-bed,0)
                s=refined(bed,h,mask,boundary='closed',spatial_order=order)
                original=s.leaf['h'].copy();s.advance(2)
                np.testing.assert_allclose(s.leaf['h'],original,atol=2e-12)
                np.testing.assert_allclose(s.leaf_velocity(),0,atol=2e-11)
                self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)

    def test_dam_break_sources_and_sinks_conserve_across_hanging_faces(self):
        shape=(8,16);bed=np.zeros(shape);h=bed.copy();h[:,:5]=1
        mask=np.zeros(shape,bool);mask[:,5:10]=True
        for method in ('rusanov','knp'):
            s=refined(bed,h,mask,boundary='closed',flux_method=method)
            s.advance(1,rain=.001,rain_until=.3,infiltration=.0001,inflow=.2,source=(3,7))
            self.assertGreater(s.h[:,6].max(),0)
            self.assertGreaterEqual(s.leaf['h'].min(),0)
            self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)
            self.assertAlmostEqual(s.balance['rainfall_m3'],.001*.3*bed.size)
            self.assertAlmostEqual(s.balance['inflow_m3'],.2)
            # Children evolve independently: this is not a colored base-grid mask.
            start=s.mesh.start[3*shape[1]+6]
            self.assertGreater(np.ptp(s.leaf['h'][start:start+4]),1e-6)
            self.assertLessEqual(s.diagnostics['max_stage_courant'],s.cfl+1e-8)

    def test_subfaces_apply_prescribed_boundary_flow_once_with_momentum(self):
        bed=np.zeros((8,12));h=np.ones_like(bed);mask=np.zeros_like(bed,bool);mask[2:5,:2]=True
        base=FloodSolver(bed,h,1,boundary='closed',manning=0)
        base.set_external_inflows([dict(id='upstream',enabled=True,delivery='boundary_discharge_with_momentum',
            hydrograph=[[0,.8],[.05,0]],inlet_velocity_m_s=2,
            boundary_faces=[dict(edge='west',offset=3,z=3,x=0,width_m=1.,weight=1.)])])
        s=RefinedFloodSolver.from_solver(base,mask);s.advance(.05)
        self.assertAlmostEqual(s.balance['external_inflow_m3'],.04,places=12)
        self.assertAlmostEqual(s.external_volume_by_id['upstream'],.04,places=12)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)
        m=s.mesh;faces=(m.edge==0)&(m.offset==3)
        self.assertEqual(faces.sum(),2)
        self.assertAlmostEqual(float((s.face_mass_flux[faces]*m.length[faces]).sum()),.8)
        self.assertGreater(float((s.leaf['mx']*m.area).sum()),0)

    def test_sea_exchange_stays_stationary_and_never_refines_ocean(self):
        z,x=np.indices((8,12));bed=.1*x-.4;h=np.maximum(-bed,0)
        types=np.zeros_like(x);types[:,:4]=4;types[2:5,5:8]=2
        _,mask=inland_mask(types,{'0':'land','2':'lake','4':'ocean'},buffer=2)
        self.assertFalse(mask[:,:4].any())
        s=refined(bed,h,mask,boundary='ocean',sea_level=0,sea_edges={'west':np.ones(8,bool)})
        before=s.leaf['h'].copy();s.advance(1)
        np.testing.assert_allclose(s.leaf['h'],before,atol=1e-12)
        np.testing.assert_allclose(s.leaf_velocity(),0,atol=1e-11)

    def test_periodic_rotation_and_global_mass_conservation(self):
        z,x=np.indices((6,10));bed=np.zeros_like(x,dtype=float);h=.5+.02*np.cos(2*np.pi*x/10)
        mask=(x>=3)&(x<7)
        a=refined(bed,h,mask,boundary='periodic',manning=0)
        b=refined(bed.T,h.T,mask.T,boundary='periodic',manning=0)
        a.advance(.3);b.advance(.3)
        np.testing.assert_allclose(a.h,b.h.T,atol=1e-12)
        self.assertLess(abs(a.summary()['balance']['residual_m3']),1e-10)

    def test_solid_wall_and_drying_sinks_are_respected(self):
        bed=np.zeros((8,12));h=np.zeros_like(bed);h[:,:6]=1;mask=np.zeros_like(bed,bool);mask[:,3:9]=True
        base=FloodSolver(bed,h,1,boundary='closed');wall=np.zeros((8,13),bool);wall[:,6]=True
        base.set_environment(wall_x=wall)
        s=RefinedFloodSolver.from_solver(base,mask);s.advance(.5)
        np.testing.assert_allclose(s.h,h,atol=1e-12)
        np.testing.assert_array_equal(s.qx[:,6],0)
        s=refined(bed,np.full_like(bed,.001),mask,boundary='closed')
        s.advance(.1,infiltration=1)
        np.testing.assert_allclose(s.leaf['h'],0,atol=1e-15)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-11)

    def test_small_amplitude_wave_converges_on_a_mixed_mesh(self):
        errors=[]
        for n in (8,16,32):
            z,x=np.indices((n,n));mask=(x>=n//4)&(x<3*n//4)
            s=RefinedFloodSolver.from_solver(FloodSolver(np.zeros((n,n)),np.ones((n,n)),16/n,boundary='periodic',manning=0),mask)
            eps=1e-5;c=np.sqrt(9.81);phase=2*np.pi*s.mesh.x/16
            s.leaf['h']=1+eps*np.sin(phase);s.leaf['mx']=s.leaf['h']*c*eps*np.sin(phase)
            s.start_scenario();momentum=float((s.leaf['mx']*s.mesh.area).sum());s.advance(.3)
            exact=1+eps*np.sin(2*np.pi*(s.mesh.x-c*.3)/16)
            errors.append(float((abs(s.leaf['h']-exact)*s.mesh.area).sum()/256))
            self.assertAlmostEqual(float((s.leaf['mx']*s.mesh.area).sum()),momentum,places=12)
        self.assertLess(errors[1],errors[0]*.5);self.assertLess(errors[2],errors[1]*.5)

    def test_legacy_refinement_request_runs_and_exports_native_grid(self):
        data=dict(mode='demo',flood=dict(refine_inland_water=True,refinement_buffer=0,
            duration_s=1,rain_duration_s=0,rain_mm_h=0,frames=3))
        world=map_sessions.create(ROOT,dict(shape=[32,32]),data)
        s=world.solver;self.assertIs(type(s),FloodSolver)
        self.assertFalse(world.config['refine_inland_water'])
        self.assertEqual(world.config['refinement_buffer'],0)
        self.assertEqual(world.meta['refinement']['refined_parent_cells'],0)
        self.assertEqual(world.meta['refinement']['leaf_cell_count'],32**2)
        self.assertEqual(world.meta['refinement']['minimum_cell_width_m'],s.dx)
        self.assertNotIn('M_mesh',world.snapshot())
        self.assertNotIn('refinement_level',world.snapshot()['M_base'])
        # Export fixtures need no repository-local writes.
        with tempfile.TemporaryDirectory(prefix='hyra-refinement-') as tmp:
            root=Path(tmp);reply=map_sessions.browser_scene(root,'mesh-test',world,pack_threshold=0)
            self.assertNotIn('M_mesh',reply)
            result=map_sessions.run(root,'mesh-test',world,lambda **_:None,lambda:False)
            frame=result['frames'][-1]
            self.assertNotIn('mesh_url',frame)
            actual=np.fromfile(root/frame['url'].lstrip('/'),dtype='<f4').reshape(4,32,32)
            np.testing.assert_array_equal(actual[0],s.h.astype('<f4'))
            with zipfile.ZipFile(root/result['archive_url'].lstrip('/')) as archive:
                self.assertFalse(any(name.startswith('mesh-frame-') for name in archive.namelist()))
                state=json.loads(archive.read('map_model.json'));self.assertNotIn('M_mesh',state)
                process=json.loads(archive.read('process.json'))
                self.assertIsNone(process['mesh_geometry'])
                self.assertFalse(process['refinement']['enabled'])
                self.assertEqual(process['refinement']['leaf_cell_count'],32**2)
                arrays=np.load(io.BytesIO(archive.read('layers.npz')),allow_pickle=False)
                self.assertFalse(any(key.startswith('M_mesh.') for key in arrays.files))

    def test_disabled_refinement_and_no_inland_class_keep_native_solver(self):
        for enabled,classes in ((False,np.full((32,32),2)),(True,np.zeros((32,32)))):
            world=map_sessions.create(ROOT,dict(shape=[32,32]),dict(mode='demo',layers=dict(terrain_class=classes),
                flood=dict(refine_inland_water=enabled,refinement_buffer=4)))
            self.assertNotIsInstance(world.solver,RefinedFloodSolver)
            self.assertFalse(world.meta['refinement']['enabled'])
            self.assertEqual(world.config['refinement_buffer'],0)
        for invalid in ({'refine_inland_water':'true'},{'refinement_buffer':1.5},{'refinement_buffer':True}):
            with self.assertRaises(ValueError):flood_run.settings(invalid,[32,32])


if __name__=='__main__':unittest.main()
