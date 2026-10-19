"""Finite barriers: original C parity, hydraulic invariants and authored objects."""
from pathlib import Path
import io
import json
import os
import sys
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hydraulic_structures import Structures,normalize,snap_path,weir_flux
from flood_solver import FloodSolver
import map_sessions


def levee(**kw):
    return dict(id='levee',kind='permanent_levee',points_m=[[6,0],[6,8]],crest_m=1,**kw)


def solver(crest=1.,depth=2.,backend='numpy',axis=1,kind='permanent_levee',**params):
    bed=np.zeros((8,12));h=np.zeros_like(bed);h[:,:6]=depth
    obj=levee();obj.update(kind=kind,crest_m=crest,**params)
    if axis==0:
        bed=bed.T.copy();h=h.T.copy();obj['points_m']=[[0,6],[8,6]]
    with patch.dict(os.environ,HYRA_FLOOD_BACKEND=backend):
        s=FloodSolver(bed,h,1,boundary='closed',manning=0)
    s.structures=Structures([obj],bed,1)
    return s


class StructureTests(unittest.TestCase):
    def test_weir_translation_matches_unmodified_c(self):
        data=json.loads((ROOT/'tests/data/anuga-weir-vectors.json').read_text())
        self.assertGreater(len(data['vectors']),100)
        for v in data['vectors']:
            np.testing.assert_allclose(weir_flux(*v['input']),v['expected'],rtol=2e-13,atol=2e-13)

    def test_low_water_blocks_and_stationary_pressure_is_preserved(self):
        for backend in ('numpy','auto'):
            for axis in (0,1):
                s=solver(2.,1.,backend,axis);initial=s.h.copy();s.advance(2)
                np.testing.assert_allclose(s.h,initial,atol=1e-12)
                np.testing.assert_allclose(s.velocity(),0,atol=1e-12)
                self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)

    def test_overtopping_matches_analytic_volume_and_both_axes(self):
        expected=(2/3)*.5*np.sqrt(2/3*9.81*.5)*8*.001
        for backend in ('numpy','auto'):
            for axis in (0,1):
                s=solver(2,2.5,backend,axis);s.advance(.001)
                downstream=s.h[:,6:] if axis==1 else s.h[6:,:]
                self.assertAlmostEqual(float(downstream.sum()),expected,delta=expected*.002)
                self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)

    def test_longer_overtopping_is_positive_conservative_and_cfl_bounded(self):
        s=solver();s.advance(3)
        self.assertGreater(s.h[:,6:].sum(),0)
        self.assertGreaterEqual(s.h.min(),0)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)
        self.assertLess(s.diagnostics['max_stage_courant'],.45)

    def test_reverse_flow_and_rotational_symmetry(self):
        a=solver();b=solver();b.h=np.flip(a.h,axis=1).copy();a.advance(.5);b.advance(.5)
        np.testing.assert_allclose(a.h,np.flip(b.h,axis=1),atol=1e-12)
        c=solver(axis=0);c.advance(.5);np.testing.assert_allclose(a.h,c.h.T,atol=1e-12)

    def test_numpy_and_accelerated_agree(self):
        a=solver(backend='numpy');b=solver(backend='auto');a.advance(.5);b.advance(.5)
        np.testing.assert_allclose(a.h,b.h,atol=1e-12)
        np.testing.assert_allclose(a.mx,b.mx,atol=1e-12)

    def test_crest_at_ground_is_not_an_artificial_weir(self):
        a=solver(0,1,crest_mode='absolute');b=solver(0,1,crest_mode='absolute')
        b.structures=None;a.advance(.3);b.advance(.3)
        np.testing.assert_array_equal(a.h,b.h)

    def test_submerged_equal_stage_over_uneven_bed_is_at_rest(self):
        zz,xx=np.indices((8,12));bed=.03*xx+.01*zz;h=4-bed
        s=FloodSolver(bed,h,1,boundary='closed');s.structures=Structures([levee()],bed,1);s.advance(2)
        np.testing.assert_allclose(s.h,h,atol=1e-12);np.testing.assert_allclose(s.velocity(),0,atol=1e-11)

    def test_temporary_removal_aligns_to_event_and_opens_flow(self):
        s=solver(2,1,kind='temporary_levee',start_s=0,end_s=.37)
        s.advance(.37);np.testing.assert_array_equal(s.h[:,6:],0)
        s.advance(.4);self.assertGreater(s.h[:,6:].sum(),0)
        self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-10)

    def test_deployment_event_matches_manually_split_reference(self):
        a=solver(2,1,kind='temporary_levee',start_s=.37,end_s=.8)
        b=solver(2,1,kind='temporary_levee',start_s=.37,end_s=.8)
        a.advance(1);b.advance(.37);b.advance(.8);b.advance(1)
        np.testing.assert_allclose(a.h,b.h,atol=1e-12)
        self.assertFalse(a.structures.active(a.structures.barriers[0],.2))
        self.assertTrue(a.structures.active(a.structures.barriers[0],.37))

    def test_warmup_ignores_temporary_barriers(self):
        s=solver(2,1,kind='temporary_levee',start_s=0,end_s=1);s.structure_warmup=True
        s.advance(.1);self.assertGreater(s.h[:,6:].sum(),0)
        s.start_scenario();s.structure_warmup=False
        self.assertTrue(s.structures.faces(0)[1])

    def test_absolute_elevation_and_relative_height_on_raised_terrain(self):
        a=solver();b=solver();b.bed+=150
        obj=levee(crest_mode='absolute');obj['crest_m']=151;b.structures=Structures([obj],b.bed,1)
        a.advance(.2);b.advance(.2);np.testing.assert_allclose(a.h,b.h,atol=1e-12)

    def test_snap_path_is_connected_and_keeps_all_diagonal_segments(self):
        edges=snap_path([[2,1],[6,5]],(8,12),1)
        self.assertEqual(len(edges),8)
        self.assertEqual(set(edges),set(snap_path([[6,5],[2,1]],(8,12),1)))

    def test_building_geometry_and_metres_survive_grid_changes(self):
        o=dict(id='b',kind='building',x_m=4,z_m=4,width_m=8,length_m=8,height_m=6)
        for dx in (1,2,4):
            s=Structures([o],np.zeros((32//dx,32//dx)),dx)
            self.assertEqual(s.buildings.sum()*dx**2,64)
            self.assertEqual(s.items[0]['width_m'],8)
            self.assertEqual(s.compiled[0]['roof_elevation_m'],6)

    def test_invalid_geometry_and_timing_are_rejected(self):
        for o in [levee(crest_mode='bad'),dict(levee(),points_m=[[0,0],[0,4]]),
                  dict(levee(),kind='temporary_levee',start_s=2,end_s=1),
                  dict(levee(),points_m=[[2,2],[2.1,2.1]]),dict(levee(),qfactor=3)]:
            with self.assertRaises(ValueError):Structures([o],np.zeros((8,12)),1)
        with self.assertRaises(ValueError):normalize([levee(),levee()],(8,12),1)

    def test_scene_export_roundtrip_and_building_removal(self):
        o=dict(id='b',kind='building',x_m=40,z_m=10,width_m=10,length_m=10,height_m=6)
        config=dict(mode='demo',flood=dict(baseflow=False,warmup_s=0,rain_mm_h=0),structures=[o,
            dict(id='l',kind='permanent_levee',points_m=[[100,0],[100,160]],crest_m=2)])
        w=map_sessions.create(ROOT,{},config)
        self.assertEqual(w.base.solid[2:4,8:10].sum(),4)
        snap=w.snapshot();self.assertEqual(len(snap['M_structures']['objects']),2)
        with zipfile.ZipFile(io.BytesIO(w.export_zip())) as archive:
            model=json.loads(archive.read('map_model.json'));self.assertEqual(len(model['M_structures']['objects']),2)
        w2=map_sessions.create(ROOT,{},w.scenario_config)
        np.testing.assert_array_equal(w.base.solid,w2.base.solid)
        config['structures']=[];w3=map_sessions.create(ROOT,{},config)
        self.assertEqual(w3.base.solid[2:4,8:10].sum(),0)

    def test_static_solid_snapshot_keeps_legacy_overlap_separate(self):
        o=dict(id='b',kind='building',x_m=60,z_m=45,width_m=15,length_m=15,height_m=6)
        w=map_sessions.create(ROOT,{},dict(mode='demo',structures=[o]))
        b=w.snapshot()['M_base']
        self.assertTrue(b['solid'][9][12])
        self.assertFalse(b['static_solid'][9][12])
        self.assertTrue(b['solid'][10][13])
        self.assertTrue(b['static_solid'][10][13])


if __name__=='__main__':unittest.main()
