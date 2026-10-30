"""Opt-in HLLC: exact contacts, balance, dry fronts and compiled parity."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from flood_solver import FloodSolver
from flood_numerics import face_flux, G
import flood_run

def flux(vals, **kw):
    return face_flux(*[np.asarray(v,dtype=float) for v in vals],method='hllc',**kw)

class HllcTests(unittest.TestCase):
    def test_uniform_state_and_supersonic_upwind(self):
        for normal in ((1.,0.),(0.,-1.),(.6,.8)):
            h,u,v=.3,2.,-3.
            a=flux((h,h,u,v,u,v),normal=normal)
            q=h*(u*normal[0]+v*normal[1])
            np.testing.assert_allclose(a[:4],[q,q*u,q*v,.5*G*h*h],atol=1e-13)
        a=flux((1.,.9,10.,.3,11.,-.2))
        np.testing.assert_allclose(a[:4],[10.,100.,3.,.5*G],atol=1e-13)

    def test_stationary_and_moving_shear_contact(self):
        for u in (0.,.4,-.4):
            a=flux((1.,1.,u,1.,u,-1.))
            expected=u*(1. if u>=0 else -1.)
            self.assertAlmostEqual(float(a[2]),expected,places=14)
            old=face_flux(*map(np.asarray,(1.,1.,u,1.,u,-1.)),method='rusanov')
            self.assertGreater(abs(float(old[2])-expected),1.)

    def test_orientation_reversal_and_rotation(self):
        vals=[np.array(v) for v in ([.2,1.,0.,.5],[.7,.3,.5,0.],
              [1.,-.5,9.,.2],[-.3,.8,8.,-.1],[.4,-.1,.3,7.],[0.,.2,-.2,8.])]
        hl,hr,ul,vl,ur,vr=vals
        a=flux(vals,normal=(.6,.8));b=flux((hr,hl,ur,vr,ul,vl),normal=(-.6,-.8))
        for k in range(3): np.testing.assert_allclose(a[k],-b[k],atol=1e-14)
        for k in (3,4): np.testing.assert_allclose(a[k],b[k],atol=1e-14)
        # 90 degree rotation of all vectors and face normal.
        c=flux(vals);d=flux((hl,hr,-vl,ul,-vr,ur),normal=(0.,1.))
        np.testing.assert_allclose(c[0],d[0],atol=1e-14)
        np.testing.assert_allclose(c[1],d[2],atol=1e-14)
        np.testing.assert_allclose(c[2],-d[1],atol=1e-14)

    def test_dry_and_vacuum_fallback_finite_and_counted(self):
        audit=np.zeros((4,5),dtype=np.int64)
        vals=([0.,1.,0.,1e-9,1.],[0.,0.,1.,1e-9,1.],
              [99.,.2,99.,.1,-10.],[99.,.4,99.,.2,.4],
              [99.,99.,.2,.1,10.],[99.,99.,.4,.2,-.4])
        a=flux(vals,audit=audit)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal([x[0] for x in a],0.)
        np.testing.assert_array_equal(audit.sum(axis=0),1)
        np.testing.assert_array_equal(audit[1],[1,1,1,1,0])
        self.assertEqual(audit[3,4],1)

    def test_entry_config_and_incompatible_mode(self):
        config=flood_run.settings(dict(flux_method='hllc'),(8,12))
        self.assertEqual(config['flux_method'],'hllc')
        self.assertEqual(flood_run.DEFAULTS['flux_method'],'rusanov')
        with self.assertRaises(ValueError):flood_run.settings(dict(flux_method='hllc',low_froude=1),(8,12))
        with self.assertRaises(ValueError):FloodSolver(np.zeros((3,4)),np.ones((3,4)),1.,flux_method='hllc',low_froude=1)

    def test_numba_numpy_faces_steps_and_reason_maps(self):
        import test_flood_acceleration as reference
        if reference.flood_acceleration is None:self.skipTest('optional numba absent')
        helper=reference.AccelerationTests()
        for boundary in ('closed','open','ocean','periodic'):
            for order in (1,2):
                with self.subTest(boundary=boundary,order=order):
                    a,b=helper.pair(boundary,'hllc',order,0)
                    for axis in (0,1):
                        fa,fb=a._flux(axis,details=True),b._flux(axis,details=True)
                        for key in fa:np.testing.assert_allclose(fa[key],fb[key],rtol=1e-11,atol=1e-12)
                    for _ in range(4):
                        dt=min(a.timestep(.01),b.timestep(.01))
                        for s in (a,b):s.step(dt,rain=.001,infiltration=.0001,inflow=.01,source=(2,2))
                    for key in ('h','mx','mz','qx','qy','maximum'):
                        np.testing.assert_allclose(getattr(a,key),getattr(b,key),rtol=1e-10,atol=1e-12)
                    for axis in (0,1):np.testing.assert_array_equal(a.hllc_face_audit[axis],b.hllc_face_audit[axis])
                    self.assertGreater(a.hllc_audit_summary()['fallback'],0)
                    self.assertGreater(a.hllc_audit_summary()['counts']['contact'],0)

    def test_static_lake_dry_shore_walls_and_offset(self):
        z,x=np.indices((9,14));raw=.08*x+.03*z+.09*np.sin(x)
        wall=np.zeros((9,15),bool);wall[2:7,6]=True
        for backend in ('numpy','numba'):
            for offset in (0.,10000.):
                with patch.dict(os.environ,HYRA_FLOOD_BACKEND=backend):
                    bed=raw+offset;h=np.maximum(.9+offset-bed,0.)
                    s=FloodSolver(bed,h,2.,boundary='closed',flux_method='hllc')
                    s.set_environment(wall_x=wall);s.advance(2.)
                    np.testing.assert_allclose(s.h,h,atol=1e-12)
                    np.testing.assert_allclose(s.velocity(),0.,atol=1e-11)
                    self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)

    def test_rk_flux_reconstructs_mass_and_dry_dam_stable(self):
        z,x=np.indices((8,32));h=np.where(x<12,1.,0.)
        for backend in ('numpy','numba'):
            with patch.dict(os.environ,HYRA_FLOOD_BACKEND=backend):
                s=FloodSolver(np.zeros_like(h),h,1.,boundary='closed',manning=0.,flux_method='hllc')
                for _ in range(30):
                    before=s.h.copy();dt=s.timestep(.04);s.step(dt)
                    expected=before-dt/s.dx*(np.diff(s.qx,axis=1)+np.diff(s.qy,axis=0))
                    np.testing.assert_allclose(s.h,expected,atol=1e-13)
                self.assertTrue(np.isfinite([s.h,s.mx,s.mz]).all())
                self.assertGreaterEqual(s.h.min(),0.)
                self.assertLess(abs(s.summary()['balance']['residual_m3']),1e-9)
                self.assertLess(abs(s.balance['numerical_adjustment_m3']),1e-10)

    def test_stationary_shear_preserved_in_solver_and_audit_reset(self):
        z,x=np.indices((4,32));h=np.ones_like(x,dtype=float)
        s=FloodSolver(h*0,h,1.,boundary='periodic',manning=0.,flux_method='hllc')
        s.mz[:]=np.where(x<16,1.,-1.);before=s.mz.copy();s.advance(.5)
        np.testing.assert_array_equal(s.mz,before)
        self.assertGreater(s.hllc_audit_summary()['total'],0)
        s.start_scenario();self.assertEqual(s.hllc_audit_summary()['total'],0)

if __name__=='__main__':unittest.main()
