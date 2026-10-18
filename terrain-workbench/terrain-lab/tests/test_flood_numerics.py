"""Numerical invariants, hydrostatic balance, flux integration and accuracy."""
import sys
import unittest
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flood_numerics import face_flux, rotate_momentum, G
from flood_solver import FloodSolver
import flood_run


class FaceAlgebraTests(unittest.TestCase):
    def test_knp_matches_original_anuga_c_vectors(self):
        reference=json.loads((ROOT/'tests/data/anuga-knp-vectors.json').read_text(encoding='utf-8'))
        for vector in reference['vectors']:
            hl,hr,hle,hre,ul,vl,ur,vr,nx,nz=vector['input']
            actual=face_flux(*[np.array([v]) for v in (hl,hr,ul,vl,ur,vr)],
                normal=(nx,nz),method='knp',low_froude=vector['low_froude'])
            np.testing.assert_allclose([float(a[0]) for a in actual],vector['expected'],
                                       rtol=2e-13,atol=1e-13)

    def test_uniform_state_physical_flux_at_oblique_normals(self):
        for method in ('rusanov', 'knp'):
            for normal in ((1., 0.), (0., -1.), (.6, .8)):
                h, u, v = np.array([.3]), np.array([2/.3]), np.array([3/.3])
                mass, fx, fz, pressure, _ = face_flux(h, h, u, v, u, v,
                    normal=normal, method=method, low_froude=2)
                q = h*(u*normal[0]+v*normal[1])
                np.testing.assert_allclose(mass, q, atol=1e-14)
                np.testing.assert_allclose(fx, q*u, atol=1e-13)
                np.testing.assert_allclose(fz, q*v, atol=1e-13)
                np.testing.assert_allclose(pressure, .5*G*h*h, atol=1e-14)

    def test_reversing_face_orientation_reverses_transport(self):
        values = [np.array(v) for v in ([.2, 1., 0.], [.7, .3, 0.],
                  [1., -.5, 0.], [-.3, .8, 0.], [.4, -.1, 0.], [0., .2, 0.])]
        hl, hr, ul, vl, ur, vr = values
        for method in ('rusanov', 'knp'):
            a = face_flux(hl, hr, ul, vl, ur, vr, normal=(.6, .8), method=method)
            b = face_flux(hr, hl, ur, vr, ul, vl, normal=(-.6, -.8), method=method)
            for x, y in zip(a[:3], b[:3]):
                np.testing.assert_allclose(x, -y, atol=1e-14)
            np.testing.assert_allclose(a[3], b[3], atol=1e-14)
            np.testing.assert_allclose(a[4], b[4], atol=1e-14)

    def test_rotation_inverse_and_low_froude_leave_mass_unchanged(self):
        mx, mz = np.array([.2, -.8]), np.array([.5, .3])
        mn, mt = rotate_momentum(mx, mz, .6, .8)
        rx, rz = rotate_momentum(mn, mt, .6, -.8)
        np.testing.assert_allclose(rx, mx, atol=1e-15)
        np.testing.assert_allclose(rz, mz, atol=1e-15)
        h = np.ones(2)
        for method in ('rusanov', 'knp'):
            a = face_flux(h, h+.1, mx*.01, mz*.01, mx*.02, mz*.02, method=method)
            b = face_flux(h, h+.1, mx*.01, mz*.01, mx*.02, mz*.02, method=method, low_froude=2)
            np.testing.assert_array_equal(a[0], b[0])
            self.assertGreater(np.max(np.abs(a[1]-b[1])), 0.)


class TransportTests(unittest.TestCase):
    def test_lake_at_rest_across_dry_shore_walls_and_elevation_offsets(self):
        z, x = np.indices((9, 14))
        raw = .08*x+.03*z+.09*np.sin(x)
        wall = np.zeros((9, 15), bool); wall[2:7, 6] = True
        for method in ('rusanov', 'knp'):
            for offset in (0., 10000.):
                bed = offset+raw; depth = np.maximum(offset+.9-bed, 0.)
                s = FloodSolver(bed, depth, 2., boundary='closed', flux_method=method)
                s.set_environment(wall_x=wall)
                s.advance(5.)
                np.testing.assert_allclose(s.h, depth, atol=1e-12)
                np.testing.assert_allclose(s.velocity(), 0., atol=1e-11)
                self.assertLess(abs(s.summary()['balance']['residual_m3']), 1e-10)

    def test_exported_rk_flux_exactly_reconstructs_mass_update(self):
        z, x = np.indices((8, 12)); h = 1.+.05*np.sin(x*.4)
        for method in ('rusanov', 'knp'):
            s = FloodSolver(np.zeros_like(h), h, 2., boundary='closed', manning=0,
                            flux_method=method)
            s.mx[:] = .1*np.sin(x*.3); s.mz[:] = .04*np.cos(z*.4)
            before = s.h.copy(); dt = .02
            s.step(dt)
            predicted = before-dt/s.dx*(np.diff(s.qx, axis=1)+np.diff(s.qy, axis=0))
            np.testing.assert_allclose(s.h, predicted, atol=3e-16)
            self.assertLess(abs(s.summary()['balance']['residual_m3']), 1e-10)

    def test_draining_limiter_preserves_nonnegative_mass_without_clamp_water(self):
        h = np.ones((5, 9))*.01
        for method in ('rusanov', 'knp'):
            s = FloodSolver(np.zeros_like(h), h, 1., boundary='closed', manning=0,
                            flux_method=method)
            s.mx[:, 4] = .5
            s.step(.1)  # Deliberately exceeds CFL to exercise draining, not stability.
            self.assertGreaterEqual(s.h.min(), 0.)
            self.assertGreater(s.diagnostics['limited_faces'], 0)
            self.assertLess(s.balance['numerical_adjustment_m3'], 1e-12)
            self.assertLess(abs(s.summary()['balance']['residual_m3']), 1e-12)

    def test_small_timestep_error_identifies_cell_and_speed(self):
        s = FloodSolver(np.zeros((5, 7)), np.ones((5, 7)), 1., manning=0)
        s.mx[2, 3] = 1e12
        with self.assertRaisesRegex(RuntimeError, 'limiting_cell'):
            s.advance(1.)
        self.assertEqual(s.last_timestep['limiting_cell'], [2, 3])
        self.assertEqual(s.last_timestep['cause'], 'cfl_grid')

    def test_config_validated_and_applied_to_current_terrain(self):
        import server
        terrain = server.describe('example/raw')
        config = flood_run.settings(dict(flux_method='knp', low_froude=0, spatial_order=1,
            time_integrator='euler', resolution=32, baseflow=False), terrain['shape'])
        s, metadata = flood_run.prepare(ROOT, terrain, config)
        self.assertEqual(s.numerical_config()['flux_method'], 'knp')
        self.assertEqual(metadata['numerics']['spatial_order'], 1)
        self.assertEqual(metadata['numerics']['low_froude'], 0)
        for change in (dict(flux_method='bad'), dict(spatial_order=True), dict(cfl=.9),
                       dict(low_froude=1.5), dict(time_integrator='bad')):
            with self.assertRaises(ValueError):
                flood_run.settings(change, terrain['shape'])


def shear_error(n, *, order=2, integrator='rk2', low_froude=2, method='rusanov'):
    dx = 100./n; x = (np.arange(n)+.5)*dx
    h = np.ones((max(2, n//16), n))
    s = FloodSolver(np.zeros_like(h), h, dx, boundary='periodic', manning=0,
        spatial_order=order, time_integrator=integrator, low_froude=low_froude,
        flux_method=method)
    speed, amplitude, end = .03, .05, 5.
    average = np.sinc(dx/100.)
    s.mx[:] = speed
    s.mz[:] = amplitude*np.sin(2*np.pi*x/100.)*average
    s.advance(end)
    exact = amplitude*np.sin(2*np.pi*(x-speed*end)/100.)*average
    return float(np.mean(np.abs(s.mz-exact)))


class AccuracyTests(unittest.TestCase):
    def test_cell_average_references_obey_integral_identities(self):
        sys.path.insert(0, str(ROOT/'benchmarks'))
        from analytic_cases import dam_reference, thacker_reference
        edges = np.linspace(0.,160.,129); dx = edges[1]-edges[0]
        for right in (0., .2):
            initial, _ = dam_reference(edges, 0., right)
            h, m = dam_reference(edges, 6., right)
            self.assertAlmostEqual(float(h.sum()*dx),float(initial.sum()*dx),places=10)
            self.assertAlmostEqual(float(m.sum()*dx),.5*G*(1-right**2)*6,places=10)
        edges = np.linspace(-30.,30.,129); dx = edges[1]-edges[0]
        for t in (0.,3.,7.):
            h, _ = thacker_reference(edges,t)
            self.assertAlmostEqual(float(h.sum()*dx),4*20/3,places=10)

    def test_second_order_converges_and_low_froude_reduces_shear_diffusion(self):
        for method in ('rusanov', 'knp'):
            coarse = shear_error(32, method=method)
            fine = shear_error(64, method=method)
            self.assertGreater(coarse/fine, 2.5)
            uncorrected = shear_error(32, low_froude=0, method=method)
            self.assertLess(coarse, uncorrected)
            first = shear_error(32, order=1, integrator='euler', method=method)
            self.assertLess(coarse, first)


if __name__ == '__main__':
    unittest.main()
