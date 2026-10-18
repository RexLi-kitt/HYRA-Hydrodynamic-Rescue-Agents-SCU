"""Grid-independent shallow-water face algebra and Cartesian MUSCL limiters.

KNP and low-Froude formulas follow ANUGA's gpu_device_helpers.h at
e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b. This is a NumPy formulation,
not ANUGA's triangular-domain solver. Hydrostatic pressure is returned
separately so the caller can couple it to well-balanced bed source terms.
"""
from __future__ import annotations
import numpy as np

G = 9.81


def rotate_momentum(mx, mz, nx, nz):
    """Components along a unit outward normal and its tangent."""
    return nx*mx + nz*mz, -nz*mx + nx*mz


def minmod_slope(backward, forward, theta=1.5):
    """Generalised minmod; theta in [1,2] preserves neighbour bounds."""
    centre = .5*(backward+forward)
    magnitude = np.minimum(np.minimum(theta*np.abs(backward),
                                     theta*np.abs(forward)), np.abs(centre))
    return np.where(backward*forward > 0, np.sign(centre)*magnitude, 0.)


def low_froude_factor(ul, vl, ur, vr, hl, hr, mode):
    if mode == 0:
        return 1.
    fr2 = (ul*ul+vl*vl+ur*ur+vr*vr)/(G*(hl+hr)+1e-10)
    if mode == 1:
        return np.sqrt(np.clip(fr2, .001, 1.))
    if mode == 2:
        return np.sqrt(np.minimum(1., .01+np.maximum(np.sqrt(fr2)-.01, 0.)))
    raise ValueError('low_froude must be 0, 1 or 2')


def face_flux(hl, hr, ul, vl, ur, vr, *, normal=(1., 0.),
              method='rusanov', low_froude=0):
    """Mass, two advective momentum fluxes, pressure and wave-speed bound.

    hl/hr are non-negative hydrostatically reconstructed depths; velocities
    come from the original (unclipped) face depths. The pressure contribution
    is normal*pressure, and is NOT included in the two momentum outputs.
    Low-Froude correction acts on momentum diffusion only, for either flux.
    """
    nx, nz = normal
    un_l, ut_l = rotate_momentum(ul, vl, nx, nz)
    un_r, ut_r = rotate_momentum(ur, vr, nx, nz)
    mn_l, mn_r = hl*un_l, hr*un_r
    mt_l, mt_r = hl*ut_l, hr*ut_r
    cl, cr = np.sqrt(G*hl), np.sqrt(G*hr)
    correction = low_froude_factor(un_l, ut_l, un_r, ut_r, hl, hr, low_froude)
    if method == 'rusanov':
        speed = np.maximum(np.abs(un_l)+cl, np.abs(un_r)+cr)
        mass = .5*(mn_l+mn_r)-.5*speed*(hr-hl)
        fn = .5*(mn_l*un_l+mn_r*un_r)-.5*speed*correction*(mn_r-mn_l)
        ft = .5*(mn_l*ut_l+mn_r*ut_r)-.5*speed*correction*(mt_r-mt_l)
        pressure = .25*G*(hl*hl+hr*hr)
    elif method == 'knp':
        sp = np.maximum(np.maximum(un_l+cl, un_r+cr), 0.)
        sm = np.minimum(np.minimum(un_l-cl, un_r-cr), 0.)
        denominator = sp-sm
        inverse = np.divide(1., denominator, out=np.zeros_like(denominator),
                            where=denominator > 1e-14)
        mass = (sp*mn_l-sm*mn_r+sp*sm*(hr-hl))*inverse
        fn = (sp*mn_l*un_l-sm*mn_r*un_r+correction*sp*sm*(mn_r-mn_l))*inverse
        ft = (sp*mn_l*ut_l-sm*mn_r*ut_r+correction*sp*sm*(mt_r-mt_l))*inverse
        pressure = .5*G*(sp*hl*hl-sm*hr*hr)*inverse
        pressure = np.where(denominator > 1e-14, pressure, .25*G*(hl*hl+hr*hr))
        speed = np.maximum(sp, -sm)
    else:
        raise ValueError('flux_method must be rusanov or knp')
    # Inverse rotation: the tangent is (-nz,nx).
    return mass, nx*fn-nz*ft, nz*fn+nx*ft, pressure, speed
