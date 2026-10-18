"""Independent exact finite-volume references for the flux comparison.

Ritter dry-bed and Stoker wet-bed Riemann solutions; translating transverse
shear (an exact nonlinear SWE solution); Thacker's parabolic canal. These
functions supply cell averages, not point samples. No numerical solver is
used to produce a reference.
"""
from __future__ import annotations
import numpy as np

G = 9.81
CASES = ('lake', 'dam_dry', 'dam_wet', 'shear', 'slow_shear', 'thacker')


def dam_reference(edges, t, right_depth=0., left_depth=1., centre=80.):
    a, b = edges[:-1], edges[1:]
    dx = b-a
    if t == 0:
        left_length = np.clip(centre-a, 0., dx)
        return (left_depth*left_length+right_depth*(dx-left_length))/dx, np.zeros_like(a)
    c = np.sqrt(G*left_depth)
    if right_depth == 0:
        fan_right, shock, hs, us = 2*c, 2*c, 0., 0.
    else:
        low, high = right_depth, left_depth
        for _ in range(90):
            mid = .5*(low+high)
            value = (2*(np.sqrt(G*mid)-c)
                     +(mid-right_depth)*np.sqrt(.5*G*(1/mid+1/right_depth)))
            if value > 0:
                high = mid
            else:
                low = mid
        hs = .5*(low+high); us = 2*(c-np.sqrt(G*hs))
        fan_right = us-np.sqrt(G*hs)
        shock = hs*us/(hs-right_depth)
    left_edge, fan_edge, shock_edge = centre-c*t, centre+fan_right*t, centre+shock*t
    left_length = np.clip(np.minimum(b, left_edge)-a, 0., dx)
    h_integral = left_depth*left_length
    momentum_integral = np.zeros_like(a)
    lo, hi = np.maximum(a, left_edge), np.minimum(b, fan_edge)
    active = hi > lo
    x0, x1 = (lo[active]-centre)/t, (hi[active]-centre)/t
    h_integral[active] += t*((2*c-x0)**3-(2*c-x1)**3)/(27*G)
    def integral_m(x):
        return 2/(27*G)*(4*c**3*x-c*x**3+x**4/4)
    momentum_integral[active] += t*(integral_m(x1)-integral_m(x0))
    length_star = np.clip(np.minimum(b, shock_edge)-np.maximum(a, fan_edge), 0., dx)
    h_integral += hs*length_star
    momentum_integral += hs*us*length_star
    h_integral += right_depth*np.clip(b-np.maximum(a, shock_edge), 0., dx)
    return h_integral/dx, momentum_integral/dx


def thacker_reference(edges, t, d0=1., radius=20., amplitude=2.):
    omega = np.sqrt(2*G*d0)/radius
    centre = amplitude*np.cos(omega*t)
    speed = -amplitude*omega*np.sin(omega*t)
    lo = np.maximum(edges[:-1], centre-radius)
    hi = np.minimum(edges[1:], centre+radius)
    active = hi > lo
    h = np.zeros_like(lo)
    h[active] = d0*((hi[active]-lo[active])
        - ((hi[active]-centre)**3-(lo[active]-centre)**3)/(3*radius**2))/np.diff(edges)[active]
    return h, speed*h


def make_case(name, n):
    if name not in CASES or n < 32 or n % 16:
        raise ValueError('Use a known case and a resolution >=32 divisible by 16')
    length = 60. if name == 'thacker' else (100. if 'shear' in name else 160.)
    origin = -30. if name == 'thacker' else 0.
    edges = origin+np.arange(n+1)*length/n
    x = .5*(edges[:-1]+edges[1:]); dx = length/n
    shape = (n//16, n)  # Fixed physical width length/16; cells stay square.
    bed = np.zeros(shape)
    if name == 'lake':
        z = (np.arange(shape[0])+.5)*dx
        bed[:] = .02*x+.15*np.sin(2*np.pi*x/length)+.02*z[:, None]
        h = np.maximum(1.5-bed, 0.)
        zero = np.zeros(shape)
        return dict(name=name, bed=bed, h=h, mx=zero.copy(), mz=zero.copy(),
                    exact_h=h.copy(), exact_mx=zero.copy(), exact_mz=zero.copy(),
                    dx=dx, end=5., boundary='closed', x=x)
    if name.startswith('dam_'):
        right = .2 if name == 'dam_wet' else 0.
        h, mx = dam_reference(edges, 0., right)
        eh, emx = dam_reference(edges, 6., right)
        mz = np.zeros(n); emz = mz.copy(); end = 6.; boundary = 'closed'
    elif 'shear' in name:
        speed = .03 if name == 'slow_shear' else 1.
        h, mx = np.ones(n), np.full(n, speed)
        mz = .05*np.sin(2*np.pi*x/length)*np.sinc(dx/length)
        end = 10.; boundary = 'periodic'
        eh, emx = h.copy(), mx.copy()
        emz = .05*np.sin(2*np.pi*(x-speed*end)/length)*np.sinc(dx/length)
    else:
        omega = np.sqrt(2*G)/20.
        end = np.pi/(2*omega); boundary = 'closed'
        # Exact area-average bed, matching the reference cell-average depth.
        bed[:] = (x*x+dx*dx/12.)/20.**2
        h, mx = thacker_reference(edges, 0.)
        eh, emx = thacker_reference(edges, end)
        mz = np.zeros(n); emz = mz.copy()
    tile = lambda a: np.broadcast_to(a, shape).copy()
    return dict(name=name, bed=bed, h=tile(h), mx=tile(mx), mz=tile(mz),
                exact_h=tile(eh), exact_mx=tile(emx), exact_mz=tile(emz),
                dx=dx, end=end, boundary=boundary, x=x)
