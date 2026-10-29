"""Differentiable closed-boundary SWE step matching the existing solver.

Same MUSCL/hydrostatic reconstruction, Rusanov flux, donor limiter, RK2,
water-source integration, dry-sink completion and split Manning drag. Supports
static walls/solids and volumetric sources; rejects unsupported physics rather
than treating it as this operator. This is an auxiliary constraint, not a source
of imagined policy experience.
"""
from dataclasses import dataclass
import math

from .runtime import enable_existing_packages
enable_existing_packages()
import torch
from torch.nn import functional as F

G = 9.81
DRY = 1e-8


def divide(numerator, denominator, threshold=0.):
    return torch.where(denominator > threshold, numerator/denominator.clamp_min(1e-30), 0.)


def padded(value, axis, constant=None):
    pads = (1, 1, 0, 0) if axis == -1 else (0, 0, 1, 1)
    source = value[:, None]
    if constant is None:
        return F.pad(source, pads, mode='replicate')[:, 0]
    return F.pad(source, pads, mode='constant', value=constant)[:, 0]


def neighbours(value, axis):
    return (value[..., :-1], value[..., 1:]) if axis == -1 else (value[:, :-1, :], value[:, 1:, :])


def faces(value, slope, axis):
    left, _ = neighbours(padded(value+.5*slope, axis), axis)
    _, right = neighbours(padded(value-.5*slope, axis), axis)
    return left, right


def minmod(backward, forward, theta=1.5):
    centre = .5*(backward+forward)
    magnitude = torch.minimum(torch.minimum(theta*backward.abs(), theta*forward.abs()), centre.abs())
    return torch.where(backward*forward > 0, centre.sign()*magnitude, 0.)


@dataclass(frozen=True)
class SWEOperator:
    dx: float = 5.
    velocity_depth: float = 1e-6
    limiter_theta: float = 1.5
    boundary: str = 'closed'
    flux_method: str = 'rusanov'
    spatial_order: int = 2
    time_integrator: str = 'rk2'

    def __post_init__(self):
        if (self.boundary != 'closed' or self.flux_method != 'rusanov' or
            self.spatial_order != 2 or self.time_integrator != 'rk2'):
            raise ValueError('This validated operator supports closed/Rusanov/MUSCL/RK2 only')
        if (not all(math.isfinite(v) for v in (self.dx, self.velocity_depth, self.limiter_theta)) or
            self.dx <= 0 or not DRY <= self.velocity_depth or not 1 <= self.limiter_theta <= 2):
            raise ValueError('Invalid numerical constants')

    def _masks(self, h, solid, wall_x, wall_y):
        if solid is None:
            solid = torch.zeros_like(h, dtype=torch.bool)
        else:
            solid = torch.broadcast_to(solid.bool(), h.shape)
        result = {}
        for axis, wall in ((-1, wall_x), (-2, wall_y)):
            a, b = neighbours(padded(solid.to(h.dtype), axis), axis)
            blocked = (a > 0) | (b > 0)
            if wall is not None:
                blocked = blocked | wall.bool()
            result[axis] = blocked
        return solid, result

    def _slope(self, value, axis, blocked, differences=None):
        if differences is None:
            differences = value-torch.roll(value, 1, axis), torch.roll(value, -1, axis)-value
        slope = minmod(*differences, self.limiter_theta)
        a, b = neighbours(blocked, axis)
        allowed = ~(a | b)
        edge = torch.ones_like(allowed)
        if axis == -1:
            edge[..., 0] = False
            edge[..., -1] = False
        else:
            edge[:, 0, :] = False
            edge[:, -1, :] = False
        return torch.where(allowed & edge, slope, 0.)

    def _reconstruction(self, h, mx, mz, bed, blocked):
        u, v = divide(mx, h, self.velocity_depth), divide(mz, h, self.velocity_depth)
        result = {}
        for axis in (-1, -2):
            hm, hp = torch.roll(h, 1, axis), torch.roll(h, -1, axis)
            backward, forward = h-hm, hp-h
            sh = self._slope(h, axis, blocked[axis], (backward, forward))
            sh = sh*torch.minimum(torch.ones_like(h), 2*h.clamp_min(0)/sh.abs().clamp_min(1e-30))
            stage = (backward+(bed-torch.roll(bed, 1, axis)),
                     forward+(torch.roll(bed, -1, axis)-bed))
            seta = self._slope(h, axis, blocked[axis], stage)
            su, sv = self._slope(u, axis, blocked[axis]), self._slope(v, axis, blocked[axis])
            dry = (h <= self.velocity_depth) | (hm <= self.velocity_depth) | (hp <= self.velocity_depth)
            sh, seta, su, sv = (torch.where(dry, 0., x) for x in (sh, seta, su, sv))
            result[axis] = sh, seta-sh, su, sv
        return u, v, result

    def _flux(self, h, u, v, bed, axis, reconstruction, blocked):
        sh, sb, su, sv = reconstruction
        zero = torch.zeros_like(h)
        hl, hr = faces(h, sh, axis)
        bl, br = faces(bed, zero, axis)
        ol, or_ = faces(zero, sb, axis)
        ul, ur = faces(u, su, axis)
        vl, vr = faces(v, sv, axis)
        first, last = torch.zeros_like(hl, dtype=torch.bool), torch.zeros_like(hr, dtype=torch.bool)
        if axis == -1:
            first[..., 0], last[..., -1] = True, True
            ul, ur = torch.where(first, -ul, ul), torch.where(last, -ur, ur)
        else:
            first[:, 0, :], last[:, -1, :] = True, True
            vl, vr = torch.where(first, -vl, vl), torch.where(last, -vr, vr)
        jump = (br-bl)+(or_-ol)
        dl, dr = (hl-jump.clamp_min(0)).clamp_min(0), (hr-(-jump).clamp_min(0)).clamp_min(0)
        un_l, un_r = (ul, ur) if axis == -1 else (vl, vr)
        # The dry branch must also have finite derivatives: sqrt(0) otherwise
        # creates inf*0 in reverse mode even when the forward flux is zero.
        cl = torch.where(dl > 0, (G*dl).clamp_min(1e-30).sqrt(), 0.)
        cr = torch.where(dr > 0, (G*dr).clamp_min(1e-30).sqrt(), 0.)
        speed = torch.maximum(un_l.abs()+cl, un_r.abs()+cr)
        mass = .5*(dl*un_l+dr*un_r)-.5*speed*(dr-dl)
        advx = .5*(dl*un_l*ul+dr*un_r*ur)-.5*speed*(dr*ur-dl*ul)
        advz = .5*(dl*un_l*vl+dr*un_r*vr)-.5*speed*(dr*vr-dl*vl)
        pressure = .25*G*(dl.square()+dr.square())
        corr_l = torch.where(blocked, .5*G*hl.square(), .5*G*(hl-dl)*(hl+dl))
        corr_r = torch.where(blocked, .5*G*hr.square(), .5*G*(hr-dr)*(hr+dr))
        mass, advx, advz, pressure = (torch.where(blocked, 0., x) for x in (mass, advx, advz, pressure))
        return dict(mass=mass, advx=advx, advz=advz, pressure=pressure, corr_l=corr_l, corr_r=corr_r)

    def _transport(self, state, bed, dt, blocked):
        h, mx, mz = state.unbind(1)
        u, v, recon = self._reconstruction(h, mx, mz, bed, blocked)
        fx = self._flux(h, u, v, bed, -1, recon[-1], blocked[-1])
        fz = self._flux(h, u, v, bed, -2, recon[-2], blocked[-2])
        outgoing = (fx['mass'][..., 1:].clamp_min(0)+(-fx['mass'][..., :-1]).clamp_min(0)
                    +fz['mass'][:, 1:].clamp_min(0)+(-fz['mass'][:, :-1]).clamp_min(0))
        available, requested = h*self.dx, dt*outgoing
        theta = torch.where(requested > available, available/requested.clamp_min(1e-30), 1.)
        for axis, field in ((-1, fx), (-2, fz)):
            left, right = neighbours(padded(theta, axis, constant=1.), axis)
            factor = torch.where(field['mass'] >= 0, left, right)
            for name in ('mass', 'advx', 'advz'):
                field[name] = field[name]*factor
        factor = dt/self.dx
        hn = h-factor*(fx['mass'][..., 1:]-fx['mass'][..., :-1]+fz['mass'][:, 1:]-fz['mass'][:, :-1])
        bx, bz = fx['advx']+fx['pressure'], fz['advx']
        cx, cz = fx['advz'], fz['advz']+fz['pressure']
        mxn = mx-factor*(bx[..., 1:]+fx['corr_l'][..., 1:]-bx[..., :-1]-fx['corr_r'][..., :-1]
                         +bz[:, 1:]-bz[:, :-1])-factor*G*h*recon[-1][1]
        mzn = mz-factor*(cx[..., 1:]-cx[..., :-1]+cz[:, 1:]+fz['corr_l'][:, 1:]
                         -cz[:, :-1]-fz['corr_r'][:, :-1])-factor*G*h*recon[-2][1]
        return torch.stack((hn.clamp_min(0), mxn, mzn), dim=1)

    def _sources(self, state, dt, source, infiltration, drains):
        h, mx, mz = state.unbind(1)
        h = h+dt*source
        removals = []
        for amount in (dt*infiltration, dt*drains/self.dx**2):
            removed = torch.minimum(h, amount)
            retained = divide(h-removed, h)
            mx, mz, h = mx*retained, mz*retained, h-removed
            removals.append(removed)
        return torch.stack((h, mx, mz), dim=1), removals

    def _drag(self, state, dt, manning):
        h, mx, mz = state.unbind(1)
        magnitude = torch.where(mx.square()+mz.square() > 0,
                                (mx.square()+mz.square()).clamp_min(1e-30).sqrt(), 0.)
        drag = 1+dt*G*manning.square()*magnitude/h.clamp_min(DRY).pow(7/3)
        return torch.stack((h, mx/drag, mz/drag), dim=1)

    def step(self, state, bed, dt, *, source=None, infiltration=0., drains=None, manning=None,
             solid=None, wall_x=None, wall_y=None):
        if state.ndim != 4 or state.shape[1] != 3 or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Expected [batch,h/mx/mz,z,x] and a positive fixed step')
        h = state[:, 0]
        bed = torch.broadcast_to(bed, h.shape)
        # Preserve the raw static bed in float64 (or remove its datum before
        # casting). Once 10km elevations are rounded to float32, millimetre
        # terrain differences cannot be recovered by the flux operator.
        if bed.dtype == torch.float32 and bool((bed.abs() > 1000).any()):
            raise ValueError('High absolute bed datum needs float64 input or prior float64 centering')
        bed = (bed-bed[:, :1, :1]).to(h.dtype)
        solid, blocked = self._masks(h, solid, wall_x, wall_y)
        source = torch.zeros_like(h) if source is None else torch.broadcast_to(source, h.shape)
        drains = torch.zeros_like(h) if drains is None else torch.broadcast_to(drains, h.shape)
        manning = torch.full_like(h, .035) if manning is None else torch.broadcast_to(manning, h.shape)
        infiltration = torch.as_tensor(infiltration, dtype=h.dtype, device=h.device)
        initial = self._drag(state, .5*dt, manning)
        first, removed_first = self._sources(self._transport(initial, bed, dt, blocked), dt, source, infiltration, drains)
        second, removed_second = self._sources(self._transport(first, bed, dt, blocked), dt, source, infiltration, drains)
        state = self._drag(.5*(initial+second), .5*dt, manning)
        h, mx, mz = state.unbind(1)
        for requested, a, b in zip((dt*infiltration, dt*drains/self.dx**2), removed_first, removed_second):
            extra = torch.minimum(h, (requested-.5*(a+b)).clamp_min(0))
            retained = divide(h-extra, h)
            h, mx, mz = h-extra, mx*retained, mz*retained
        wet = (h > self.velocity_depth) & ~solid
        return torch.stack((torch.where(solid, 0., h), torch.where(wet, mx, 0.), torch.where(wet, mz, 0.)), dim=1)
