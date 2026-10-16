"""Declared physical-domain variants of the existing ANUGA experiment.

Reuses ANUGA's rectangular mesh and existing physical pulse law. Unlike changing
the mesh, changing length deliberately changes the physical propagation regime.
The total longitudinal bed drop and fixed y-boundary support are held constant.
"""
from __future__ import annotations

import numpy as np

from .lead import XY, pulse
from .numerical_stability import boundary_footprint


def sampling_xy(length_m):
    if not np.isfinite(length_m) or length_m <= 0:
        raise ValueError('Invalid physical domain length')
    return XY.astype(float)*[length_m/100., 1.]


def scaled_domain(params, mesh, length_m, factor=1., initial=None, start=0.,
                  amplitude_m=0., anchor_s=180., center_y_m=20.):
    import anuga
    if length_m <= 0 or amplitude_m < 0 or mesh % 10:
        raise ValueError('Unsupported domain or boundary grid')
    domain = anuga.rectangular_cross_domain(mesh, mesh, len1=float(length_m), len2=100.)
    domain.set_store(False); domain.set_flow_algorithm('DE0')
    scale = length_m/100.
    domain.set_quantity('elevation', lambda x, y: -params['slope']*x/scale+
                        params['ridge']*np.exp(-((y-50)/12)**2))
    domain.set_quantity('friction', params.get('friction', .03))
    domain.set_quantity('stage', expression='elevation + 0.15')
    if initial is not None:
        for name in ('stage', 'xmomentum', 'ymomentum'):
            domain.set_quantity(name, initial[name], location='centroids')
    domain.set_starttime(float(start))
    if amplitude_m:
        for edge, tag in list(domain.boundary.items()):
            if tag == 'left':
                _, y = domain.get_edge_midpoint_coordinate(*edge)
                if center_y_m-10 <= y < center_y_m+10:
                    domain.boundary[edge] = 'pulse'
        tagged = {tag: [] for tag in set(domain.boundary.values())}
        for j, (cell, edge) in enumerate(zip(domain.boundary_cells, domain.boundary_edges)):
            tagged[domain.boundary[int(cell), int(edge)]].append(j)
        domain.tag_boundary_cells = tagged; domain.mesh.tag_boundary_cells = tagged
        np.testing.assert_allclose(boundary_footprint(domain), [center_y_m-10, center_y_m+10], atol=1e-8)
    def base_level(t):
        return .15+params['rate']*factor*max(t-params['onset'], 0.)
    wall = anuga.Reflective_boundary(domain)
    boundaries = dict(left=anuga.Time_boundary(domain, lambda t: [base_level(t), 0., 0.]),
                      right=wall, top=wall, bottom=wall)
    if amplitude_m:
        boundaries['pulse'] = anuga.Time_boundary(domain, lambda t: [
            base_level(t)+pulse(t, anchor_s, amplitude_m, 20), 0., 0.])
    domain.set_boundary(boundaries)
    return domain
