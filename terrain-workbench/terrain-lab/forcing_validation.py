"""Validate compiled external forcing before it changes solver state."""
from copy import deepcopy
import math
import numpy as np

def external_entries(solver, entries, solid=None, wall_x=None, wall_y=None):
    if not isinstance(entries, (list, tuple)):
        raise ValueError('External inflows must be a sequence')
    solid = solver.solid if solid is None else solid
    wall_x = solver.wall_x if wall_x is None else wall_x
    wall_y = solver.wall_y if wall_y is None else wall_y
    nz,nx=solid.shape
    clean=deepcopy(entries);ids=set();occupied={}
    def number(v):
        return isinstance(v,(int,float,np.integer,np.floating)) and not isinstance(v,(bool,np.bool_)) and np.isfinite(v)
    def cell(z,x):
        return all(isinstance(v,(int,np.integer)) and not isinstance(v,(bool,np.bool_)) and 0<=v<n
                   for v,n in ((z,nz),(x,nx)))
    for e in clean:
        if not isinstance(e,dict) or not isinstance(e.get('id'),str) or not e['id'] or e['id'] in ids:
            raise ValueError('External inflow IDs must be nonempty and unique')
        ids.add(e['id'])
        if not isinstance(e.get('enabled'),bool):raise ValueError('Invalid external inflow enabled flag')
        mode=e.get('delivery','volume_source_at_inlet_cells')
        if mode not in ('volume_source_at_inlet_cells','boundary_discharge_with_momentum'):
            raise ValueError('Invalid external delivery mode')
        # Reuse the event validator with a harmless dummy cell; geometry follows below.
        dummy=np.zeros_like(solid)
        e['hydrograph']=solver._validated_source_schedule(
            [dict(cells=[(0,0)],hydrograph=e.get('hydrograph'))],dummy)[0]['hydrograph']
        key='boundary_faces' if mode=='boundary_discharge_with_momentum' else 'cells'
        parts=e.get(key)
        if not isinstance(parts,(list,tuple)) or not parts:raise ValueError('External inflow requires '+key)
        weights=[];seen=set()
        if mode=='boundary_discharge_with_momentum':
            v=e.get('inlet_velocity_m_s',0.)
            if not number(v) or v<0:raise ValueError('Invalid inlet velocity')
            e['inlet_velocity_m_s']=float(v)
        for p in parts:
            if not isinstance(p,dict) or not cell(p.get('z'),p.get('x')):raise ValueError('Invalid external inflow cell')
            z,x=p['z'],p['x'];w=p.get('weight')
            if not number(w) or w<=0:raise ValueError('External weights must be positive and finite')
            weights.append(float(w))
            if e['enabled'] and solid[z,x]:raise ValueError('External inflow cannot be inside a solid cell')
            identity=(z,x)
            if mode=='boundary_discharge_with_momentum':
                edge=p.get('edge');offset=p.get('offset');width=p.get('width_m')
                if edge not in ('west','east','north','south') or not isinstance(offset,(int,np.integer)) or isinstance(offset,(bool,np.bool_)):
                    raise ValueError('Invalid boundary face')
                expected={'west':(offset,0),'east':(offset,nx-1),'north':(0,offset),'south':(nz-1,offset)}[edge]
                if expected!=(z,x) or not number(width) or not 0<width<=solver.dx:
                    raise ValueError('Invalid boundary face geometry')
                wall={'west':wall_x[z,0],'east':wall_x[z,nx],'north':wall_y[0,x],'south':wall_y[nz,x]}[edge]
                if e['enabled'] and wall:raise ValueError('External boundary face is blocked by a wall')
                normal={'west':(1.,0.),'east':(-1.,0.),'north':(0.,1.),'south':(0.,-1.)}[edge]
                direction=p.get('momentum_direction_xz',normal)
                if (not isinstance(direction,(list,tuple)) or len(direction)!=2 or not all(number(v) for v in direction)
                        or not math.isclose(math.hypot(*direction),1.,rel_tol=1e-9,abs_tol=1e-9)
                        or sum(a*b for a,b in zip(direction,normal))<=1e-8):
                    raise ValueError('Inlet direction must be a unit vector pointing inward')
                identity=(edge,int(offset))
                if e['enabled']:
                    occupied[identity]=occupied.get(identity,0.)+width
                    if occupied[identity]>solver.dx*(1+1e-10):raise ValueError('Boundary inlet widths exceed the face width')
            if identity in seen:raise ValueError('Duplicate external inflow cells or faces')
            seen.add(identity)
        if not math.isclose(math.fsum(weights),1.,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('External inflow weights must sum to one')
    return clean
