"""Explicit containing-cell gauge sampling; never interpolate through barriers."""
import math
import numpy as np


def sample_gauge(solver, x_m, z_m, *, origin_center_m):
    """Coordinates in metres; origin is FIRST PARENT CELL CENTRE (x,z).

    Cells use half-open extents [low, high); an internal face belongs to
    the cell on its positive side. Returns missing velocities for dry/solid
    points, not a fabricated zero-speed observation. No spatial interpolation.
    """
    coordinates=(x_m,z_m,*origin_center_m)
    if len(origin_center_m)!=2 or not all(isinstance(v,(int,float,np.number)) and not isinstance(v,(bool,np.bool_)) and math.isfinite(v) for v in coordinates):
        raise ValueError('Gauge coordinates and origin must be finite metre values')
    dx=solver.dx;ox,oz=origin_center_m
    fx=(x_m-(ox-dx/2))/dx;fz=(z_m-(oz-dx/2))/dx
    result=dict(x_m=float(x_m),z_m=float(z_m),time_s=float(solver.time),method='containing_cell',
                origin_center_m=[float(ox),float(oz)],status='outside',depth_m=None,stage_m=None,
                u_m_s=None,v_m_s=None,speed_m_s=None)
    nz,nx=solver.h.shape
    if not (0<=fx<nx and 0<=fz<nz):return result
    x,z=math.floor(fx),math.floor(fz)
    result.update(parent_x=x,parent_z=z,cell_width_m=dx)
    if hasattr(solver,'leaf'):
        parent=z*nx+x;mesh=solver.mesh
        child=(int(fx-x>=.5)+2*int(fz-z>=.5)) if mesh.mask[z,x] else -1
        index=int(mesh.start[parent])+(child if child>=0 else 0)
        h,bed,mx,mz=[float(solver.leaf[k][index]) for k in ('h','bed','mx','mz')]
        solid=bool(solver.leaf_solid[index]);result.update(leaf_index=index,child=child,cell_width_m=float(mesh.width[index]))
    else:
        h,bed,mx,mz=[float(getattr(solver,k)[z,x]) for k in ('h','bed','mx','mz')]
        solid=bool(solver.solid[z,x])
    if solid:
        result['status']='solid';return result
    result.update(depth_m=h,stage_m=bed+h,status='dry' if h<=solver.velocity_depth else 'wet')
    if result['status']=='wet':
        result.update(u_m_s=mx/h,v_m_s=mz/h,speed_m_s=math.hypot(mx/h,mz/h))
    return result
