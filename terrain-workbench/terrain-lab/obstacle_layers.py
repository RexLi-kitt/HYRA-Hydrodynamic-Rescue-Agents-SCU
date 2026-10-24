"""Separate ground and explicit obstacle footprints before grid aggregation.

Merewether extraction is a DEM interpretation, not surveyed geometry. Fractional
coverage is audit data only; the solver continues to use whole solid cells.
"""
import numpy as np
from scipy import ndimage


def separate(surface, footprint):
    surface=np.asarray(surface,float);mask=np.asarray(footprint,bool)
    if surface.ndim!=2 or mask.shape!=surface.shape or not np.isfinite(surface).all():
        raise ValueError('Ground and obstacle rasters must be finite matching arrays')
    if mask.all():raise ValueError('Ground estimation needs at least one known ground cell')
    ground=surface.copy()
    if mask.any():
        nearest=ndimage.distance_transform_edt(mask,return_distances=False,return_indices=True)
        ground[mask]=surface[tuple(nearest[:,mask])]
    return dict(ground=ground,obstacle=mask,obstacle_top=surface.copy(),estimated_ground=mask.copy())


def merewether_layers(surface):
    """Identify artificial plateaus/raised features without consulting gauges.

The repeated artificial 30 m plateau is specific to this DEM, not a generic
height threshold. A morphological residual retains narrower sub-plateau piers.
"""
    surface=np.asarray(surface,float)
    seed=np.isclose(surface,30.,rtol=0,atol=1e-6)
    if not seed.any():raise ValueError('Merewether artificial 30 m plateau absent; supply an explicit footprint')
    opened=ndimage.grey_opening(surface,size=(9,9),mode='nearest')
    candidate=seed | (surface-opened>.75)
    guard=ndimage.binary_dilation(seed,iterations=2)
    initial_ground=separate(surface,candidate | guard)['ground']
    mask=candidate | (guard & (surface-initial_ground>.75))
    result=separate(surface,mask)
    result['metadata']=dict(method='merewether-dem-plateau-and-narrow-feature-v1',
        source_kind='inferred_from_combined_dem_not_surveyed_footprints',plateau_m=30.,
        opening_window_native_cells=9,minimum_residual_m=.75,plateau_halo_native_cells=2,
        ground_estimation='nearest known unmasked ground; unchanged outside extracted footprint',
        obstacle_behavior='impermeable and assumed unsubmerged as in experiment setup',
        semantic_classes='obstacle only; walls, piers and buildings not automatically distinguished',
        inferred_ground_cells=int(mask.sum()),ground_under_obstacles_measured=False,
        observations_used_for_extraction=False)
    return result


def aggregate(layers,factor,native_dx=.175):
    """Average known fluid ground, then rasterize majority whole-cell solids.

Every component that would disappear keeps its maximum-overlap coarse cell.
This preserves small features' presence, not their exact width/connectivity.
"""
    ground=layers['ground'];mask=layers['obstacle'];nz,nx=ground.shape
    if isinstance(factor,bool) or not isinstance(factor,int) or factor<1:raise ValueError('Invalid factor')
    top=(-nz)%factor;right=(-nx)%factor
    def padded(a):return np.pad(a,((top,0),(0,right)),mode='edge')
    g=padded(ground);m=padded(mask);shape=(g.shape[0]//factor,factor,g.shape[1]//factor,factor)
    coverage=m.reshape(shape).mean(axis=(1,3));known=(~m).reshape(shape)
    counts=known.sum(axis=(1,3))
    floor=np.divide(np.where(m,0,g).reshape(shape).sum(axis=(1,3)),counts,
        out=g.reshape(shape).mean(axis=(1,3)),where=counts>0)
    solid=coverage>=.5
    labels,n=ndimage.label(mask,structure=np.ones((3,3),bool));rescued=0
    for j,sl in enumerate(ndimage.find_objects(labels),start=1):
        if sl is None:continue
        zz,xx=np.nonzero(labels[sl]==j);zz+=sl[0].start;xx+=sl[1].start
        cz,cx=(zz+top)//factor,xx//factor
        if not solid[cz,cx].any():
            ids,weights=np.unique(cz*solid.shape[1]+cx,return_counts=True)
            solid.ravel()[int(ids[np.argmax(weights)])]=True;rescued+=1
    projected=np.repeat(np.repeat(solid,factor,axis=0),factor,axis=1)[top:top+nz,:nx]
    audit=dict(method='known-fluid-ground-mean + majority solid + minimum component footprint',
        factor=factor,padded_north_cells=top,padded_east_cells=right,
        native_obstacle_cells=int(mask.sum()),native_components=int(n),minimum_component_cells_added=rescued,
        native_obstacle_area_m2=float(mask.sum()*native_dx**2),
        projected_obstacle_area_m2=float(projected.sum()*native_dx**2),
        extra_blocked_native_area_m2=float((projected&~mask).sum()*native_dx**2),
        lost_obstacle_native_area_m2=float((mask&~projected).sum()*native_dx**2),
        partial_coverage_cells=int(((coverage>0)&(coverage<1)).sum()),partial_coverage_used_by_solver=False,
        resolution_status='native footprint' if factor==1 else 'coarse whole-cell approximation; inspect lost footprint and channels')
    return dict(ground=floor,solid=solid,coverage=coverage,audit=audit)


def probe(layers,coarse,x,y,dx,native_dx=.175):
    native=layers['ground'];nz,nx=native.shape
    rz,rx=int((nz*native_dx-y)/native_dx),int(x/native_dx)
    z,col=int((coarse['ground'].shape[0]*dx-y)/dx),int(x/dx)
    return dict(native_ground_m=float(native[rz,rx]),native_obstacle=bool(layers['obstacle'][rz,rx]),
        native_ground_estimated=bool(layers['estimated_ground'][rz,rx]),
        coarse_ground_m=float(coarse['ground'][z,col]),coarse_solid=bool(coarse['solid'][z,col]),
        coarse_obstacle_fraction=float(coarse['coverage'][z,col]))
