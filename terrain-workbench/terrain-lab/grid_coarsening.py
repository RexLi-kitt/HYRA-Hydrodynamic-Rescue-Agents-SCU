"""Explicit uniform grid aggregation; never invent finer terrain samples."""
import numpy as np


def blocks(a,factor,operation='mean'):
    a=np.asarray(a);nz,nx=a.shape
    if nz%factor or nx%factor:
        raise ValueError('地图尺寸必须能被计算格合并倍数整除')
    grouped=a.reshape(nz//factor,factor,nx//factor,factor)
    if operation=='mean':return grouped.mean(axis=(1,3))
    if operation=='sum':return grouped.sum(axis=(1,3))
    if operation=='any':return grouped.any(axis=(1,3))
    if operation=='mode':
        result=np.zeros((nz//factor,nx//factor),a.dtype);best=np.full(result.shape,-1)
        for value in np.unique(a):
            count=(grouped==value).sum(axis=(1,3));choose=count>best
            result[choose]=value;best[choose]=count[choose]
        return result
    raise ValueError('Unknown aggregation')


def resize_layers(layers,native_bed,native_dx,factor,manning):
    """Apply authored patches at their original grid, then aggregate quantities."""
    old=layers.get('grid_factor',1)
    if isinstance(old,bool) or old not in (1,2,4):raise ValueError('图层网格倍数需要为1、2或4')
    if old==factor:return layers
    from multilayer import BaseMap
    base=BaseMap(blocks(native_bed,old),native_dx*old,layers,manning)
    source=base.snapshot();result=dict(hydrographs=layers.get('hydrographs',{}),grid_factor=factor)
    if factor>old:
        ratio=factor//old
        for key in ('z','n'):result[key]=blocks(source[key],ratio)
        result['solid']=blocks(source['solid'],ratio,'any')
        result['drain_capacity']=blocks(source['drain_capacity'],ratio,'sum')
        result['terrain_class']=blocks(source['terrain_class'],ratio,'mode')
        ids=np.full(result['solid'].shape,-1,np.int32)
        for sid in np.unique(source['source_id']):
            if sid<0:continue
            selected=blocks(source['source_id']==sid,ratio,'any')
            if np.any(selected&(ids>=0)):
                raise ValueError('较大计算格合并了不同编号的水源，请使用更小格宽或重新编辑水源')
            ids[selected]=sid
        result['source_id']=ids
        nz,nx=ids.shape
        wx=np.zeros((nz,nx+1),bool);wy=np.zeros((nz+1,nx),bool)
        # Snap impermeable faces to the nearest coarse face; retain every wall.
        z,x=np.nonzero(source['wall_x']);np.logical_or.at(wx,(z//ratio,np.rint(x/ratio).astype(int)),True)
        z,x=np.nonzero(source['wall_y']);np.logical_or.at(wy,(np.rint(z/ratio).astype(int),x//ratio),True)
        result.update(wall_x=wx,wall_y=wy)
    else:
        ratio=old//factor
        for key in ('z','n','solid','source_id','terrain_class','drain_capacity'):
            result[key]=np.repeat(np.repeat(source[key],ratio,axis=0),ratio,axis=1)
        result['drain_capacity']/=ratio**2
        nz,nx=result['z'].shape
        wx=np.zeros((nz,nx+1),bool);wy=np.zeros((nz+1,nx),bool)
        wx[:,::ratio]=np.repeat(source['wall_x'],ratio,axis=0)
        wy[::ratio,:]=np.repeat(source['wall_y'],ratio,axis=1)
        result.update(wall_x=wx,wall_y=wy)
    if 'z' not in layers:
        # Returning to a finer grid restores the original terrain samples.
        result['z']=blocks(native_bed,factor)
    return result


def coarsen_water_state(bed, depth, factor, metres_per_unit=1.):
    """Conserve volume; reject mixed shoreline cells that shift wet-cell stage.

    A single coarse bed/depth pair cannot in general retain both wet-subcell
    free surface and storage. Reject instead of silently deleting water.
    Inputs use terrain units; tolerance is in physical metres.
    """
    bed=np.asarray(bed,float); depth=np.asarray(depth,float)
    if bed.shape!=depth.shape or not np.isfinite(bed).all() or not np.isfinite(depth).all() or (depth<0).any():
        raise ValueError('Invalid bed/depth for water coarsening')
    if not np.isfinite(metres_per_unit) or metres_per_unit<=0:
        raise ValueError('Invalid terrain unit')
    if factor==1:return bed.copy(),depth.copy()
    coarse_bed=blocks(bed,factor);coarse_depth=blocks(depth,factor)
    wet=depth>0
    coverage=blocks(wet.astype(float),factor)
    wet_stage=np.divide(blocks(np.where(wet,bed+depth,0.),factor),coverage,
                        out=np.zeros_like(coverage),where=coverage>0)
    shift=(coarse_bed+coarse_depth-wet_stage)*metres_per_unit
    risky=(coverage>0)&(coverage<1)&(np.abs(shift)>1e-6)
    if risky.any():
        raise ValueError(f'初始水体粗化会改变干湿岸线水面（{int(risky.sum())}个格，最大偏移{float(np.abs(shift[risky]).max()):.6g}米）；请选择原始格宽重新模拟。')
    return coarse_bed,coarse_depth
