"""ESRI ASCII -> float-metre raster terrain, independent of the world generator.

Rows remain north-to-south. Original x/y coordinates and datum live in metadata;
the workbench uses local x-right/z-down coordinates. No vertical scaling occurs.
"""
from pathlib import Path
import json
import math
import numpy as np


def read_ascii(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        header={}
        for _ in range(6):
            parts=stream.readline().split()
            if len(parts)!=2:raise ValueError('ASCII raster needs six header rows')
            header[parts[0].lower()]=float(parts[1])
        nx,nz=header.get('ncols',0.),header.get('nrows',0.)
        if not nx.is_integer() or not nz.is_integer() or min(nx,nz)<2:raise ValueError('Invalid raster dimensions')
        dx=header.get('cellsize',0)
        if not math.isfinite(dx) or dx<=0:raise ValueError('Invalid raster cell size')
        x=header.get('xllcorner',header.get('xllcenter',float('nan'))-dx/2)
        y=header.get('yllcorner',header.get('yllcenter',float('nan'))-dx/2)
        if not math.isfinite(x+y):raise ValueError('Missing raster origin')
        bed=np.loadtxt(stream).reshape(int(nz),int(nx))
    valid=np.isfinite(bed)&(bed!=header.get('nodata_value',-9999))
    if not valid.any():raise ValueError('Raster has no valid elevation')
    return bed,valid,dict(shape=list(bed.shape),cellsize_m=dx,xllcorner=x,yllcorner=y,
                         nodata=header.get('nodata_value',-9999))


def coarsen(bed,valid,factor,method):
    if isinstance(factor,bool) or not isinstance(factor,int) or factor<1:raise ValueError('Factor must be a positive integer')
    if method not in ('mean','max'):raise ValueError('Use mean or max elevation aggregation')
    nz,nx=bed.shape;top=(-nz)%factor;right=(-nx)%factor
    # Extend north/east only, preserving the south inflow and original origin.
    b=np.pad(bed,((top,0),(0,right)),mode='edge')
    v=np.pad(valid,((top,0),(0,right)),mode='edge')
    shape=(b.shape[0]//factor,factor,b.shape[1]//factor,factor)
    values=np.where(v,b,np.nan).reshape(shape)
    count=v.reshape(shape).sum(axis=(1,3))
    if method=='max':
        result=np.where(v,b,-np.inf).reshape(shape).max(axis=(1,3))
    else:result=np.nansum(values,axis=(1,3))/np.maximum(count,1)
    result=np.where(count>0,result,0.)
    # Unknown terrain is impermeable, never silently interpolated into a route.
    solid=count<factor*factor
    return result,solid,dict(padded_north_cells=top,padded_east_cells=right,method=method,
                             invalid_native_cells=int((~valid).sum()))


def write(root,name,bed,dx,metadata,*,solid=None,wall_x=None,wall_y=None,scenario=None,crossings=None,
          obstacle_top=None,obstacle_fraction=None):
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Import name must be a safe identifier')
    root=Path(root).resolve();folder=root/'generated'/name/'raw';folder.mkdir(parents=True,exist_ok=True)
    bed=np.asarray(bed,float)
    if bed.ndim!=2 or min(bed.shape)<2 or not np.isfinite(bed).all() or not math.isfinite(dx) or dx<=0:
        raise ValueError('Invalid imported elevation')
    nz,nx=bed.shape;fields={}
    def field(key,array,dtype):
        array=np.asarray(array,dtype=dtype);filename=key+'.bin';array.tofile(folder/filename)
        fields[key]=dict(file=filename,dtype=dtype,shape=list(array.shape))
    field('ground_height',bed,'<f8');fields['height']=dict(fields['ground_height'])
    field('terrain',np.zeros(bed.shape),'<i4')
    field('water_surface',np.full(bed.shape,np.nan),'<f4')
    for key,a,dtype in [('obstacle_top',obstacle_top,'<f8'),('obstacle_fraction',obstacle_fraction,'<f4')]:
        if a is not None:
            a=np.asarray(a,float)
            if a.shape!=bed.shape or not np.isfinite(a).all():raise ValueError('Invalid imported '+key)
            if key=='obstacle_fraction' and ((a<0)|(a>1)).any():raise ValueError('Obstacle fractions must be in [0,1]')
            field(key,a,dtype)
    for key,a,shape in [('solid',solid,bed.shape),('wall_x',wall_x,(nz,nx+1)),('wall_y',wall_y,(nz+1,nx))]:
        if a is not None:
            if np.asarray(a).shape!=shape:raise ValueError('Imported '+key+' shape mismatch')
            field(key,a,'|u1')
    descriptor=dict(schema='hyra-raster-import-v1',name=name,shape=list(bed.shape),dx_m=float(dx),
        fields=fields,metadata=metadata,scenario=scenario or {},crossings=crossings or [])
    (folder/'imported.json').write_text(json.dumps(descriptor,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return str(folder.relative_to(root)).replace('\\','/')


def describe(root,relative):
    root=Path(root).resolve();folder=(root/relative).resolve()
    if root not in folder.parents:raise ValueError('Imported raster outside workspace')
    data=json.loads((folder/'imported.json').read_text(encoding='utf-8'))
    if data.get('schema')!='hyra-raster-import-v1':raise ValueError('Unsupported raster schema')
    fields={}
    for name,field in data['fields'].items():
        p=(folder/field['file']).resolve()
        if p.parent!=folder:raise ValueError('Imported field path must stay in its data directory')
        fields[name]=dict(field,url='/'+str(p.relative_to(root)).replace('\\','/'))
    return dict(id=relative,shape=data['shape'],origin=[0,0],step=data['dx_m'],scale=1.,sea_level=-1e6,
        full=True,fields=fields,terrain_ids={'0':'unclassified_experiment_ground'},block_ids={},biome_ids={},
        config=dict(seed='imported',mode='imported_raster'),source_kind='imported_experiment_raster',
        imported=data['metadata'],case_defaults=data['scenario'],
        river_inlets=dict(crossings=data['crossings'],message='Experiment boundary, not inferred river network'),
        river_network=dict(status='not_applicable',message='实验边界已明确；未提供自然河网分类',warnings=[]))


def defaults(root,terrain,data):
    """Merge imported boundary defaults without overriding explicit edits."""
    result=dict(terrain['case_defaults'],**data)
    result['flood']=dict(terrain['case_defaults'].get('flood',{}),**data.get('flood',{}))
    if terrain.get('imported',{}).get('obstacle_layers') and result['flood'].get('grid_factor',1)!=1:
        raise ValueError('分层障碍地图需要按目标格宽重新导入，不能再次直接合并计算格')
    layers={}
    for name in ('solid','wall_x','wall_y'):
        if name in terrain['fields']:
            f=terrain['fields'][name];layers[name]=np.fromfile(Path(root)/f['url'].lstrip('/'),dtype=f['dtype']).reshape(f['shape']).astype(bool)
    result['layers']=dict(layers,**data.get('layers',{}))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--name',required=True)
    parser.add_argument('--factor',type=int,default=1)
    parser.add_argument('--aggregation',choices=['mean','max'],default='mean')
    parser.add_argument('--manning',type=float,default=.035)
    parser.add_argument('--activate',action='store_true')
    args=parser.parse_args()
    if not .01<=args.manning<=.2:parser.error('Manning n must be between 0.01 and 0.2')
    root=Path(__file__).resolve().parent
    native,valid,header=read_ascii(args.input)
    bed,solid,aggregation=coarsen(native,valid,args.factor,args.aggregation)
    scene=dict(mode='current',flood=dict(metres_per_block=1,roughness_mode='uniform',manning=args.manning,
        boundary='open',initial_water=False,baseflow=False,warmup_s=0,rain_mm_h=0,rain_duration_s=0,inflow_m3_s=0))
    terrain_id=write(root,args.name,bed,header['cellsize_m']*args.factor,
        dict(native=header,aggregation=aggregation,coordinate_system='source raster x/y; local workbench z=-y',
             vertical_datum='source raster, unchanged',source_filename=args.input.name),solid=solid,scenario=scene)
    scene['terrain_id']=terrain_id
    (root/terrain_id).parent.joinpath('scenario.json').write_text(json.dumps(scene,indent=2),encoding='utf-8')
    if args.activate:(root/'current.json').write_text(json.dumps(dict(directory=terrain_id)),encoding='utf-8')
    print(json.dumps(dict(terrain_id=terrain_id,shape=list(bed.shape),dx_m=header['cellsize_m']*args.factor)))
