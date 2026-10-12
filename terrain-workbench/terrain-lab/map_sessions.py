"""Local multi-layer scene construction and persistence; no generator mutations."""
from pathlib import Path
import json
import numpy as np
import flood_run
from flood_solver import FloodSolver
from multilayer import MapWorld, DATA_DICTIONARY


def create(root,terrain,data):
    if not isinstance(data,dict):raise ValueError('场景需要是对象')
    mode=data.get('mode','current')
    if mode=='demo':
        nz=nx=32;zz,xx=np.indices((nz,nx));bed=.035*xx+.01*zz
        depth=np.zeros_like(bed);depth[10:22,:6]=.25
        config=flood_run.settings(dict(resolution=32,source_x=2,source_z=16,rain_mm_h=60,
            rain_duration_s=300,duration_s=600,baseflow=False,warmup_s=0,boundary='closed'),[32,32])
        config.update({k:v for k,v in flood_run.settings(dict(config,**data.get('flood',{})),[32,32]).items()})
        config['resolution']=32;unit=config['metres_per_block']
        solver=FloodSolver(bed*unit,depth*unit,5*unit,boundary=config['boundary'],manning=config['manning'],sea_level=0,sea_edges={})
        meta=dict(shape=[32,32],dx_m=5*unit,origin_blocks=[0,0],coarsening_factor=1,config=config,
                  terrain_source='synthetic_multilayer_demo',terrain_id='multilayer-demo')
        default=dict(patches=[dict(kind='building',x=13,z=10,width=4,height=8),
            dict(kind='wall_x',x=22,z=8,height=7),dict(kind='wall_x',x=22,z=18,height=7),
            dict(kind='roughness',x=8,z=20,width=7,height=7,value=.08),
            dict(kind='no_fly',x=20,z=3,width=5,height=4),
            dict(kind='drain',x=27,z=16,width=2,height=2,value=.5),
            dict(kind='person',x=20,z=14,value=1),dict(kind='person',x=25,z=22,value=2),
            dict(kind='source',x=2,z=16,width=2,height=2,value=0)],hydrographs={'0':[[0,8],[120,3],[300,0]]})
        # Explicit synthetic demonstration field; actual generated maps do not
        # invent a signal field or population. This is not a GMRF estimator.
        default['signal_field']=(np.exp(-((xx-20)**2+(zz-14)**2)/18)
                                 +.6*np.exp(-((xx-25)**2+(zz-22)**2)/18)).tolist()
        # Caller-supplied arrays take precedence, patches extend the demonstration.
        layers=dict(default,**data.get('layers',{}))
        if 'patches' in data.get('layers',{}):layers['patches']=default['patches']+data['layers']['patches']
        agents=data.get('agents',[dict(id='uav-1',x=7,z=16,radius=4),dict(id='uav-2',x=26,z=22,radius=4)])
    elif mode=='current':
        if data.get('terrain_id')!=terrain['id']:raise ValueError('地图已更新，请重新读取地图')
        config=flood_run.settings({'warmup_s':0,**data.get('flood',{})},terrain['shape'])
        if config['warmup_s']!=0:raise ValueError('实时多层场景从初始状态开始；可先关闭降雨推进基流预热')
        solver,meta=flood_run.prepare(root,terrain,config)
        layers=dict(data.get('layers',{}));agents=data.get('agents')
        raw=terrain['fields']['terrain'];types=np.fromfile(root/raw['url'].lstrip('/'),dtype=raw['dtype']).reshape(terrain['shape'])
        f=meta['coarsening_factor'];nz,nx=meta['shape'];tiles=types.reshape(nz,f,nx,f).transpose(0,2,1,3).reshape(nz,nx,f*f)
        ids=np.unique(types);votes=np.stack([(tiles==i).sum(axis=-1) for i in ids])
        layers.setdefault('terrain_class',ids[votes.argmax(axis=0)].tolist())
    else:raise ValueError('请选择当前地图或多层示例')
    if 'initial_depth' in data:
        h=np.asarray(data['initial_depth'],float)
        if h.shape!=solver.h.shape or not np.isfinite(h).all() or (h<0).any():raise ValueError('初始水深数组需要同网格、非负且有限')
        solver.h=h.copy()
    return MapWorld(solver,meta,layers,agents,legacy=data.get('legacy_exploration'))


def persist(root,session_id,world):
    folder=root/'generated'/session_id/'maps';folder.mkdir(parents=True,exist_ok=True)
    tmp=folder/'map-result.zip.tmp';tmp.write_bytes(world.export_zip());tmp.replace(folder/'map-result.zip')
    return f'/generated/{session_id}/maps/map-result.zip'
