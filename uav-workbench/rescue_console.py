"""Registered-scene and checkpoint contract for the independent rescue page."""
from dataclasses import asdict
import json,math,zipfile
from pathlib import Path
import numpy as np
from sur_rescue.hydraulics import metadata,file_digest
from sur_rescue.environment import RescueConfig,ACTIONS
ROOT=Path(__file__).resolve().parent
MODELS={'rescue:QMIX8':'QMIX8','rescue:Sur-QMIX8':'Sur-QMIX8'}

def sources():return {p.name:file_digest(p) for p in sorted((ROOT/'sur_rescue').glob('*.py'))}
def model_record(method):
    if method not in MODELS:raise ValueError('Unknown eight-action checkpoint')
    folder=ROOT/'generated/sur-rescue-pilot'/(MODELS[method]+'-s11')
    meta=json.loads((folder/'actor.json').read_text(encoding='utf-8'))
    if meta['n_actions']!=8 or meta['image_shape']!=[9,5,5] or meta['action_names']!=list(ACTIONS):raise ValueError('Checkpoint is not a nine-channel/eight-action model')
    if meta['sources']!=sources() or meta['actor_sha256']!=file_digest(folder/'actor.pt'):raise ValueError('Rescue checkpoint/source changed')
    return folder,meta

def catalog():
    result=[]
    for method in MODELS:
        try:
            _,m=model_record(method);result.append(dict(id=method,available=True,episodes=m['trained_episodes'],seed=m['seed'],scope=m['scope']))
        except (OSError,ValueError,KeyError) as e:result.append(dict(id=method,available=False,error=str(e)))
    return dict(version='hyra-dual-gmrf-eight-v1',models=result,actions=list(ACTIONS),channels=9,default_config=asdict(RescueConfig()),scope='工程贯通模型，尚未验证收敛或论文性能。')

def registered_scenes():
    from console_app import registered_scenes as old_scenes
    result=old_scenes();terrain=ROOT.parent/'terrain-workbench/terrain-lab/generated'
    existing={str(Path(s['path']).resolve()) for s in result}
    for p in sorted(terrain.glob('maps-*/maps/map-result.zip'),key=lambda p:p.stat().st_mtime,reverse=True):
        if str(p.resolve()) in existing:continue
        try:metadata(p)
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
        id_=p.parent.parent.name;result.append(dict(id='flood:'+id_,name='洪水工作台 / '+id_,path=str(p),kind='direct_hydraulic_workbench_export'))
    return result

def scene_metadata(scene):
    p=metadata(scene['path']);return dict(**{k:v for k,v in scene.items() if k!='path'},grid=p['grid'],duration_s=p['frames'][-1]['time_s'],actual_frame_count=len(p['frames']))

def validate(data):
    if not isinstance(data,dict):raise ValueError('Expected an object')
    method=data.get('method','rescue:rule')
    if method!='rescue:rule' and method not in MODELS:raise ValueError('Unknown rescue policy')
    scene=next((s for s in registered_scenes() if s['id']==data.get('scene')),None)
    if scene is None:raise ValueError('Unknown hydraulic scene')
    config=RescueConfig(**data.get('rescue_config',{}));p=metadata(scene['path']);times=np.asarray([f['time_s'] for f in p['frames']],dtype=float)
    class Header:
        def index(self,t):
            found=np.flatnonzero(np.isclose(times,t,atol=1e-7,rtol=0))
            if len(found)!=1:raise ValueError(f'{t}s没有实际洪水帧，请缩短任务或重新生成足够长的过程')
    config.validate(Header());cells=math.prod(p['grid']['shape']);count=round(config.duration_s/config.step_s)+1
    if cells*count>2_000_000:raise ValueError('当前回放上限为200万格点时刻，请缩短任务；水力帧按需读取，不缩小地图')
    if method in MODELS:
        _,model=model_record(method)
        if p['grid']['shape']!=[32,32] or p['grid']['dx_m']!=5 or asdict(config)!=model['config']:raise ValueError('工程模型仅验证32×32、5米格距及登记的250秒/5秒任务设置；大地图请先用八向规则核查')
    return dict(scene=scene,method=method,rescue_config=asdict(config),source_sha256=file_digest(scene['path']))
