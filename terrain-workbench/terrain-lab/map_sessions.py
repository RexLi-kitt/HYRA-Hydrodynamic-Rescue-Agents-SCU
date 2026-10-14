"""Local multi-layer scene construction and persistence; no generator mutations."""
from pathlib import Path
import json
import numpy as np
import flood_run
from flood_solver import FloodSolver
from multilayer import MapWorld, DATA_DICTIONARY, json_safe


def create(root,terrain,data):
    if not isinstance(data,dict):raise ValueError('场景需要是对象')
    if 'agents' in data or 'legacy_exploration' in data:raise ValueError('地图工作台已移除无人机与探索体系，请使用地图场景配置')
    mode=data.get('mode','current')
    incoming=dict(data.get('flood',{}));frames=incoming.pop('frames',121)
    if isinstance(frames,bool) or not isinstance(frames,int) or not 2<=frames<=241:raise ValueError('过程帧数需要在2到241之间')
    if mode=='demo':
        nz=nx=32;zz,xx=np.indices((nz,nx));bed=.035*xx+.01*zz
        depth=np.zeros_like(bed);depth[10:22,:6]=.25
        config=flood_run.settings(dict(resolution=32,source_x=2,source_z=16,rain_mm_h=60,
            rain_duration_s=300,duration_s=600,baseflow=False,warmup_s=0,boundary='closed'),[32,32])
        config.update({k:v for k,v in flood_run.settings(dict(config,**incoming),[32,32]).items()})
        config['resolution']=32;unit=config['metres_per_block']
        if not config['initial_water']:depth.fill(0)
        solver=FloodSolver(bed*unit,depth*unit,5*unit,boundary=config['boundary'],manning=config['manning'],sea_level=0,sea_edges={})
        meta=dict(shape=[32,32],dx_m=5*unit,origin_blocks=[0,0],coarsening_factor=1,config=config,
                  terrain_source='synthetic_multilayer_demo',terrain_id='multilayer-demo')
        default=dict(patches=[dict(kind='building',x=13,z=10,width=4,height=8),
            dict(kind='wall_x',x=22,z=8,height=7),dict(kind='wall_x',x=22,z=18,height=7),
            dict(kind='roughness',x=8,z=20,width=7,height=7,value=.08),
            dict(kind='drain',x=27,z=16,width=2,height=2,value=.5),
            dict(kind='source',x=2,z=16,width=2,height=2,value=0)],hydrographs={'0':[[0,8],[120,3],[300,0]]})
        # Caller-supplied arrays take precedence, patches extend the demonstration.
        layers=dict(default,**data.get('layers',{}))
        if 'patches' in data.get('layers',{}):layers['patches']=default['patches']+data['layers']['patches']
    elif mode=='current':
        if data.get('terrain_id')!=terrain['id']:raise ValueError('地图已更新，请重新读取地图')
        config=flood_run.settings({'warmup_s':0,**incoming},terrain['shape'])
        solver,meta=flood_run.prepare(root,terrain,config)
        layers=dict(data.get('layers',{}))
        raw=terrain['fields']['terrain'];types=np.fromfile(root/raw['url'].lstrip('/'),dtype=raw['dtype']).reshape(terrain['shape'])
        f=meta['coarsening_factor'];nz,nx=meta['shape'];tiles=types.reshape(nz,f,nx,f).transpose(0,2,1,3).reshape(nz,nx,f*f)
        ids=np.unique(types);votes=np.stack([(tiles==i).sum(axis=-1) for i in ids])
        layers.setdefault('terrain_class',ids[votes.argmax(axis=0)].tolist())
    else:raise ValueError('请选择当前地图或多层示例')
    if 'initial_depth' in data:
        h=np.asarray(data['initial_depth'],float)
        if h.shape!=solver.h.shape or not np.isfinite(h).all() or (h<0).any():raise ValueError('初始水深数组需要同网格、非负且有限')
        solver.h=h.copy()
    config['frames']=frames
    world=MapWorld(solver,meta,layers)
    import river_inlets
    crossings=terrain.get('river_inlets',{}).get('crossings',[]) if mode=='current' else []
    river_inlets.configure(world,crossings,data.get('external_inflows',[]),terrain['shape'] if mode=='current' else [32,32])
    world.scenario_config=json_safe(data)
    return world


def run(root,session_id,world,update,cancelled):
    """Compute a whole scenario in a worker; publish immutable frame files."""
    s=world.solver;c=world.config;meta=world.meta
    if s.time!=0:raise ValueError('请重新建立场景后开始模拟')
    if s.baseflow_m3_s>0 and c['warmup_s']>0:
        from river_baseflow import spin_up
        schedule=s.source_schedule;external=s.external_schedule;s.source_schedule=[];s.external_schedule=[];s.external_rate.fill(0)
        try:
            meta['initial_conditions']=spin_up(s,c['warmup_s'],c['infiltration_mm_h']/3600000,update,cancelled)
        finally:s.source_schedule=schedule;s.external_schedule=external;s.external_rate=s._external_rate(0)
        s.start_scenario();s.qx.fill(0);s.qy.fill(0)
    else:
        meta['initial_conditions']=dict(duration_s=0,converged=False,
            reason='no_active_baseflow' if s.baseflow_m3_s==0 else 'warmup_disabled')
    folder=root/'generated'/session_id/'maps';folder.mkdir(parents=True,exist_ok=True)
    url=f'/generated/{session_id}/maps/'
    frames=[]
    for i,end in enumerate(np.linspace(0,c['duration_s'],c['frames'])):
        if cancelled():raise InterruptedError('模拟已取消，已计算时段仍可播放')
        world.advance_to(float(end),cancelled)
        u,v=s.velocity();name=f'frame-{i:03d}.f32'
        tmp=folder/(name+'.tmp')
        np.stack((s.h,u,v,s.maximum)).astype('<f4').tofile(tmp);tmp.replace(folder/name)
        frames.append(dict(index=i,url=url+name,**s.summary()))
        update(state='running',progress=round(i/(c['frames']-1)*100,1),
            message=f'已计算 {end:.1f} / {c["duration_s"]:.0f} 秒',
            frames=list(frames),initial_conditions=meta['initial_conditions'])
    if cancelled():raise InterruptedError('模拟已取消，已计算时段仍可播放')
    update(message='正在保存完整地图与洪水过程…')
    save_process(root,session_id,world,frames)
    return dict(frames=frames,archive_url=url+'map-result.zip')


def save_process(root,session_id,world,frames):
    import zipfile
    folder=root/'generated'/session_id/'maps';folder.mkdir(parents=True,exist_ok=True)
    c=world.config;meta=world.meta
    # ZIP stores physical layers, calculated frames and final shared face fluxes.
    tmp=folder/'map-result.zip.tmp'
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('map_model.json',json.dumps(world.snapshot(),ensure_ascii=False,allow_nan=False))
        archive.writestr('data_dictionary.json',json.dumps(DATA_DICTIONARY,ensure_ascii=False,indent=2))
        import io
        data=io.BytesIO();np.savez_compressed(data,**world.arrays());archive.writestr('layers.npz',data.getvalue())
        archive.writestr('process.json',json.dumps(dict(grid=world.grid(),config=c,frames=frames,
            frame_fields=['depth','u','v','maximum_depth'],frame_dtype='<f4',
            initial_conditions=meta.get('initial_conditions',{}),external_inflows=world.external_inflows,
            river_crossings=world.river_crossings),ensure_ascii=False,allow_nan=False))
        for f in frames:archive.write(folder/Path(f['url']).name,Path(f['url']).name)
    tmp.replace(folder/'map-result.zip')
    if hasattr(world,'scenario_config'):
        url=f'/generated/{session_id}/maps/'
        complete=bool(frames and frames[-1]['time_s']>=c['duration_s']-1e-8)
        status=dict(state='done' if complete else 'cancelled',progress=100 if complete else
            (100*frames[-1]['time_s']/c['duration_s'] if frames else 0),frames=frames,
            message='已保存的洪水过程，可回放或重新模拟' if complete else '已保存的部分过程，可回放已计算时段',
            initial_conditions=meta.get('initial_conditions',{}),archive_url=url+'map-result.zip')
        replay=dict(world.snapshot(),id=session_id,scenario_config=world.scenario_config,
            process=status,archive_status_url=url+'status.json')
        for name,value in [('status.json',status),('replay.json',replay)]:
            file=folder/name;pending=file.with_suffix('.json.tmp')
            pending.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
            pending.replace(file)
    return f'/generated/{session_id}/maps/map-result.zip'
