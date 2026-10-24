"""Local multi-layer scene construction and persistence; no generator mutations."""
from pathlib import Path
import json
import time
import numpy as np
import flood_run
from flood_solver import FloodSolver
from multilayer import MapWorld, DATA_DICTIONARY, json_safe


def browser_scene(root,session_id,world,scope='initial',pack_threshold=256**2):
    """Send large grid arrays as binary fields instead of millions of JSON numbers."""
    size=world.solver.mesh.count if hasattr(world.solver,'mesh') else world.solver.h.size
    if size<=pack_threshold:return world.snapshot(compact=True)
    folder=root/'generated'/session_id/'maps'/scope;folder.mkdir(parents=True,exist_ok=True)
    url=f'/generated/{session_id}/maps/{scope}/';index=0
    def encode(a):
        nonlocal index
        name=f'field-{index:02d}.bin';index+=1
        # Explicit little-endian types are also readable in the browser.
        dtype=a.dtype.newbyteorder('<')
        a.astype(dtype,copy=False).tofile(folder/name)
        return dict(array_url=url+name,dtype=dtype.str,shape=list(a.shape))
    return world.snapshot(array_encoder=encode,compact=True)


def write_snapshot(stream,world):
    """Stream the portable JSON export row by row, without a full Python-list copy."""
    def write(value):
        if isinstance(value,np.ndarray):
            stream.write('[')
            for i,row in enumerate(value):
                if i:stream.write(',')
                stream.write(json.dumps(json_safe(row),ensure_ascii=False,allow_nan=False,separators=(',',':')))
            stream.write(']')
        elif isinstance(value,dict):
            stream.write('{')
            for i,(key,item) in enumerate(value.items()):
                if i:stream.write(',')
                stream.write(json.dumps(str(key),ensure_ascii=False)+':');write(item)
            stream.write('}')
        elif isinstance(value,(list,tuple)):
            stream.write('[')
            for i,item in enumerate(value):
                if i:stream.write(',')
                write(item)
            stream.write(']')
        else:stream.write(json.dumps(json_safe(value),ensure_ascii=False,allow_nan=False))
    write(world.snapshot(array_encoder=lambda a:a))


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
        config['grid_factor']=1  # The authored demonstration has fixed grid coordinates.
        if not config['initial_water']:depth.fill(0)
        solver=FloodSolver(bed*unit,depth*unit,5*unit,boundary=config['boundary'],manning=config['manning'],
            sea_level=0,sea_edges={},**flood_run.numerical_options(config))
        meta=dict(shape=[32,32],dx_m=5*unit,origin_blocks=[0,0],coarsening_factor=1,config=config,
                  terrain_source='synthetic_multilayer_demo',terrain_id='multilayer-demo',
                  terrain_ids={'0':'land','1':'river','2':'lake','3':'wetland','4':'ocean'},
                  numerics=solver.numerical_config())
        types=np.zeros((32,32),np.int32);types[10:22,:6]=2
        default=dict(terrain_class=types,patches=[dict(kind='building',x=13,z=10,width=4,height=8),
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
        prepare_config=dict(config,initial_water=False) if 'initial_depth' in data else config
        solver,meta=flood_run.prepare(root,terrain,prepare_config)
        meta['config']=config
        meta['terrain_ids']=dict(terrain['terrain_ids'])
        layers=dict(data.get('layers',{}))
        raw=terrain['fields']['terrain'];types=np.fromfile(root/raw['url'].lstrip('/'),dtype=raw['dtype']).reshape(terrain['shape'])
        old_factor=layers.get('grid_factor',1)
        if old_factor not in (1,2,4) or isinstance(old_factor,bool):raise ValueError('图层网格倍数需要为1、2或4')
        explicit_n='n' in layers
        if not explicit_n:
            from grid_coarsening import blocks
            if old_factor==config['grid_factor']:
                layers['n']=solver.manning.copy()
            elif config['roughness_mode']=='landcover':
                from surface_roughness import from_terrain
                native_n,_=from_terrain(root,terrain,config['manning'])
                layers['n']=blocks(native_n,old_factor) if old_factor>1 else native_n
            else:
                layers['n']=np.full((terrain['shape'][0]//old_factor,terrain['shape'][1]//old_factor),config['manning'])
        meta['roughness']['base_source']='explicit_n_layer' if explicit_n else config['roughness_mode']
        if old_factor>1:
            from grid_coarsening import blocks
            layers.setdefault('terrain_class',blocks(types,old_factor,'mode'))
        else:layers.setdefault('terrain_class',types)
        layers=dict(layers,patches=[normalize_patch(p,old_factor) for p in layers.get('patches',[])])
        if old_factor!=config['grid_factor']:
            from grid_coarsening import resize_layers
            rough_patches=[p for p in layers.get('patches',[]) if p.get('kind')=='roughness']
            name='ground_height' if terrain['full'] else 'height';field=terrain['fields'][name]
            native_bed=np.fromfile(root/field['url'].lstrip('/'),dtype=field['dtype']).reshape(terrain['shape'])
            native_bed=native_bed*config['metres_per_block']*(1 if terrain['full'] else terrain['scale'])
            layers=resize_layers(layers,native_bed,terrain['step']*config['metres_per_block'],
                config['grid_factor'],config['manning'])
            if not explicit_n and config['roughness_mode']=='landcover' and old_factor>config['grid_factor']:
                # Restore native automatic detail outside authored coarse patches.
                authored=np.zeros((terrain['shape'][0]//old_factor,terrain['shape'][1]//old_factor),bool)
                for p in rough_patches:
                    authored[p['z']:p['z']+p.get('height',1),p['x']:p['x']+p.get('width',1)]=True
                ratio=old_factor//config['grid_factor']
                authored=np.repeat(np.repeat(authored,ratio,axis=0),ratio,axis=1)
                fresh=blocks(native_n,config['grid_factor']) if config['grid_factor']>1 else native_n
                layers['n']=np.where(authored,layers['n'],fresh)
            # Rebin default terrain classes even when the authored layers use a coarse grid.
        elif old_factor>1 and np.asarray(layers['terrain_class']).shape!=solver.h.shape:
            from grid_coarsening import blocks
            layers['terrain_class']=blocks(types,old_factor,'mode')
    else:raise ValueError('请选择当前地图或多层示例')
    if 'initial_depth' in data:
        h=np.asarray(data['initial_depth'],float)
        if mode=='current' and h.shape==tuple(terrain['shape']) and config['grid_factor']>1:
            from grid_coarsening import coarsen_water_state
            name='ground_height' if terrain['full'] else 'height';field=terrain['fields'][name]
            native_bed=np.fromfile(root/field['url'].lstrip('/'),dtype=field['dtype']).reshape(terrain['shape'])
            native_bed=native_bed*config['metres_per_block']*(1 if terrain['full'] else terrain['scale'])
            _,h=coarsen_water_state(native_bed,h,config['grid_factor'])
        if h.shape!=solver.h.shape or not np.isfinite(h).all() or (h<0).any():raise ValueError('初始水深数组需要同网格、非负且有限')
        solver.h=h.copy()
    config['frames']=frames
    world=MapWorld(solver,meta,layers,data.get('structures',[]),data.get('people'))
    if mode=='current':
        meta['roughness'].update(effective_min_n=float(world.base.n.min()),effective_max_n=float(world.base.n.max()),
            manual_patch_count=sum(p.get('kind')=='roughness' for p in data.get('layers',{}).get('patches',[])))
    if mode=='current':
        import river_network
        meta['river_network_grid_check']=river_network.audit_world(root,terrain,world)
    data=dict(data,structures=world.structures.items,people=world.people,
              flood=dict(data.get('flood',{}),roughness_mode=config['roughness_mode']))
    import river_inlets
    crossings=terrain.get('river_inlets',{}).get('crossings',[]) if mode=='current' else []
    river_inlets.configure(world,crossings,data.get('external_inflows',[]),terrain['shape'] if mode=='current' else [32,32])
    use_native_grid(world)
    world.scenario_config=json_safe(data)
    return world


def normalize_patch(p,factor):
    """Keep editor patch coordinates tied to the grid where they were authored."""
    source=p.get('grid_factor',factor)
    if isinstance(source,bool) or source not in (1,2,4):raise ValueError('编辑网格倍数需要为1、2或4')
    if source==factor:return p
    result=dict(p,grid_factor=factor)
    for coordinate,extent in (('x','width'),('z','height')):
        start=p[coordinate]*source/factor;end=(p[coordinate]+p.get(extent,1))*source/factor
        result[coordinate]=int(np.floor(start));result[extent]=int(np.ceil(end))-result[coordinate]
    if p['kind']=='drain':result['value']=p.get('value',1)*(factor/source)**2
    return result


def use_native_grid(world):
    """Use one uniform hydraulic grid, including legacy refinement requests."""
    s=world.solver
    world.config.update(refine_inland_water=False,refinement_buffer=0)
    world.meta['refinement']=dict(requested=False,enabled=False,ratio=1,
        target_classes=[],buffer_cells=0,selected_parent_cells=0,refined_parent_cells=0,
        base_cell_count=s.h.size,leaf_cell_count=s.h.size,minimum_cell_width_m=s.dx,
        time_stepping='shared_global_timestep',display_projection='native_grid_state')
    world.meta['numerics']=world.solver.numerical_config()


def run(root,session_id,world,update,cancelled):
    """Compute a whole scenario in a worker; publish immutable frame files."""
    s=world.solver;c=world.config;meta=world.meta
    if s.time!=0:raise ValueError('请重新建立场景后开始模拟')
    update(state='running',progress=0,message='正在准备洪水计算，首次计算可能稍慢…')
    if s.baseflow_m3_s>0 and c['warmup_s']>0:
        from river_baseflow import spin_up
        schedule=s.source_schedule;external=s.external_schedule;s.source_schedule=[];s.external_schedule=[];s.external_rate.fill(0)
        s.structure_warmup=True
        try:
            meta['initial_conditions']=spin_up(s,c['warmup_s'],c['infiltration_mm_h']/3600000,update,cancelled)
        finally:
            s.source_schedule=schedule;s.external_schedule=external;s.external_rate=s._external_rate(0)
            s.structure_warmup=False
        s.start_scenario();s.qx.fill(0);s.qy.fill(0)
    else:
        meta['initial_conditions']=dict(duration_s=0,converged=False,
            reason='no_active_baseflow' if s.baseflow_m3_s==0 else 'warmup_disabled')
    folder=root/'generated'/session_id/'maps';folder.mkdir(parents=True,exist_ok=True)
    url=f'/generated/{session_id}/maps/'
    frames=[]
    started=time.perf_counter()
    def progress(s):
        elapsed=time.perf_counter()-started
        rate=s.time/max(elapsed,1e-9)
        remaining=(c['duration_s']-s.time)/rate if rate>0 else None
        update(state='running',progress=round(100*s.time/c['duration_s'],1),
            computed_time_s=float(s.time),steps=s.steps,dt_s=s.last_timestep.get('dt_s'),
            wall_elapsed_s=elapsed,simulation_rate=rate,estimated_remaining_s=remaining,
            message=f'已计算 {s.time:.2f} / {c["duration_s"]:.0f} 秒 · 已算 {s.steps} 步 · 每秒推进 {rate:.2f} 秒模拟')
    for i,end in enumerate(np.linspace(0,c['duration_s'],c['frames'])):
        if cancelled():raise InterruptedError('模拟已取消，已计算时段仍可播放')
        world.advance_to(float(end),cancelled,progress=progress)
        u,v=s.velocity();name=f'frame-{i:03d}.f32'
        tmp=folder/(name+'.tmp')
        np.stack((s.h,u,v,s.maximum)).astype('<f4').tofile(tmp);tmp.replace(folder/name)
        frame=dict(index=i,url=url+name,**s.summary())
        if hasattr(s,'mesh'):
            mu,mv=s.leaf_velocity();mesh_name=f'mesh-frame-{i:03d}.f32'
            pending=folder/(mesh_name+'.tmp')
            np.stack((s.leaf['h'],mu,mv,s.leaf['maximum'])).astype('<f4').tofile(pending);pending.replace(folder/mesh_name)
            frame.update(mesh_url=url+mesh_name,mesh_shape=[4,s.mesh.count],base_frame_projection='area_averaged_conserved_state')
        frames.append(frame)
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
        import io
        with archive.open('map_model.json','w',force_zip64=True) as output:
            with io.TextIOWrapper(output,encoding='utf-8') as text:write_snapshot(text,world)
        archive.writestr('data_dictionary.json',json.dumps(DATA_DICTIONARY,ensure_ascii=False,indent=2))
        with archive.open('layers.npz','w',force_zip64=True) as output:
            np.savez_compressed(output,**world.arrays())
        archive.writestr('process.json',json.dumps(dict(grid=world.grid(),config=c,frames=frames,
            frame_fields=['depth','u','v','maximum_depth'],frame_dtype='<f4',
            initial_conditions=meta.get('initial_conditions',{}),external_inflows=world.external_inflows,
            river_crossings=world.river_crossings,refinement=meta.get('refinement'),
            mesh_frame_fields=['depth','u','v','maximum_depth'] if hasattr(world.solver,'mesh') else None,
            mesh_geometry='map_model.json:M_mesh' if hasattr(world.solver,'mesh') else None),ensure_ascii=False,allow_nan=False))
        for f in frames:
            for key in ('url','mesh_url'):
                if f.get(key):archive.write(folder/Path(f[key]).name,Path(f[key]).name)
    tmp.replace(folder/'map-result.zip')
    if hasattr(world,'scenario_config'):
        url=f'/generated/{session_id}/maps/'
        complete=bool(frames and frames[-1]['time_s']>=c['duration_s']-1e-8)
        status=dict(state='done' if complete else 'cancelled',progress=100 if complete else
            (100*frames[-1]['time_s']/c['duration_s'] if frames else 0),frames=frames,
            message='已保存的洪水过程，可回放或重新模拟' if complete else '已保存的部分过程，可回放已计算时段',
            initial_conditions=meta.get('initial_conditions',{}),archive_url=url+'map-result.zip')
        replay=dict(browser_scene(root,session_id,world,scope='replay'),id=session_id,scenario_config=world.scenario_config,
            process=status,archive_status_url=url+'status.json')
        for name,value in [('status.json',status),('replay.json',replay)]:
            file=folder/name;pending=file.with_suffix('.json.tmp')
            pending.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
            pending.replace(file)
    return f'/generated/{session_id}/maps/map-result.zip'
