"""Terrain adapter and disk-backed flood jobs; NumPy is loaded on demand."""
from __future__ import annotations
import json
import math
from pathlib import Path
import zipfile

DEFAULTS = dict(metres_per_block=1.0, duration_s=600.0,
                rain_mm_h=80.0, rain_duration_s=600.0, infiltration_mm_h=0.0,
                inflow_m3_s=0.0, source_x=128, source_z=128,
                manning=0.035, roughness_mode='landcover', boundary='ocean', initial_water=True, frames=31,
                baseflow=True, baseflow_m3_s=5.0, warmup_s=600.0,
                refine_inland_water=False, refinement_buffer=0, grid_factor=1,
                flux_method='rusanov', spatial_order=2, time_integrator='rk2',
                low_froude=0, cfl=.4, limiter_theta=1.5, velocity_depth=1e-6,
                max_timestep=1., max_steps=500000)

NUMERICAL_KEYS = ('flux_method', 'spatial_order', 'time_integrator', 'low_froude',
                  'cfl', 'limiter_theta', 'velocity_depth', 'max_timestep', 'max_steps')


def numerical_options(config):
    return {key: config.get(key, DEFAULTS[key]) for key in NUMERICAL_KEYS}


def settings(data, shape):
    if not isinstance(data, dict): raise ValueError('洪水参数格式不正确')
    config = dict(DEFAULTS)
    config['source_x']=min(config['source_x'],shape[1]-1)
    config['source_z']=min(config['source_z'],shape[0]-1)
    specs = dict(metres_per_block=(.1, 100), duration_s=(1, 3600), rain_mm_h=(0, 1000),
                 rain_duration_s=(0, 3600), infiltration_mm_h=(0, 500), inflow_m3_s=(0, 10000),
                 manning=(.01, .2), source_x=(0, shape[1]-1), source_z=(0, shape[0]-1),
                 frames=(2, 61), baseflow_m3_s=(0, 10000), warmup_s=(0, 3600),
                 refinement_buffer=(0,4),
                 spatial_order=(1, 2), low_froude=(0, 2), cfl=(.0001, .45),
                 limiter_theta=(1, 2), velocity_depth=(1e-8, .01),
                 max_timestep=(1e-5, 60), max_steps=(1, 5000000))
    for name, (low, high) in specs.items():
        if name in NUMERICAL_KEYS and isinstance(data.get(name), bool):
            raise ValueError(f'{name} 需要数值，不能是布尔值')
        try: value = float(data.get(name, config[name]))
        except (ValueError, TypeError): raise ValueError(f'{name} 需要是数值')
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{name} 超出允许范围')
        if name in ('source_x', 'source_z', 'frames', 'spatial_order', 'low_froude', 'max_steps','refinement_buffer'):
            if value != int(value): raise ValueError(f'{name} 需要是整数')
            value = int(value)
        config[name] = value
    # Legacy resolution settings never resample a newly constructed scene.
    config['resolution'] = max(shape)
    factor=data.get('grid_factor',1)
    if isinstance(factor,bool) or not isinstance(factor,int) or factor not in (1,2,4):
        raise ValueError('计算格合并倍数需要为1、2或4')
    if any(n%factor or n//factor<2 for n in shape):
        raise ValueError('地图尺寸必须能被计算格合并倍数整除，且至少保留2×2格')
    config['grid_factor']=factor
    config['roughness_mode']=data.get('roughness_mode',config['roughness_mode'])
    if config['roughness_mode'] not in ('landcover','uniform'):raise ValueError('糙率模式需要为landcover或uniform')
    if config['rain_duration_s'] > config['duration_s']: raise ValueError('降雨持续时间不能超过模拟时间')
    config['boundary'] = data.get('boundary', config['boundary'])
    if config['boundary'] not in ('ocean', 'open', 'closed'): raise ValueError('请选择边界方式')
    config['initial_water'] = data.get('initial_water', True)
    if not isinstance(config['initial_water'], bool): raise ValueError('初始水体参数格式不正确')
    config['baseflow'] = data.get('baseflow', True)
    if not isinstance(config['baseflow'], bool): raise ValueError('基流参数格式不正确')
    config['refine_inland_water']=data.get('refine_inland_water',False)
    if not isinstance(config['refine_inland_water'],bool):raise ValueError('内陆水域加密开关需要布尔值')
    if isinstance(data.get('refinement_buffer'),bool):raise ValueError('加密过渡带需要整数')
    # Accept legacy configuration files, but all new runs use native cells.
    config['refine_inland_water']=False
    config['refinement_buffer']=0
    for name, allowed in (('flux_method', ('rusanov', 'knp')), ('time_integrator', ('euler', 'rk2'))):
        config[name] = data.get(name, config[name])
        if config[name] not in allowed: raise ValueError(f'{name} 参数无效')
    return config


def prepare(root, terrain, config):
    try: import numpy as np
    except ImportError: raise RuntimeError('洪水模拟需要 NumPy，请先安装 terrain-workbench/requirements.txt')
    from flood_solver import FloodSolver
    nz, nx = terrain['shape']
    fields = terrain['fields']
    def read(name):
        field = fields[name]
        path = (root / field['url'].lstrip('/')).resolve()
        if root.resolve() not in path.parents: raise ValueError('地图字段路径不正确')
        a = np.fromfile(path, dtype=field['dtype'])
        if a.size != nz*nx: raise ValueError(f'{name} 数据不完整')
        return a.astype(np.float64).reshape(nz, nx)
    full = terrain['full']
    bed = read('ground_height') if full else read('height')*terrain['scale']
    if not np.isfinite(bed).all(): raise ValueError('地图包含无效高程')
    types = read('terrain').astype(int)
    ocean = np.zeros_like(bed, dtype=bool)
    river = np.zeros_like(bed, dtype=bool)
    for code, name in terrain['terrain_ids'].items():
        if 'ocean' in name: ocean |= types == int(code)
        if name == 'river': river |= types == int(code)
    river_water = read('water_surface') if full else read('riverWaterLevel')*terrain['scale']
    native_bed = bed.copy()
    if config['initial_water']:
        if full:
            water = read('water_surface')
            if np.isinf(water).any(): raise ValueError('地图包含无效水面')
            depth = np.maximum(0, np.where(np.isfinite(water), water-bed, 0))
        else:
            water = read('riverWaterLevel')*terrain['scale']
            depth = np.zeros_like(bed)
            for code, name in terrain['terrain_ids'].items():
                mask = types == int(code)
                if 'ocean' in name:
                    depth[mask] = np.maximum(0, terrain['sea_level']-bed[mask])
                elif name in ('river', 'lake', 'wetland'):
                    depth[mask] = np.maximum(0, water[mask]-bed[mask])
            depth = np.maximum(depth, np.maximum(0, terrain['sea_level']-bed))
            if not np.isfinite(depth).all(): raise ValueError('快速预览初始水体数据不完整')
    else: depth = np.zeros_like(bed)
    factor = config.get('grid_factor',1)
    if config.get('roughness_mode','uniform')=='landcover':
        from surface_roughness import from_terrain
        native_n,roughness=from_terrain(root,terrain,config['manning'])
        from grid_coarsening import blocks
        manning=blocks(native_n,factor) if factor>1 else native_n
    else:
        manning=config['manning'];roughness=dict(mode='uniform',n=float(manning))
    if factor>1:
        from grid_coarsening import blocks
        from grid_coarsening import coarsen_water_state
        bed,depth=coarsen_water_state(bed,depth,factor,config['metres_per_block'])
    unit = config['metres_per_block']
    bed, depth = bed*unit, depth*unit
    dx = terrain['step']*unit*factor
    sea_edges = {name: a.copy() for name, a in
                 (('west', ocean[:, 0]), ('east', ocean[:, -1]),
                  ('north', ocean[0, :]), ('south', ocean[-1, :]))}
    if factor>1:
        sea_edges={name:mask.reshape(-1,factor).any(axis=1) for name,mask in sea_edges.items()}
    sea_level = terrain['sea_level']*unit
    solver = FloodSolver(bed, depth, dx, boundary=config['boundary'], manning=manning,
                         sea_level=sea_level, sea_edges=sea_edges, **numerical_options(config))
    from river_baseflow import plan
    groups, sources = plan(river, river_water, native_bed, factor,
                           config['baseflow_m3_s'] if config['baseflow'] else 0)
    solver.set_baseflow(sources)
    meta = dict(schema='hyra-flood-v1', solver='Cartesian 2D SWE / MUSCL / selectable Rusanov or KNP / SSP-RK2',
                numerics=solver.numerical_config(),
                shape=list(bed.shape), axis_order=['z', 'x'], dx_m=dx, coarsening_factor=factor,
                origin_blocks=[v+(factor-1)*terrain['step']/2 for v in terrain['origin']],
                step_blocks=terrain['step']*factor, terrain_id=terrain['id'], seed=terrain['config']['seed'],
                terrain_source='actual_ground' if full else 'continuous_preview', config=config,
                units=dict(bed='m', depth='m', u='m/s', v='m/s', maximum_depth='m', time='s'),
                frame_fields=['depth', 'u', 'v', 'maximum_depth'], frame_dtype='<f4',
                wet_threshold_m=.01, frames=[])
    meta['terrain_mode'] = terrain['config'].get('terrain_mode', 'custom')
    meta['roughness']=roughness
    meta['grid_aggregation']=dict(factor=factor,native_shape=list(terrain['shape']),
        bed='area_mean',initial_depth='area_mean_preserving_volume',
        sea_boundary='coarse_face_touches_native_ocean')
    if 'terrain_profile' in terrain: meta['terrain_profile'] = terrain['terrain_profile']
    meta['boundary_conditions'] = dict(mode=config['boundary'], sea_level_m=sea_level,
        sea_edges={name: mask.tolist() for name, mask in solver.sea_edges.items()},
        sea_face_count=sum(int(mask.sum()) for mask in solver.sea_edges.values()),
        other_edges='closed' if config['boundary'] == 'closed' else 'outflow_only')
    meta['river_baseflow'] = dict(enabled=config['baseflow'], requested_m3_s=config['baseflow_m3_s'],
        actual_m3_s=solver.baseflow_m3_s, river_native_cells=int(river.sum()),
        reach_count=len(groups), groups=groups, sources=sources,
        method='highest native water surface per connected river reach; area-weighted synthetic discharge')
    return solver, meta


def run(root, jobid, terrain, config, update, cancelled):
    import numpy as np
    solver, meta = prepare(root, terrain, config)
    generated_initial = solver.h.copy()
    if solver.baseflow_m3_s > 0 and config['warmup_s'] > 0:
        from river_baseflow import spin_up
        meta['initial_conditions'] = spin_up(solver, config['warmup_s'],
            config['infiltration_mm_h']/3600000, update, cancelled)
        solver.start_scenario()
    else:
        meta['initial_conditions'] = dict(duration_s=0, converged=False,
            reason='no_active_baseflow' if solver.baseflow_m3_s == 0 else 'warmup_disabled')
    folder = root / 'generated' / jobid / 'flood'
    folder.mkdir(parents=True, exist_ok=True)
    url = f'/generated/{jobid}/flood/'
    solver.bed.astype('<f4').tofile(folder / 'bed.f32')
    meta['bed_url'] = url+'bed.f32'
    source = (config['source_z']//meta['coarsening_factor'], config['source_x']//meta['coarsening_factor'])
    for index, end in enumerate(np.linspace(0, config['duration_s'], config['frames'])):
        if cancelled(): raise InterruptedError('模拟已取消')
        solver.advance(float(end), rain=config['rain_mm_h']/3600000,
                       rain_until=config['rain_duration_s'], infiltration=config['infiltration_mm_h']/3600000,
                       inflow=config['inflow_m3_s'], source=source, cancelled=cancelled)
        u, v = solver.velocity()
        filename = f'frame-{index:03d}.f32'
        np.stack((solver.h, u, v, solver.maximum)).astype('<f4').tofile(folder / filename)
        frame = dict(index=index, url=url+filename, **solver.summary())
        meta['frames'].append(frame)
        update(state='running', progress=round(index/(config['frames']-1)*100),
               message=f'已模拟 {end:.0f} / {config["duration_s"]:.0f} 秒', result=dict(meta, frames=list(meta['frames'])))
    # The complete archive remains usable without the running web server.
    (folder / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    np.savez_compressed(folder / 'final-state.npz', bed=solver.bed, depth=solver.h,
                        u=u, v=v, maximum_depth=solver.maximum, initial_depth=solver.initial,
                        generated_initial_depth=generated_initial, baseflow_rate_m_s=solver.baseflow_rate,
                        sea_level_m=meta['boundary_conditions']['sea_level_m'],
                        **{'sea_edge_'+name: mask for name, mask in solver.sea_edges.items()})
    archive = folder / 'flood-result.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(folder.iterdir()):
            if path != archive: z.write(path, path.name)
    meta['archive_url'] = url+archive.name
    return meta
