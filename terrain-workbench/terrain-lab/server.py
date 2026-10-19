"""Local terrain controls connected to the preserved FreeTerraForged generator."""
from __future__ import annotations
import argparse
import copy
import json
import math
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import flood_run
import generation_process

ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT.parent / 'freeterraforged-reproduction'
BASE = json.loads((ROOT / 'base-preset.json').read_text())
DEFAULTS = dict(seed='42', origin_x=512, origin_z=1536, span=1024, vertical_scale=0.8,
    continent_scale=1200, region_size=800, river_count=30, main_width=12, branch_width=8,
    river_depth=7, main_bank_width=28, branch_bank_width=20, branch_depth=5,
    lake_chance=0.8, wetland_chance=0.85, lake_distance=0.35,
    biome_size=225, moisture_bias=0.0, temperature_bias=0.05, mode='quick')
DEFAULTS.update(plains_weight=2.0, hills_weight=2.0, mountains_weight=2.5)
DEFAULTS['terrain_mode'] = 'custom'
SPECS = {
    'origin_x': (-30000000, 30000000, int), 'origin_z': (-30000000, 30000000, int),
    'span': (32, 4096, int), 'vertical_scale': (0.25, 2.0, float),
    'continent_scale': (500, 6000, int), 'region_size': (200, 2400, int),
    'river_count': (0, 30, int), 'main_width': (1, 20, int), 'branch_width': (1, 20, int),
    'river_depth': (1, 10, int), 'lake_chance': (0, 1, float), 'wetland_chance': (0, 1, float),
    'main_bank_width': (1, 50, int), 'branch_bank_width': (1, 50, int), 'branch_depth': (1, 10, int),
    'lake_distance': (0, 1, float), 'biome_size': (100, 1000, int),
    'moisture_bias': (-1, 1, float), 'temperature_bias': (-1, 1, float),
    'plains_weight': (0, 10, float), 'hills_weight': (0, 10, float), 'mountains_weight': (0, 10, float),
}
LOCK = threading.Lock()
JOBS = {}
FLOOD_JOBS = {}
MAP_SESSIONS = {}

def map_reply(session_id,entry):
    # The initial physical layers are immutable while the worker runs.
    return dict(id=session_id,**entry['scene'],process=dict(entry['process']))


def simulate_maps(session_id,entry):
    import map_sessions
    def update(**values):
        with entry['lock']:entry['process'].update(values)
    def cancelled():
        with entry['lock']:return entry['process'].get('cancel_requested',False)
    try:
        result=map_sessions.run(ROOT,session_id,entry['world'],update,cancelled)
        update(state='done',progress=100,message='洪水过程计算完成，可播放或拖动时间轴',**result)
    except InterruptedError as error:
        try:
            archive=map_sessions.save_process(ROOT,session_id,entry['world'],entry['process']['frames'])
            update(state='cancelled',message=str(error),archive_url=archive)
        except Exception as export_error:
            update(state='failed',message=f'计算已停止，但保存失败：{export_error}')
    except Exception as error:
        update(state='failed',message=str(error))


def current_terrain():
    pointer = ROOT / 'current.json'
    directory = json.loads(pointer.read_text())['directory'] if pointer.is_file() else 'example/raw'
    return describe(directory)

def simulate_flood(jobid, terrain, config):
    def update(**values):
        with LOCK: FLOOD_JOBS[jobid].update(values)
    def cancelled():
        with LOCK: return FLOOD_JOBS[jobid].get('cancel_requested', False)
    try:
        update(state='running', message='正在准备洪水计算网格…')
        result = flood_run.run(ROOT, jobid, terrain, config, update, cancelled)
        if cancelled(): raise InterruptedError('模拟已取消')
        update(state='done', progress=100, message='洪水模拟完成', result=result)
    except InterruptedError:
        update(state='cancelled', message='模拟已取消')
    except ImportError:
        update(state='failed', message='洪水模拟需要 NumPy，请先安装 terrain-workbench/requirements.txt')
    except Exception as error:
        update(state='failed', message=str(error))

def settings(data):
    result = dict(DEFAULTS)
    if not isinstance(data, dict): raise ValueError('参数格式不正确')
    terrain_mode = data.get('terrain_mode', 'custom')
    from terrain_modes import PROFILES
    if terrain_mode not in ('custom', *PROFILES): raise ValueError('请选择地形类型')
    result['terrain_mode'] = terrain_mode
    if terrain_mode != 'custom':
        data = dict(data, **PROFILES[terrain_mode]['parameters'])
    for key, (low, high, cast) in SPECS.items():
        value = float(data.get(key, result[key]))
        if not math.isfinite(value) or not low <= value <= high or cast is int and value != int(value):
            raise ValueError(f'{key} 超出允许范围')
        result[key] = cast(value)
    seed = str(data.get('seed', result['seed'])).strip()
    if not re.fullmatch(r'-?\d{1,19}', seed) or not -(1 << 63) <= int(seed) < (1 << 63):
        raise ValueError('种子需要是 64 位整数')
    result['seed'] = seed
    result['mode'] = data.get('mode', 'quick')
    if result['mode'] not in ('quick', 'surface'): raise ValueError('请选择生成方式')
    if result['mode'] == 'surface' and result['span'] > 1024:
        raise ValueError('完整地表边长不能超过1024方块；更大范围可使用快速预览')
    return result

def preset_for(config):
    p = copy.deepcopy(BASE)
    p['world']['continent']['continentScale'] = config['continent_scale']
    p['terrain']['general'].update(globalVerticalScale=config['vertical_scale'], terrainRegionSize=config['region_size'])
    for terrain in ('plains', 'hills', 'mountains'):
        p['terrain'][terrain]['weight'] = config[terrain + '_weight']
    p['rivers']['riverCount'] = config['river_count']
    p['rivers']['mainRivers'].update(bedWidth=config['main_width'], bedDepth=config['river_depth'], bankWidth=max(config['main_bank_width'], config['main_width']))
    p['rivers']['branchRivers'].update(bedWidth=config['branch_width'], bedDepth=config['branch_depth'], bankWidth=max(config['branch_bank_width'], config['branch_width']))
    p['rivers']['lakes'].update(chance=config['lake_chance'], maxStartDistance=config['lake_distance'])
    p['rivers']['wetlands']['chance'] = config['wetland_chance']
    p['climate']['biomeShape'].update(biomeSize=config['biome_size'], undergroundBiomeSize=config['biome_size'])
    p['climate']['moisture']['bias'] = config['moisture_bias']
    p['climate']['temperature']['bias'] = config['temperature_bias']
    from terrain_modes import configure_preset
    configure_preset(p, config.get('terrain_mode', 'custom'))
    return p

def describe(relative):
    folder = ROOT / relative
    grid = json.loads((folder / 'grid.json').read_text())
    result = dict(id=relative, shape=grid['shape'], origin=grid['origin_blocks'], step=grid['step_blocks'],
        scale=grid['terrain_scale_factor'], sea_level=grid['sea_level_blocks'], terrain_ids=grid['fields']['terrain']['values'],
        block_ids={}, biome_ids={}, full=False, fields={})
    for name in ('height', 'terrain', 'riverZone', 'riverMask', 'riverWaterLevel', 'temperature', 'moisture'):
        field = dict(grid['fields'][name]); field['url'] = f'/{relative}/{field["file"]}'
        result['fields'][name] = field
    sf = folder / 'surface/surface.json'
    if sf.is_file():
        surface = json.loads(sf.read_text())
        result.update(full=True, block_ids=surface['block_ids'], biome_ids=surface['biome_ids'])
        for name, value in surface['fields'].items():
            field = dict(value); field['url'] = f'/{relative}/surface/{field["file"]}'
            result['fields'][name] = field
    config_file = folder.parent / 'config.json'
    result['config'] = json.loads(config_file.read_text()) if config_file.is_file() else dict(DEFAULTS)
    report = folder.parent / 'terrain-profile.json'
    if report.is_file(): result['terrain_profile'] = json.loads(report.read_text())
    network = folder / 'river-network.json'
    if network.is_file(): result['river_network'] = json.loads(network.read_text(encoding='utf-8'))
    try:
        import river_inlets
        result['river_inlets']=river_inlets.annotate(ROOT,result)
        result['fields']['terrain']=result['river_inlets']['terrain_field']
    except ImportError:
        result['river_inlets']=dict(crossings=[],message='安装 NumPy 后可识别河道入口')
    return result

def generation_request(config):
    """One original terrain sample per block, shared with the flood grid."""
    request = dict(seed=config['seed'], preset_file='preset.json', natural_surface=True,
        generate_structures=False, unbounded_height=True,
        origin_x=config['origin_x'], origin_z=config['origin_z'],
        nx=config['span'], nz=config['span'], step_blocks=1,
        fields=['height','terrain','riverZone','riverMask','riverWaterLevel','temperature','moisture'],
        output='raw', export_datapack=config['mode'] == 'surface')
    if config['mode'] == 'surface':
        request['surface'] = {key: request[key] for key in ('origin_x', 'origin_z', 'nx', 'nz', 'step_blocks')}
    return request


def generate(jobid, config):
    directory = ROOT / 'generated' / jobid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    (directory / 'preset.json').write_text(json.dumps(preset_for(config), indent=2))
    request = generation_request(config)
    path = directory / 'request.json'; path.write_text(json.dumps(request, indent=2))
    job = JOBS[jobid]
    job.update(state='running', progress=0, progress_known=False, stage='dependencies',
               message='正在检查 Java 21 与首次构建依赖…', log_url=f'/generated/{jobid}/generation.log', log_tail='')
    pending = ''
    def report(chunk, elapsed, idle):
        nonlocal pending
        job['elapsed_s'] = round(elapsed)
        if not chunk:
            if idle >= 15 and job['stage'] == 'dependencies':
                job['message'] = f'正在准备首次构建依赖（{round(idle)} 秒无新输出）；源码转换可能持续数分钟，可取消或查看记录'
            return
        job['log_tail'] = (job['log_tail'] + chunk)[-6000:]
        pending = (pending + chunk)[-16000:]
        for line in pending.splitlines():
            if 'Uncaught exception in thread "main"' in line or 'Exception in thread "main"' in line:
                raise RuntimeError('生成器启动异常，已停止本次任务；请查看生成记录中的具体错误')
            tile = re.search(r'Original terrain tiles: (\d+)/(\d+)', line)
            surface = re.search(r'Original surface rows: (\d+)/(\d+)', line)
            if 'HYRA_STAGE:' in line:
                job.update(message=line.split('HYRA_STAGE:', 1)[1].strip(), progress_known=False)
            if 'Downloading ' in line:
                job.update(stage='dependencies', progress_known=False, message='首次启动正在下载构建工具与依赖，尚未开始生成地形…')
            if '> Task ' in line and not tile and not surface:
                job.update(stage='dependencies', progress_known=False, message='正在首次构建地形生成器；完成后开始生成地图…')
            if ':remapping sources' in line:
                job.update(stage='dependencies', progress_known=False, message='正在首次转换生成器依赖源码，可能需要数分钟…')
            if tile:
                job.update(stage='terrain', progress_known=True, progress=round(100*int(tile[1])/int(tile[2])), message='正在生成地形与河网…')
            if 'Preparing start region' in line:
                job.update(stage='surface', progress_known=False, progress=0, message='正在生成实际地表…')
            if surface:
                job.update(stage='surface', progress_known=True, progress=round(100*int(surface[1])/int(surface[2])), message='正在读取地面方块与群系…')
        pending = pending.rsplit('\n', 1)[-1]
    try:
        from terrain_modes import candidate_origins, assess, PROFILES
        target = config.get('terrain_mode', 'custom')
        origins = candidate_origins(config)
        requested_origin = [config['origin_x'], config['origin_z']]
        attempts = []
        import terrain_cache
        cached = terrain_cache.previews(ROOT,directory,preset_for(config),request) if target != 'custom' else {}
        started = time.monotonic()
        def export(candidate_request, log_name):
            nonlocal pending
            if job.get('cancel_requested'): raise InterruptedError('生成已取消，原地图保留')
            remaining = 1800 - (time.monotonic() - started)
            if remaining <= 0: raise TimeoutError('寻找目标地形超过 1800 秒，原地图保留；请更换种子或缩小范围')
            pending = ''
            path.write_text(json.dumps(candidate_request, indent=2))
            job['log_url'] = f'/generated/{jobid}/{log_name}'
            code = generation_process.run([sys.executable, '-u', str(GENERATOR / 'run.py'), 'export', '--request', str(path)],
                GENERATOR, directory / log_name, report, lambda: job.get('cancel_requested', False), total_timeout=remaining)
            if code:
                diagnostic = next((s.strip() for s in job['log_tail'].splitlines() if '生成新地图需要' in s or '--java-home' in s), None)
                raise RuntimeError(diagnostic or '地形生成失败；请打开生成记录查看具体构建或下载错误')
            if job.get('cancel_requested'): raise InterruptedError('生成已取消，原地图保留')
        for number, (x,z) in enumerate(origins,1):
            candidate = dict(request, origin_x=x, origin_z=z)
            if target != 'custom':
                candidate.pop('surface',None)
                candidate.update(export_datapack=False, output=f'candidate-{number}/raw')
                job.update(message=f'正在寻找{PROFILES[target]["label"]}区域（{number}/{len(origins)}），保留原种子…',
                           candidate_index=number, candidate_count=len(origins))
            source = cached.get((x,z))
            if source is not None:
                if job.get('cancel_requested'): raise InterruptedError('生成已取消，原地图保留')
                job.update(stage='cache',progress_known=False,message='正在重新检查已生成的区域…')
                terrain_cache.copy_preview(source,directory/candidate['output'])
                path.write_text(json.dumps(candidate,indent=2))
                log_name = f'candidate-{number}.log'
                (directory/log_name).write_text(f'Reused native preview: {source.relative_to(ROOT)}\n')
                job['log_url'] = f'/generated/{jobid}/{log_name}'
            else:
                export(candidate, f'candidate-{number}.log' if target != 'custom' else 'generation.log')
            if target == 'custom': break
            check = assess(directory/candidate['output'],target)
            if source is not None: check['reused_preview'] = str(source.relative_to(ROOT))
            attempts.append(dict(origin=[x,z], **check))
            (directory/'terrain-search.json').write_text(json.dumps(attempts,indent=2))
            if check['accepted']:
                config = dict(config,origin_x=x,origin_z=z)
                (directory/'config.json').write_text(json.dumps(config,indent=2))
                if config['mode'] == 'surface':
                    final_request = dict(request,origin_x=x,origin_z=z)
                    final_request['surface'] = {key: final_request[key] for key in ('origin_x','origin_z','nx','nz','step_blocks')}
                    export(final_request,'generation.log')
                    check = assess(directory/'raw',target)
                    if not check['accepted']:
                        raise RuntimeError('实际地表未达到所选地形标准，原地图保留；可使用快速预览或更换种子')
                else:
                    (directory/candidate['output']).replace(directory/'raw')
                    path.write_text(json.dumps(dict(candidate,output='raw'),indent=2))
                check.update(requested_origin=requested_origin, selected_origin=[x,z],
                             candidates_checked=number, seed=config['seed'])
                (directory/'terrain-profile.json').write_text(json.dumps(check,indent=2))
                break
        else:
            closest=min(attempts,key=lambda a:len(a.get('rejection_reasons',[])))
            reasons='；'.join(closest.get('rejection_reasons',[]))
            detail=f'。最接近的区域 {tuple(closest["origin"])} 未达标项：{reasons}' if reasons else ''
            raise RuntimeError(f'已检查 {len(origins)} 个区域，未找到符合“{PROFILES[target]["label"]}”标准的地图{detail}。原地图保留；相同种子与坐标会重复相同搜索，请更换种子或区域起点')
        result = describe(f'generated/{jobid}/raw')
        if config['mode'] == 'surface' and not result['full']: raise RuntimeError('地表生成尚未完成')
        if job.get('cancel_requested'): raise InterruptedError('生成已取消，原地图保留')
        job.update(stage='river_network', progress_known=False, message='正在用 RivGraph 检查原始河网拓扑…')
        import river_network
        result['river_network'] = river_network.audit_generation(ROOT, result, lambda: job.get('cancel_requested', False))
        if job.get('cancel_requested'): raise InterruptedError('生成已取消，原地图保留')
        pointer = ROOT / 'current.json.tmp'
        pointer.write_text(json.dumps({'directory': result['id']}))
        pointer.replace(ROOT / 'current.json')
        job.update(state='done', progress=100, progress_known=True, message='生成完成', result=result)
    except InterruptedError as error:
        job.update(state='cancelled', message=str(error), progress_known=False)
    except Exception as error:
        job.update(state='failed', message=str(error), progress=0)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs): super().__init__(*args, directory=str(ROOT), **kwargs)
    def json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/api/config':
                from terrain_modes import PROFILES
                return self.json({'defaults': DEFAULTS, 'terrain_profiles': PROFILES, 'flood_defaults': flood_run.DEFAULTS})
            if path == '/api/current':
                return self.json(current_terrain())
            if path=='/api/maps/schema':
                from multilayer import DATA_DICTIONARY
                return self.json(DATA_DICTIONARY)
            if path.startswith('/api/maps/'):
                parts=path.strip('/').split('/');sid=parts[2]
                with LOCK:entry=MAP_SESSIONS.get(sid)
                if entry is None:return self.json({'error':'未找到这次多层场景'},404)
                with entry['lock']:
                    if len(parts)==4 and parts[3]=='status':return self.json(dict(entry['process']))
                    if len(parts)==3:return self.json(map_reply(sid,entry))
                return self.json({'error':'未找到地图操作'},404)
            if path.startswith('/api/flood-jobs/'):
                with LOCK:
                    job = FLOOD_JOBS.get(path.rsplit('/', 1)[1])
                    snapshot = dict(job) if job else None
                return self.json(snapshot if snapshot else {'error': '未找到这次洪水模拟'}, 200 if snapshot else 404)
            if path.startswith('/api/jobs/'):
                job = JOBS.get(path.rsplit('/', 1)[1])
                return self.json(job if job else {'error': '未找到这次生成'}, 200 if job else 404)
            return super().do_GET()
        except (ValueError,KeyError) as error: return self.json({'error': str(error)}, 400)
        except Exception as error: return self.json({'error': str(error)}, 500)
    def do_POST(self):
        # Only same-origin/local requests may trigger the desktop generator.
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.json({'error': '请从本地页面生成'}, 403)
        try:
            if self.path=='/api/maps' or self.path.startswith('/api/maps/'):
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=8*1024*1024:raise ValueError('地图数据过长或为空')
                data=json.loads(self.rfile.read(length))
                if not isinstance(data,dict):raise ValueError('地图操作参数需要是对象')
                import map_sessions
                if self.path=='/api/maps':
                    world=map_sessions.create(ROOT,current_terrain(),data)
                    sid='maps-'+uuid.uuid4().hex[:12]
                    entry=dict(world=world,scene=dict(map_sessions.browser_scene(ROOT,sid,world),scenario_config=data),lock=threading.Lock(),
                        process=dict(state='ready',progress=0,message='地图已建立，点击开始模拟',frames=[]))
                    with LOCK:
                        if len(MAP_SESSIONS)>=3:
                            old=next((key for key,value in MAP_SESSIONS.items()
                                if value['process']['state'] not in ('queued','running')),None)
                            if old is None:return self.json({'error':'运行中的地图过多，请先取消计算'},409)
                            MAP_SESSIONS.pop(old)
                        MAP_SESSIONS[sid]=entry
                    return self.json(map_reply(sid,entry),201)
                parts=self.path.strip('/').split('/')
                if len(parts)!=4:return self.json({'error':'未找到地图操作'},404)
                sid,action=parts[2:]
                with LOCK:entry=MAP_SESSIONS.get(sid)
                if entry is None:return self.json({'error':'未找到这次多层场景'},404)
                if action=='run':
                    with LOCK:
                        running=next((key for key,value in MAP_SESSIONS.items()
                            if value['process']['state'] in ('queued','running')),None)
                        if running is not None:
                            return self.json({'error':'已有多层洪水过程正在计算，已恢复该过程，可等待或取消',
                                              'process_id':running},409)
                        with entry['lock']:
                            if entry['process']['state']!='ready':return self.json({'error':'请重新建立场景后模拟'},409)
                            entry['process'].update(state='queued',message='正在开始洪水计算…')
                    threading.Thread(target=simulate_maps,args=(sid,entry),daemon=True).start()
                    return self.json({'id':sid},202)
                if action=='cancel':
                    with entry['lock']:
                        if entry['process']['state'] in ('queued','running'):
                            entry['process'].update(cancel_requested=True,message='正在停止计算…')
                    return self.json({'id':sid})
                return self.json({'error':'未找到地图操作'},404)
            if self.path.startswith('/api/jobs/') and self.path.endswith('/cancel'):
                jobid = self.path.split('/')[-2]
                with LOCK:
                    job = JOBS.get(jobid)
                    if job is None: return self.json({'error': '未找到这次生成'}, 404)
                    if job['state'] in ('queued', 'running'):
                        job.update(cancel_requested=True, message='正在停止生成…')
                return self.json({'id': jobid})
            if self.path.startswith('/api/flood-jobs/') and self.path.endswith('/cancel'):
                jobid = self.path.split('/')[-2]
                with LOCK:
                    job = FLOOD_JOBS.get(jobid)
                    if job is None: return self.json({'error': '未找到这次洪水模拟'}, 404)
                    if job['state'] in ('queued', 'running'):
                        job.update(cancel_requested=True, message='正在停止计算…')
                return self.json({'id': jobid})
            if self.path not in ('/api/generate', '/api/flood'):
                return self.json({'error': '未找到操作'}, 404)
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 16384: raise ValueError('参数过长或为空')
            data = json.loads(self.rfile.read(length))
            if self.path == '/api/flood':
                terrain = current_terrain()
                # Bind the submitted scenario to the map actually shown in the browser.
                if not isinstance(data, dict) or data.get('terrain_id') != terrain['id']:
                    return self.json({'error': '当前地图已更新，请读取最新地图后再模拟'}, 409)
                config = flood_run.settings(data, terrain['shape'])
                with LOCK:
                    if any(v['state'] in ('queued', 'running') for v in FLOOD_JOBS.values()):
                        return self.json({'error': '已有洪水模拟正在计算，请等待或取消'}, 409)
                    # Keep bounded in-memory status; exported files are preserved.
                    if len(FLOOD_JOBS) >= 10: FLOOD_JOBS.pop(next(iter(FLOOD_JOBS)))
                    jobid = 'flood-' + uuid.uuid4().hex[:12]
                    FLOOD_JOBS[jobid] = dict(id=jobid, state='queued', progress=0, message='正在开始模拟…')
                threading.Thread(target=simulate_flood, args=(jobid, terrain, config), daemon=True).start()
                return self.json({'id': jobid}, 202)
            config = settings(data)
            with LOCK:
                active = next((v for v in JOBS.values() if v['state'] in ('queued', 'running')), None)
                if active: return self.json({'error': '已有地图正在生成，请等待完成'}, 409)
                jobid = uuid.uuid4().hex[:12]
                JOBS[jobid] = dict(id=jobid, state='queued', progress=0, message='正在开始生成…')
            threading.Thread(target=generate, args=(jobid, config), daemon=True).start()
            return self.json({'id': jobid}, 202)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            return self.json({'error': str(error)}, 400)
        except Exception as error:
            return self.json({'error': str(error)}, 500)
    def log_message(self, format, *args):
        if '/api/jobs/' not in self.path: super().log_message(format, *args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    print(f'Local: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
