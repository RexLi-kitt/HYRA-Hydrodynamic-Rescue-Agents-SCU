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
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
GENERATOR = ROOT.parent / 'freeterraforged-reproduction'
BASE = json.loads((ROOT / 'base-preset.json').read_text())
DEFAULTS = dict(seed='42', origin_x=512, origin_z=1536, span=1024, vertical_scale=0.8,
    continent_scale=1200, region_size=800, river_count=30, main_width=12, branch_width=8,
    river_depth=7, main_bank_width=28, branch_bank_width=20, branch_depth=5,
    lake_chance=0.8, wetland_chance=0.85, lake_distance=0.35,
    biome_size=225, moisture_bias=0.0, temperature_bias=0.05, mode='quick')
DEFAULTS.update(plains_weight=2.0, hills_weight=2.0, mountains_weight=2.5)
SPECS = {
    'origin_x': (-30000000, 30000000, int), 'origin_z': (-30000000, 30000000, int),
    'span': (512, 4096, int), 'vertical_scale': (0.25, 2.0, float),
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

def settings(data):
    result = dict(DEFAULTS)
    if not isinstance(data, dict): raise ValueError('参数格式不正确')
    for key, (low, high, cast) in SPECS.items():
        value = float(data.get(key, result[key]))
        if not math.isfinite(value) or not low <= value <= high or cast is int and value != int(value):
            raise ValueError(f'{key} 超出允许范围')
        result[key] = cast(value)
    if result['span'] not in (512, 1024, 2048, 4096): raise ValueError('请选择预设地图范围')
    seed = str(data.get('seed', result['seed'])).strip()
    if not re.fullmatch(r'-?\d{1,19}', seed) or not -(1 << 63) <= int(seed) < (1 << 63):
        raise ValueError('种子需要是 64 位整数')
    result['seed'] = seed
    result['mode'] = data.get('mode', 'quick')
    if result['mode'] not in ('quick', 'surface'): raise ValueError('请选择生成方式')
    if result['mode'] == 'surface' and result['span'] > 1024:
        raise ValueError('完整地表请使用 512 或 1024 的地图范围；更大范围可先快速预览')
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
    return result

def generate(jobid, config):
    directory = ROOT / 'generated' / jobid
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'config.json').write_text(json.dumps(config, indent=2))
    (directory / 'preset.json').write_text(json.dumps(preset_for(config), indent=2))
    request = dict(seed=config['seed'], preset_file='preset.json', natural_surface=True, generate_structures=False, unbounded_height=True,
        origin_x=config['origin_x'], origin_z=config['origin_z'], nx=256, nz=256,
        step_blocks=config['span'] // 256, output='raw', export_datapack=config['mode'] == 'surface')
    if config['mode'] == 'surface':
        request['surface'] = {key: request[key] for key in ('origin_x', 'origin_z', 'nx', 'nz', 'step_blocks')}
    path = directory / 'request.json'; path.write_text(json.dumps(request, indent=2))
    job = JOBS[jobid]
    job.update(state='running', progress=0, message='正在准备地形生成器…')
    try:
        with (directory / 'generation.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(GENERATOR / 'run.py'), 'export', '--request', str(path)],
                cwd=GENERATOR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                log.write(line); log.flush()
                tile = re.search(r'Original terrain tiles: (\d+)/(\d+)', line)
                surface = re.search(r'Original surface rows: (\d+)/(\d+)', line)
                if tile:
                    job.update(progress=round(100 * int(tile[1]) / int(tile[2])), message='正在生成地形与河网…')
                if 'Preparing start region' in line:
                    job.update(progress=0, message='正在生成实际地表…')
                if surface:
                    job.update(progress=round(100 * int(surface[1]) / int(surface[2])), message='正在读取地面方块与群系…')
            code = process.wait()
        if code: raise RuntimeError('生成失败；请查看该次生成记录')
        result = describe(f'generated/{jobid}/raw')
        if config['mode'] == 'surface' and not result['full']: raise RuntimeError('地表生成尚未完成')
        job.update(state='done', progress=100, message='生成完成', result=result)
        (ROOT / 'current.json').write_text(json.dumps({'directory': result['id']}))
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
            if path == '/api/config': return self.json({'defaults': DEFAULTS})
            if path == '/api/current':
                pointer = ROOT / 'current.json'
                directory = json.loads(pointer.read_text())['directory'] if pointer.is_file() else 'example/raw'
                return self.json(describe(directory))
            if path.startswith('/api/jobs/'):
                job = JOBS.get(path.rsplit('/', 1)[1])
                return self.json(job if job else {'error': '未找到这次生成'}, 200 if job else 404)
            return super().do_GET()
        except Exception as error: return self.json({'error': str(error)}, 500)
    def do_POST(self):
        if self.path != '/api/generate': return self.json({'error': '未找到操作'}, 404)
        # Only same-origin/local requests may trigger the desktop generator.
        origin = self.headers.get('Origin')
        if origin and urlparse(origin).netloc != self.headers.get('Host'):
            return self.json({'error': '请从本地页面生成'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 16384: raise ValueError('参数过长或为空')
            config = settings(json.loads(self.rfile.read(length)))
            with LOCK:
                active = next((v for v in JOBS.values() if v['state'] in ('queued', 'running')), None)
                if active: return self.json({'error': '已有地图正在生成，请等待完成'}, 409)
                jobid = uuid.uuid4().hex[:12]
                JOBS[jobid] = dict(id=jobid, state='queued', progress=0, message='正在开始生成…')
            threading.Thread(target=generate, args=(jobid, config), daemon=True).start()
            return self.json({'id': jobid}, 202)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            return self.json({'error': str(error)}, 400)
    def log_message(self, format, *args):
        if '/api/jobs/' not in self.path: super().log_message(format, *args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    print(f'Local: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
