"""Reuse native previews only when generation settings and grid metadata match."""
import json
import shutil
from pathlib import Path

REVISION = 'c5d322f5e44539ca6d3a6059c5de440af9abcfe6'


def previews(root, directory, preset, request):
    result = {}
    for job in sorted((Path(root)/'generated').iterdir()):
        if not job.is_dir() or job == directory: continue
        try:
            if json.loads((job/'preset.json').read_text()) != preset: continue
            if str(json.loads((job/'config.json').read_text())['seed']) != request['seed']: continue
        except (OSError,ValueError,KeyError): continue
        for folder in [job/'raw', *sorted(job.glob('candidate-*/raw'))]:
            try:
                grid = json.loads((folder/'grid.json').read_text())
                if (str(grid['seed']) != request['seed'] or grid['shape'] != [request['nz'],request['nx']]
                    or grid['step_blocks'] != request['step_blocks'] or grid['upstream_revision'] != REVISION
                    or grid.get('height_policy') != 'simulation_without_game_ceiling'): continue
                fields = grid['fields']
                if not {'height','terrain','riverWaterLevel','riverZone'} <= fields.keys(): continue
                count = request['nx']*request['nz']
                for field in fields.values():
                    name = field['file']; dtype = field['dtype']
                    if Path(name).name != name or dtype not in ('<f4','<i4','|u1'):
                        raise ValueError('Invalid native field')
                    if (folder/name).stat().st_size != count*(1 if dtype == '|u1' else 4):
                        raise ValueError('Incomplete native field')
                result.setdefault(tuple(grid['origin_blocks']),folder)
            except (OSError,ValueError,KeyError,TypeError): continue
    return result


def copy_preview(source, target):
    """Copy native fields, excluding URL-bearing annotations and actual surfaces."""
    grid = json.loads((source/'grid.json').read_text())
    target.mkdir(parents=True,exist_ok=True)
    for name in ['grid.json', *[field['file'] for field in grid['fields'].values()]]:
        shutil.copy2(source/name,target/name)
    if (source/'preset.json').is_file(): shutil.copy2(source/'preset.json',target/'preset.json')
