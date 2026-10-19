"""Generation-time RivGraph audit and a separate check of the effective flood grid.

The graph describes channel geometry, not a hydraulic validity certificate.
Never condition the source DEM or discard disconnected channels here.
"""
from __future__ import annotations
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

SCHEMA = 'hyra-river-network-v1'
RIVGRAPH_REVISION = 'f8745a645d822377c75e16f97ae995da32587695'


def components(mask, wall_x=None, wall_y=None):
    import numpy as np
    mask = np.asarray(mask, bool)
    nz, nx = mask.shape
    labels = np.zeros(mask.shape, np.int32)
    sizes = []
    for z, x in zip(*np.nonzero(mask)):
        if labels[z, x]:
            continue
        label = len(sizes) + 1
        labels[z, x] = label
        queue = deque([(int(z), int(x))]); size = 0
        while queue:
            zz, xx = queue.popleft(); size += 1
            for dz, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                z1, x1 = zz + dz, xx + dx
                if not (0 <= z1 < nz and 0 <= x1 < nx) or not mask[z1, x1] or labels[z1, x1]:
                    continue
                if dz == 0 and wall_x is not None and wall_x[zz, max(xx, x1)]:
                    continue
                if dx == 0 and wall_y is not None and wall_y[max(zz, z1), xx]:
                    continue
                labels[z1, x1] = label; queue.append((z1, x1))
        sizes.append(size)
    return labels, sizes


def compare_grids(native_mask, effective_mask, factor, wall_x=None, wall_y=None):
    """Compare channel membership, including new joins and losses; no water-state assumptions."""
    import numpy as np
    native, native_sizes = components(native_mask)
    effective, effective_sizes = components(effective_mask, wall_x, wall_y)
    z, x = np.nonzero(native_mask)
    a = native[z, x]; b = effective[z // factor, x // factor]
    pairs = np.unique(np.column_stack((a, b)), axis=0)
    forward = {}; backward = {}
    for old, new in pairs:
        forward.setdefault(int(old), set()).add(int(new))
        if new:
            backward.setdefault(int(new), set()).add(int(old))
    lost = int(np.count_nonzero(b == 0))
    split = sum(len(values - {0}) > 1 for values in forward.values())
    merged = sum(len(values) > 1 for values in backward.values())
    warnings = []
    if lost:
        warnings.append(f'{lost} 个原始河道格在当前河道分类或建筑占地下消失')
    if split:
        warnings.append(f'{split} 片原始河道在当前网格或阻水边作用下分裂')
    if merged:
        warnings.append(f'{merged} 片当前河道合并了原本分离的河道，需检查是否产生捷径')
    return dict(native_components=len(native_sizes), effective_components=len(effective_sizes),
                lost_native_channel_cells=lost, split_native_components=int(split),
                merged_effective_components=int(merged), warnings=warnings,
                scope='channel-mask topology through unsealed faces; not a hydraulic convergence test')


def read_native(root, terrain):
    import numpy as np
    fields = terrain['fields']
    def read(name):
        f = fields[name]
        return np.fromfile(Path(root) / f['url'].lstrip('/'), dtype=f['dtype']).reshape(terrain['shape'])
    codes = [int(k) for k, v in terrain['terrain_ids'].items() if v == 'river']
    mask = np.isin(read('terrain'), codes)
    bed = read('ground_height' if terrain['full'] else 'height')
    water = read('water_surface' if terrain['full'] else 'riverWaterLevel')
    scale = 1 if terrain['full'] else terrain['scale']
    return mask, bed * scale, water * scale


def audit_generation(root, terrain, cancelled=lambda: False):
    """Run the pinned engine in isolation before publishing a newly generated map."""
    import numpy as np
    root = Path(root); folder = root / terrain['id']
    mask, bed, water = read_native(root, terrain)
    if not mask.any():
        report = dict(schema=SCHEMA, status='no_river', message='无河流，河网检查不适用', warnings=[])
    else:
        local = Path(__file__).parent / '.runtime/rivgraph-env'
        executable = os.environ.get('HYRA_RIVGRAPH_PYTHON') or str(local / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
        if not Path(executable).is_file():
            raise RuntimeError('河网检查需要独立 RivGraph 环境；请按 docs/river-network.md 安装，或设置 HYRA_RIVGRAPH_PYTHON')
        with tempfile.TemporaryDirectory(dir=folder, prefix='river-audit-') as temporary:
            temporary = Path(temporary)
            np.savez_compressed(temporary / 'input.npz', mask=mask, bed=bed, water=water)
            command = [executable, str(Path(__file__).with_name('river_network_worker.py')),
                       str(temporary / 'input.npz'), str(temporary / 'result.json'), str(terrain['step'])]
            with (temporary / 'worker.log').open('w', encoding='utf-8') as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                started = time.monotonic()
                try:
                    while process.poll() is None:
                        if cancelled():
                            raise InterruptedError('生成已取消，原地图保留')
                        if time.monotonic() - started > 180:
                            raise RuntimeError('RivGraph 河网检查超过180秒，原地图保留；请缩小地图范围')
                        time.sleep(.1)
                finally:
                    if process.poll() is None:
                        process.kill(); process.wait()
            if process.returncode:
                raise RuntimeError('RivGraph 河网检查失败，原地图保留：' + (temporary / 'worker.log').read_text(encoding='utf-8')[-1800:])
            payload = json.loads((temporary / 'result.json').read_text(encoding='utf-8'))
            graph = payload.pop('graph')
            graph_file = folder / 'river-network-graph.json'
            graph_file.write_text(json.dumps(graph, ensure_ascii=False), encoding='utf-8')
            report = dict(schema=SCHEMA, **payload,
                          graph_url=f'/{terrain["id"]}/river-network-graph.json',
                          requested_upstream_revision=RIVGRAPH_REVISION)
    report.update(shape=terrain['shape'], step_blocks=terrain['step'],
                  scope='native channel geometry; no DEM edits; flood-grid validity is checked separately')
    target = folder / 'river-network.json'
    tmp = target.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'); tmp.replace(target)
    return report


def audit_world(root, terrain, world):
    import numpy as np
    native, bed, water = read_native(root, terrain)
    codes = [int(k) for k, v in terrain['terrain_ids'].items() if v == 'river']
    effective = np.isin(world.base.terrain_class, codes) & ~world.base.solid
    from people import barrier_masks
    edges = [(e['axis'], e['z'], e['x']) for obj in world.structures.compiled
             if obj['kind'] != 'building' and world.structures.active(obj, 0.) for e in obj['edges']]
    wx, wy = barrier_masks(world.base.shape, world.base.wall_x, world.base.wall_y, edges)
    factor = world.config['grid_factor']
    report = compare_grids(native, effective, factor, wx, wy)
    # At native river locations, compare the *effective* bed against the original
    # stage. This flags potential channel burial; it does not prohibit overtopping.
    z, x = np.nonzero(native & np.isfinite(water) & (water > bed + .01))
    raised = world.base.z[z // factor, x // factor] >= water[z, x] * world.config['metres_per_block']
    report['native_wet_channel_cells_bed_above_original_stage'] = int(raised.sum())
    if raised.any():
        report['warnings'].append(f'{int(raised.sum())} 个原始有水河道格映射后的河床不低于原水面，需检查河槽是否被平均抬高')
    report.update(schema=SCHEMA, factor=factor, dx_m=world.base.dx,
                  status='review' if report['warnings'] else 'no_component_change_detected',
                  native_audit=terrain.get('river_network', {}).get('status', 'not_available'))
    return report
