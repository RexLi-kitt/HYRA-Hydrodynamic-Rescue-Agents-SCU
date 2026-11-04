"""Generate native 10 m SWE frames and interval-wide service eligibility."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'generated' / 'flood-dqn-v1'
TERRAIN = ROOT.parent / 'terrain-workbench' / 'terrain-lab'
sys.path.insert(0, str(TERRAIN))
os.environ['HYRA_FLOOD_BACKEND'] = 'numpy'
from flood_solver import FloodSolver
from large_flood_routes.dataset import parent_parameters

SPLITS = {'train': range(21001, 21025), 'validation': range(31001, 31007),
          'test': range(41001, 41051)}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, data):
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


class IntervalSolver(FloodSolver):
    def allowed(self):
        u, v = self.velocity()
        return (self.h <= .08) & (np.hypot(u, v) <= .3)

    def step(self, dt, **kw):
        super().step(dt, **kw)
        self.interval_safe &= self.allowed()


def make_map(seed):
    folder = DATA / 'maps'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{seed}.npz'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if digest(path) != record['sha256'] or record['generator'] != digest(__file__):
            raise ValueError(f'map hash/version mismatch: {seed}')
        return record
    terrain = parent_parameters(seed)
    hydro_seed = seed + 100000
    flow = parent_parameters(hydro_seed)
    z, x = (np.indices((100, 100)) + .5) * 10
    river = (500 + 5 * (terrain['center_cells'] - 120)
             + 5 * terrain['meander_cells'] * np.sin(z / (5 * terrain['wavelength_cells']) + terrain['phase']))
    bed = (terrain['longitudinal_slope'] * (995-z) + terrain['valley_curvature'] * (x-river)**2
           + terrain['roughness_amplitude_m'] * np.sin(x/100) * np.sin(z/150))
    bed -= bed.min()
    solver = IntervalSolver(bed, np.maximum(0, .18-bed), 10., boundary='closed',
                            max_timestep=2., spatial_order=1, time_integrator='euler')
    peak, q = flow['peak_time_s'], flow['peak_flow_m3_s']
    source = [[97, int(river[97, 0] // 10) + offset] for offset in (-1, 0, 1)]
    schedule = [[0, 1.], [peak, q], [peak+600, .4*q], [1500, flow['final_flow_m3_s']]]
    solver.set_source_schedule([dict(cells=source, hydrograph=schedule)])
    fields = np.empty((301, 3, 100, 100), dtype='<f4')
    safe = np.empty((300, 100, 100), dtype=bool)
    for i in range(301):
        solver.interval_safe = solver.allowed()
        solver.advance(i*5.)
        fields[i] = np.stack((solver.h, *solver.velocity()))
        if i:
            safe[i-1] = solver.interval_safe
    if not np.isfinite(fields).all() or fields[:, 0].min() < -1e-7:
        raise ValueError('invalid SWE output')
    tmp = path.with_suffix('.partial.npz')
    np.savez_compressed(tmp, fields=fields, safe=safe, bed=bed, time_s=np.arange(301)*5)
    tmp.replace(path)
    record = dict(terrain_seed=seed, hydro_seed=hydro_seed, sha256=digest(path),
                  generator=digest(__file__), solver=digest(TERRAIN/'flood_solver.py'),
                  source=source, hydrograph=schedule, wet_fraction=float((fields[:, 0]>=.05).mean()))
    save_json(receipt, record)
    return record


def prepare(split):
    if split == 'test' and not (DATA/'seal.json').is_file():
        raise ValueError('test batch requires frozen model')
    with ProcessPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(make_map, SPLITS[split]))
    save_json(DATA/f'{split}-maps.json', records)


@lru_cache(maxsize=6)
def load_map(seed):
    with np.load(DATA/'maps'/f'{seed}.npz') as f:
        return f['fields'], f['safe']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('split', choices=SPLITS)
    prepare(parser.parse_args().split)
