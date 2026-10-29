"""Create a synthetic 1.2 km development event with the terrain workbench solver."""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
from .simulate import peak_memory_bytes

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ROOT.parent / 'terrain-workbench' / 'terrain-lab'
sys.path.insert(0, str(TERRAIN))
from flood_solver import FloodSolver


def generate(path, seed=0):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    z, x = np.indices((240, 240), dtype=float)
    # A sloping valley with a meandering river corridor, in metres.
    river = 115 + 22 * np.sin(z / 40 + seed * .4)
    bed = 0.004 * (239-z)*5 + 0.000035 * ((x-river)*5)**2
    bed += 0.12 * np.sin(x/20) * np.sin(z/30)
    bed -= bed.min()
    initial = np.maximum(0., .18-bed)
    solver = FloodSolver(bed, initial, 5., boundary='closed', max_timestep=2., spatial_order=1, time_integrator='euler')
    cells = [[235, int(river[235, 0]) + offset] for offset in range(-2, 3)]
    schedule = [[0, 1.], [300, 5.], [900, 2.], [1500, .3], [1800, 0.]]
    solver.set_source_schedule([dict(cells=cells, hydrograph=schedule)])
    frames, wet, first, last = [], [], None, None
    start = time.perf_counter()
    try:
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as out:
            for t in range(0, 1801, 5):
                solver.advance(float(t))
                u, v = solver.velocity()
                frame = np.stack((solver.h, u, v, solver.maximum)).astype('<f4')
                name = f'frame-{t:04d}.bin'
                out.writestr(name, frame.tobytes())
                frames.append(dict(index=len(frames), time_s=t, url=name))
                labels = solver.h >= .05
                wet.append(int(labels.sum()))
                if first is None: first = labels.copy()
                last = labels.copy()
                if t % 300 == 0: print(f'calculated {t}/1800 s', flush=True)
            grid = dict(shape=[240, 240], dx_m=5., axis_order=['z','x'])
            process = dict(grid=grid, frame_fields=['depth','u','v','maximum_depth'], frame_dtype='<f4', frames=frames,
                           units=dict(time='s', depth='m', velocity='m/s', distance='m'))
            out.writestr('process.json', json.dumps(process))
            out.writestr('bed.npy', _npy(bed))
            out.writestr('case.json', json.dumps(dict(schema='hyra-large-flood-development-v1',
                scope='synthetic, uncalibrated development event', seed=seed, source_cells=cells, source_schedule_m3_s=schedule,
                solver_sha256=hashlib.sha256((TERRAIN/'flood_solver.py').read_bytes()).hexdigest(),
                numerical_config=solver.numerical_config(), wet_cells=wet,
                net_changed_cells=int(np.count_nonzero(first != last)), elapsed_s=time.perf_counter()-start,
                peak_process_memory_bytes=peak_memory_bytes())))
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    print(json.dumps(dict(path=str(path), size_bytes=path.stat().st_size, elapsed_s=time.perf_counter()-start, wet_min=min(wet), wet_max=max(wet), peak_process_memory_bytes=peak_memory_bytes()), ensure_ascii=False), flush=True)


def _npy(array):
    import io
    result = io.BytesIO()
    np.save(result, array, allow_pickle=False)
    return result.getvalue()

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    args=p.parse_args()
    generate(args.output, args.seed)
