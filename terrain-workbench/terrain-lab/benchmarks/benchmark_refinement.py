"""Controlled refinement experiment; production defaults are never changed."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flood_solver import FloodSolver
from refined_solver import RefinedFloodSolver
from refined_mesh import inland_mask


class ExperimentalSolver(RefinedFloodSolver):
    factor = .5

    def timestep(self, *args, **kwargs):
        original = self.cfl
        try:
            self.cfl = original * self.factor / .5
            return super().timestep(*args, **kwargs)
        finally:
            self.cfl = original


def build(case, n, buffer, factor):
    z, x = np.indices((n, n))
    classes = np.zeros((n, n), int)
    classes[:, n//3:n//3+2] = 1
    classes[n//3:2*n//3, n//3:2*n//3] = 2
    mask = inland_mask(classes, {'0': 'land', '1': 'river', '2': 'lake'}, buffer=buffer)[1]
    bed = np.zeros((n, n))
    h = np.where(x < n//2, 2., .001)
    boundary = 'closed'
    if case == 'lake':
        bed = .04*x + .02*z
        h = np.maximum(1.2-bed, 0)
    elif case == 'wave':
        boundary = 'periodic'
        h = 1 + .05*np.sin(2*np.pi*x/n)
    base = FloodSolver(bed, h, 1., boundary=boundary, manning=0.)
    s = ExperimentalSolver.from_solver(base, mask)
    s.factor = factor
    if case == 'wave':
        s.leaf['h'] = 1 + .05*np.sin(2*np.pi*s.mesh.x/n)
        s.leaf['mx'] = s.leaf['h']*np.sqrt(9.81)*.05*np.sin(2*np.pi*s.mesh.x/n)
        s._sync()
        s.start_scenario()
    return s


def run(case, n, buffer, factor, duration):
    s = build(case, n, buffer, factor)
    initial = s.leaf['h'].copy()
    started = time.perf_counter()
    failure = None
    try:
        s.advance(duration)
    except RuntimeError as error:
        failure = str(error)
    elapsed = time.perf_counter()-started
    summary = s.summary()
    row = dict(case=case, buffer=buffer, factor=factor, cells=s.mesh.count,
               refined_parents=int(s.mesh.mask.sum()), steps=s.steps,
               simulated_s=s.time, elapsed_s=elapsed, failure=failure,
               min_depth=float(s.leaf['h'].min()), max_depth=float(s.leaf['h'].max()),
               max_speed=summary['max_speed_m_s'],
               relative_mass_error=abs(summary['balance']['residual_m3'])/max(float((initial*s.mesh.area).sum()), 1e-30),
               max_depth_change=float(np.max(np.abs(s.leaf['h']-initial))),
               **s.diagnostics)
    return s, row


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--size', type=int, default=48)
    p.add_argument('--duration', type=float, default=1.)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    rows = []
    run('dam', 12, 1, .5, .05)  # warm up
    for case in ('lake', 'dam', 'wave'):
        broad, _ = run(case, args.size, 4, .25, args.duration)
        for buffer in (0, 1, 2, 4):
            reference, _ = run(case, args.size, buffer, .25, args.duration)
            timings = {factor: [] for factor in (.5, .75, 1.)}
            results = {}
            for repeat in range(args.repeats):
                factors = (.5, .75, 1.) if repeat % 2 == 0 else (1., .75, .5)
                for factor in factors:
                    s, row = run(case, args.size, buffer, factor, args.duration)
                    timings[factor].append(row['elapsed_s'])
                    row['depth_mae_vs_small_dt'] = float(np.sum(np.abs(s.leaf['h']-reference.leaf['h'])*s.mesh.area)/(args.size**2))
                    row['depth_max_error_vs_small_dt'] = float(np.max(np.abs(s.leaf['h']-reference.leaf['h'])))
                    row['projected_depth_mae_vs_buffer4_small_dt'] = float(np.mean(np.abs(s.h-broad.h)))
                    results[factor] = row
            for factor, row in results.items():
                row['timings_s'] = timings[factor]
                row['elapsed_s'] = statistics.median(timings[factor])
                rows.append(row)
            print(f'{case} buffer={buffer} completed', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(numpy=np.__version__, size=args.size, duration=args.duration,
        repeats=args.repeats, reference_factor=.25, rows=rows), indent=2), encoding='utf-8')
    print(str(args.output), flush=True)
