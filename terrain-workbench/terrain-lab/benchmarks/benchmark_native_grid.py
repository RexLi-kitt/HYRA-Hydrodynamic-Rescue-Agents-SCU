"""Time identical SWE steps; optionally verify against a supplied pre-change solver."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from flood_solver import FloodSolver


def measure(cls,n,repeats,steps):
    z,x=np.mgrid[:n,:n];bed=.002*x+.001*z;depth=np.maximum(1.5-bed,0)
    timings=[]
    for _ in range(repeats):
        solver=cls(bed,depth,1,boundary='closed');solver.mx=depth*.2
        started=time.perf_counter()
        for _ in range(steps):solver.step(solver.timestep(.01),rain=1e-5,infiltration=1e-6)
        timings.append(time.perf_counter()-started)
    return solver,statistics.median(timings)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--sizes',type=int,nargs='+',default=[256,512])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--steps',type=int,default=4)
    args=parser.parse_args();baseline=None
    if args.baseline:
        spec=importlib.util.spec_from_file_location('baseline_solver',args.baseline)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);baseline=module.FloodSolver
    rows=[]
    measure(FloodSolver,64,1,2)
    if baseline:measure(baseline,64,1,2)
    for n in args.sizes:
        timings={'current':[],'baseline':[]};classes=[('current',FloodSolver)]
        if baseline:classes.append(('baseline',baseline))
        for repeat in range(args.repeats):
            states={}
            for name,cls in classes[::1 if repeat%2==0 else -1]:
                states[name],elapsed=measure(cls,n,1,args.steps);timings[name].append(elapsed)
            if baseline:
                for key in ('h','mx','mz','qx','qy','maximum'):
                    np.testing.assert_array_equal(getattr(states['baseline'],key),getattr(states['current'],key))
            del states
        elapsed=statistics.median(timings['current'])
        row=dict(side=n,steps=args.steps,repeats=args.repeats,current_median_s=elapsed,timings_s=timings)
        if baseline:row.update(baseline_median_s=statistics.median(timings['baseline']),
            speedup=statistics.median(timings['baseline'])/elapsed,arrays_exactly_equal=True)
        rows.append(row)
    print(json.dumps(dict(numpy_version=np.__version__,results=rows),indent=2))
