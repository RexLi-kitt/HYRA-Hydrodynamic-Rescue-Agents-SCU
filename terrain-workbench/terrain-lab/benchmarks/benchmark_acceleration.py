"""Time reference/compiled kernels and optional uniform coarse cells on one scene."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import map_sessions
import server


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=.2)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    supplied=json.loads(args.scenario.read_text(encoding='utf-8'))
    scenario=supplied.get('scenario_config',supplied)
    terrain=server.describe(scenario['terrain_id']);rows=[];states={}
    variants=[('numpy',1),('numba',1),('numba',2),('numba',4)]
    # Cache compilation before timing, including source and friction kernels.
    os.environ['HYRA_FLOOD_BACKEND']='numba'
    warm=map_sessions.create(ROOT,terrain,dict(scenario,flood=dict(scenario['flood'],grid_factor=4)))
    warm.advance_to(.001);del warm
    for phase in ('warmup','flood'):
        for repeat in range(args.repeats):
            for backend,factor in variants[::1 if repeat%2==0 else -1]:
                os.environ['HYRA_FLOOD_BACKEND']=backend
                c=dict(scenario,flood=dict(scenario['flood'],grid_factor=factor))
                world=map_sessions.create(ROOT,terrain,c);s=world.solver
                assert s.numerical_config()['compute_backend']==backend
                if phase=='warmup':
                    s.external_schedule=[];s.external_rate.fill(0);s.source_schedule=[]
                start=time.perf_counter()
                if phase=='warmup':s.advance(args.duration,infiltration=world.config['infiltration_mm_h']/3600000)
                else:world.advance_to(args.duration)
                wall=time.perf_counter()-start
                summary=s.summary()
                row=dict(phase=phase,backend=backend,factor=factor,repeat=repeat,
                    cells=s.h.size,steps=s.steps,simulated_s=s.time,wall_s=wall,
                    max_stage_courant=s.diagnostics['max_stage_courant'],
                    mass_residual_m3=summary['balance']['residual_m3'])
                rows.append(row)
                if factor==1:states[backend]=tuple(getattr(s,key).copy() for key in ('h','mx','mz','qx','qy'))
                print(json.dumps(row),flush=True)
            for a,b in zip(states['numpy'],states['numba']):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
    summary=[]
    for phase in ('warmup','flood'):
        baseline=statistics.median(r['wall_s'] for r in rows if r['phase']==phase and r['backend']=='numpy')
        for backend,factor in variants:
            wall=statistics.median(r['wall_s'] for r in rows if (r['phase'],r['backend'],r['factor'])==(phase,backend,factor))
            summary.append(dict(phase=phase,backend=backend,factor=factor,median_wall_s=wall,speedup_vs_numpy_native=baseline/wall))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(dict(scenario=scenario,duration_s=args.duration,repeats=args.repeats,
        native_backend_states_match=True,rows=rows,summary=summary),indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
