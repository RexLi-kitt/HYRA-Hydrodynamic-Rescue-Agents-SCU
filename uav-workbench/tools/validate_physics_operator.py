"""Compare the differentiable step with actual-fixture NumPy restarts."""
import os
os.environ['HYRA_FLOOD_BACKEND'] = 'numpy'
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'terrain-workbench/terrain-lab'))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from flood_solver import FloodSolver
from multilayer import BaseMap
from hyra_learning.physics import SWEOperator


def main():
    plan = json.loads((ROOT/'learning_configs/physics_operator_validation_20261025.json').read_text(encoding='utf-8'))
    for name, expected in plan['source_files'].items():
        assert sha256((ROOT.parent/'terrain-workbench/terrain-lab'/name).read_bytes()).hexdigest() == expected
    results = []
    for fixture in plan['actual_fixture_cases']['fixtures']:
        path = ROOT/'fixtures'/fixture/'map-result.zip'
        with zipfile.ZipFile(path) as archive:
            process, model = (json.loads(archive.read(name)) for name in ('process.json', 'map_model.json'))
            assert process['config']['boundary'] == 'closed'
            assert not process['config']['inflow_m3_s'] and not process.get('external_inflows')
            assert not model.get('M_structures', {}).get('objects')
            grid, config = process['grid'], process['config']
            base_config = dict(model['M_base'], hydrographs=model.get('source_hydrographs', {}))
            base = BaseMap(model['M_base']['z'], grid['dx_m'], base_config, config['manning'])
            for now in plan['actual_fixture_cases']['times_s']:
                entry = next(f for f in process['frames'] if f['time_s'] == now)
                for key, value in dict(spatial_order=2, time_integrator='rk2', flux_method='rusanov', low_froude=0).items():
                    assert entry['numerics'][key] == value
                frame = np.frombuffer(archive.read(entry['url'].replace('\\','/').rsplit('/',1)[-1]), dtype='<f4').reshape(4, *grid['shape'])
                initial = np.stack((frame[0], frame[0].astype(float)*frame[1], frame[0].astype(float)*frame[2])).astype(float)
                reference = FloodSolver(base.z, initial[0], base.dx, boundary='closed', manning=base.n)
                base.configure(reference)
                reference.mx, reference.mz = initial[1].copy(), initial[2].copy()
                reference.time = float(now)
                reference.source_rate = reference._scheduled_rate(now)
                rain = config['rain_mm_h']/3600000 if now < config['rain_duration_s'] else 0.
                infiltration = config['infiltration_mm_h']/3600000
                dt = reference.timestep(limit=plan['actual_fixture_cases']['max_step_s'], rain=rain)*plan['actual_fixture_cases']['step_fraction_of_CFL']
                source = rain*(~base.solid)+reference.source_rate
                reference.step(dt, rain=rain, infiltration=infiltration)
                expected = np.stack((reference.h, reference.mx, reference.mz))
                for dtype in (torch.float64, torch.float32):
                    tensor = lambda value: torch.as_tensor(value, dtype=dtype)
                    actual = SWEOperator(dx=base.dx).step(tensor(initial)[None], torch.as_tensor(base.z, dtype=torch.float64), dt,
                        source=tensor(source), infiltration=infiltration, drains=tensor(base.drain_capacity), manning=tensor(base.n),
                        solid=torch.as_tensor(base.solid), wall_x=torch.as_tensor(base.wall_x), wall_y=torch.as_tensor(base.wall_y))
                    error = float(np.abs(actual[0].detach().numpy()-expected).max())
                    tolerance = plan['thresholds']['float64_max_absolute_state_error' if dtype == torch.float64 else 'float32_max_absolute_state_error']
                    assert error <= tolerance, (fixture, now, str(dtype), error, tolerance)
                    results.append(dict(fixture=fixture, time_s=now, dt_s=dt, dtype=str(dtype), max_abs_error=error,
                                        tolerance=tolerance, passed=True, source_sha256=sha256(path.read_bytes()).hexdigest()))
    output = ROOT/'docs/evidence/20261025-physics-operator'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed', comparisons=len(results), results=results,
                  operator_sha256=sha256((ROOT/'hyra_learning/physics.py').read_bytes()).hexdigest(),
                  validation_plan_sha256=sha256((ROOT/'learning_configs/physics_operator_validation_20261025.json').read_bytes()).hexdigest(),
                  max_error_float64=max(r['max_abs_error'] for r in results if r['dtype'] == 'torch.float64'),
                  max_error_float32=max(r['max_abs_error'] for r in results if r['dtype'] == 'torch.float32'),
                  scope='Single accepted substep from actual saved states; not a five/ten-second coarse difference or a policy-benefit result.',
                  engineering_amendment='Initial high-datum float32 comparison failed because terrain precision was lost before operator input; preserve static bed float64 or center in float64 first. Original tolerance unchanged; failed log retained.')
    (output/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','comparisons','max_error_float64','max_error_float32')}))


if __name__ == '__main__':
    main()
