"""Small CPU DG2 point-source contract probe; no field-validation claims."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(exe, out):
    exe, out = Path(exe), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cases = [
        ('qfix_5m', 5., .02, .005, False, 1),
        ('qfix_50m_unscaled', 50., .02, .005, False, 1),
        ('qfix_50m_scaled', 50., .002, .005, False, 1),
        ('qvar_constant', 5., .02, .005, True, 1),
        ('qfix_half_dt', 5., .02, .0025, False, 1),
        ('qfix_duplicate_cell', 5., .01, .005, False, 2),
    ]
    results = []
    for name, dx, q, maxdt, variable, count in cases:
        case = out/name
        case.mkdir()
        header = f'ncols 20\nnrows 20\nxllcorner 0\nyllcorner 0\ncellsize {dx}\nNODATA_value -9999\n'
        for filename in ('flat.dem','flat.dem1x','flat.dem1y'):
            (case/filename).write_text(header + (' '.join(['0']*20)+'\n')*20)
        source = f'P {10.5*dx} {10.5*dx} ' + ('QVAR pulse' if variable else f'QFIX {q}') + '\n'
        (case/'input.bci').write_text(source*count)
        (case/'input.bdy').write_text(f'# constant test, not observation\npulse\n2 seconds\n{q} 0\n{q} 1\n')
        (case/'case.par').write_text(
            'DEMfile flat.dem\nbcifile input.bci\n' + ('bdyfile input.bdy\n' if variable else '') +
            f'dirroot results\nresroot check\ndg2\nsim_time .2\ninitial_tstep {maxdt}\nCFL .3\n'
            'massint .05\nsaveint .2\nfpfric 0\nlimitslopes\noutput_precision 12\n')
        env = dict(os.environ, OMP_NUM_THREADS='1')
        started = time.perf_counter()
        with (case/'run.log').open('w') as log:
            p = subprocess.run([str(exe), '-v', 'case.par'], cwd=case, env=env,
                               stdout=log, stderr=subprocess.STDOUT, timeout=60)
        if p.returncode:
            raise RuntimeError(f'{name}: exit {p.returncode}; see run.log')
        steps = np.loadtxt(case/'accepted_timesteps.txt', ndmin=2)
        mass = np.loadtxt(case/'results/check.mass', skiprows=1, ndmin=2)
        depth = np.loadtxt(case/'results/check-0000.wd', skiprows=6)
        end = float(steps[-1].sum())
        expected = q*dx*count*end
        # Prediction from source-code ordering: first 1e-6, then previous accepted dt.
        old_dt_volume = q*dx*count*float(1e-6 + steps[:-1,1].sum())
        actual = float(depth.sum()*dx*dx)
        item = dict(name=name, dx_m=dx, q_input_m2_s=q, points=count,
                    physical_Q_m3_s=q*dx*count, max_dt_s=maxdt,
                    end_s=end, steps=len(steps), expected_volume_m3=expected,
                    final_raster_volume_m3=actual, old_dt_prediction_m3=old_dt_volume,
                    volume_error_m3=actual-expected,
                    max_abs_reported_mass_error_m3=float(abs(mass[:,10]).max()),
                    reported_final_volume_m3=float(mass[-1,5]),
                    finite_nonnegative=bool(np.isfinite(depth).all() and (depth >= 0).all()),
                    elapsed_s=time.perf_counter()-started,
                    input_sha256={f.name:sha(f) for f in case.iterdir() if f.suffix in ('.dem','.bci','.bdy','.par')})
        assert abs(end-.2)<1e-12 and item['finite_nonnegative']
        item['volume_contract_pass'] = abs(actual-expected) <= 1e-7
        results.append(item)
        print(json.dumps(item), flush=True)
    summary = dict(purpose='internal numerical/input test, not real-flood validation',
                   exe=str(exe), exe_sha256=sha(exe), tolerance_m3=1e-7, cases=results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False), encoding='utf8')
    return summary


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('exe'); p.add_argument('out')
    a=p.parse_args(); run(a.exe,a.out)
