"""Audit the fixed 400-cell inlet fixture; this is not a physical benchmark."""
from pathlib import Path
import argparse, json
import numpy as np


def audit(before, after, output):
    before, after, output = map(Path, (before, after, output))
    old = np.loadtxt(before/'results/check.mass', skiprows=1)
    mass = np.loadtxt(after/'results/check.mass', skiprows=1)
    steps = np.loadtxt(after/'accepted_timesteps.txt')
    times = np.arange(1, 5)*.05
    assert mass.shape == (4, 12), mass.shape
    assert np.isfinite(mass).all() and np.isfinite(steps).all()
    assert np.all(steps[:,1] > 0)
    assert abs(steps[0,0]) < 1e-14
    assert np.max(abs(steps[1:,0] - (steps[:-1,0]+steps[:-1,1]))) < 1e-14
    accepted_end = float(steps[-1,0] + steps[-1,1])
    assert abs(accepted_end-.2) < 1e-14
    assert abs(steps[:,1].sum()-.2) < 1e-14
    ends = steps[:,0] + steps[:,1]
    assert max(min(abs(ends-t)) for t in times) < 1e-14
    assert np.allclose(mass[:,0], times, rtol=0, atol=1e-12)
    assert np.allclose(mass[:,6], .05, rtol=0, atol=1e-12)
    assert np.all(mass[:,8] == 0)
    assert np.max(abs(mass[:,5] - times*.05)) < 1e-7
    assert np.max(abs(mass[:,10])) < 1e-12
    depth = np.loadtxt(after/'results/check-0000.wd', skiprows=6)
    assert depth.shape == (40, 10) and np.isfinite(depth).all() and np.min(depth) >= 0
    assert abs(depth.sum()*.1**2 - .01) < 1e-6  # ASCII depth rounding
    raw = (after/'results/check.velocity').read_text().splitlines()
    rows = [line.split() for line in raw if len(line.split()) == 4 and line.split()[0] in ('0.050','0.100','0.150','0.200')]
    speed = np.array(rows, dtype=float)
    assert speed.shape == (4,4) and np.isfinite(speed).all()
    assert np.all(speed[:,2:] == 0)
    result = {'purpose':'engineering event/inlet test, not accuracy validation',
      'accepted_steps':len(steps),'actual_end_s':accepted_end,'output_times_s':mass[:,0].tolist(),
      'requested_Q_m3_s':.05,'reported_Q_m3_s':mass[:,6].tolist(),
      'final_volume_m3':float(mass[-1,5]),'expected_volume_m3':.01,
      'max_abs_water_residual_m3':float(max(abs(mass[:,10]))),
      'old_reported_output_times_s':old[:,0].tolist(),'old_reported_final_volume_m3':float(old[-1,5]),
      'finite_nonnegative':True,'dry_gauge_speeds_zero':True,
      'limits':['single dry flat rectangular inlet fixture','no rainfall or point sources or restart validation','no full Merewether reference score','CPU FP64 only']}
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(result))
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('before'); p.add_argument('after'); p.add_argument('output')
    a = p.parse_args(); audit(a.before, a.after, a.output)
