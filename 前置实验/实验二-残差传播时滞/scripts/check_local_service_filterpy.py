"""Use the pinned FilterPy source as an independent reference, without installing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.local_service import load_design, conventional_predictions, OFFSETS


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pinned = json.loads((ROOT/'third_party/filterpy/SOURCE.json').read_text(encoding='utf-8'))
    source = ROOT/'third_party/filterpy'
    sys.path.insert(0,str(source))
    from filterpy.kalman import KalmanFilter
    import filterpy.kalman.kalman_filter as reference_module
    assert Path(reference_module.__file__).resolve().is_relative_to(source.resolve())
    config, _ = load_design(ROOT)
    rng = np.random.default_rng(2026101519)
    history = np.zeros((32, 6, 4))
    history[:, :, 0] = rng.normal(0, .05, (32, 6))
    history[:, :, 1] = (np.arange(6)[None]*10+rng.uniform(0, 9, (32, 6)))/60
    history[:, :, 2] = rng.uniform(20, 30, (32, 6))/30
    history[:, :, 3] = rng.random((32, 6)) > .25
    history[0, :, 3] = 0
    ours = conventional_predictions(history, config)
    count = 0; maximum = 0.
    for kind, qs in [('rw', config['models']['kf_rw_process_variance_rate_m2_per_s']),
                     ('cv', config['models']['kf_cv_process_spectral_density_m2_per_s3'])]:
        for q in qs:
            for r in config['models']['kf_measurement_std_m']:
                name = f'kf_{kind}_q{q:g}_r{r:g}'
                for row in range(len(history)):
                    f = KalmanFilter(dim_x=2, dim_z=1)
                    f.x = np.zeros(2); f.H = np.array([[1., 0.]])
                    f.P = np.diag([.05**2, .002**2 if kind == 'cv' else 0]); f.R = np.array([[r*r]])
                    last = None
                    for b in range(5, -1, -1):
                        if not history[row, b, 3]: continue
                        t = -60*history[row, b, 1]; dt = 0 if last is None else t-last
                        if kind == 'cv':
                            f.F = np.array([[1., dt], [0., 1.]])
                            f.Q = q*np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]])
                        else:
                            f.F = np.eye(2); f.Q = np.diag([q*dt, 0])
                        f.predict(); f.update(np.array([history[row, b, 0]])); last = t
                    expected = f.x[0]+(OFFSETS-(last or 0))*f.x[1]
                    error = float(abs(ours[name][row]-expected).max())
                    maximum = max(maximum, error); count += 1
                    np.testing.assert_allclose(ours[name][row], expected, atol=1e-12, rtol=1e-12)
    result = dict(passed=True, reference_repository=pinned['repository'], reference_commit=pinned['commit'],
        source_path=str(source), reference_module=str(reference_module.__file__),
        reference_module_sha256=hashlib.sha256(Path(reference_module.__file__).read_bytes()).hexdigest(),
        compared_parameter_settings=21, histories=32, reference_prediction_curves=count,
        maximum_absolute_difference_m=maximum, dependency_installed=False,
        usage='Pinned FilterPy executes independent Kalman reference; vectorized equivalent is used for bulk replays')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__': main()
