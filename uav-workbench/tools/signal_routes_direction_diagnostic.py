"""Development-only direction checks outside the frozen training sources.

The aligned route, a no-refit joint-water permutation, and the separately
trained matched permutation control use already frozen parameters. Nothing
under signal_routes/, no parameter file, and no confirmation data is changed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import signal_routes  # Enable the study's existing local numerical packages.
import numpy as np

from signal_routes.belief import Parameters, SignalBeliefEstimator
from signal_routes.common import DEFAULT, digest, read, sources, write
from signal_routes.offline import estimator, load_records, synthetic_record


DIRECTIONS = ('forward', 'reverse', 'isotropic')
ARMS = ('aligned', 'frozen_parameter_shuffle', 'matched_trained_shuffle')
METRICS = ('rmse', 'nll', 'coverage95')


def parameter_status(parameters, method):
    p = parameters[method]
    if method in ('R1', 'C1'):
        effect = p['alpha'] > 0 and p['kappa'] > 0
    else:
        effect = p['strength'] > 0 and abs(p['cross']) > 0
    return dict(null_signal_effect=not effect,
                directional_parameter_active=bool(effect and p['kappa'] > 0),
                parameters={key: p[key] for key in ('alpha', 'kappa', 'strength', 'cross')})


def numerical_tests():
    """A nonzero directional toy must respond; zero effects must be identical."""
    shape = (4, 5)
    signal = np.arange(20, dtype=float).reshape(shape) / 20.
    valid = np.zeros(shape, bool)
    valid[:2] = True
    valid[2, 1:4] = True
    water = np.zeros((3, *shape))
    water[0] = .1
    water[1] = .2
    stamps = np.where(valid, 0., -1.)
    obstacles = (np.indices(shape).sum(axis=0) % 3) == 0
    rows = []
    for mode, active, null in (
        ('R1', Parameters(alpha=.8, kappa=3.), Parameters(alpha=0., kappa=3.)),
        ('R3', Parameters(strength=2., cross=.7, kappa=3.),
         Parameters(strength=0., cross=.7, kappa=3.)),
    ):
        def predict(p, direction):
            return SignalBeliefEstimator(mode, p, direction).reset(shape).update(
                signal, water, valid, stamps, 5., obstacles).posterior(True)

        forward = predict(active, 'forward')
        reverse = predict(active, 'reverse')
        change = float(np.max(np.abs(forward['mean'] - reverse['mean'])))
        if not change > 1e-6:
            raise AssertionError(f'{mode} nonzero toy did not respond to reversal')
        zero = [predict(null, direction) for direction in DIRECTIONS]
        exact = all(np.array_equal(zero[0][key], other[key])
                    for other in zero[1:] for key in ('mean', 'variance'))
        if not exact:
            raise AssertionError(f'{mode} zero-effect directions differ')
        rows.append(dict(method=mode, nonzero_forward_reverse_max_change=change,
                         zero_mean_and_variance_bitwise_equal=exact))
    return dict(passed=True, rows=rows)


def grouped(rows):
    result = []
    keys = sorted({(r['condition'], r['route'], r['arm'], r['direction']) for r in rows})
    for condition, route, arm, direction in keys:
        part = [r for r in rows if (r['condition'], r['route'], r['arm'], r['direction'])
                == (condition, route, arm, direction)]
        parents = sorted({r['parent'] for r in part})
        values = {parent: {metric: float(np.mean([r[metric] for r in part
                    if r['parent'] == parent])) for metric in METRICS} for parent in parents}
        result.append(dict(condition=condition, route=route, arm=arm, direction=direction,
            sample_count=len(part), parent_count=len(parents), parent_values=values,
            means={metric: float(np.mean([values[parent][metric] for parent in parents]))
                   for metric in METRICS}))
    return result


def paired_comparisons(rows, predictions):
    lookup = {(r['condition'], r['route'], r['arm'], r['direction'],
               r['parent'], r['forcing'], r['seed']): r for r in rows}
    comparisons = []
    for condition in ('original', 'directional'):
        for route in ('R1', 'R3'):
            aligned = [r for r in rows if r['condition'] == condition and r['route'] == route
                       and r['arm'] == 'aligned' and r['direction'] == 'forward']
            for label, other_route, arm, direction in (
                ('reverse_minus_forward', route, 'aligned', 'reverse'),
                ('isotropic_minus_forward', route, 'aligned', 'isotropic'),
                ('no_refit_shuffle_minus_aligned', route, 'frozen_parameter_shuffle', 'forward'),
                ('trained_shuffle_minus_aligned', route, 'matched_trained_shuffle', 'forward'),
                ('baseline_minus_aligned', 'Bpost', 'baseline', 'isotropic'),
            ):
                paired = []
                for a in aligned:
                    ka = (condition, route, 'aligned', 'forward', a['parent'], a['forcing'], a['seed'])
                    kb = (condition, other_route, arm, direction, a['parent'], a['forcing'], a['seed'])
                    b = lookup[kb]
                    change = np.abs(predictions[kb] - predictions[ka])
                    paired.append(dict(parent=a['parent'],
                        **{metric: b[metric] - a[metric] for metric in METRICS},
                        posterior_mean_abs_change=float(change.mean()),
                        posterior_max_abs_change=float(change.max())))
                parents = sorted({r['parent'] for r in paired})
                values = {parent: {metric: float(np.mean([r[metric] for r in paired
                          if r['parent'] == parent])) for metric in METRICS} for parent in parents}
                comparisons.append(dict(condition=condition, route=route, contrast=label,
                    sample_count=len(paired), parent_count=len(parents), parent_deltas=values,
                    mean_deltas={metric: float(np.mean([values[parent][metric] for parent in parents]))
                                 for metric in METRICS},
                    posterior_mean_abs_change=float(np.mean([r['posterior_mean_abs_change'] for r in paired])),
                    posterior_max_abs_change=float(max(r['posterior_max_abs_change'] for r in paired))))
    return comparisons


def main(study=DEFAULT):
    study = Path(study)
    output = study / 'independent-direction-diagnostics.json'
    if output.exists():
        raise FileExistsError(f'Preserve the existing diagnostic: {output}')
    source_before = sources()
    parameter_before = {p.name: digest(p) for p in study.glob('parameters*.json')}
    records_path = study / 'development-records.npz'
    records = [r for r in load_records(records_path) if r['step'] == 12]
    parents = sorted({r['parent'] for r in records})
    if len(records) != 18 or len(parents) != 3:
        raise AssertionError('Expected all 3 development parents, 2 forcings and 3 layouts')
    parameters = {condition: read(study / filename)['parameters'] for condition, filename in (
        ('original', 'parameters.json'), ('directional', 'parameters-directional.json'))}
    tests = numerical_tests()
    rows, predictions = [], {}
    for condition in ('original', 'directional'):
        p = parameters[condition]
        for original in records:
            record = synthetic_record(original, condition)
            specs = [('Bpost', 'baseline', 'isotropic', 'Bpost', p['Bpost'])]
            for route, control in (('R1', 'C1'), ('R3', 'C3')):
                for direction in DIRECTIONS:
                    specs.extend(((route, 'aligned', direction, route, p[route]),
                                  (route, 'frozen_parameter_shuffle', direction, control, p[route]),
                                  (route, 'matched_trained_shuffle', direction, control, p[control])))
            for route, arm, direction, mode, param in specs:
                model = estimator(mode, param, record, direction)
                score = model.score(record['truth'])
                posterior = model.posterior()
                key = (condition, route, arm, direction, record['parent'],
                       record['mechanism'], record['seed'])
                predictions[key] = posterior['mean']
                rows.append(dict(condition=condition, route=route, arm=arm, direction=direction,
                    estimator_mode=mode, parameter_method=mode if arm != 'frozen_parameter_shuffle' else route,
                    parent=record['parent'], forcing=record['mechanism'], seed=record['seed'],
                    **score, residual_inf=posterior['diagnostics']['residual_inf'],
                    posterior_sha256=hashlib.sha256(posterior['mean'].tobytes()).hexdigest()))
        print(f'direction diagnostic {condition}: completed {len(rows)} scores', flush=True)
    source_after = sources()
    parameter_after = {p.name: digest(p) for p in study.glob('parameters*.json')}
    if source_before != source_after or parameter_before != parameter_after:
        raise AssertionError('Frozen training sources or parameter artifacts changed during diagnostic')
    result = dict(schema='hyra-independent-direction-diagnostics-v1', split='development',
        confirmation_data_read=False, policies_or_parameter_files_modified=False,
        development_records_sha256=digest(records_path), parameter_sha256=parameter_before,
        frozen_training_sources=source_before, tool_sha256=digest(__file__),
        numerical_tests=tests, sample_count_per_condition=len(records), parent_count=len(parents),
        scoring_rows=len(rows), sample_unit='one step-12 snapshot per parent/forcing/layout',
        nll='conditional marginal NLL on at most 32 deterministic unobserved cells, per cell',
        aggregation='within-parent average first; equal weight across the 3 development parents',
        arms=dict(aligned='route parameters and correctly aligned arrived water',
                  frozen_parameter_shuffle='same route parameters; existing C-mode joint h/u/v/age permutation; no refit',
                  matched_trained_shuffle='existing C-mode and its separately frozen fitted control parameters'),
        parameter_status={condition: {method: parameter_status(parameters[condition], method)
                          for method in ('R1', 'C1', 'R3', 'C3')} for condition in parameters},
        rows=rows, grouped=grouped(rows), contrasts=paired_comparisons(rows, predictions),
        interpretation=[
            'Numerical sensitivity of a nonzero toy does not prove learned directional predictive benefit.',
            'Null-selected routes yield direction-invariant baseline predictions; a null direction contrast is uninformative about physical propagation.',
            'Directional synthetic snapshots encode an artificial local signal response, not movement of fixed people or a real detector law.',
            'The no-refit permutation is an inference perturbation; its difference is distinct from the separately fitted matched-control comparison.',
            'Three development parents support descriptive diagnostics only; layouts, forcings and snapshots are not additional independent parents.',
            'No confidence intervals or hypothesis tests are selected from these development diagnostics.',
        ])
    write(output, result)
    lines = ['# 冻结参数的开发集方向诊断', '',
        '本诊断只读3个开发父地形，各含2种来水、3种布局，共18个固定阶段快照；不读取确认性能。',
        '分别使用原任务和人工方向响应条件各自已经冻结的参数，比较正向、反向、各向同性、同参数水情置乱及已训练匹配置乱控制。',
        '各行先在父地形内平均，再对3个父地形等权；不将快照或布局当作独立重复，不进行显著性检验。', '',
        '| 条件 | 路线 | 零退化 | 方向参数活跃 |', '|---|---|---|---|']
    for condition, modes in result['parameter_status'].items():
        for method in ('R1', 'R3'):
            status = modes[method]
            lines.append(f"| {condition} | {method} | {status['null_signal_effect']} | {status['directional_parameter_active']} |")
    lines.extend(['', '| 条件 | 路线 | 对照减正向 | RMSE差 | NLL差 | 95%覆盖率差 | 最大后验变化 |',
                  '|---|---|---|---:|---:|---:|---:|'])
    for contrast in result['contrasts']:
        d = contrast['mean_deltas']
        lines.append(f"| {contrast['condition']} | {contrast['route']} | {contrast['contrast']} | {d['rmse']:.6g} | {d['nll']:.6g} | {d['coverage95']:.6g} | {contrast['posterior_max_abs_change']:.6g} |")
    lines.extend(['', 'RMSE和NLL的正差值表示正向配准一侧更小；覆盖率差仅报告校准变化，不能按数值越大越好解释。',
        'NLL按每个快照最多32个固定选定未观测格的条件边缘高斯密度计算；RMSE覆盖全部未观测格。',
        '非零toy的方向翻转响应及零系数逐位一致性均通过。主任务的零退化与合成条件的活跃状态须分别解释。',
        '人工方向响应不能证明固定人员发生漂流，也不能证明真实探测物理。全部训练源码、主参数及合成参数的前后摘要一致。'])
    (study / 'independent-direction-diagnostics.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {output}; {len(rows)} rows; frozen sources and parameters unchanged', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, default=DEFAULT)
    main(parser.parse_args().study)
