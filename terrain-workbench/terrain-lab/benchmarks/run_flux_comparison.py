"""Reproducible Cartesian flux experiment; no ANUGA runtime required.

Run from any directory: python run_flux_comparison.py --repeats 3
All methods use identical physical domains and cell-average analytic targets.
The legacy baseline is a frozen, unmodified file from the recorded Git commit.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import statistics
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
LAB = HERE.parent
sys.path.insert(0, str(LAB))
from flood_solver import FloodSolver
from analytic_cases import CASES, make_case
from river_baseflow import spin_up

BASELINE_COMMIT = '95a080e0a8f35283a97fcc426a6f9432e51ba0c2'
ANUGA_REFERENCE = 'e2e7bf67b5ed0f54d1c001b9dbc403d7567b8a9b'
METHODS = {
    'legacy_rusanov': dict(),
    'muscl_rusanov_lf0': dict(flux_method='rusanov', low_froude=0),
    'muscl_knp_lf0': dict(flux_method='knp', low_froude=0),
    'muscl_rusanov_lf2': dict(flux_method='rusanov', low_froude=2),
    'muscl_knp_lf2': dict(flux_method='knp', low_froude=2),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_solver():
    spec = importlib.util.spec_from_file_location('hyra_frozen_rusanov', HERE/'baselines/rusanov_v1.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.FloodSolver


def metrics(solver, case):
    h = solver.h; mx = solver.mx; mz = solver.mz
    dh = np.abs(h-case['exact_h'])
    dm = np.hypot(mx-case['exact_mx'], mz-case['exact_mz'])
    reference_m = float(np.mean(np.hypot(case['exact_mx'], case['exact_mz'])))
    summary = solver.summary(); balance = summary['balance']
    return dict(depth_l1_m=float(dh.mean()),
        depth_l1_relative=float(dh.mean()/case['exact_h'].mean()),
        momentum_l1_m2_s=float(dm.mean()),
        momentum_l1_relative=float(dm.mean()/reference_m) if reference_m > 1e-14 else None,
        max_depth_error_m=float(dh.max()), minimum_depth_m=float(h.min()),
        raw_mass_residual_m3=balance['residual_m3'],
        numerical_adjustment_m3=balance['numerical_adjustment_m3'],
        max_speed_m_s=summary['max_speed_m_s'], steps=solver.steps,
        diagnostics=summary.get('diagnostics', {}))


def run_case(case, method, repeats, old_solver):
    if method == 'legacy_rusanov' and case['boundary'] == 'periodic':
        # The baseline never supported periodic edges. Do not alter it to
        # manufacture a result for this case, and do not rank missing rows.
        return dict(status='unsupported', reason='frozen legacy solver has no periodic boundary')
    times, hashes, last = [], [], None
    for _ in range(repeats):
        cls = old_solver if method == 'legacy_rusanov' else FloodSolver
        options = METHODS[method]
        solver = cls(case['bed'], case['h'], case['dx'], boundary=case['boundary'],
                     manning=0., **options)
        solver.mx[:] = case['mx']; solver.mz[:] = case['mz']
        begin = time.perf_counter(); solver.advance(case['end'])
        times.append(time.perf_counter()-begin)
        if not all(np.isfinite(a).all() for a in (solver.h, solver.mx, solver.mz)):
            raise RuntimeError('Non-finite benchmark output')
        hashes.append(hashlib.sha256(b''.join(a.tobytes() for a in (solver.h,solver.mx,solver.mz))).hexdigest())
        last = solver
    if len(set(hashes)) != 1:
        raise RuntimeError('Repeat runs disagree: '+case['name']+'/'+method)
    config = (last.numerical_config() if hasattr(last, 'numerical_config') else
              dict(flux_method='rusanov', spatial_order=1, time_integrator='euler',
                   low_froude=0, cfl=.45, dry_depth_m=1e-8, max_timestep_s=1.,
                   source_commit=BASELINE_COMMIT))
    result = dict(status='ok', **metrics(last, case), runtime_s=times,
                  runtime_median_s=statistics.median(times),
                  runtime_min_s=min(times), runtime_max_s=max(times),
                  final_state_sha256=hashes[0], configuration=config)
    # Store one-dimensional profiles; these cases are invariant in z.
    profiles = dict(h=last.h.mean(axis=0), mx=last.mx.mean(axis=0), mz=last.mz.mean(axis=0))
    return result, profiles


def channel_check(method, old_solver):
    bed=np.full((8,24),4.); bed[3:5,1:]=.02*np.arange(22,-1,-1)
    depth=np.zeros_like(bed); depth[3:5,1:]=.3
    cls=old_solver if method=='legacy_rusanov' else FloodSolver
    if method == 'legacy_rusanov':
        # Adapt only the progress callback API; the frozen numerical code is unchanged.
        class LegacyProgressAdapter(old_solver):
            def advance(self, end, *, progress=None, **kwargs):
                return super().advance(end, **kwargs)
        cls = LegacyProgressAdapter
    solver=cls(bed,depth,1.,manning=.035,**METHODS[method])
    solver.set_baseflow([dict(z=3,x=1,discharge_m3_s=.15),dict(z=4,x=1,discharge_m3_s=.15)])
    report=spin_up(solver,600.,0.,lambda **kw:None,lambda:False)
    return dict(method=method,converged=report['converged'],duration_s=report['duration_s'],
                checks=report['checks'],mass_residual_m3=solver.summary()['balance']['residual_m3'])


def write_report(rows, manifest, channel_checks, target):
    lines = ['# 方格水动力升级与 Rusanov/KNP 对比', '',
        '本报告由实验脚本根据实际运行结果生成。所有升级版本共用二阶 MUSCL、SSP-RK2、CFL=0.4、限制器 θ=1.5；仅改变通量和 low_froude。', '',
        '旧版是提交 '+BASELINE_COMMIT+' 的原始一阶 Rusanov/Euler，CFL=0.45，作为整套升级的历史基线；不把它与新版之间的差异归因于 KNP。', '',
        '这是一组解析数值验证，不代表真实洪水校准，也不是运行原版 ANUGA。所有数据为同一进程顺序执行；计时覆盖 advance()，不含构造、解析参考及文件导出。', '',
        '## 实验设置', '',
        '- 分辨率：'+', '.join(map(str, manifest['levels']))+'；每个案例固定物理域，使用原生正方格。',
        '- 每组重复 '+str(manifest['repeats'])+' 次，保存各次耗时；结果表采用中位数。重复结果的最终状态 SHA-256 必须一致。',
        '- Python '+manifest['python']+'；NumPy '+manifest['numpy']+'；'+manifest['platform']+'。',
        '- 干/湿床溃坝使用 Ritter/Stoker 解；剪切流为精确非线性浅水解；抛物盆地使用 Thacker 解；参考均为单元平均值。',
        '- 剪切案例需要周期边界；旧版不支持，明确记为缺失，不修改旧版凑齐对比。', '',
        '解析公式来源：[干床溃坝](https://github.com/anuga-community/anuga_core/blob/'+ANUGA_REFERENCE+'/validation_tests/analytical_exact/dam_break_dry/analytical_dam_break_dry.py)、[湿床溃坝](https://github.com/anuga-community/anuga_core/blob/'+ANUGA_REFERENCE+'/validation_tests/analytical_exact/dam_break_wet/analytical_dam_break_wet.py)、[Thacker 盆地](https://github.com/anuga-community/anuga_core/blob/'+ANUGA_REFERENCE+'/validation_tests/analytical_exact/parabolic_basin/analytical_parabolic_basin.py)。本地实现对这些函数做了单元平均解析积分。', '',
        '剪切流使用 h=1、u=U、v=a·sin(k(x−Ut))；它满足连续方程与两个动量方程。参考在每格上积分，周期边界只用于此验证案例。', '',
        '## 全部实测结果', '',
        '| 案例 | Nx×Nz | 方法 | 水深 L1/m | 动量 L1/(m²/s) | 水量残差/m³ | 时间中位数/s | 步数 |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        if row['status'] != 'ok':
            lines.append(f"| {row['case']} | {row['nx']}×{row['nz']} | {row['method']} | 不支持周期边界 | — | — | — | — |")
        else:
            lines.append(f"| {row['case']} | {row['nx']}×{row['nz']} | {row['method']} | {row['depth_l1_m']:.5e} | {row['momentum_l1_m2_s']:.5e} | {row['raw_mass_residual_m3']:.3e} | {row['runtime_median_s']:.4f} | {row['steps']} |")
    lines += ['', '## 网格细化误差阶', '',
        'p=log₂(E粗/E细)。湖泊静水的误差接近浮点量级，不据此估计收敛阶；激波/干湿前沿不要求处处二阶。', '',
        '| 案例 | 方法 | 网格对 | 水深 p | 动量 p |',
        '| --- | --- | --- | ---: | ---: |']
    for case in manifest['cases']:
        for method in METHODS:
            group = sorted([r for r in rows if r['status']=='ok' and r['case']==case and r['method']==method], key=lambda r:r['nx'])
            for a,b in zip(group, group[1:]):
                if case == 'lake' or b['nx'] != 2*a['nx']:
                    continue
                def order(key):
                    return f"{np.log2(a[key]/b[key]):.3f}" if min(a[key],b[key])>1e-13 else '—'
                lines.append(f"| {case} | {method} | {a['nx']}→{b['nx']} | {order('depth_l1_m')} | {order('momentum_l1_m2_s')} |")
    lines += ['', '## 有摩阻河道的稳态检查', '',
        '8×24 方格、dx=1 m、两格宽河道、床坡0.02、Manning n=0.035、总基流0.3 m³/s。最多预热600 s，每60 s检查，连续两窗水深变化≤0.01 m且速度变化≤0.01 m/s才判定收敛。这是稳态行为检查，没有独立解析精度结论。', '',
        '| 方法 | 达到判据 | 预热/s | 最后一窗水深变化/m | 最后一窗速度变化/(m/s) | 水量残差/m³ |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for c in channel_checks:
        last=c['checks'][-1]
        lines.append(f"| {c['method']} | {'是' if c['converged'] else '否'} | {c['duration_s']:.0f} | {last['max_depth_change_m']:.4e} | {last['max_velocity_change_m_s']:.4e} | {c['mass_residual_m3']:.3e} |")
    lines += ['', '默认使用 Rusanov、MUSCL、RK2，low_froude=0；模式1/2保留为显式实验选项。低弗劳德数修正应同时检查解析误差和含地形/摩阻场景，不能仅凭剪切流误差下降就默认开启。', '',
        '## 数据与适用范围', '',
        '- 原始配置、代码哈希、每次计时、误差、限流诊断见 `../terrain-lab/benchmarks/results/flux-comparison.json`。',
        '- 平面表格见同目录 `flux-comparison.csv`；精确与数值剖面见 `flux-profiles.npz`。',
        '- 直接比较 KNP 与 Rusanov 时，应固定同一个 low_froude 值；固定通量后，lf0/lf2 比较才反映修正的影响。',
        '- 报告中的时间是本机测量，不外推到其他网格、硬件、真实地形或不同精度目标。',
        '- Manning、降雨、排水与事件计量另由物理回归测试覆盖；这些无源解析案例不证明所有源项组合都达到二阶。', '']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--levels', type=int, nargs='+', default=[64,128,256])
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=HERE/'results')
    parser.add_argument('--report', type=Path, default=LAB.parent/'docs/flood-numerics-comparison.md')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('repeats must be positive')
    manifest = dict(schema='hyra-flux-comparison-v1', python=platform.python_version(),
        numpy=np.__version__, platform=platform.platform(), repeats=args.repeats,
        levels=args.levels, cases=args.cases, baseline_commit=BASELINE_COMMIT,
        anuga_formula_reference_commit=ANUGA_REFERENCE,
        code_sha256={str(p.relative_to(LAB)):sha256(p) for p in
            (LAB/'flood_solver.py', LAB/'flood_numerics.py', HERE/'analytic_cases.py',
             Path(__file__), HERE/'baselines/rusanov_v1.py')})
    rows, profiles = [], {}
    old = legacy_solver()
    for name in args.cases:
        for n in args.levels:
            case = make_case(name, n)
            prefix = name+'_'+str(n)
            profiles[prefix+'_x'] = case['x']
            for field in ('h','mx','mz'):
                profiles[prefix+'_exact_'+field] = case['exact_'+field].mean(axis=0)
            for method in METHODS:
                print(f'{name} n={n} {method}', flush=True)
                outcome = run_case(case, method, args.repeats, old)
                if isinstance(outcome, tuple):
                    result, values = outcome
                    for key, value in values.items():
                        profiles[prefix+'_'+method+'_'+key] = value
                else:
                    result = outcome
                rows.append(dict(case=name, nx=n, nz=case['h'].shape[0],
                    dx_m=case['dx'], finaltime_s=case['end'], method=method, **result))
    channel_checks=[]
    for method in METHODS:
        print('steady channel '+method,flush=True)
        channel_checks.append(channel_check(method,old))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'flux-comparison.json').write_text(json.dumps(dict(manifest=manifest,results=rows,
        channel_checks=channel_checks),
        ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    fields = ['case','nx','nz','dx_m','finaltime_s','method','status','depth_l1_m',
              'momentum_l1_m2_s','raw_mass_residual_m3','numerical_adjustment_m3',
              'runtime_median_s','runtime_min_s','runtime_max_s','steps']
    with (args.output/'flux-comparison.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    np.savez_compressed(args.output/'flux-profiles.npz', **profiles)
    write_report(rows, manifest, channel_checks, args.report)
    print('Saved '+str(args.report), flush=True)


if __name__ == '__main__':
    main()
