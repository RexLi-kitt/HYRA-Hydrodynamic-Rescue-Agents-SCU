"""Generate a descriptive audited study report, tables, and scientific figures."""
import argparse
import csv
import json
from pathlib import Path
from scale100_publish import validated_bundle, ROOT, FORMAL, EVALUATION, read, digest, require
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ('B0', 'B1', 'B2', 'B3', 'D', 'P', 'frontier', 'sweep_refresh', 'water_aware')
COLORS = dict(zip(METHODS, ('#71808b', '#397fa0', '#8b6ca5', '#217d80', '#bc8a36',
                          '#b4554e', '#38464d', '#38464d', '#38464d')))
TASK_NAMES = {'E1': '自主探索', 'E2': '探索—搜救'}
PRIMARY = {'E1': 'first_coverage_auc', 'E2': 'rescue_fraction'}


def export_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return '—' if value is None else f'{value:.4f}'


def scientific(value):
    return '—' if value is None else f'{value:.6g}'


def figures(results, training, output):
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'})
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for axis, task in zip(axes, ('E1', 'E2')):
        key = PRIMARY[task]
        for position, method in enumerate(METHODS):
            rows = [r for r in results['parent_tables'] if r['task'] == task and r['method'] == method]
            values = [r[key] for r in rows]
            require(len(values) == 2 and all(v is not None for v in values), 'Missing parent means in primary figure')
            axis.scatter([position-.09, position+.09], values, color=COLORS[method], s=35, alpha=.8)
            average = float(np.mean(values))
            axis.plot([position-.22, position+.22], [average, average], color=COLORS[method], lw=2)
        axis.set(title='E1: exploration' if task == 'E1' else 'E2: exploration and rescue',
                 ylabel='First coverage AUC' if task == 'E1' else 'Rescued fraction', ylim=(-.03, 1.04))
        axis.set_xticks(range(len(METHODS)), ('B0', 'B1', 'B2', 'B3', 'D', 'P', 'frontier', 'sweep', 'water'), rotation=40)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    for extension in ('png', 'svg'):
        figure.savefig(output / ('primary-results.' + extension), dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for axis, task in zip(axes, ('E1', 'E2')):
        for method in METHODS[:6]:
            histories = [r['returns'] for r in training if r['task'] == task and r['method'] == method]
            require(len(histories) == 3 and all(len(h) == 1024 for h in histories), 'Learning curve matrix incomplete')
            mean = np.mean(histories, axis=0)
            smoothed = np.convolve(mean, np.ones(32)/32, mode='valid')
            axis.plot(range(32, 1025), smoothed, color=COLORS[method], label=method, lw=1.3)
        axis.set(title=task + ': fit rollouts during learning', xlabel='Training episode',
                 ylabel='Return, rolling32 mean')
        axis.grid(alpha=.2)
        axis.legend(ncol=3, frameon=False)
    for extension in ('png', 'svg'):
        figure.savefig(output / ('training-curves.' + extension), dpi=180)
    plt.close(figure)


def execute(output, forecast_path):
    require(not output.exists(), 'Refusing to overwrite an existing study report')
    plan, seal, proof, results, _ = validated_bundle()
    forecast = read(forecast_path)
    require(forecast['complete'] is True
            and forecast['schema'] == 'hyra-scale100-formal-auxiliary-diagnostic-v1'
            and forecast['policy_results_sha256'] == digest(EVALUATION / 'results.json')
            and forecast['model_seal_sha256'] == digest(EVALUATION / 'sealed-models.json')
            and forecast['release_audit_sha256'] == digest(EVALUATION / 'release-audit.json')
            and forecast['protocol_sha256'] == digest(ROOT / 'research/scale100-formal-forecast-protocol-20261026.json')
            and len(forecast['rows']) == 2052, 'Formal auxiliary diagnostic is missing or has different provenance')
    output.mkdir(parents=True)
    trials = [dict(trial_id=row['trial_id'], parent=row['parent'], task=row['task'], method=row['method'],
                   training_seed=row['training_seed'], scenario_seed=row['scenario_seed'],
                   return_=row['return_'], **row['metrics']) for row in results['rows']]
    export_csv(output / 'trials.csv', trials)
    export_csv(output / 'parent-means.csv', results['parent_tables'])
    export_csv(output / 'paired-effects.csv', [dict(task=row['task'], contrast=row['name'], **parent)
               for row in results['contrasts'] for parent in row['parent_differences']])
    export_csv(output / 'auxiliary-points.csv', forecast['rows'])
    training = []
    for job in plan['jobs']:
        history = read(FORMAL / job['id'] / 'history.json')
        training.append(dict(**job, returns=[row['return_'] for row in history]))
    figures(results, training, output)
    lines = ['# HyRA 100米双任务正式实验报告', '',
        '本报告来自36项实际训练、统一开发集选模、封存后的216项模型和36项规则留出运行，以及252项回放的全量复核。结果仅描述两个合成留出父场景，不能据此宣称统计显著、策略收敛、真实救援适用性或大图迁移成功。', '',
        '## 实验条件与复现范围', '',
        '地图100×100米、20×20原格、两机、900秒、每5秒一次八向相邻格动作。直向1米/秒，对角约1.41米/秒，各方向未统一物理速度。5×5感知窗口；21通道输入来自合法已到达观测和公开静态底床。E1无人员信号源及救援奖励；E2五名合成人员，连续10秒停留施救，服务条件为水深≤0.08米、流速≤0.3米/秒。这些阈值不是实际人员生存或飞行安全标准。', '',
        'B0/B1/B2/B3分别为原始观测QMIX、GMRF QMIX、原始观测Sur-QMIX、GMRF Sur-QMIX；D加入数据辅助，P在同样数据辅助上加入SWE物理约束。Sur增强只用于训练。全部方法具有同等合法信息、同一种子下的初始化、最大1024回合和1021次更新；训练种子11、29、47。选模只用开发集，在256/512/1024回合中按已冻结任务规则选择。选中的回合可以不同。', '',
        '双层GMRF及Sur-QMIX是引用和适配的已有方法。本轮改变机数、网格、观测和洪水任务等条件，不是原论文的严格性能复现；物理辅助没有进行模型内规划或生成虚拟强化学习经验。旧四方向研究及其负面结果保留。', '',
        '## 任务结果', '',
        '每个父场景先平均训练/场景种子，再对两个父场景等权平均。代理、像素、时间点和随机种子不算额外独立洪水事件。规则不具有模型训练种子。E1以首次覆盖AUC为主，E2以截止时施救比例为主；水情正确覆盖单列。', '',
        '| 任务 | 方法 | 首次覆盖AUC | 当前正确覆盖AUC | 施救比例 | 航程/米 | 冲突尝试 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in results['summary']:
        values = row['means']
        lines.append('| ' + ' | '.join([row['task'], row['method'],
            number(values['first_coverage_auc']), number(values['balanced_current_correct_coverage_auc']),
            number(values['rescue_fraction']), number(values['distance_m']),
            number(values['collision_attempt_events'])]) + ' |')
    lines += ['', f"![任务结果]({(output/'primary-results.png').resolve().as_posix()})", '',
        '图中每个点代表一个父场景的平均结果，横线为两个父场景的等权均值；没有把重复试验当成独立场景来构造误差条。', '',
        '## 物理约束的增量', '',
        'P−D为物理约束主要比较，P−B3为完整辅助机制比较。下表为P减对照，AUC和施救比例以百分点表达；正值只表示当前描述性结果更高，不代表可靠或显著改善。', '',
        '| 任务 | 比较 | 主指标差值/百分点 | 两父场景各自差值/百分点 |', '|---|---|---:|---|']
    for task in ('E1', 'E2'):
        for name in ('physics_increment', 'full_auxiliary'):
            row = next(r for r in results['contrasts'] if r['task'] == task and r['name'] == name)
            key = PRIMARY[task]
            parents = ', '.join(f"{p['parent']}: {p[key]*100:+.2f}" for p in row['parent_differences'])
            lines.append(f"| {task} | {'P−D' if name=='physics_increment' else 'P−B3'} | {row['mean_difference'][key]*100:+.2f} | {parents} |")
    lines += ['', 'GMRF、Sur及其二阶对照的逐父场景结果同时保存在paired-effects.csv；本矩阵不能辨识任意Sur×Physics交互效应。失败、未完成施救与删失记录保留在逐试验CSV和原始回放，不只统计成功回合。', '',
        '## 辅助重建与物理诊断', '',
        f"辅助诊断使用全部12个已封存的D/P任务模型，在相同规则生成的合法输入和上一动作序列上评分。实际时距为{forecast['min_horizon_s']}—{forecast['max_horizon_s']}秒求解器内部步长，不是5秒或长期预测。2052个评分点仍只来自两个相关的合成父场景。", '',
        '| 任务 | 方法 | 短步水深MAE/米 | 局部物理归一化MSE | 全域体积残差绝对均值/立方米 |',
        '|---|---|---:|---:|---:|']
    for row in forecast['summary']:
        lines.append(f"| {row['task']} | {row['method']} | {row['microstep_h_mae_m']:.6g} | {row['local_physics_normalized_mse']:.6g} | {row['volume_residual_abs_mean_m3']:.6g} |")
    lines += ['', '数据误差、局部残差和全域体积残差分别报告。合法记忆持久性保留比池化actor输入更细的原始格点记录；真实当前状态持久性使用特权真值，仅作离线短时变化参照，不能作为可部署方法。辅助误差更小不证明任务效果更好。', '',
        '## 训练与机制证据', '',
        f"![训练曲线]({(output/'training-curves.png').resolve().as_posix()})", '',
        '训练图为三个训练种子的均值再做32回合滑动平均，曲线只反映拟合场景训练过程，不能证明收敛或泛化。全部36项完整历史、初始化与所选权重摘要均由发布审计核对。', '',
        '| 辅助组任务 | 最后记录的数据共享梯度范数 | 物理共享梯度范数 | 当时机制有非零共享梯度 |',
        '|---|---:|---:|---|']
    for row in proof['training']:
        if row['shared_auxiliary_live_at_last_audited_update'] is None:
            continue
        gradient = row['last_shared_gradient'] or {}
        lines.append(f"| {row['id']} | {scientific(gradient.get('data_shared_norm'))} | {scientific(gradient.get('physics_shared_norm'))} | {row['shared_auxiliary_live_at_last_audited_update']} |")
    lines += ['', '梯度记录是已观测更新点的机制诊断，不证明所有更新均有同等作用。早期辅助头无效、精度问题和死激活版本仍保留在开发证据中；正式结果未据留出得分重选模型或重新调参。', '',
        '## 结论与边界', '',
        '本轮可以报告的是上表给出的配对描述性差异，以及所登记训练、数据权限、运动和施救流程通过了哪些核验。只有两个同一程序生成器的合成留出父场景，无法据此建立普遍、可靠的正向结论。E1和E2分别判断；残差下降不能替代任务收益。', '',
        '原始256×256地图的读取与规则接口保持可用，但本轮100米学习模型没有完成长时原始大图迁移验证；不能把页面支持大图当作学习策略大图收益。真实洪水率定、通信退化及真实无人机动力学也不在本轮验证范围内。', '',
        '## 可追溯文件', '',
        f"- 冻结方案：{(FORMAL/'plan.json').as_posix()}",
        f"- 模型封存：{(EVALUATION/'sealed-models.json').as_posix()}",
        f"- 252项逐试验结果：{(EVALUATION/'results.json').as_posix()}",
        f"- 全量回放审计：{(EVALUATION/'release-audit.json').as_posix()}",
        f"- 辅助诊断：{forecast_path.resolve().as_posix()}",
        '- 本目录CSV保留逐试验、逐父场景、配对效应及辅助诊断点；图像同时提供PNG和可编辑文字SVG。']
    (output / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    manifest = dict(schema='hyra-scale100-report-v1', complete=True, formal_models=36, policy_trials=252,
        independent_parents=2, claims_statistical_significance=False, claims_large_map_transfer=False,
        report_source_sha256=digest(__file__), formal_plan_sha256=digest(FORMAL/'plan.json'),
        release_audit_sha256=digest(EVALUATION/'release-audit.json'), auxiliary_sha256=digest(forecast_path),
        files_sha256={p.name:digest(p) for p in output.iterdir() if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--forecasts', type=Path, required=True)
    args = parser.parse_args()
    execute(args.output, args.forecasts)
