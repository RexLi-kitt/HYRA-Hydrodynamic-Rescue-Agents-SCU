"""Publish audited local-service results without changing frozen experiment code."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path); args = parser.parse_args()
    run = args.run.resolve(); result = read(run/'confirmation_summary.json')
    audit = read(run/'final_independent_audit.json'); assert audit['passed']
    condition_rows = read(run/'condition_family_scores.json')
    selected = read(run/'selection.json')['selected']
    ceilings = [v for v in audit['finite_oracle'] if v['split'] == 'confirmation']
    derived = [v for v in audit['derived'] if v['split'] == 'confirmation']
    headroom = float(np.mean([r['remaining_gap_over_A2'] for r in ceilings]))
    oracle = float(np.mean([r['finite_oracle_mean'] for r in ceilings]))
    margin_contribution = np.mean([v['A4_minus_A2_gain_contribution_by_truth_margin_bins'] for v in derived], axis=0)
    mode_rows = defaultdict(list); quality_rows = defaultdict(list)
    for r in condition_rows:
        e, mode, noise, delay = r['condition'].split('/')
        difference = (r['A4']-r['A2'])*100
        mode_rows[mode].append(difference); quality_rows[noise+'/'+delay].append(difference)
    common_windows = read(run/'common_window_diagnostic.json')
    window = {g: {'full_window_false_feasible': common_windows['means'][g]['false_feasible'],
                   'full_window_false_infeasible': common_windows['means'][g]['false_infeasible']} for g in selected}
    diagnostic = dict(finite_oracle_mean=oracle, finite_oracle_gain_over_A2_per100=headroom*100,
        practical_target_per100=1, maximum_available_gain_below_target=headroom < .01,
        margin_gain_contribution_per100=margin_contribution.tolist(), full_window_classification=window,
        A4_minus_A2_per100_by_mode={k: float(np.mean(v)) for k, v in mode_rows.items()},
        A4_minus_A2_per100_by_observation={k: float(np.mean(v)) for k, v in quality_rows.items()})
    (run/'interpretation.json').write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding='utf-8')
    font = FontProperties(family='DejaVu Sans')
    plt.rcParams.update({'axes.unicode_minus': False, 'font.family': font.get_name(), 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.25]})
    comps = result['comparisons']
    for y, c in enumerate(comps):
        mean = c['mean_gain_services_per_case']*100; lo, hi = np.asarray(c['interval'])*100
        axes[0].errorbar(mean, y, xerr=[[max(0, mean-lo)], [max(0, hi-mean)]], fmt='o',
                         color='#2563a6' if c['primary'] else '#687985', capsize=4)
    axes[0].axvline(0, color='#999', lw=1); axes[0].set_yticks(range(4), ['A4−A3：选择标准', 'A4−A2：普通更新', 'A4−A1：仅预测学习', 'A4−A0：名义预测'])
    axes[0].invert_yaxis(); axes[0].set_xlabel('每100个初始情境新增服务数')
    axes[0].set_title('48个确认家族：配对收益')
    axes[0].text(.02, -.28, '主要比较97.5%区间；次级比较95%区间\n预设实际意义目标：+1.00/100（超出左图范围）',
                 transform=axes[0].transAxes, fontsize=9)
    matrix = np.zeros((4, 4))
    qualities = ['noise0/delay0', 'noise0/delay10', 'noise0.02/delay0', 'noise0.02/delay10']
    for i, mode in enumerate(['M0', 'M1', 'M2', 'M3']):
        for j, quality in enumerate(qualities):
            values = [(r['A4']-r['A2'])*100 for r in condition_rows if '/'+mode+'/' in r['condition'] and r['condition'].endswith(quality)]
            matrix[i, j] = np.mean(values)
    limit = float(np.max(abs(matrix)))
    im = axes[1].imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit)
    for i in range(4):
        for j in range(4): axes[1].text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center', color='white' if abs(matrix[i,j]) > limit*.6 else '#222')
    axes[1].set_xticks(range(4), ['无噪声\n无延迟', '无噪声\n10秒延迟', '2厘米噪声\n无延迟', '2厘米噪声\n10秒延迟'])
    axes[1].set_yticks(range(4), ['近准确', '持续低估', '持续高估', '突发失配'])
    axes[1].set_title('A4相对普通更新：分条件收益/100')
    fig.colorbar(im, ax=axes[1], shrink=.75, label='每100情境新增服务')
    fig.subplots_adjust(left=.17, right=.97, top=.86, bottom=.28, wspace=.50)
    fig.savefig(run/'results_overview.png', dpi=180)
    fig.savefig(run/'results_overview.pdf'); plt.close(fig)
    lines = ['# HyRA 本地残差与服务遗憾：执行结果', '',
             '2026-10-15。已完成24训练、12验证、12先导和48独立确认家族；全部采用固定观测、单机两已知目标、初始选序。', '',
             '**本轮未通过推进门槛：MAE与任务收益选择器选中同一历史模型；该模型相对普通滤波有很小的平均正增益，但未达到预设实际意义目标。**', '',
             '## 主要结果', '', '|比较|新增服务/100情境|家族区间/100|判断|', '|---|---:|---|---|']
    for c in comps:
        lo, hi = np.asarray(c['interval'])*100
        decision = '两个选择器完全相同' if c['baseline'] == 'A3' else '小幅正增益，低于预设1/100' if c['baseline'] == 'A2' else '预设次级比较'
        lines.append(f"|A4−{c['baseline']}|{c['mean_gain_services_per_case']*100:+.4f}|[{lo:+.4f},{hi:+.4f}]|{decision}|")
    lines += ['', 'A3/A4/A5均为 history_ridge_10；A2为 kf_rw_q1e-07_r0.005。两个主要比较使用97.5%双侧家族bootstrap区间，次级使用95%。A4−A2区间下界略高于零，不能写成“完全没有平均收益”；其幅度仍很小，48家族中19正、11零、18负，不能称为跨家族一致优势。', '',
              f'![结果总览]({(run/"results_overview.png").as_posix()})', '',
              '## 任务本身的改进空间', '',
              f'普通滤波平均完成 {result["mean_yield"]["A2"]:.6f}/2 个服务；两候选顺序的真值最优为 {oracle:.6f}/2。即便拥有未来真值，最多也只能比普通滤波多完成 **{headroom*100:.4f}/100** 个服务，低于本轮预设目标1/100。', '',
              '这暴露了设计门槛的不足：先导只要求有顺序敏感任务和非零差值方差，没有把普通基线的剩余可挽回损失与实际意义目标联立。确认测试不能再据此重选布局或降低门槛；后续新协议应先检查可挽回空间，再决定是否开展方法比较。本轮结果不能解释成一般残差方法无用。', '',
              '## 哪些条件受益，哪些条件受损', '', '|失配条件|A4−A2，服务/100|', '|---|---:|']
    mode_labels = {'M0': '近准确', 'M1': '持续低估', 'M2': '持续高估', 'M3': '突发失配'}
    for m in mode_labels: lines.append(f'|{mode_labels[m]}|{np.mean(mode_rows[m]):+.4f}|')
    lines += ['', '这些分层是预设次级描述，不能用其中最好的一格替代主总体。近准确和突发失配条件平均受损，持续高估/低估受益；总体改善不能直接命名为突发预警优势。', '',
              '## 窗口判断与局部脉冲', '', '|组别|假可行率|假不可行率|', '|---|---:|---:|']
    for g in ['A0', 'A1', 'A2', 'A4']:
        lines.append(f'|{g}|{window[g]["full_window_false_feasible"]*100:.4f}%|{window[g]["full_window_false_infeasible"]*100:.4f}%|')
    lines += ['', '这里所有方法使用A0名义顺序及其服务开始时刻形成的同一组完整12秒区间，分母为两个共同计划区间，真值不参与挑区间。它是次级窗口诊断，与真实执行失败率分开，不能当作校准后的成功概率。方法各自预测开始时刻的补充诊断另保留在独立核查JSON，不混用两个分母。', '',
              f'固定E1真实水情、换成配对E0观测的离线检查中，A4平均脉冲观测效应为 {result["mean_pulse_observation_effect"]["A4"]*100:+.4f}/100；A0/A1负对照为零。这是当前固定模型的配对信息效应描述，没有证明物理方向、时滞或独立物理关联收益。', '',
              '## 对 plus 框架的含义', '',
              '本地残差历史能改善部分完整服务误判，且在新的受控家族上相对仅预测学习有小幅任务价值。任务选模优于MAE的主张本轮未获支持；超出普通更新的实际幅度目标未达到。水动力关联、多机滚动规划、主动观测和Sur-QMIX仍未在本轮验证。按预设门槛停止扩展，不启动更大策略训练。', '',
              '## 源码实际使用及核查', '',
              '新洪水与逐次起报使用现有ANUGA4.0.1。FilterPy从“残差收益验证/vendor”固定提交源码直接导入，完成21组参数×32种历史=672条参考预测核验；最大差约4.44×10^-16米。批量重放使用经其核对的向量化等价实现，未安装新依赖。其余已下载库按后续研究阶段保留。', '',
              f'25项相关测试通过；独立核查覆盖{audit["saved_batches"]:,}个保存批次、{audit["actual_input_references"]:,}条实际引用、{audit["scalar_order_checks"]:,}次标量顺序评分及{audit["paired_observation_difference_checks"]:,}个配对观测差。源代码冻结、家族隔离、共享观测与条件权重复算均通过。', '',
              f'运行目录：[{run.name}]({run.as_posix()})。', '',
              f'- [主要结果与完整家族差值]({(run/"confirmation_summary.json").as_posix()})',
              f'- [独立核查及窗口指标]({(run/"final_independent_audit.json").as_posix()})',
              f'- [冻结清单]({(run/"freeze_manifest.json").as_posix()})',
              f'- [FilterPy源码核验]({(run/"filterpy_reference_check.json").as_posix()})',
              f'- [共同窗口诊断]({(run/"common_window_diagnostic.json").as_posix()})',
              f'- [下载源码位置]({(ROOT/"vendor/github_20261015_220812/README.md").as_posix()})',
              '', '本轮96个基础家族及其物理/预测/观测派生版本全部保留；不将时刻、相位、噪声重复或预测版本算作独立家族。后续改布局、阈值、候选池或评价总体必须另开协议与新确认家族。']
    report = ROOT.parent/'实验报告/20261015/HyRA_本地残差确认实验结果_20261015.md'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (run/'AUDITED_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    latest_path = ROOT/'results/latest_local_service.json'
    latest = read(latest_path); latest.update(audited_report=str(report), independent_audit=str(run/'final_independent_audit.json'), audit_passed=True)
    latest_path.write_text(json.dumps(latest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(report=str(report), finite_oracle_gain_over_A2_per100=headroom*100, plot=str(run/'results_overview.png')), ensure_ascii=True))


ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__': main()
