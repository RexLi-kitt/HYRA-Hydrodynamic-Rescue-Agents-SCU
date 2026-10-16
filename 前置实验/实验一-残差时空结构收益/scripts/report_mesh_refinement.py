"""Publish the bounded review only after the saved-artifact audits pass."""
import argparse
import csv
from pathlib import Path
import sys

from audit_pulse_service_alignment import read

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    summary = read(run/'summary.json'); audit = read(run/'independent_audit.json')
    opportunity_audit = read(run/'opportunity_audit.json')
    assert audit['passed'] and opportunity_audit['passed']
    with (run/'bed_readout.csv').open(encoding='utf-8-sig', newline='') as stream:
        beds = list(csv.DictReader(stream))
    lines = ['# HyRA 有预算的40→80网格复核', '',
        '2026-10-16。复用此前按旧结果选择的4个训练诊断家族；新积分已完成，两项保存数据审核通过。未读取旧确认家族，未训练P4/P5，未开启新确认。', '',
        '**网格40→80的本轮工程门槛：'+('通过。' if summary['numerical_gate_passed'] else '未通过。')+'** 有限数值配置中的一致性不能替代真实水动力精度证明。', '',
        '## 设置与覆盖', '',
        '同一物理来水区间[10,30]/[60,80]米、同一固定取样点、180/210秒锚点、32布局、E0/E1和全部共同服务分支。最近质心与固定点质心三角剖分凸权重线性读取同时保留。新网格80采用2秒输出，与保存的网格40比较。', '',
        f"新积分{summary['new_field_integrations']}场，其中{summary['nominal_restarts']}次预测只从同网格独立名义状态重启。预算上限为56场、网格80；本轮不会继续自动加密。", '',
        '## 结果', '',
        '|读取|共同区间数|最大区间水深变化|标签变化|', '|---|---:|---:|---:|']
    for method, label in [('nearest', '最近质心'), ('linear', '固定点线性')]:
        row = summary['comparisons'][method]
        lines.append(f"|{label}|{row['windows']}|{row['max_window_change_m']*1000:.3f}毫米|{row['label_changes']}|")
    lines += ['', f"线性读取的目标全轨迹最大变化{summary['max_target_depth_change_m']*1000:.3f}毫米；名义探针时滞最大变化{summary['max_nominal_lag_change_s']:.1f}秒。原门槛分别是5毫米、2秒和共同区间标签零变化，逐项结果见summary.json，未按结果放宽。", '',
        '|预先确定的诊断分组|区间数|40→80线性标签变化|扩展配置包络不一致|', '|---|---:|---:|---:|']
    for group, label in [('old_disagreements', '全部旧不一致区间'), ('stable_controls', '预先选定稳定对照'), ('all', '全体')]:
        row = summary['coverage'][group]
        lines.append(f"|{label}|{row['windows']}|{row['label_changes']}|{row['expanded_disagreements']}|")
    lines += ['', f"全部{summary['parent_disagreements']}个旧不一致区间均覆盖。稳定对照在新积分前按家族、锚点、边界位置、事件和目标分层选最近阈值及最远阈值区间，索引保存在coverage_selection.json；实际重算仍覆盖全部18432区间，没有按新结果删除样本。", '',
        f"加入网格80两种读取后，完整历史配置集合的包络有{summary['expanded_disagreements']}个区间跨越0.8米。包络只表示测试配置范围，不是概率区间或严格误差界。", '',
        '## 床面与取样', '', '|网格|线性读取床面相对解析床面最大差|最近质心最大位置偏差|', '|---|---:|---:|']
    for mesh in (40, 80):
        selected = [r for r in beds if int(r['mesh']) == mesh]
        bed_error = max(float(r['linear_max_analytic_difference_m']) for r in selected)
        location_error = max(float(r['nearest_max_distance_m']) for r in selected)
        lines.append(f'|{mesh}|{bed_error*1000:.3f}毫米|{location_error:.3f}米|')
    lines += ['', '床面比较单独保存每个点的解析值与读取值；它不能把演化水位误差分解成独立可相加的误差来源。线性读取是项目中的固定点取样对照，不冒称ANUGA原生高阶重构。', '',
        '## 离线行动机会', '',
        f"旧有限配置集合有{summary['parent_robust_opportunities']}个一致增益情境；加入网格80两种读取后保留{summary['expanded_robust_opportunities']}个。进一步要求所有配置下配对参考响应满足预定可见性条件，零延迟有{summary['expanded_visible_D0']}个情境×相位计数，10秒延迟有{summary['expanded_visible_D10']}个。", '',
        '这些是同一替代顺序在有限配置集合中均优于冻结原顺序的离线诊断。配对参考响应不可在线获得，不能将其当检测器输出、模型收益或相对强基线的确认结果。', '',
        '## 时滞输入的可辨识性', '']
    support = read(run/'refined_kernel_support.json')
    assert support['all_families_complete']
    lines += ['使用现有同侧候选边的等权版本及细网格独立名义探针时滞，按实际采样相位和延迟比较归一化物理核与新近性核。此处只检查输入权重是否不同，不代表水情已在线检测或预测有效。', '',
        '|延迟|预测时域|权重L1差异≥0.1比例|', '|---|---:|---:|']
    for row in support['summary']:
        lines.append(f"|{row['delay_s']}秒|{row['horizon_s']}秒|{row['fraction_L1_ge_point1']*100:.1f}%|")
    lines += ['', '完整逐情境权重诊断保存在refined_kernel_support.csv。现有同侧候选边是诊断假设，后续应核对跨侧传播，不能把地理分组直接当成完整物理连通。', '',
        '## 代码与审核', '',
        f"保存数据审核复算{audit['raw_fields']}份原始场、{audit['point_projection_scalar_values']}个点投影标量、{audit['nominal_restarts']}次名义重启和{audit['scalar_window_evaluations']}次区间最大值；另独立复算{opportunity_audit['scalar_order_evaluations']}次顺序评分及可见性标记。审核范围与计数见两份audit JSON。", '',
        '新增输入契约模块仅准备了共同六源记录、实际版本与到达时刻、匹配物理/新近性归一化；它不是已完成的方法比较。相关输入测试5项、数值与复核辅助测试6项通过。', '',
        '## 下一步与结论边界', '',
        ('本轮配对工程检查通过，可以据此拟定固定数值契约，再核查可辨识性、强基线剩余空间及完整行动窗口。旧粗网格集合的不确定标签仍需报告。' if summary['numerical_gate_passed'] else
         '本轮配对工程检查仍未通过。按预算在网格80停止，将数值不确定区间保留；下一阶段先固定数值契约及跨配置风险/收益报告规则，检查物理传播、观测、移动与服务尺度是否匹配。任何新的场景尺度或标签协议都要另存训练期设计，再进入方法比较，不能把未通过写成通过。'), '',
        '完整目标仍为：无目标本地观测时，物理连通与时滞相对相同上游原始记录的强基线有额外价值，并在来得及改变动作时增加真实完成服务。当前没有完成这项确认。', '',
        f'- [运行数据]({run.as_posix()})',
        f'- [汇总]({(run/"summary.json").as_posix()})',
        f'- [全部区间]({(run/"windows.csv").as_posix()})',
        f'- [原始场审核]({(run/"independent_audit.json").as_posix()})',
        f'- [行动机会审核]({(run/"opportunity_audit.json").as_posix()})',
        '- GitHub来源检查：复用现有ANUGA实现；Empirical_LocalPatch去趋势与局地化、FloodGNN-GRU图时间编码仍属于方法参考，未冒称完成原算法复现。', '']
    report = ROOT.parent/'实验报告/20261016/HyRA_有预算网格复核结果_20261016.md'
    report.write_text('\n'.join(lines), encoding='utf-8')
    pointer = read(ROOT/'results/latest_numerical_refinement.json')
    pointer.update(report=str(report), report_pending=False, opportunity_audit_passed=True,
                   next_action='freeze_fixed_point_contract_then_actionable_time_and_strong_baseline_headroom_calibration'
                   if summary['numerical_gate_passed'] else summary['next_action'])
    write_json(ROOT/'results/latest_numerical_refinement.json', pointer)
    print(report)


if __name__ == '__main__': main()
