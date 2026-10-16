"""Finalize the calibration report, retaining generated status before adjudication."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, data): Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve(); root = run.parents[2]
    audit = read(run/'independent_audit.json'); assert audit['passed']
    summary = read(run/'summary.json'); grid = read(run/'positive_case_grid_review.json')
    families = [read(p) for p in sorted((run/'families').glob('local_v1_train_???.json'))]
    assert len(families) == 24
    rows = [r for f in families for r in f['opportunities'] if r['population'] == 'critical']
    gap = sum(r['gap_sum'] for r in rows); retained = sum(r['retained_cases'] for r in rows)
    visible_families = [f['family'] for f in families if any(a['critical_paired_response_and_oracle_times_s']
                        for a in f['actions'] if a['delay_s'] == 0)]
    offsets = Counter(); anchors = Counter()
    for f in families:
        for a in f['actions']:
            if a['delay_s'] == 0:
                for t in a['critical_paired_response_and_oracle_times_s']:
                    offsets[t-a['anchor_s']] += 1; anchors[a['anchor_s']] += 1
    macro = next(r for r in summary['opportunity'] if r['population'] == 'critical' and r['condition'] == 'both')
    next_action = 'repair_numerical_stability_before_model_or_H5_confirmation'
    assessment = dict(stage_A_completed=True, training_families=24, confirmation_opened=False,
        initial_fixed_grid_existence_gate_passed=summary['grid_gate_passed'],
        positive_case_numerical_sensitivity_gate_passed=grid['core_existence_reversals'] == 0,
        ready_for_model_or_H5_confirmation=False,
        critical_headroom_over_nominal_family_macro_per100=macro['headroom_over_nominal_per100_family_macro'],
        critical_headroom_over_nominal_pooled_per100=100*gap/retained,
        critical_supported_families=macro['supported_families'],
        visible_opportunity_families=visible_families, visible_counts_by_offset=dict(offsets),
        visible_counts_by_anchor=dict(anchors),
        positive_case_coarse_count=grid['coarse_visible_total'],
        positive_case_fine_count=grid['fine_visible_total_on_selected_pairs'],
        positive_case_existence_reversals=grid['core_existence_reversals'],
        no_full_fine_grid_population_estimate=True,
        B_star_headroom_gate_tested=False,
        next_action=next_action,
        generated_status_correction='The runner unconditionally suggested expansion after n=24; source preserved. Final status incorporates completed expansion and exploratory positive-case numerical failure.')
    for filename in ('summary.json', 'completion.json', 'REPORT.md'):
        original = run/(Path(filename).stem+'.generated'+Path(filename).suffix)
        if not original.exists(): shutil.copyfile(run/filename, original)
    summary['next_action'] = next_action
    summary['final_numerical_gate_passed'] = False
    summary['positive_case_grid_review'] = 'positive_case_grid_review.json'
    summary['B_star_headroom_gate_tested'] = False
    write(run/'summary.json', summary)
    completion = read(run/'completion.json'); completion.update(next_action=next_action,
        final_numerical_gate_passed=False, initial_fixed_grid_gate_passed=completion['grid_gate_passed'])
    write(run/'completion.json', completion)
    write(run/'postrun_assessment.json', assessment)
    counts = audit['counters']
    links = lambda filename: f'[{filename}]({(run/filename).as_posix()})'
    lines = ['# HyRA 扰动—服务窗口校准结果', '',
        '结论：A阶段已完成24个训练家族；当前数值稳定性不足，不进入物理模型收益确认或QMIX训练。', '',
        '这是旧训练家族上的设计校准，不是新确认实验。集中式单机、两个已知目标、六个共同上游传感器、目标无本地观测；每次完整服务12秒。考察90/120/150/180/210秒锚点，两个边界脉冲位置及锚点后0/6/12秒的初始选序，绝对截止为锚点+60秒。没有执行途中改航。', '',
        '## 任务空间', '',
        '|统计口径|相对名义预测的有限顺序真值上限增量（服务/100情境）|', '|---|---:|',
        '|一般任务，24家族等权|0.1389|',
        f"|临界任务，有支持的18家族等权|{macro['headroom_over_nominal_per100_family_macro']:.4f}|",
        f'|临界任务，合并全部保留情境|{100*gap/retained:.4f}|', '',
        f'临界筛选只读取锚点已发布名义预测，保留{retained}/46080个派生情境。1440个家族×锚点×位置×事件×决策偏移单元中1224个为空，6个家族没有临界任务。两种临界均值对应不同权重；没有选较大者替换原先家族等权口径。它们仍是相对名义预测的设计诊断，强基线B*尚未拟合，正式B*门槛尚未检验。', '',
        '家族等权的空间低于预设最低实际增益1/100；合并口径虽高于1/100，仍未给出2/100缓冲的设计依据。以上是这批训练设计的描述，不能推出所有新家族或所有残差方法的理论上限。', '',
        '## 可行动时间', '',
        '|观测延迟|临界任务×相位决策|真值改序可能改善|配对源响应已可见且仍可能改善|', '|---:|---:|---:|---:|',
        '|0秒|9472|192|105|', '|10秒|9472|192|0|', '',
        '此处“已可见”是离线必要条件上界：保留的实际历史记录中，相关上游源的E1−E0无噪声响应至少1毫米。它使用参考配对轨迹，不能部署为检测器，也不证明有噪声时模型能发现扰动。相位和噪声重复不增加独立家族数。', '',
        f'105个计数全部来自210秒锚点及4个家族：{", ".join(visible_families)}；其中决策偏移6秒为33，12秒为72。10秒延迟的零值只适用于本轮三个初始决策偏移，不能外推为所有后续时刻或途中状态均无机会。', '',
        '## 数值检查改变了放行判断', '',
        '预先固定的网格10→20抽查覆盖首批中的2个家族、8个锚点×位置对照，未发现“机会是否存在”的翻转，因此按计划扩展到24家族。但是这些对照没有覆盖随后出现正例的4个家族，不能据此宣称正例稳定。', '',
        '为检查正例，另做探索性数值敏感性复算：保留粗网格名义筛选与动作，重算全部4个正例家族的8个正例锚点×位置组合。该检查是在训练结果出现后追加，不冒称预先固定的确认检验。', '',
        '|检查项|结果|', '|---|---:|',
        f"|正例组合的可行动机会存在性翻转|{grid['core_existence_reversals']}/8|",
        f"|相同选定组合的粗网格乐观可行动计数|{grid['coarse_visible_total']}|",
        f"|相同选定组合的细网格乐观可行动计数|{grid['fine_visible_total_on_selected_pairs']}|", '',
        '47不是整个细网格总体的机会估计：本次仅复算了粗网格正例所在组合，没有完整扫描细网格可能新增的正例。翻转已经足以说明当前少量正例对数值设置敏感；尚未区分水动力离散、最近质心取样映射和阈值附近标签变化各自的贡献。', '',
        '名义传播时滞也随网格改变：首批抽查最大变化10秒，追加正例抽查最大变化8秒。与主要6/12秒预测时域相比不可忽略，时滞先验尚未证明稳定。当前不应据此评价物理时滞学习器的优劣。', '',
        '## 对 plus 的含义与后续顺序', '',
        '- H2：名义有响应边上的物理核与新近性在6/12秒通常可以区分，但尚未进行P4/P5误差比较，时滞数值稳定性也未通过。输入有差异不等于物理约束有效。',
        '- H3：测试网格中存在少量无延迟的离线机会；通信延迟及数值离散均会削弱它。尚未证明可部署检测或模型实际可利用。',
        '- H4/H5：本轮未训练方法，未验证完整窗口预测改善或真实增益。机会表不是方法收益表。', '',
        '下一项应先做固定正例及配对负例的网格10/20/40、时间输出步长与一致点位取样检查；把服务阈值附近的数值不确定区间单列，报告全体覆盖，不按是否有收益删除案例。参考真值与独立名义预测应在一致的数值精度下重新核验，继续禁止用真值初始化预测。', '',
        '数值结果稳定后，才重新审查任务总体及上游传感—目标布局能否提供足够时间和强基线剩余空间。若调整物理/任务尺度，另开协议和新确认家族；不降低已设目标或在旧确认集调参。随后才比较P0—P5，再考虑多机。', '',
        '## 核查与复现', '',
        '8项针对性测试通过。独立审核的“通过”表示算术、版本与引用复算一致，不表示数值收敛或研究假设通过。', '',
        f"- 原始物理轨迹哈希：{counts['raw_trajectory_hashes']}条（包含加密复算，仍只有24个基础家族）。",
        f"- 独立BFS与标量顺序评分：{counts['independent_scalar_orders']:,}次。",
        f"- 因果观测通道引用：{counts['causal_channel_references']:,}条；观测标量值重建：{counts['observed_scalar_values_checked']:,}个。",
        f"- 机会单元：{counts['opportunity_cells']:,}；行动集合：{counts['action_sets']:,}；传播边：{counts['propagation_edges']:,}；核权重比较：{counts['kernel_rows']:,}。",
        f"- 追加正例网格比较独立复算：{counts.get('posthoc_grid_positive_pair_checks', 0)}组。", '',
        '本阶段实际运行ANUGA；没有借用未接入的其他下载库冒称算法实现。FilterPy用于后续强普通更新对照，其前轮源码核验继续保留。', '',
        '原始传播、任务机会和行动时刻表：'+', '.join(links(n) for n in ('edges.csv', 'opportunities.csv', 'actions.csv'))+'。', '',
        '复核文件：'+', '.join(links(n) for n in ('independent_audit.json', 'positive_case_grid_review.json', 'postrun_assessment.json'))+'。', '',
        '脚本：'+', '.join(f'[{n}]({(root/"scripts"/n).as_posix()})' for n in (
            'calibrate_pulse_service_alignment.py', 'audit_pulse_service_alignment.py', 'stresscheck_alignment_positive_cases.py'))+'。', '',
        '原执行器在24家族完成后仍输出“继续扩展至24”的静态提示；本次保留*.generated原文件，修正最终状态，并将追加数值检查纳入放行判断。冻结的运行源代码及原始数据没有改写。']
    content = '\n'.join(lines)+'\n'
    (run/'FINAL_REPORT.md').write_text(content, encoding='utf-8')
    auto_report = (run/'REPORT.generated.md').read_text(encoding='utf-8')
    (run/'REPORT.md').write_text('> 最终放行状态：追加正例网格检查未通过。请以[最终报告](FINAL_REPORT.md)为准。\n\n'+
        auto_report.replace('下一步：按计划继续其余旧训练家族校准。', '下一步：24个训练家族已完成；先修复数值稳定性，不进入方法收益确认。'), encoding='utf-8')
    report = root.parent/'实验报告/20261015/HyRA_扰动窗口校准结果_20261015.md'
    report.write_text(content, encoding='utf-8')
    files = [p for p in run.rglob('*') if p.is_file() and p.name != 'final_integrity_manifest.json']
    manifest = dict(stage_A_finalized=True, final_report=str(report),
        report_sha256=sha(report), files={str(p.relative_to(run)):sha(p) for p in files})
    write(run/'final_integrity_manifest.json', manifest)
    write(root/'results/latest_pulse_service_alignment.json', dict(run=str(run), report=str(report),
        completed=True, numerical_gate_passed=False, next_action=next_action))
    print(json.dumps(dict(report=str(report), final_numerical_gate_passed=False,
        trajectories=counts['raw_trajectory_hashes'], audited_positive_pairs=counts.get('posthoc_grid_positive_pair_checks', 0)), ensure_ascii=False))


if __name__ == '__main__': main()
