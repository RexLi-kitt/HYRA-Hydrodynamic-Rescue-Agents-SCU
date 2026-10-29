"""Independent completeness and aggregation audit for the two finished P3 rounds."""
from collections import defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyra_uav.experiments import write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def score_audit(root):
    plan, summary = read(root/'plan.json'), read(root/'summary.json')
    assert summary['status'] == 'completed'
    holdout = [e for e in read(root/'index.json') if e['split'] == 'holdout']
    by_run = {(e['id'], e['route']): e for e in holdout}
    seen, aggregates, na = set(), defaultdict(lambda: [0., 0, 0., 0, 0., 0]), defaultdict(lambda: [0, 0])
    with (root/'metrics.jsonl').open(encoding='utf-8') as file:
        for line in file:
            row = json.loads(line)
            case, route, group, seed = (row[k] for k in ('case', 'route', 'group', 'seed'))
            fraction, band, now = (row[k] for k in ('fraction', 'band', 'time_s'))
            assert (case, route) in by_run and group in plan['groups'] and seed in plan['seeds']
            assert fraction in plan['fractions'] and band in ('all', 'age_0_60', 'age_60_180', 'age_180_plus')
            assert now in range(10, 600, 10)
            key = case, route, group, seed, fraction, band, now
            assert key not in seen
            seen.add(key)
            n, positive, k = row['n'], row['positives'], row['k']
            assert 0 <= positive <= n and k == (max(1, math.ceil(n*fraction)) if n else 0)
            if n:
                assert 0 <= row['precision'] <= 1
                hit = round(row['precision']*k)
                assert abs(row['precision']*k-hit) < 1e-8 and 0 <= hit <= min(k, positive)
                assert abs(row['prevalence']-positive/n) < 1e-12
                if positive:
                    assert abs(row['recall']-hit/positive) < 1e-12
                else:
                    assert row['recall'] is None
            else:
                assert row['precision'] is None and row['recall'] is None and row['prevalence'] is None
            event_key = case, group, fraction, band
            na[event_key][0] += n == 0
            na[event_key][1] += positive == 0
            accumulator = aggregates[event_key+(seed, route)]
            for i, field in enumerate(('precision', 'recall', 'prevalence')):
                if row[field] is not None:
                    accumulator[2*i] += row[field]
                    accumulator[2*i+1] += 1
    expected = 9*3*59*5*3*3*4
    assert len(seen) == summary['metrics_rows'] == summary['expected_metrics_rows'] == expected
    assert len(summary['events']) == 540
    for event in summary['events']:
        key = tuple(event[k] for k in ('case', 'group', 'fraction', 'band'))
        assert na[key] == [event['empty_times'], event['no_positive_times']]
        for i, field in enumerate(('precision', 'recall', 'prevalence')):
            seeds = []
            for seed in plan['seeds']:
                routes = []
                for route in ('R0', 'R1', 'R2'):
                    total = aggregates[key+(seed, route)]
                    if total[2*i+1]:
                        routes.append(total[2*i]/total[2*i+1])
                if routes:
                    seeds.append(sum(routes)/len(routes))
            value = sum(seeds)/len(seeds) if seeds else None
            assert (value is None and event[field] is None) or abs(value-event[field]) < 1e-12
    return dict(status='passed', rows=expected, event_summaries=540,
                checks=['all score keys unique and in registered domain', 'exact ceil selection budget',
                        'precision/recall/positive-count consistency', 'empty and no-positive NA preserved',
                        'independent time-route-seed aggregation of precision, recall and prevalence'])


def archive_audit(root):
    provenance, plan = read(root/'runtime-provenance.json'), read(root/'plan.json')
    assert provenance['plan_sha256'] == digest(root/'plan.json')
    assert provenance['source_zip_sha256'] == digest(root/'frozen-sources.zip')
    with zipfile.ZipFile(root/'frozen-sources.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == set(provenance['files_sha256'])
        for name, expected in provenance['files_sha256'].items():
            assert sha256(archive.read(name)).hexdigest() == expected
        for name, expected in plan['version']['source_files_sha256'].items():
            assert sha256(archive.read('uav-workbench/'+name.replace('\\', '/'))).hexdigest() == expected
    return dict(status='passed', files=len(provenance['files_sha256']), source_zip_sha256=provenance['source_zip_sha256'])


def gate_audit(root):
    summary, plan = read(root/'summary.json'), read(root/'plan.json')
    main = {(r['case'], r['group']): r['precision'] for r in summary['events']
            if r['fraction'] == .1 and r['band'] == 'all'}
    for row in summary['paired']:
        for group in ('B1', 'B2', 'A1', 'A2'):
            delta = main[row['case'], 'B3']-main[row['case'], group]
            assert abs(delta-row['delta_'+group]) < 1e-12
    means = {g: float(np.mean([r['delta_'+g] for r in summary['paired']]))
             for g in ('B1', 'B2', 'A1', 'A2')}
    assert all(abs(means[g]-summary['mean_deltas'][g]) < 1e-12 for g in means)
    positive = sum(r['delta_B1'] > 0 and r['delta_B2'] > 0 for r in summary['paired'])
    assert positive == summary['positive_events']
    controls = [a['relation'] for a in read(root/'case-audits.json') if a['split'] == 'holdout']
    valid = sum(a['mean_operator_total_variation'] >= .05 and a['changed_positive_weight_fraction'] >= .5
                and a['support_exact'] and a['distance_weighted_mass_exact'] for a in controls)
    assert valid == summary['valid_control_events']
    criterion = plan['gate']
    checks = {f'mean_increment_{g}': means[g] >= criterion['min_mean_increment'] for g in ('B1', 'B2')}
    checks.update(positive_events=positive >= criterion['min_positive_events'],
                  A2_increment=means['A2'] >= criterion['min_A2_increment'],
                  A1_increment=means['A1'] >= criterion['min_A1_increment'],
                  valid_controls=valid >= criterion['min_valid_control_events'])
    for mechanism in ('slow', 'rapid', 'async'):
        rows = [r for r in summary['paired'] if r['mechanism'] == mechanism]
        checks[mechanism+'_positive_events'] = sum(r['delta_B1'] > 0 and r['delta_B2'] > 0 for r in rows) >= criterion['min_positive_per_mechanism']
        for group in ('B1', 'B2'):
            delta = float(np.mean([r['delta_'+group] for r in rows]))
            assert abs(delta-summary['mechanism_deltas'][mechanism][group]) < 1e-12
            checks[mechanism+'_positive_mean_'+group] = delta > 0
    assert all(checks.values()) == summary['gate_passed']
    return dict(status='passed', criteria=checks, gate_passed=all(checks.values()),
                failed_criteria=[key for key, passed in checks.items() if not passed])


def report_round(root, summary):
    round_id = summary['round']
    main = [r for r in summary['events'] if r['band'] == 'all' and r['fraction'] == .1]
    means = {g: float(np.mean([r['precision'] for r in main if r['group'] == g]))
             for g in ('B1', 'B2', 'B3', 'A1', 'A2')}
    controls = [a['relation'] for a in read(root/'case-audits.json') if a['split'] == 'holdout']
    lines = [f'# P3 非线性第{round_id}轮结果（2026-10-25）', '',
             f"完整训练和留出评分已完成；预登记升级门槛{'通过' if summary['gate_passed'] else '未通过'}。"
             f"B3 相对 B1 为 {summary['mean_deltas']['B1']*100:+.3f} 个百分点，相对 B2 为 {summary['mean_deltas']['B2']*100:+.3f} 个百分点；"
             f"{summary['positive_events']}/9 个事件同时正向。", '',
             '| 方法 | 事件等权 Precision@10% |', '|---|---:|']
    lines += [f'| {g} | {means[g]*100:.3f}% |' for g in means]
    lines += ['', '## 逐事件差异', '', '| 事件 | B3−B1 百分点 | B3−B2 百分点 | B3−A1 百分点 | B3−A2 百分点 |', '|---|---:|---:|---:|---:|']
    for row in summary['paired']:
        lines.append('| '+row['case']+' | '+' | '.join(f"{row['delta_'+g]*100:+.3f}" for g in ('B1','B2','A1','A2'))+' |')
    lines += ['', f'![逐事件配对差异](evidence/20261025-p3-nonlinear-round{round_id}/paired-events.png)', '',
              '## 机制、消融与预算', '', '| 变化机制 | B3−B1 百分点 | B3−B2 百分点 |', '|---|---:|---:|']
    lines += [f"| {m} | {v['B1']*100:+.3f} | {v['B2']*100:+.3f} |" for m, v in summary['mechanism_deltas'].items()]
    lines += ['', f"A2 在 {summary['valid_control_events']}/9 个事件达到登记扰动门槛；算子总变差范围 "
              f"{min(a['mean_operator_total_variation'] for a in controls):.3f}—{max(a['mean_operator_total_variation'] for a in controls):.3f}，"
              f"正权重改变比例 {min(a['changed_positive_weight_fraction'] for a in controls)*100:.1f}%—{max(a['changed_positive_weight_fraction'] for a in controls)*100:.1f}%。"
              '支持、未加权度数及按距离分层的向外权重质量匹配；向内加权质量未精确匹配，其差异在 case-audits.json 披露。', '',
              '本轮 A1 为目标与全部邻区只保留最新观测的严格消融，过去标签和推导变化均移除。'
              '它故意减少历史信息；不能声称五组原始信息完全相同。B1/B2/B3/A2 的原始历史与静态先验一致。', '',
              '每组两个候选、64 棵树选型，最终三个种子各 128 棵树，10 次选型拟合和 15 个最终模型。'
              '所有最终拟合仅用拟合集，选型集不加入最终训练。模型维数均为 1448；实际树节点数不同并完整披露，不能称相同独立神经参数数。'
              '全量缓存、候选资格、标签、历史、严格 A1 和实际模型已由独立审计器检查。', '',
              f"评分 {summary['metrics_rows']} 行、540 项事件/年龄/比例汇总，空候选和无正例 NA 保留。"
              '九个留出事件只属于三个母地形家族；同家族事件相关，路线与种子不能充当额外独立事件。'
              '来水采用三类合成开发模板，结果不代表真实洪水或任意来水泛化。', '',
              '运行中发生一次会话中断，保留 interruptions 记录；已保存模型复用，未完成的相同配置拟合重试，'
              '没有追加候选或留出选模。报告的完成拟合耗时不含中断前未完成尝试的未知开销。', '',
              f'![各预算排序结果](evidence/20261025-p3-nonlinear-round{round_id}/ranking.png)', '',
              '## 证据与复现', '',
              f'- [完整摘要](evidence/20261025-p3-nonlinear-round{round_id}/summary.json)',
              f'- [实际缓存与模型审计](evidence/20261025-p3-nonlinear-round{round_id}/input-audit-final/cache-model-audit.json)',
              f'- [实际帧与运动审计](evidence/20261025-p3-nonlinear-round{round_id}/input-audit-final/replay-audit.json)',
              f'- [独立评分聚合审计](evidence/20261025-p3-closeout/round{round_id}-score-audit.json)',
              f'- [实际容量与预算](evidence/20261025-p3-nonlinear-round{round_id}/capacity-budget.json)',
              '- [启动与复现命令](REPRODUCE.md)', '',
              '本轮为原 P3 的第二轮和最后一轮。结合两轮结果作阶段结论，不再自动新增第三轮。'
              '用户另行授权的 Sur-QMIX 与物理辅助世界模型属于独立扩展，不能冒充原 P3 已通过或改写其结果。', '']
    capacity = read(ROOT/f'docs/evidence/20261025-p3-nonlinear-round{round_id}/capacity-budget.json')
    choices = {r['group']: r['params'] for r in read(root/'model-freeze.json')['choices']}
    lines += ['## 实际选型与容量限制', '', '| 组 | 选中 min_samples_leaf | 平均树节点数 |', '|---|---:|---:|']
    lines += [f"| {r['group']} | {choices[r['group']]['min_samples_leaf']} | {r['mean_nodes']:.0f} |" for r in capacity]
    lines += ['', '候选空间、树数、深度上限和选型机会相同，但选中叶节点约束不同，实际容量并不相同。'
              '本轮 B3 比 B1/B2 约多一倍树节点；增量包含相同选型协议下的模型复杂度差异，'
              '不能直接全部归因于地形联系组织。此项作为解释限制保留，不在已知留出上补调参或重选模型。', '',
              '第二轮在慢变和异步事件上呈现正向增量；唯一未满足的登记条件为快速机制对B2的平均差异仍为负。'
              'radial-notch-rapid 的负向差异保留，不将其删除或以其余8事件代替完整结果。', '']
    (ROOT/f'docs/P3_NONLINEAR_ROUND{round_id}_20261025.md').write_text('\n'.join(lines), encoding='utf-8')
    return means


def main():
    output = ROOT/'docs/evidence/20261025-p3-closeout'
    output.mkdir(parents=True, exist_ok=True)
    rounds, beds, recipes = [], set(), set()
    for number in (1, 2):
        root = ROOT/f'generated/p3-nonlinear-round{number}'
        summary, plan = read(root/'summary.json'), read(root/'plan.json')
        cache_audit = read(root/'input-audit-final/cache-model-audit.json')
        replay_audit = read(root/'input-audit-final/replay-audit.json')
        assert cache_audit['status'] == replay_audit['status'] == 'passed'
        assert cache_audit['replay_count'] == 54 and len(cache_audit['actual_models']) == 15
        assert len(replay_audit['runs']) == 54
        case_audits = read(root/'case-audits.json')
        assert len(case_audits) == 18
        current = {a['bed_sha256'] for a in case_audits}
        current_recipes = {c['terrain_recipe'] for c in plan['cases']}
        assert len(current) == 18 and not beds & current and not recipes & current_recipes
        beds |= current
        recipes |= current_recipes
        if number == 2:
            assert plan['strict_change_ablation']
            assert sum(r['raw_histories_removed_rows'] for r in cache_audit['runs']) > 0
        scores, sources, gate = score_audit(root), archive_audit(root), gate_audit(root)
        write_json(output/f'round{number}-score-audit.json', scores)
        write_json(output/f'round{number}-source-audit.json', sources)
        write_json(output/f'round{number}-gate-audit.json', gate)
        evidence = ROOT/f'docs/evidence/20261025-p3-nonlinear-round{number}/input-audit-final'
        evidence.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root/'input-audit-final/cache-model-audit.json', evidence/'cache-model-audit.json')
        rounds.append(dict(round=number, gate_passed=summary['gate_passed'], mean_deltas=summary['mean_deltas'],
                           positive_events=summary['positive_events'], valid_control_events=summary['valid_control_events'],
                           metrics_rows=summary['metrics_rows'], source_archive=sources, gate_audit=gate,
                           audited_cache_rows=cache_audit['cache_rows'], summary_sha256=digest(root/'summary.json')))
        if number == 2:
            means = report_round(root, summary)
            rounds[-1]['mean_precision_at_10'] = means
    ledger = read(ROOT/'research/rounds.json')
    assert ledger['maximum_rounds'] == 2 and len(ledger['rounds']) == 2
    for row, result in zip(ledger['rounds'], rounds):
        assert row['round'] == result['round']
        row.update(state='completed', summary_sha256=result['summary_sha256'], gate_passed=result['gate_passed'])
    ledger['rounds'][-1]['decision'] = ('P4 gate requires full review' if rounds[-1]['gate_passed']
                                      else 'original P3 closed without sufficient B3 support; no third round')
    write_json(ROOT/'research/rounds.json', ledger)
    write_json(output/'rounds.json', ledger)
    write_json(output/'audit.json', dict(status='passed', rounds=rounds, distinct_parent_arrays=len(beds),
                                        distinct_terrain_recipes=len(recipes),
                                        original_P4_gate_passed=rounds[-1]['gate_passed'],
                                        auditor_sha256=digest(Path(__file__)),
                                        limits='P3 complete only; Sur-QMIX and independent physical-world-model goal remain incomplete.'))
    lines = ['# P3 两轮最终阶段结论', '',
             '两轮均按各自冻结协议完成；不增加第三轮。以下差异单位为 Precision@10% 的百分点。', '',
             '| 轮次 | B3−B1 | B3−B2 | B3−A1 | B3−A2 | 同时正向事件 | 有效 A2 | 升级门槛 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for row in rounds:
        lines.append(f"| {row['round']} | "+' | '.join(f"{row['mean_deltas'][g]*100:+.3f}" for g in ('B1','B2','A1','A2'))+
                     f" | {row['positive_events']}/9 | {row['valid_control_events']}/9 | {'通过' if row['gate_passed'] else '未通过'} |")
    lines += ['', '只有第二轮 A1 移除了全部旧观测历史；第一轮为显式传播通道消融，两轮不能视为完全相同的 A1 重复实验。'
              '两轮拟合、选型、留出使用不同母地形配方，36 个母地形数组无重复。各轮各有三个留出家族和九个事件，'
              '不能将两轮拼接后重新调门槛或挑选有利事件。', '',
              '第二轮的慢变、异步机制有正向增量；快速机制相对B2平均−0.187个百分点，未满足每机制均值均正的条件。'
              '其实际选型也产生容量差异：B3约326184节点，B1/B2约159000节点，不能将全部增量归因于地形组织。', '',
              ('第二轮达到预登记门槛；是否推进原 P4 仍须结合全项验收与资源评估。' if rounds[-1]['gate_passed'] else
               '本次两轮证据不足以支持候选静态地形联系超越强历史和距离基线的稳定增量；原路线不进入条件 P4。'
               '这不证明所有水动力知识都无效，也不证明尚未实现的物理世界模型一定有效。'), '',
              '用户另行授权的 Sur-QMIX 基线与物理辅助世界模型继续按独立研究推进；其训练不是原 P3 阳性结果的后续证明。', '',
              '- [第一轮报告](P3_NONLINEAR_ROUND1_20261025.md)', '- [第二轮报告](P3_NONLINEAR_ROUND2_20261025.md)',
              '- [两轮独立审计](evidence/20261025-p3-closeout/audit.json)',
              '- [逐项验收表](CONTINUOUS_ACCEPTANCE.md)', '- [复现命令](REPRODUCE.md)', '']
    (ROOT/'docs/P3_FINAL_20261025.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(rounds, ensure_ascii=False))


if __name__ == '__main__':
    main()
