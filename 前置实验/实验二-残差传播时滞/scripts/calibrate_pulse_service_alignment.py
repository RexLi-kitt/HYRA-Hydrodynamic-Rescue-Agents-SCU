"""Training-only ANUGA alignment campaign with saved causal references and grid gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import new_run, write_json
from hyra_validation.alignment import (anchor_selection, evaluate_trajectory, nominal_actions,
    sampled_records, causal_bin_ids, propagation_edges, kernel_distance, gained_targets,
    paired_response_available)
from hyra_validation.local_service import task_table
from hyra_validation.local_service_data import load_family, nominal_domain
from hyra_validation.lead import XY, TARGETS, SOURCES, hydraulic_domain
from hyra_validation.adapters.anuga import simulate
from hyra_validation.diagnostics import score_fixed_order


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_csv(path, rows):
    if not rows: return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def trajectory(run, meta, mesh, kind, anchor=0, center=25):
    folder = run/'trajectories'; folder.mkdir(exist_ok=True)
    key = f"{meta['family_id']}_mesh{mesh}_{kind}_a{anchor}_y{center}"
    path = folder/(key+'.npz'); marker = folder/(key+'.complete.json')
    if marker.exists():
        completed = read(marker)
        assert sha(path) == completed['sha256']
        with np.load(path, allow_pickle=False) as z: return z['time_s'], z['depth_m'], completed
    if path.exists(): raise FileExistsError('Preserve incomplete trajectory: '+str(path))
    started = time.perf_counter()
    factor = 1 if kind.startswith('truth') else meta['nominal_specifications'][0]['factor']
    amplitude = .1 if kind.endswith('pulse') else 0
    domain = hydraulic_domain(meta['params'], mesh, factor=factor, pulse_amplitude=amplitude,
                               pulse_start=anchor, pulse_center=center, plateau=20)
    times, depths, states, mapping = simulate(domain, 300, XY)
    arrays = dict(time_s=times, depth_m=depths, centroid_xy_m=domain.get_centroid_coordinates(),
                  bed_m=domain.quantities['elevation'].centroid_values.copy())
    for key2 in states[0]: arrays[key2] = np.asarray([s[key2] for s in states])
    assert np.isfinite(depths).all() and depths.min() >= 0
    np.savez_compressed(path, **arrays)
    completed = dict(file=str(path.relative_to(run)), family=meta['family_id'], nmesh=mesh,
        kind=kind, anchor_s=anchor, center_y_m=center, sha256=sha(path), mapping=mapping,
        seconds=time.perf_counter()-started)
    write_json(marker, completed)
    print(f"TRAJECTORY {meta['family_id']} mesh={mesh} {kind} a={anchor} y={center}", flush=True)
    return times, depths, completed


def verify_family(parent, entry):
    assert entry['split'] == 'train'
    result = {}
    for name, checksum in entry['sha256'].items():
        path = parent/'families'/name
        assert sha(path) == checksum, name
        result[str(path)] = checksum
    return result


def family_campaign(run, parent, entry, layouts):
    folder = run/'families'; folder.mkdir(exist_ok=True)
    result_path = folder/(entry['family_id']+'.json')
    if result_path.exists(): return read(result_path)
    source_hashes = verify_family(parent, entry)
    arrays, meta = load_family(parent/'families', entry)
    assert meta['split'] == 'train' and meta['nmesh'] == 10
    event = dict(time_s=np.arange(0, 301, 2), xy_m=XY, release_s=arrays['release_s'],
                 forecast_valid_s=arrays['forecast_valid_s'], forecast_depth_m=arrays['M0_forecast_depth_m'])
    times, base, _ = trajectory(run, meta, 10, 'truth_base')
    nt, nominal, _ = trajectory(run, meta, 10, 'nominal_base')
    np.testing.assert_array_equal(times, nt)
    np.testing.assert_allclose(base, arrays['E0_truth_depth_m'][:len(times)], atol=1e-8, rtol=0)
    np.testing.assert_allclose(nominal, arrays['M0_nominal_depth_m'][:len(times)], atol=1e-8, rtol=0)
    opportunities = []; actions = []; edges_all = []; kernels = []; refs = {}; samples = {}
    audits = dict(scalar_orders=0, causal_record_channel_references=0, noise_pairs=0,
                  published_anchor_checks=0, target_sensor_references=0)
    for anchor in (90, 120, 150, 180, 210):
        cases, critical, anchor_score = anchor_selection(event, layouts, anchor)
        baseline, current_scores = nominal_actions(event, cases)
        fi = int(np.flatnonzero(event['release_s'] == anchor)[0]); ti = int(anchor/2)
        np.testing.assert_allclose(event['forecast_depth_m'][fi, 0], nominal[ti], atol=1e-8, rtol=0)
        audits['published_anchor_checks'] += 1
        for center in (25, 75):
            tt, pulse, _ = trajectory(run, meta, 10, 'truth_pulse', anchor, center)
            nn, npulse, _ = trajectory(run, meta, 10, 'nominal_pulse', anchor, center)
            np.testing.assert_array_equal(tt, times); np.testing.assert_array_equal(nn, times)
            np.testing.assert_allclose(pulse[times <= anchor], base[times <= anchor], atol=1e-8, rtol=0)
            np.testing.assert_allclose(npulse[times <= anchor], nominal[times <= anchor], atol=1e-8, rtol=0)
            edges = propagation_edges(times, npulse-nominal, pulse-base, meta['family_id'], anchor, center)
            edges_all.extend(edges)
            scores = [evaluate_trajectory(event, water, cases) for water in (base, pulse)]
            for condition, water, score in zip(('E0', 'E1'), (base, pulse), scores):
                for j, case in enumerate(cases):
                    for order in (0, 1):
                        target_order, ds = case['orders'][order]
                        scalar = score_fixed_order(times, water, target_order, ds, case['now_s'], 6, 12, .8,
                                                   end_time=anchor+60)
                        assert scalar['service_yield'] == score['count'][j, order]
                        np.testing.assert_allclose([x['start_s'] for x in scalar['trace']],
                            case['now_s']+score['starts'][j, order], atol=1e-10)
                        audits['scalar_orders'] += 1
                counts = score['count']; actual = counts[np.arange(len(cases)), baseline]
                for population, mask in [('general', np.ones(len(cases), bool)), ('critical', critical)]:
                    for offset in (0, 6, 12):
                        keep = mask & np.array([c['now_s'] == anchor+offset for c in cases])
                        n = int(keep.sum())
                        opportunities.append(dict(family=meta['family_id'], anchor_s=anchor, center_y_m=center,
                            condition=condition, population=population, decision_offset_s=offset,
                            available_cases=int(len(cases)//3), retained_cases=n,
                            nominal_actual_sum=int(actual[keep].sum()),
                            finite_oracle_sum=int(counts[keep].max(axis=1).sum()),
                            gap_sum=int((counts.max(axis=1)-actual)[keep].sum()),
                            order_sensitive_cases=int(np.count_nonzero(counts[keep, 0] != counts[keep, 1])),
                            pulse_changes_counts_cases=int(np.count_nonzero(np.any(scores[0]['count'][keep] != scores[1]['count'][keep], axis=1)))))
            prefix = f'a{anchor}_y{center}'
            samples[prefix+'_baseline_order'] = baseline
            samples[prefix+'_critical_mask'] = critical
            samples[prefix+'_anchor_maxima'] = anchor_score['maxima'][0]
            samples[prefix+'_current_nominal_maxima'] = current_scores['maxima']
            samples[prefix+'_E0_counts'] = scores[0]['count']
            samples[prefix+'_E1_counts'] = scores[1]['count']
            samples[prefix+'_E1_maxima'] = scores[1]['maxima']
            for phase_index, phase in enumerate((0, .25, .5, .75)):
                records = sampled_records(event, base, pulse, phase, 0, 0,
                    [meta['index'], anchor, center, phase_index])
                key = prefix+f'_p{phase_index}'
                for name in ('valid_s', 'prior_release_s', 'actual_lead_s', 'prior_m', 'paired_reference_response_m'):
                    refs[key+'_'+name] = records[name]
                for noise in (0, .02):
                    for repeat in range(3 if noise else 1):
                        noisy = sampled_records(event, base, pulse, phase, noise, repeat,
                            [meta['index'], anchor, center, phase_index])
                        for condition in ('E0', 'E1'):
                            refs[key+f'_n{noise}_rep{repeat}_{condition}_observed_m'] = noisy[condition+'_observed_m']
                        np.testing.assert_allclose(noisy['E1_observed_m']-noisy['E0_observed_m'],
                            records['paired_reference_response_m'], atol=1e-14, rtol=0)
                        audits['noise_pairs'] += 1
                for delay in (0, 10):
                    ids_all = np.array([causal_bin_ids(records['valid_s'], anchor+d, delay) for d in (0, 6, 12)])
                    refs[key+f'_d{delay}_bin_ids'] = ids_all
                    for d, ids in zip((0, 6, 12), ids_all):
                        valid_ids = ids[ids >= 0]; now = anchor+d
                        assert np.all(records['valid_s'][valid_ids]+delay <= now)
                        assert np.all(records['prior_release_s'][valid_ids] <= records['valid_s'][valid_ids]-20)
                        audits['causal_record_channel_references'] += len(valid_ids)*len(SOURCES)*8
                        ages = now-records['valid_s'][valid_ids]
                        for edge in edges:
                            if edge['kernel_lag_s'] is None: continue
                            for tau in (6, 12):
                                distance = kernel_distance(ages, edge['kernel_lag_s'], tau)
                                kernels.append(dict(family=meta['family_id'], anchor_s=anchor, center_y_m=center,
                                    source=edge['source'], target=edge['target'], decision_offset_s=d,
                                    phase_index=phase_index, delay_s=delay, horizon_s=tau,
                                    weight_L1=distance))
                    for layout_index in range(len(layouts)):
                        indices = np.arange(layout_index*3, layout_index*3+3)
                        gap_times = []; visible_times = []; critical_gap = []; critical_visible = []
                        for j in indices:
                            case = cases[j]; gain = gained_targets(case, scores[1], j, int(baseline[j]))
                            if gain:
                                gap_times.append(case['now_s'])
                                if critical[j]: critical_gap.append(case['now_s'])
                                if paired_response_available(records, case['now_s'], delay, gain):
                                    visible_times.append(case['now_s'])
                                    if critical[j]: critical_visible.append(case['now_s'])
                        actions.append(dict(family=meta['family_id'], anchor_s=anchor, center_y_m=center,
                            layout=layouts[layout_index]['layout'], phase_index=phase_index, delay_s=delay,
                            critical_eligible_times_s=[cases[j]['now_s'] for j in indices if critical[j]],
                            oracle_improvement_times_s=gap_times, critical_oracle_improvement_times_s=critical_gap,
                            paired_response_and_oracle_times_s=visible_times,
                            critical_paired_response_and_oracle_times_s=critical_visible,
                            noise_invariant_necessary_bound_not_detector=True,
                            actual_physical_model_action_tested=False))
    np.savez_compressed(folder/(meta['family_id']+'_causal_records.npz'), source_region_ids=SOURCES, **refs)
    np.savez_compressed(folder/(meta['family_id']+'_task_scores.npz'), **samples)
    result = dict(family=meta['family_id'], source_hashes=source_hashes, audits=audits,
                  opportunities=opportunities, actions=actions, edges=edges_all, kernels=kernels)
    write_json(result_path, result)
    print('FAMILY_DONE '+meta['family_id'], flush=True)
    return result


def grid_check(run, parent, entries, results, layouts):
    # First family plus largest NOMINAL target response, selected without reference benefit.
    largest = max(results, key=lambda r: max(e['nominal_target_peak_m'] for e in r['edges']))['family']
    selected = list(dict.fromkeys([entries[0]['family_id'], largest])); comparisons = []
    for family in selected:
        entry = next(e for e in entries if e['family_id'] == family)
        arrays, meta = load_family(parent/'families', entry)
        event = dict(time_s=np.arange(0, 301, 2), release_s=arrays['release_s'],
            forecast_valid_s=arrays['forecast_valid_s'], forecast_depth_m=arrays['M0_forecast_depth_m'])
        times, b10, _ = trajectory(run, meta, 10, 'truth_base')
        _, nb10, _ = trajectory(run, meta, 10, 'nominal_base')
        _, b20, _ = trajectory(run, meta, 20, 'truth_base')
        _, nb20, _ = trajectory(run, meta, 20, 'nominal_base')
        for anchor in (90, 210):
            cases, keep, _ = anchor_selection(event, layouts, anchor)
            chosen, _ = nominal_actions(event, cases)
            for center in (25, 75):
                _, p10, _ = trajectory(run, meta, 10, 'truth_pulse', anchor, center)
                _, np10, _ = trajectory(run, meta, 10, 'nominal_pulse', anchor, center)
                _, p20, _ = trajectory(run, meta, 20, 'truth_pulse', anchor, center)
                _, np20, _ = trajectory(run, meta, 20, 'nominal_pulse', anchor, center)
                scores = [evaluate_trajectory(event, p, cases) for p in (p10, p20)]
                counts = [s['count'] for s in scores]
                gaps = [c.max(axis=1)-c[np.arange(len(cases)), chosen] for c in counts]
                visible_flags = []
                for b, p, s in zip((b10, b20), (p10, p20), scores):
                    yes = False
                    for pi, phase in enumerate((0, .25, .5, .75)):
                        rec = sampled_records(event, b, p, phase, 0, 0, [meta['index'], anchor, center, pi])
                        yes |= any(keep[j] and paired_response_available(rec, c['now_s'], 0,
                            gained_targets(c, s, j, int(chosen[j]))) for j, c in enumerate(cases))
                    visible_flags.append(bool(yes))
                edges10 = propagation_edges(times, np10-nb10, p10-b10, family, anchor, center)
                edges20 = propagation_edges(times, np20-nb20, p20-b20, family, anchor, center)
                changes = [abs(a['kernel_lag_s']-b['kernel_lag_s']) for a, b in zip(edges10, edges20)
                           if a['kernel_lag_s'] is not None and b['kernel_lag_s'] is not None]
                opportunity_flags = [bool(np.any(g[keep] > 0)) for g in gaps]
                comparisons.append(dict(family=family, anchor_s=anchor, center_y_m=center,
                    retained_cases=int(keep.sum()), shared_nominal_selection_and_actions=True,
                    full_interval_classifications_changed=int(np.count_nonzero(
                        (scores[0]['maxima'] <= .8) != (scores[1]['maxima'] <= .8))),
                    order_yields_changed=int(np.count_nonzero(counts[0] != counts[1])),
                    nominal_lag_absolute_change_max_s=max(changes) if changes else None,
                    coarse_critical_opportunity_exists=opportunity_flags[0],
                    fine_critical_opportunity_exists=opportunity_flags[1],
                    coarse_critical_visible_oracle_opportunity_exists=visible_flags[0],
                    fine_critical_visible_oracle_opportunity_exists=visible_flags[1],
                    core_existence_reversed=opportunity_flags[0] != opportunity_flags[1] or visible_flags[0] != visible_flags[1]))
    result = dict(selected_families=selected, coarse_mesh=10, fine_mesh=20, comparisons=comparisons,
        core_existence_reversal_count=sum(r['core_existence_reversed'] for r in comparisons),
        engineering_gate_passed=not any(r['core_existence_reversed'] for r in comparisons),
        is_full_convergence_study=False)
    write_json(run/'grid_refinement.json', result)
    return result


def summarize(run, results, grid):
    flat = {name: [r for family in results for r in family[name]]
            for name in ('opportunities', 'actions', 'edges', 'kernels')}
    for name, rows in flat.items(): save_csv(run/(name+'.csv'), rows)
    summary = []
    for population in ('general', 'critical'):
        for condition in ('E0', 'E1', 'both'):
            rows = [r for r in flat['opportunities'] if r['population'] == population
                    and (condition == 'both' or r['condition'] == condition)]
            grouped = defaultdict(lambda: [0, 0])
            for row in rows:
                grouped[row['family']][0] += row['gap_sum']
                grouped[row['family']][1] += row['retained_cases']
            values = [100*g/n for g, n in grouped.values() if n]
            summary.append(dict(population=population, condition=condition,
                retained=sum(r['retained_cases'] for r in rows), total=sum(r['available_cases'] for r in rows),
                empty_cells=sum(r['retained_cases'] == 0 for r in rows), cells=len(rows),
                supported_families=len(values), headroom_over_nominal_per100_family_macro=float(np.mean(values)) if values else None))
    timing = []
    for delay in (0, 10):
        for tau in (6, 12):
            rows = [r for r in flat['kernels'] if r['delay_s'] == delay and r['horizon_s'] == tau
                    and r['weight_L1'] is not None]
            timing.append(dict(delay_s=delay, horizon_s=tau, edges_times=len(rows),
                mean_weight_L1=float(np.mean([r['weight_L1'] for r in rows])),
                fraction_weight_L1_ge_point1=float(np.mean([r['weight_L1'] >= .1 for r in rows]))))
    action = []
    for delay in (0, 10):
        rows = [r for r in flat['actions'] if r['delay_s'] == delay]
        eligible = sum(len(r['critical_eligible_times_s']) for r in rows)
        gap = sum(len(r['critical_oracle_improvement_times_s']) for r in rows)
        visible = sum(len(r['critical_paired_response_and_oracle_times_s']) for r in rows)
        action.append(dict(delay_s=delay, critical_task_phase_decisions=eligible,
            critical_oracle_improvement_decisions=gap, critical_paired_response_available_and_oracle_decisions=visible,
            optimistic_fraction_of_critical=visible/eligible if eligible else None,
            fraction_of_oracle_improvement=visible/gap if gap else None,
            operational_detector_or_learned_action=False))
    out = dict(training_families=len(results), opportunity=summary, kernel=timing, action=action,
        grid_gate_passed=grid['engineering_gate_passed'],
        audits={k: sum(r['audits'][k] for r in results) for k in results[0]['audits']},
        old_confirmation_labels_read=0, fitted_models=0,
        aggregation='Within-family pooled fixed-grid cases; empty cells and family support reported. Not equal-cell macro.',
        next_action='expand_training_calibration_to_24' if grid['engineering_gate_passed'] else 'stop_for_numerical_design_review')
    write_json(run/'summary.json', out)
    lines = ['# 扰动—完整服务窗口对齐校准', '',
        f"本轮{len(results)}个旧训练家族；集中式单机两目标初始选序。没有训练物理模型或检测器，没有打开旧确认标签。",
        '机会相对名义预测计算，尚非相对强基线B*；下表不能作为物理关联效果。', '',
        '|总体|事件|保留/总任务|空条件单元|家族平均上限增量/100|', '|---|---|---:|---:|---:|']
    for r in summary:
        value = '无支持' if r['headroom_over_nominal_per100_family_macro'] is None else f"{r['headroom_over_nominal_per100_family_macro']:.4f}"
        lines.append(f"|{r['population']}|{r['condition']}|{r['retained']}/{r['total']}|{r['empty_cells']}/{r['cells']}|{value}|")
    lines += ['', '按家族内保留的固定网格情境聚合，再对有支持家族等权。空单元没有补零；这不是条件单元等权总体，也不代表一般洪涝任务。E0/E1及两处边界成对；采样/噪声重复不增加独立家族数。', '',
        '|延迟|时域|物理/新近性权重L1均值|L1≥0.1比例|', '|---:|---:|---:|---:|']
    for r in timing:
        lines.append(f"|{r['delay_s']}|{r['horizon_s']}|{r['mean_weight_L1']:.4f}|{100*r['fraction_weight_L1_ge_point1']:.2f}%|")
    lines += ['', '|延迟|临界任务×相位决策|真值可改善|配对源响应已可见且真值可改善|', '|---:|---:|---:|---:|']
    for r in action:
        lines.append(f"|{r['delay_s']}|{r['critical_task_phase_decisions']}|{r['critical_oracle_improvement_decisions']}|{r['critical_paired_response_available_and_oracle_decisions']}|")
    lines += ['', '最后一列是离线必要条件上界：可见历史中至少一条相关上游记录的配对无噪声响应≥1毫米。它使用E0/E1参考差，不能部署为检测器；有噪声是否能可靠检测仍未知。该上界对噪声重复不变，未将其复制成独立样本。实际物理模型改序收益尚未测试。', '',
        f"网格10→20抽查：{len(grid['comparisons'])}个探针对照，{grid['core_existence_reversal_count']}处核心存在性判断翻转。工程门槛：{'通过' if grid['engineering_gate_passed'] else '未通过'}。该抽查不是完整网格收敛证明。", '',
        '完整传播表、机会表、行动时刻集合及核差异见同目录CSV；原始轨迹、状态、观测、引用索引和任务分数见trajectories与families。', '',
        '下一步：'+('按计划继续其余旧训练家族校准。' if grid['engineering_gate_passed'] else '停止扩大主实验，先审查网格/采样映射和阈值附近数值稳定性。'), '',
        '本轮尚未检验P4相对P5/B*，不能宣称H2/H3/H4/H5已经成立。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--family-count', type=int, choices=(4, 24), default=4)
    args = parser.parse_args(); parent = args.parent.resolve()
    design = read(ROOT/'configs/hidden_target_physics.alignment.design.v1.json')
    entries = [e for e in read(parent/'development_family_manifest.json') if e['split'] == 'train']
    entries.sort(key=lambda e: e['index']); entries = entries[:args.family_count]
    assert len(entries) == args.family_count
    geometry = read(parent/'geometry.json'); unique = {}
    for row in task_table(geometry): unique.setdefault(row['layout'], row)
    layouts = list(unique.values()); assert len(layouts) == 32
    config = dict(design=design, parent=str(parent), training_ids=[e['family_id'] for e in entries],
        stage='alignment_calibration_not_confirmation',
        action_bound='paired_noiseless_source_response_ge_1mm_not_operational_detector',
        absolute_task_deadline='anchor_plus_60_s', anchors=[90, 120, 150, 180, 210])
    run = args.run.resolve() if args.run else new_run('pulse_service_alignment', config)
    if args.run:
        previous = read(run/'config.json')
        assert previous['parent'] == str(parent)
        # Resume/expansion cannot alter source after seeing outputs.
        provenance = read(run/'provenance.json')
        for relative, checksum in provenance['files'].items():
            assert sha(ROOT/relative) == checksum, 'Source changed: '+relative
    write_json(run/f'campaign_request_n{args.family_count}.json', config)
    started = time.perf_counter(); results = []
    for entry in entries[:4]: results.append(family_campaign(run, parent, entry, layouts))
    grid = read(run/'grid_refinement.json') if (run/'grid_refinement.json').exists() else grid_check(run, parent, entries[:4], results, layouts)
    if grid['engineering_gate_passed'] and args.family_count == 24:
        for entry in entries[4:]: results.append(family_campaign(run, parent, entry, layouts))
    summary = summarize(run, results, grid)
    write_json(run/'completion.json', dict(completed=True, requested_families=args.family_count,
        completed_families=len(results), expanded_to_24=len(results) == 24,
        seconds_this_invocation=time.perf_counter()-started, grid_gate_passed=grid['engineering_gate_passed'],
        next_action=summary['next_action'], old_confirmation_labels_read=0))
    write_json(ROOT/'results/latest_pulse_service_alignment.json', dict(run=str(run), report=str(run/'REPORT.md'), completed=True))
    print('REPORT '+str(run/'REPORT.md'), flush=True)


if __name__ == '__main__': main()
