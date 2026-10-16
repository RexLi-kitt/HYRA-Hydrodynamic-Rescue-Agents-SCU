"""Training-only opportunity and nominal timing checks before hidden-target learning.

No candidate model is fitted, no confirmation labels are used, and no new
observation policy is claimed. Nominal hydraulic probes are offline priors.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import new_run, write_json
from hyra_validation.local_service import task_table, event_foundation, score_orders, choose_order
from hyra_validation.local_service_data import load_family, family_events, nominal_domain
from hyra_validation.lead import hydraulic_domain, XY, TARGETS, SOURCES
from hyra_validation.adapters.anuga import simulate


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def all_geometry_cases(preview):
    layouts = {c['layout']: c for c in task_table(preview)}
    return [dict(c, now_s=now) for c in layouts.values() for now in range(24, 241, 6)]


def arrival(times, delta, start):
    after = times >= start
    peak = float(np.max(abs(delta[after])))
    if peak < .001: return None
    indices = np.flatnonzero(after & (abs(delta) >= .1*peak))
    return float(times[indices[0]]) if len(indices) else None


def nominal_probes(run, meta):
    """Known nominal model, two boundary segments and two calendar times."""
    factor = meta['nominal_specifications'][0]['factor']
    params = meta['params']; mesh = meta['nmesh']
    times, base, _, _ = simulate(nominal_domain(params, mesh, factor), 260, XY)
    arrays = {'time_s': times, 'nominal_depth_m': base}
    edges = []
    for start in (90, 180):
        for center in (25, 75):
            domain = hydraulic_domain(params, mesh, factor=factor, pulse_amplitude=.1,
                                      pulse_start=start, pulse_center=center, plateau=20)
            tt, perturbed, _, _ = simulate(domain, 260, XY)
            np.testing.assert_array_equal(tt, times)
            delta = perturbed-base
            assert np.max(abs(delta[times < start])) < 1e-8
            arrays[f'probe_s{start}_y{center}_depth_m'] = perturbed
            for target in TARGETS:
                if (XY[target, 1] < 50) != (center == 25): continue
                target_at = arrival(times, delta[:, target], start)
                for source in SOURCES:
                    if (XY[source, 1] < 50) != (center == 25): continue
                    source_at = arrival(times, delta[:, source], start)
                    if source_at is None or target_at is None: continue
                    edges.append(dict(family=meta['family_id'], probe_start_s=start,
                        center_y_m=center, source=source, target=target,
                        source_marker_s=source_at, target_marker_s=target_at,
                        signed_response_lag_s=target_at-source_at,
                        nonnegative_kernel_lag_s=max(target_at-source_at, 0),
                        target_peak_response_m=float(np.max(abs(delta[:, target])))))
    path = run/f"{meta['family_id']}_nominal_probes.npz"
    np.savez_compressed(path, **arrays)
    return edges


def opportunity(arrays, meta, cases):
    events = {(e, m): event for e, m, event in family_events(arrays, meta)}
    truths = {}
    for condition in ('E0', 'E1'):
        foundation = event_foundation(events[condition, 'M0'])
        truths[condition] = score_orders(foundation['truth_future'][None], cases,
                                         foundation['row_times'], foundation['row_regions'])['count'][0]
    best0 = truths['E0'] == truths['E0'].max(axis=1, keepdims=True)
    best1 = truths['E1'] == truths['E1'].max(axis=1, keepdims=True)
    pulse_changes_counts = np.any(truths['E0'] != truths['E1'], axis=1)
    pulse_forces_reversal = ~np.any(best0 & best1, axis=1)
    output = []
    for mode in ('M0', 'M1', 'M2', 'M3'):
        foundation = event_foundation(events['E0', mode])
        nominal = score_orders(foundation['base'][None], cases, foundation['row_times'], foundation['row_regions'])
        chosen = choose_order(nominal, cases)[0]
        nominal_distance_to_threshold = np.min(abs(.8-nominal['maxima'][0]), axis=(1, 2))
        for band in (None, .05, .10, .15):
            keep = np.ones(len(cases), bool) if band is None else nominal_distance_to_threshold <= band
            for condition in ('E0', 'E1'):
                yy = truths[condition]
                actual = yy[np.arange(len(cases)), chosen]
                row = dict(family=meta['family_id'], condition=condition, mode=mode,
                    nominal_band_m=band, available_cases=len(cases), retained_cases=int(keep.sum()))
                if np.any(keep):
                    row.update(nominal_mean_yield=float(actual[keep].mean()),
                        finite_oracle_mean=float(yy[keep].max(axis=1).mean()),
                        maximum_gain_over_nominal=float((yy.max(axis=1)-actual)[keep].mean()),
                        order_sensitive_fraction=float(np.mean(yy[keep, 0] != yy[keep, 1])),
                        pulse_changes_order_counts_fraction=float(pulse_changes_counts[keep].mean()),
                        pulse_forces_best_order_reversal_fraction=float(pulse_forces_reversal[keep].mean()))
                output.append(row)
    return output


def kernel_and_sampling(edges):
    ages_by_delay = {}
    for delay in (0, 10):
        all_ages = []; all_masks = []
        for phase in (0, .25, .5, .75):
            samples = np.arange(20+phase*5, 301, 5)
            for now in range(24, 241, 6):
                ages = now-samples
                bins = np.zeros(6); mask = np.zeros(6, bool)
                for b in range(6):
                    allowed = np.flatnonzero((ages >= b*10) & (ages < (b+1)*10) & (samples+delay <= now))
                    if len(allowed): bins[b] = ages[allowed[-1]]; mask[b] = True
                all_ages.append(bins); all_masks.append(mask)
        ages_by_delay[delay] = (np.asarray(all_ages), np.asarray(all_masks))
    lags = np.array([e['nonnegative_kernel_lag_s'] for e in edges])
    rows = []
    for tau in (6, 12, 20, 30):
        for delay, (ages, masks) in ages_by_delay.items():
            center = np.maximum(lags-tau, 0)
            recent = np.exp(-.5*(ages/10)**2)*masks
            recent = np.divide(recent, recent.sum(axis=1, keepdims=True), out=np.zeros_like(recent), where=recent.sum(axis=1, keepdims=True)>0)
            physical = np.exp(-.5*((ages[None]-center[:, None, None])/10)**2)*masks[None]
            denominator = physical.sum(axis=2, keepdims=True)
            physical = np.divide(physical, denominator, out=np.zeros_like(physical), where=denominator>0)
            distance = abs(physical-recent[None]).sum(axis=2)
            distance = distance[:, masks.any(axis=1)]
            rows.append(dict(future_horizon_s=tau, delay_s=delay,
                exact_center_collapse_fraction=float(np.mean(center == 0)),
                mean_weight_L1_distance=float(distance.mean()),
                fraction_L1_at_least_point1=float(np.mean(distance >= .1)),
                no_history_settings=int(sum(~masks.any(axis=1)))))
    sampling = []
    for delay in (0, 10):
        leads = []
        for edge in edges:
            for phase in (0, .25, .5, .75):
                first_sample = 20+phase*5+5*max(0, np.ceil((edge['source_marker_s']-20-phase*5)/5))
                first_decision = 24+6*max(0, np.ceil((first_sample+delay-24)/6))
                leads.append(edge['target_marker_s']-first_decision)
        values = np.asarray(leads)
        sampling.append(dict(delay_s=delay, median_marker_lead_s=float(np.median(values)),
            fraction_nonnegative=float(np.mean(values >= 0)),
            fraction_at_least_6s=float(np.mean(values >= 6)),
            fraction_at_least_12s=float(np.mean(values >= 12))))
    paired = defaultdict(dict)
    for edge in edges: paired[edge['family'], edge['source'], edge['target']][edge['probe_start_s']] = edge['signed_response_lag_s']
    differences = [abs(value[180]-value[90]) for value in paired.values() if 90 in value and 180 in value]
    return dict(kernel=rows, sampling_marker_bound=sampling,
        lag_min_s=float(lags.min()), lag_max_s=float(lags.max()),
        probe_time_absolute_lag_change_median_s=float(np.median(differences)),
        probe_time_absolute_lag_change_max_s=float(np.max(differences)),
        marker_bound_is_not_a_detector_or_actual_service_deadline=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--parent', type=Path, required=True); args = parser.parse_args()
    parent = args.parent.resolve(); manifest = read(parent/'development_family_manifest.json')
    training = [f for f in manifest if f['split'] == 'train']
    assert len(training) == 24 and all(f['split'] == 'train' for f in training)
    preview = read(parent/'geometry.json'); cases = all_geometry_cases(preview)
    run = new_run('hidden_target_physics_precheck', dict(parent=str(parent), training_families_only=True,
        training_ids=[f['family_id'] for f in training], target_observations_proposed_removed=True,
        proposed_source_pool=SOURCES, no_new_model_fit=True, no_confirmation_labels_used=True,
        candidate_nominal_bands_m=[None, .05, .10, .15], opportunity_headroom_design_target=.02,
        future_horizon_s=[6, 12, 20, 30], nominal_probe_start_s=[90, 180]))
    started = time.perf_counter(); opportunities = []; edges = []
    for info in training:
        arrays, meta = load_family(parent/'families', info)
        assert meta['split'] == 'train'
        opportunities += opportunity(arrays, meta, cases)
        edges += nominal_probes(run, meta)
        print('PRECHECK '+meta['family_id'], flush=True)
    summary = []
    for band in (None, .05, .10, .15):
        rows = [r for r in opportunities if r['nominal_band_m'] == band]
        support = [r for r in rows if r['retained_cases'] > 0]
        # Family macro means; empty strata are explicitly reported, never padded.
        per_family = defaultdict(list)
        for r in support: per_family[r['family']].append(r)
        summary.append(dict(nominal_band_m=band,
            retained_cases=sum(r['retained_cases'] for r in rows), total_cases=sum(r['available_cases'] for r in rows),
            nonempty_family_condition_cells=len(support), total_family_condition_cells=len(rows),
            maximum_gain_over_nominal_per100=float(np.mean([np.mean([r['maximum_gain_over_nominal'] for r in rr])*100 for rr in per_family.values()])),
            pulse_changes_counts_fraction=float(np.mean([np.mean([r['pulse_changes_order_counts_fraction'] for r in rr]) for rr in per_family.values()])),
            pulse_forces_reversal_fraction=float(np.mean([np.mean([r['pulse_forces_best_order_reversal_fraction'] for r in rr]) for rr in per_family.values()])),
            headroom_at_least_twice_previous_target=bool(np.mean([np.mean([r['maximum_gain_over_nominal'] for r in rr]) for rr in per_family.values()]) >= .02)))
    timing = kernel_and_sampling(edges)
    write_json(run/'opportunity_cells.json', opportunities); write_json(run/'opportunity_summary.json', summary)
    write_json(run/'nominal_probe_edges.json', edges); write_json(run/'timing_identifiability.json', timing)
    write_json(run/'completion.json', dict(completed=True, training_families=24, geometry_cases=1184,
        nominal_probe_simulations=120, new_prediction_or_policy_models=0, confirmation_labels_read=0,
        seconds=time.perf_counter()-started, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    lines = ['# 无本地观测与物理关联：下一阶段预检', '',
        '本轮只用前一轮24个训练家族；不读取确认家族标签、不训练新模型。目标本地观测将遮蔽、六个上游监测点为拟定共同输入池，但正式物理表征比较尚未开始。', '',
        '## 任务空间', '', '|名义临界带|保留/总任务|非空家族条件单元|真值上限相对名义增量/100|脉冲改变顺序结果比例|脉冲迫使最优顺序反转|', '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{'全部' if r['nominal_band_m'] is None else str(r['nominal_band_m'])+'米'}|{r['retained_cases']}/{r['total_cases']}|{r['nonempty_family_condition_cells']}/{r['total_family_condition_cells']}|{r['maximum_gain_over_nominal_per100']:.4f}|{r['pulse_changes_counts_fraction']*100:.4f}%|{r['pulse_forces_reversal_fraction']*100:.4f}%|")
    lines += ['', '临界带只用名义预测中完整服务区间最大水深距0.8米的距离选任务。表中真值只用于训练期设计诊断；各带是不同条件总体，不能拿最好的带直接宣布方法有效。空单元已报告，未用真值补任务。这个上限相对名义预测，加入强上游普通更新后剩余空间可能更小。', '',
        '## 时间与核可辨识性', '',
        f"名义探针时滞范围 {timing['lag_min_s']:.1f}—{timing['lag_max_s']:.1f} 秒；90/180秒探针的配对时滞变化中位数 {timing['probe_time_absolute_lag_change_median_s']:.1f} 秒、最大 {timing['probe_time_absolute_lag_change_max_s']:.1f} 秒。", '',
        '|未来时域|延迟|物理核中心与新近性核相同|归一化权重L1距离均值|L1≥0.1比例|', '|---:|---:|---:|---:|---:|']
    for r in timing['kernel']:
        lines.append(f"|{r['future_horizon_s']}|{r['delay_s']}|{r['exact_center_collapse_fraction']*100:.2f}%|{r['mean_weight_L1_distance']:.4f}|{r['fraction_L1_at_least_point1']*100:.2f}%|")
    lines += ['', '|延迟|采样及下一决策后标记提前量中位数|≥6秒比例|≥12秒比例|', '|---:|---:|---:|---:|']
    for r in timing['sampling_marker_bound']:
        lines.append(f"|{r['delay_s']}|{r['median_marker_lead_s']:.1f}秒|{r['fraction_at_least_6s']*100:.2f}%|{r['fraction_at_least_12s']*100:.2f}%|")
    lines += ['', '标记使用名义探针响应达到自身峰值10%的时刻，依赖离线校准；它只是乐观的时间可行性诊断，不是可部署检测器，不是实际服务最后可改节点。噪声下是否可检测仍未验证。物理联系、统计联系和上游原始历史必须共享原始记录及预算，残差变换不增加原始信息。', '',
        '## 接下来的实施顺序', '',
        '先依据这些训练诊断明确服务任务总体及脉冲与窗口的时间关系，冻结新协议；再做无本地观测下的名义预测、六源原始历史、普通空间更新、统计联系、物理联系和新近性/错时滞比较。若物理与新近性输入相同，则该时域不用于证明独立时滞收益。', '',
        '候选关联先验可参考vendor中的Empirical_LocalPatch去趋势、空间依赖和局地化思路；本轮实际使用ANUGA名义探针。该仓库针对CaMa-Flood，尚未宣称移植完成。只有在强普通更新之上存在可辨识、可行动的改进，才进入新的集中式任务评价。', '',
        '当前没有开启新确认测试，也没有选择新的主模型或训练QMIX。所有下一阶段结论仍限于训练期可行性诊断。']
    report = run/'REPORT.md'; report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(ROOT/'results/latest_hidden_target_precheck.json', dict(run=str(run), report=str(report), completed=True))
    print('REPORT '+str(report), flush=True)


if __name__ == '__main__': main()
