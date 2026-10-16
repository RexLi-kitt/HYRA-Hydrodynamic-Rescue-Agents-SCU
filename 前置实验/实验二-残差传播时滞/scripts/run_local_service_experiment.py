"""Staged, resumable execution of the local residual / service-regret protocol."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

# Small linear models must not spawn dozens of BLAS threads per replay.
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np

from hyra_validation.artifacts import ROOT, new_run, write_json
from hyra_validation.local_service import (load_design, task_table, schedules, build_batch,
    event_foundation, RidgeMoments, predict_candidates, evaluate_batch, select_models,
    learned_predict)
from hyra_validation.local_service_data import generate_family, load_family, family_events
from hyra_validation.decision_value import order_scores, OFFSETS


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = list((ROOT/'src').rglob('*.py'))+[Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def status(run, stage, **extra):
    row = dict(stage=stage, updated_unix_s=time.time(), **extra)
    write_json(run/'status.json', row)
    print(json.dumps(row, ensure_ascii=True), flush=True)


def save_array(archive, name, array):
    buf = io.BytesIO(); np.save(buf, array, allow_pickle=False)
    archive.writestr(name+'.npy', buf.getvalue())


def freeze_sources(run):
    hashes = source_hashes()
    for relative in hashes:
        target = run/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/relative, target)
    return hashes


def training_add(moments, event, identity, config):
    foundation = event_foundation(event)
    for setting in schedules(config):
        batch = build_batch(event, setting, identity, config['master_seed'], foundation)
        for kind in moments:
            moments[kind].add(batch['inputs'][kind], batch['labels'], setting['weight'])


def fit_models(moments, config):
    return {f'{kind}_ridge_{alpha:g}': values.fit(alpha)
            for kind, values in moments.items() for alpha in config['models']['ridge_alpha']}


def evaluate_family(run, arrays, meta, config, cases, models, selected=None):
    identifier = meta['family_id']; code_hash = hashlib.sha256(json.dumps(source_hashes(), sort_keys=True).encode()).hexdigest()[:12]
    suffix = 'all' if selected is None else hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()[:10]
    directory = run/f'evaluation_{code_hash}_{suffix}'; directory.mkdir(exist_ok=True)
    result_path = directory/f'{identifier}.json'
    if result_path.exists(): return read(result_path)
    archive_path = directory/f'{identifier}.zip'
    if archive_path.exists():
        archive_path = directory/f'{identifier}_retry_{time.time_ns()}.zip'
    wanted = None if selected is None else list(dict.fromkeys(['nominal']+[v['candidate'] for v in selected.values()]))
    total_weight = 0.; totals = {}; strata = {}; audit = []; counterfactual_orders = {}
    started = time.perf_counter(); names = groups = None
    pulse_sum = None; pulse_weight = 0
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for condition, mode, event in family_events(arrays, meta):
            foundation = event_foundation(event)
            for setting_index, setting in enumerate(schedules(config)):
                batch = build_batch(event, setting, meta['seed_identity'], config['master_seed'], foundation)
                candidate_names, errors, candidate_groups = predict_candidates(batch, models, config, wanted)
                if names is None:
                    names, groups = candidate_names, candidate_groups
                    totals = {k: np.zeros(len(names)) for k in ('yield', 'regret', 'mae', 'failed_rate')}
                    pulse_sum = np.zeros(len(names))
                assert names == candidate_names
                evaluated = evaluate_batch(batch, errors, cases)
                values = dict(yield_=evaluated['actual'].mean(axis=1), regret=evaluated['regret'].mean(axis=1),
                              mae=evaluated['mae'], failed_rate=(2-evaluated['actual']).mean(axis=1)/2)
                weight = setting['weight']; total_weight += weight
                for key in totals: totals[key] += weight*values['yield_' if key == 'yield' else key]
                stratum_key = f"{condition}/{mode}/noise{setting['noise_std_m']:g}/delay{setting['delay_s']}"
                if stratum_key not in strata:
                    strata[stratum_key] = dict(weight=0., **{k: np.zeros(len(names)) for k in totals},
                        opportunity_sum=0., nominal_critical_sum=0.)
                entry = strata[stratum_key]; entry['weight'] += weight
                for key in totals: entry[key] += weight*values['yield_' if key == 'yield' else key]
                entry['opportunity_sum'] += weight*evaluated['order_sensitive'].mean()
                entry['nominal_critical_sum'] += weight*evaluated['nominal_critical'].mean()
                pair_key = (mode, setting_index)
                if condition == 'E0': counterfactual_orders[pair_key] = evaluated['chosen'].copy()
                else:
                    old_choice = counterfactual_orders[pair_key]
                    truth = evaluated['true_yields']
                    counter_y = np.take_along_axis(np.broadcast_to(truth, (*old_choice.shape, 2)), old_choice[..., None], axis=-1)[..., 0]
                    pulse_sum += weight*(evaluated['actual']-counter_y).mean(axis=1)
                    pulse_weight += weight
                prefix = f'{condition}_{mode}_s{setting_index:02}'
                for key in ('chosen', 'actual', 'regret', 'mae', 'true_yields', 'oracle', 'order_sensitive',
                            'nominal_critical', 'predicted_counts', 'predicted_maxima', 'predicted_starts',
                            'truth_maxima', 'truth_starts'):
                    save_array(archive, prefix+'/'+key, evaluated[key])
                for key in ('valid_s', 'arrival_s', 'values_m', 'prior_m', 'residual_m', 'issue_index', 'actual_lead_s'):
                    save_array(archive, prefix+'/observations_'+key, batch['sensors'][key])
                save_array(archive, prefix+'/actual_input_record_indices', batch['record_ids'])
                audit.append(dict(condition=condition, mode=mode, **setting,
                    observation_sha256=batch['observation_sha256'],
                    actual_references=int(np.sum(batch['record_ids'] >= 0)), input_arrival_and_age_checked=True))
    assert len(audit) == 256 and abs(total_weight-128) < 1e-8
    aggregate = {key: value/total_weight for key, value in totals.items()}
    for entry in strata.values():
        for key in totals: entry[key] /= entry['weight']
        entry['order_sensitive_fraction'] = entry.pop('opportunity_sum')/entry['weight']
        entry['nominal_critical_fraction'] = entry.pop('nominal_critical_sum')/entry['weight']
    pulse = pulse_sum/pulse_weight
    for i, name in enumerate(names):
        if name == 'nominal' or name.startswith('forecast_ridge_'):
            assert pulse[i] == 0, 'Negative control: no-observation decisions changed under paired observations'
    result = dict(family_id=identifier, split=meta['split'], candidate_names=names, groups=groups,
        aggregate=aggregate, strata=strata, pulse_observation_effect=pulse,
        audit_batches=len(audit), causal_references=sum(r['actual_references'] for r in audit),
        raw_results=str(archive_path), raw_results_sha256=sha(archive_path),
        seconds=time.perf_counter()-started, source_code_hash=code_hash)
    write_json(directory/f'{identifier}.audit.json', audit)
    write_json(result_path, result)
    return read(result_path)


def average_validation(families):
    names = families[0]['candidate_names']; groups = families[0]['groups']
    assert all(f['candidate_names'] == names and f['groups'] == groups for f in families)
    mean = {key: np.mean([f['aggregate'][key] for f in families], axis=0) for key in ('yield', 'regret', 'mae', 'failed_rate')}
    return names, groups, mean


def old_event(path):
    with np.load(path, allow_pickle=False) as z: event = {key: z[key] for key in z.files}
    event['metadata'] = read(path.with_suffix('.json'))
    return event


def preflight(run, config, cases):
    marker = run/'preflight.json'
    if marker.exists(): return read(marker)
    old = ROOT/'results/runs/20261015_014922_600331_lead_stage1'
    moments = {kind: RidgeMoments() for kind in ('forecast', 'latest', 'history')}
    start = time.perf_counter()
    for event_name in ('lead_000_C0_stable', 'lead_000_C1_local'):
        training_add(moments, old_event(old/f'{event_name}.npz'), [9, 0], config)
    models = fit_models(moments, config)
    event = old_event(old/'lead_006_C1_local.npz')
    setting = next(s for s in schedules(config) if s['delay_s'] == 10 and s['noise_std_m'] > 0)
    batch = build_batch(event, setting, [9, 6], config['master_seed'])
    names, errors, groups = predict_candidates(batch, models, config)
    evaluated = evaluate_batch(batch, errors, cases)
    assert groups['A3'] == groups['A4'] and len(names) == 38
    scalar_checks = 0
    lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(batch['row_times'], batch['row_regions']))}
    for model_index in range(len(names)):
        for case_index in range(0, len(cases), 4):
            case = cases[case_index]; depth = np.zeros((26, 100))
            for target in case['targets']:
                row = lookup[case['now_s'], target]
                depth[:, target] = np.maximum(0, batch['base'][row]+errors[model_index, row])
            scalar = order_scores(case['now_s']+OFFSETS, depth, case['orders'], case['now_s'])
            assert [s['service_yield'] for s in scalar] == list(evaluated['predicted_counts'][model_index, case_index])
            for order_index, score in enumerate(scalar):
                assert [t['start_s']-case['now_s'] for t in score['trace']] == list(evaluated['predicted_starts'][model_index, case_index, order_index])
            scalar_checks += 2
    selected = select_models(names, groups, dict(yield_=None, **{'yield': evaluated['actual'].mean(axis=1), 'mae': evaluated['mae']}))
    wanted = list(dict.fromkeys(['nominal']+[v['candidate'] for v in selected.values()]))
    reduced_names, reduced, _ = predict_candidates(batch, models, config, wanted)
    for j, name in enumerate(reduced_names):
        np.testing.assert_array_equal(reduced[j], errors[names.index(name)])
    result = dict(passed=True, old_data_debug_only=True, no_new_effect_claim=True,
                  candidates=38, shared_local_candidates=10, ordinary_candidates=26,
                  independent_scalar_order_checks=scalar_checks,
                  reduced_evaluation_exactly_reproduces_all_candidates=True,
                  seconds=time.perf_counter()-start)
    write_json(marker, result)
    return result


def develop(run, config, cases):
    status(run, 'preflight'); checks = preflight(run, config, cases)
    assert checks['passed']; status(run, 'preflight_passed', checks=checks)
    data_dir = run/'families'; data_dir.mkdir(exist_ok=True)
    complete = []
    for split, count in [('train', config['family_counts']['train']), ('validation', config['family_counts']['validation']),
                         ('pilot', config['family_counts']['sample_size_pilot'])]:
        for index in range(count):
            info = generate_family(data_dir, config, split, index)
            complete.append(info); write_json(run/'development_family_manifest.json', complete)
            status(run, 'generating_development', split=split, index=index, completed_families=len(complete),
                   last_family_seconds=info['seconds'])
    model_file = run/'models.json'
    if model_file.exists():
        models = read(model_file)
    else:
        moments = {kind: RidgeMoments() for kind in ('forecast', 'latest', 'history')}
        for info in [f for f in complete if f['split'] == 'train']:
            arrays, meta = load_family(data_dir, info)
            for condition, mode, event in family_events(arrays, meta):
                training_add(moments, event, meta['seed_identity'], config)
            status(run, 'fitting_training', family=info['family_id'])
        models = fit_models(moments, config); write_json(model_file, models)
    validation = []
    for info in [f for f in complete if f['split'] == 'validation']:
        arrays, meta = load_family(data_dir, info)
        result = evaluate_family(run, arrays, meta, config, cases, models)
        validation.append(result)
        status(run, 'validation', family=info['family_id'], evaluation_seconds=result['seconds'])
    names, groups, means = average_validation(validation)
    selected = select_models(names, groups, means)
    selection = dict(selected=selected, validation_family_ids=[f['family_id'] for f in validation],
                     candidate_names=names, candidate_groups=groups, all_candidate_validation=means,
                     selection_uses_pilot_or_confirmation=False, source_hashes=source_hashes())
    write_json(run/'selection.json', selection)
    pilot = []
    for info in [f for f in complete if f['split'] == 'pilot']:
        arrays, meta = load_family(data_dir, info)
        result = evaluate_family(run, arrays, meta, config, cases, models, selected)
        pilot.append(result); status(run, 'pilot', family=info['family_id'], evaluation_seconds=result['seconds'])
    pilot_values = []
    for result in pilot:
        yn = result['candidate_names']; yy = result['aggregate']['yield']
        value = {g: yy[yn.index(v['candidate'])] for g, v in selected.items()}
        pilot_values.append([value['A4']-value['A3'], value['A4']-value['A2']])
    diff = np.asarray(pilot_values)
    sd = diff.std(axis=0, ddof=1); sstar = max(sd)*config['statistics']['pilot_sd_inflation']
    target = config['statistics']['minimum_practical_gain_services_per_initial_case']
    requested = math.ceil((config['statistics']['power_planning_z_sum']*sstar/target)**2)
    counts = config['family_counts']
    required = max(counts['confirmation_min'], math.ceil(requested/12)*12)
    meaningful_variation = bool(np.any(np.ptp(diff, axis=0) > 1e-12))
    opportunity = float(np.mean([e['order_sensitive_fraction'] for f in pilot for e in f['strata'].values()]))
    allowed = meaningful_variation and opportunity > 0
    plan = dict(pilot_family_ids=[f['family_id'] for f in pilot], differences=diff,
                standard_deviation=sd, inflated_sd=sstar, normal_approximation_required=requested,
                confirmation_n=min(required, counts['confirmation_max']), planned_power_target_supported=required <= counts['confirmation_max'],
                pilot_not_in_confirmation=True, meaningful_variation=meaningful_variation,
                order_sensitive_fraction=opportunity, confirmation_gate_passed=allowed,
                reason='fixed protocol and nonzero opportunity/variation' if allowed else 'No distinguishable pilot variation or task opportunity; stop before confirmation')
    write_json(run/'sample_size_plan.json', plan)
    benchmark = dict(preflight_seconds=checks['seconds'],
        generation_seconds=[f['seconds'] for f in complete],
        validation_seconds=[f['seconds'] for f in validation], pilot_seconds=[f['seconds'] for f in pilot],
        development_bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()))
    write_json(run/'resource_benchmark.json', benchmark)
    frozen = dict(source_hashes=freeze_sources(run), config_sha256=sha(run/'config.json'),
        protocol_sha256=sha(run/'PROTOCOL.md'), geometry_sha256=sha(run/'geometry.json'),
        models_sha256=sha(model_file), selection_sha256=sha(run/'selection.json'),
        sample_size_plan_sha256=sha(run/'sample_size_plan.json'), confirmation_n=plan['confirmation_n'],
        development_manifest_sha256=sha(run/'development_family_manifest.json'),
        preflight_passed=True, confirmation_gate_passed=allowed, frozen_unix_s=time.time())
    write_json(run/'freeze_manifest.json', frozen)
    write_json(run/'development_summary.json', dict(selection=selection, sample_size=plan, resource=benchmark))
    status(run, 'development_frozen' if allowed else 'stopped_by_pilot_gate', confirmation_n=plan['confirmation_n'], reason=plan['reason'])
    development_report(run, selection, plan, benchmark)
    return allowed


def development_report(run, selection, plan, benchmark):
    lines = ['# 本地残差与服务遗憾实验：开发与先导', '',
             '训练24、验证12、独立先导12个新家族；本页不是确认测试结果。', '',
             '|组别|验证选定候选|验证完成数/情境|验证MAE(m)|', '|---|---|---:|---:|']
    for group, value in selection['selected'].items():
        lines.append(f"|{group}|{value['candidate']}|{value['validation_yield']:.6f}|{value['validation_mae_m']:.6f}|")
    lines += ['', f"确认门槛：{plan['confirmation_gate_passed']}。{plan['reason']}",
              f"按先导方差规划确认家族数：{plan['confirmation_n']}；预算内满足近似功效规划：{plan['planned_power_target_supported']}。",
              f"先导顺序敏感比例：{plan['order_sensitive_fraction']:.6f}。",
              '', '先导只估计方差与机会，没有据其收益重新选模型。全部条件、候选评价、实际观测引用与服务轨迹见运行目录。',
              '', '原始记录与归档大小：'+str(benchmark['development_bytes'])+' bytes。']
    (run/'DEVELOPMENT_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def confirm(run, config, cases):
    frozen = read(run/'freeze_manifest.json')
    if not frozen['confirmation_gate_passed']: return False
    assert frozen['source_hashes'] == source_hashes(), 'Source changed after freeze'
    for filename, key in [('config.json', 'config_sha256'), ('PROTOCOL.md', 'protocol_sha256'),
                          ('geometry.json', 'geometry_sha256'), ('models.json', 'models_sha256'),
                          ('selection.json', 'selection_sha256'), ('sample_size_plan.json', 'sample_size_plan_sha256')]:
        assert sha(run/filename) == frozen[key], 'Frozen artifact changed: '+filename
    selected = read(run/'selection.json')['selected']; models = read(run/'models.json')
    rows = []; complete = []
    for index in range(frozen['confirmation_n']):
        info = generate_family(run/'families', config, 'confirmation', index)
        complete.append(info); write_json(run/'confirmation_family_manifest.json', complete)
        arrays, meta = load_family(run/'families', info)
        result = evaluate_family(run, arrays, meta, config, cases, models, selected)
        rows.append(result); status(run, 'confirmation', completed_families=index+1, total_families=frozen['confirmation_n'],
                                  generation_seconds=info['seconds'], evaluation_seconds=result['seconds'])
    produce_confirmation_report(run, config, rows, selected)
    status(run, 'complete', confirmation_families=len(rows))
    return True


def produce_confirmation_report(run, config, rows, selected):
    groups = list(selected); values = []; strata = []; pulses = []
    for result in rows:
        names = result['candidate_names']; ids = [names.index(selected[g]['candidate']) for g in groups]
        values.append(np.asarray(result['aggregate']['yield'])[ids])
        pulses.append(np.asarray(result['pulse_observation_effect'])[ids])
        for key, entry in result['strata'].items():
            strata.append(dict(family=result['family_id'], condition=key,
                **{g: float(entry['yield'][i]) for g, i in zip(groups, ids)},
                order_sensitive_fraction=entry['order_sensitive_fraction']))
    values = np.asarray(values); count = len(rows)
    rng = np.random.default_rng(np.random.SeedSequence([config['master_seed'], 7, 0]))
    resamples = rng.integers(0, count, (config['statistics']['bootstrap_family_resamples'], count))
    comparisons = []
    for baseline in ('A3', 'A2', 'A1', 'A0'):
        diff = values[:, groups.index('A4')]-values[:, groups.index(baseline)]
        boot = diff[resamples].mean(axis=1)
        primary = baseline in ('A3', 'A2'); quantiles = [.0125, .9875] if primary else [.025, .975]
        interval = np.quantile(boot, quantiles)
        comparisons.append(dict(proposed='A4', baseline=baseline, primary=primary,
            mean_gain_services_per_case=float(diff.mean()), interval=interval,
            interval_level=.975 if primary else .95, favorable_families=int(sum(diff > 1e-12)),
            zero_families=int(sum(abs(diff) <= 1e-12)), negative_families=int(sum(diff < -1e-12)),
            family_differences=diff, hypothesis_supported=bool(primary and interval[0] > 0 and diff.mean() >= .01)))
    safety = {}
    for subset in ('M0', 'noise_and_delay'):
        losses = []
        for result in rows:
            names = result['candidate_names']; ia = names.index(selected['A4']['candidate']); ib = names.index(selected['A2']['candidate'])
            entries = [e for k, e in result['strata'].items() if ('/M0/' in k if subset == 'M0' else '/noise0.02/delay10' in k)]
            losses.append(np.mean([e['yield'][ib]-e['yield'][ia] for e in entries]))
        losses = np.asarray(losses)
        # With fixed two target slots, failure-rate increase is loss/2.
        upper = float(np.quantile((losses[resamples]/2).mean(axis=1), .95))
        safety[subset] = dict(mean_loss=float(losses.mean()), failure_rate_increase_upper_one_sided_95=upper,
            passed=bool(losses.mean() <= .005 and upper <= .005))
    result = dict(confirmation_families=count, comparisons=comparisons, safety=safety,
        mean_yield={g: float(values[:, i].mean()) for i, g in enumerate(groups)},
        family_yields=values, family_ids=[r['family_id'] for r in rows],
        mean_pulse_observation_effect={g: float(np.mean(pulses, axis=0)[i]) for i, g in enumerate(groups)},
        full_core_claim_supported=all(c['hypothesis_supported'] for c in comparisons if c['primary']),
        engineering_expansion_gate_passed=all(c['hypothesis_supported'] for c in comparisons if c['primary']) and all(v['passed'] for v in safety.values()),
        resource=str(run/'resource_benchmark.json'), source_code_frozen=True)
    write_json(run/'confirmation_summary.json', result); write_json(run/'condition_family_scores.json', strata)
    lines = ['# 本地残差与服务遗憾：独立确认结果', '',
             f'独立确认家族 {count} 个，模型按训练/验证冻结。每家族128个紧凑两目标初始情境，各条件内部配对。', '',
             '|比较|每100情境新增服务|家族区间/100|正/零/负家族|主假设获支持|', '|---|---:|---|---|---|']
    for c in comparisons:
        lo, hi = np.asarray(c['interval'])*100
        lines.append(f"|A4−{c['baseline']}|{c['mean_gain_services_per_case']*100:+.4f}|[{lo:+.4f},{hi:+.4f}]|{c['favorable_families']}/{c['zero_families']}/{c['negative_families']}|{c['hypothesis_supported'] if c['primary'] else '次级'}|")
    lines += ['', '两个主要比较各用97.5%双侧家族区间；次级比较95%区间仅作解释。',
              '', f"完整主张获支持：{result['full_core_claim_supported']}。扩展工程门槛通过：{result['engineering_expansion_gate_passed']}。",
              '', '本实验只涉及固定观测、单机两目标初始选序，不证明物理时滞、多机滚动重规划、主动观测或真实部署收益。',
              '', '全部条件、家族差值、脉冲观测配对效应、观测与评分记录见同目录JSON及评价归档。']
    (run/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(ROOT/'results/latest_local_service.json', dict(run=str(run), report=str(run/'REPORT.md'), completed=True))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path)
    parser.add_argument('--stage', choices=['init', 'develop', 'confirm', 'all'], default='all')
    args = parser.parse_args()
    if args.run is None:
        config, preview = load_design(ROOT)
        config = dict(config, execution_started=True, status='execution_protocol_v1')
        run = new_run('local_service_campaign', config)
        shutil.copy2(config['protocol_path'], run/'PROTOCOL.md')
        write_json(run/'geometry.json', preview); freeze_sources(run)
        print('RUN_DIRECTORY '+str(run), flush=True)
    else:
        run = args.run.resolve(); config = read(run/'config.json'); preview = read(run/'geometry.json')
    cases = task_table(preview); write_json(run/'task_cases.json', cases)
    if args.stage == 'init':
        result = preflight(run, config, cases); status(run, 'preflight_passed', checks=result); return
    if args.stage in ('develop', 'all'):
        allowed = develop(run, config, cases)
        if not allowed:
            write_json(ROOT/'results/latest_local_service.json', dict(run=str(run), report=str(run/'DEVELOPMENT_REPORT.md'), completed=True, stopped_at_gate=True))
            return
    if args.stage in ('confirm', 'all'): confirm(run, config, cases)


if __name__ == '__main__': main()
