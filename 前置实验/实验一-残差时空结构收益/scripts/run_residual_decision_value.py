"""Complete the exploratory residual-to-decision audit; no online policy training."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from hyra_validation.artifacts import ROOT, new_run, write_json
from hyra_validation.data import load_events
from hyra_validation.decision_value import (
    METHODS, OFFSETS, issued_curve, scheduled_records, binned_history,
    feature_groups, fit_reader, predict_reader, shortest_paths,
    candidate_orders, max_duration, order_scores, choose, prefix_checkpoints,
    record_evidence)
from hyra_validation.lead import TARGETS

EVENT_RUN = ROOT / 'results/runs/20261015_014922_600331_lead_stage1'
TIMING_RUN = ROOT / 'results/runs/20261015_021259_656652_lead_timing_diagnostic'
PRELIM_RUN = ROOT / 'results/runs/20261015_125958_501553_residual_decision_value'
RUN_METHODS = ('forecast_only',) + METHODS
TIMES = tuple(range(24, 241, 6))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump_line(handle, obj):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    handle.write(json.dumps(obj, ensure_ascii=False, default=convert, allow_nan=False) + '\n')


def schedules():
    settings = [(5, d, p) for d in (0, 10) for p in (0, .25, .5, .75)]
    settings += [(r, d, 0) for r in (10, 40) for d in (0, 10)]
    return [dict(revisit=r, delay=d, phase=p, schedule=f'r{r}_d{d}_p{int(p*4)}')
            for r, d, p in settings]


def all_features(event, records, graph, region, now, forecast):
    result = feature_groups(event, records, graph, region, now, forecast)
    result['forecast_only'] = result['raw_full'].copy()
    result['forecast_only'][:96] = 0
    result['forecast_only'][-9:] = 0
    return result


def make_layouts():
    """Geometry only; replicate the existing obstacle and target seed rules."""
    output = []
    for kind in ('general', 'calibrated_targets'):
        for index in range(32):
            blocked = np.zeros(100, dtype=bool)
            eligible = np.setdiff1d(np.arange(100), [0, 1, 10, 11])
            blocked[np.random.default_rng(815+index).choice(eligible, 10, replace=False)] = True
            rng = np.random.default_rng(9815+index)
            origin = int(rng.choice(np.flatnonzero(~blocked)))
            from_origin = shortest_paths(blocked, origin)
            possible = ([r for r in from_origin if r != origin] if kind == 'general'
                        else [r for r in TARGETS if r in from_origin and r != origin])
            if len(possible) < 2: continue
            targets = list(map(int, rng.choice(possible, 2, replace=False)))
            paths = {r: shortest_paths(blocked, r) for r in [origin, *targets]}
            output.append(dict(kind=kind, layout=index, origin=origin, targets=targets,
                               blocked=blocked, paths=paths,
                               orders=candidate_orders(paths, origin, targets)))
    return output


def response_offsets():
    grouped = defaultdict(list)
    for row in read(TIMING_RUN/'direction_audit.json')['rows']:
        if row['role'] == 'target' and row['arrival_after_pulse_s'] is not None:
            grouped[(row['center_y_m'], row['region'])].append(row['arrival_after_pulse_s'])
    return {k: float(np.median(v)) for k, v in grouped.items()}


def context(event, split, layout, now, origin, initial_time, initial_order, index,
            orders, response, forecast_cache):
    eid = event['metadata']['event_id']
    if (eid, now) not in forecast_cache:
        forecast_cache[eid, now] = issued_curve(event, now)
    f, curve = forecast_cache[eid, now]
    if now + max_duration(orders) > min(300, now+50, event['forecast_valid_s'][f, -1]):
        return None
    nominal_scores = order_scores(event['forecast_valid_s'][f], event['forecast_depth_m'][f], orders, now)
    nominal = choose(nominal_scores)
    truth_scores = order_scores(event['time_s'], event['truth_depth_m'], orders, now)
    truth = [r['service_yield'] for r in truth_scores]
    meta = event['metadata']
    arrivals = [meta['pulse_start_s'] + response[meta['pulse_center_y_m'], r]
                for r in layout['targets'] if (meta['pulse_center_y_m'], r) in response]
    first_response = min(arrivals) if arrivals else None
    period = ('unmatched_branch' if first_response is None else
              'before_nominal_response' if now < first_response else 'after_nominal_response')
    return dict(case_id=f'{eid}/L{layout["layout"]}/T{initial_time}',
        event_id=eid, split=split, family=meta['event_family_id'], condition=meta['condition'],
        layout=layout['layout'], kind=layout['kind'], initial_time_s=initial_time,
        initial_origin=layout['origin'], node_index=index, now_s=now, origin=origin,
        targets=layout['targets'], orders=orders, initial_order=initial_order,
        baseline_order=nominal, nominal_predicted_yields=[r['service_yield'] for r in nominal_scores],
        true_yields=truth, truth_traces=truth_scores, finite_oracle=max(truth),
        baseline_regret=max(truth)-truth[nominal], continue_regret=max(truth)-truth[initial_order],
        forecast_id=f, forecast_release_s=float(event['release_s'][f]),
        nominal_first_target_response_s=first_response, response_period=period,
        response_metadata_used_for_policy=False)


def build_contexts(events, layouts, run, forecast_cache):
    contexts = defaultdict(list); opportunities = []; all_general = []; excluded_prefix = 0
    response = response_offsets()
    with gzip.open(run/'task_contexts.jsonl.gz', 'wt', encoding='utf-8') as writer:
        for split, sequence in events.items():
            for event in sequence:
                eid = event['metadata']['event_id']
                for kind in ('general', 'calibrated_targets'):
                    kept = []; excluded = 0
                    for layout in (l for l in layouts if l['kind'] == kind):
                        for now in TIMES:
                            f = int(np.flatnonzero(event['release_s'] <= now)[-1])
                            end = min(300, event['forecast_valid_s'][f, -1],
                                      now+50 if kind == 'calibrated_targets' else 360)
                            if now+max_duration(layout['orders']) > end:
                                excluded += 1; continue
                            nominal = order_scores(event['forecast_valid_s'][f], event['forecast_depth_m'][f], layout['orders'], now)
                            selected = choose(nominal)
                            truth = [s['service_yield'] for s in order_scores(event['time_s'], event['truth_depth_m'], layout['orders'], now)]
                            row = dict(event_id=eid, family=event['metadata']['event_family_id'], split=split,
                                condition=event['metadata']['condition'], kind=kind, layout=layout['layout'],
                                time_s=now, origin=layout['origin'], targets=layout['targets'],
                                true_yields=truth, baseline_order=selected,
                                gap=max(truth)-truth[selected], order_sensitive=truth[0] != truth[1])
                            kept.append(row)
                            if kind == 'general':
                                all_general.append(row); continue
                            nodes = []
                            for index, (when, location) in enumerate(prefix_checkpoints(layout['paths'], layout['origin'], layout['targets'], selected, now)):
                                if location not in layout['paths']:
                                    layout['paths'][location] = shortest_paths(layout['blocked'], location)
                                orders = candidate_orders(layout['paths'], location, layout['targets'])
                                item = context(event, split, layout, when, location, now, selected, index, orders, response, forecast_cache)
                                if item is None:
                                    excluded_prefix += 1; continue
                                nodes.append(item)
                            corrective = [n['now_s'] for n in nodes if n['continue_regret'] > 0]
                            nominal_gaps = [n['now_s'] for n in nodes if n['baseline_regret'] > 0]
                            for item in nodes:
                                item['last_correctable_s'] = max(corrective) if corrective else None
                                item['last_nominal_regret_s'] = max(nominal_gaps) if nominal_gaps else None
                                contexts[eid].append(item); dump_line(writer, item)
                    opportunities.append(dict(kind=kind, split=split, family=event['metadata']['event_family_id'],
                        condition=event['metadata']['condition'], cases=len(kept), excluded=excluded,
                        positive_gap=sum(r['gap'] > 0 for r in kept), order_sensitive=sum(r['order_sensitive'] for r in kept),
                        mean_gap=float(np.mean([r['gap'] for r in kept])) if kept else 0.))
    write_json(run/'opportunity_summary.json', opportunities)
    write_json(run/'general_opportunity_cases.json', all_general)
    old = read(EVENT_RUN/'opportunity_cases.json')
    index = {(r['family'], r['condition'], r['layout'], r['time_s']): r for r in old}
    fresh = [r for r in all_general if r['split'] in ('train', 'validation')]
    mismatches = [r for r in fresh if (r['family'],r['condition'],r['layout'],r['time_s']) not in index or
        any(r[k] != index[r['family'],r['condition'],r['layout'],r['time_s']][k] for k in ('gap','order_sensitive'))]
    replication = dict(cases=len(fresh), old_cases=len(old), mismatches=len(mismatches),
        passed=len(fresh) == len(old) and not mismatches, excluded_prefix_nodes=excluded_prefix)
    write_json(run/'general_reproduction.json', replication)
    if not replication['passed']: raise ValueError('General opportunity reproduction failed')
    return contexts


def dataset(events, split, graph, records, forecast_cache):
    xs = {m: [] for m in RUN_METHODS}; ys = []; metadata = []
    for event in events[split]:
        eid = event['metadata']['event_id']
        for now in TIMES:
            if (eid, now) not in forecast_cache: forecast_cache[eid, now] = issued_curve(event, now)
            f, curve = forecast_cache[eid, now]
            labels = np.column_stack([np.interp(now+OFFSETS, event['time_s'], event['truth_depth_m'][:, r]) for r in TARGETS]) - curve[:, TARGETS]
            for j, region in enumerate(TARGETS):
                values = all_features(event, records[eid], graph, region, now, curve)
                for m in RUN_METHODS: xs[m].append(values[m])
                ys.append(labels[:, j]); metadata.append(dict(event_id=eid, family=event['metadata']['event_family_id'],
                    condition=event['metadata']['condition'], now_s=now, target=region, forecast_id=f))
    return dict(x={m: np.asarray(x) for m,x in xs.items()}, y=np.asarray(ys), meta=metadata)


def family_error(prediction, data):
    errors = np.mean(abs(prediction-data['y']), axis=1)
    families = np.array([r['family'] for r in data['meta']])
    return float(np.mean([errors[families == f].mean() for f in sorted(set(families))]))


def evaluate_schedule(events, graph, contexts, run, settings, forecast_cache, aggregates):
    name = settings['schedule']; out = run/name; out.mkdir()
    records = {e['metadata']['event_id']: scheduled_records(e, settings['revisit'], settings['delay'], settings['phase'])
               for seq in events.values() for e in seq}
    with gzip.open(out/'observations.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for eid, by_region in records.items():
            for region, rows in by_region.items():
                for row in rows: dump_line(handle, dict(event_id=eid, record_id=f'{eid}/{region}/{row["valid_time_s"]:g}', **row))
    datasets = {s: dataset(events, s, graph, records, forecast_cache) for s in events}
    readers = {}; reader_info = []; model_arrays = {}
    for method in RUN_METHODS:
        candidates = {a: fit_reader(datasets['train']['x'][method], datasets['train']['y'], a) for a in (.1, 1, 10)}
        validation = {a: family_error(predict_reader(model, datasets['validation']['x'][method]), datasets['validation']) for a,model in candidates.items()}
        alpha = min(validation, key=validation.get); model = candidates[alpha]; readers[method] = model
        z = (datasets['train']['x'][method]-model['mean'])/model['std']
        reader_info.append(dict(method=method, alpha=alpha, validation_family_mae=validation,
            allocated_features=z.shape[1], nonconstant_columns=int(np.sum(np.std(z, axis=0)>1e-8)),
            standardized_design_rank=int(np.linalg.matrix_rank(np.c_[np.ones(len(z)), z])),
            selection_used_test=False, posthoc_attribution_control=method == 'forecast_only'))
        for k in ('mean','std','weights'): model_arrays[f'{method}_{k}'] = model[k]
        for split, data in datasets.items():
            prediction = predict_reader(model, data['x'][method]); error = prediction-data['y']
            fam = np.array([r['family'] for r in data['meta']]); cond = np.array([r['condition'] for r in data['meta']])
            for f,c in sorted(set(zip(fam,cond))):
                mask = (fam == f)&(cond == c)
                aggregates['information'].append(dict(settings,split=split,family=f,condition=c,method=method,
                    samples=int(mask.sum()),mae_m=float(abs(error[mask]).mean()),rmse_m=float(np.sqrt(np.mean(error[mask]**2)))))
            if split == 'test': model_arrays[f'{method}_internal_prediction'] = prediction
    model_arrays['internal_labels'] = datasets['test']['y']
    np.savez_compressed(out/'models_and_internal_predictions.npz', **model_arrays)
    write_json(out/'readers.json',reader_info); write_json(out/'internal_prediction_index.json',datasets['test']['meta'])
    initial_groups = defaultdict(list); window_groups = defaultdict(list); phase_groups = defaultdict(list)
    old_rows = read(PRELIM_RUN/f'{name}_decision_cases.json')
    old_index = {(r['event_id'],r['layout'],r['now_s'],r['method']):r for r in old_rows}
    checked = 0; mismatches = []
    with gzip.open(out/'decisions.jsonl.gz','wt',encoding='utf-8') as decisions, \
         gzip.open(out/'evidence.jsonl.gz','wt',encoding='utf-8') as evidence, \
         gzip.open(out/'window_outcomes.jsonl.gz','wt',encoding='utf-8') as windows:
        for split, sequence in events.items():
            for event in sequence:
                eid = event['metadata']['event_id']; nodes = contexts[eid]
                keys = sorted({(n['now_s'],r) for n in nodes for r in n['targets']})
                features = {m: [] for m in RUN_METHODS}
                for now, region in keys:
                    curve = forecast_cache[eid,now][1]
                    ff = all_features(event, records[eid], graph, region, now, curve)
                    for m in RUN_METHODS: features[m].append(ff[m])
                    history = {str(source): [f'{eid}/{source}/{row["valid_time_s"]:g}' if row else None
                        for row in binned_history(records[eid],source,now)]
                        for source in [region]+[l['source'] for l in graph[str(region)]]}
                    dump_line(evidence,dict(evidence_id=f'{eid}/{now}/{region}',event_id=eid,now_s=now,target=region,
                        actual_history_record_ids=history,latest_upstream=record_evidence(records[eid],graph,region,now)))
                curve_predictions = {m: predict_reader(readers[m],np.asarray(features[m])) for m in RUN_METHODS}
                key_index = {key:i for i,key in enumerate(keys)}
                episodes = defaultdict(list)
                for n in nodes:
                    for method in RUN_METHODS:
                        forecast = forecast_cache[eid,n['now_s']][1].copy()
                        for target in n['targets']:
                            forecast[:,target] = np.maximum(0,forecast[:,target]+curve_predictions[method][key_index[n['now_s'],target]])
                        scores = order_scores(n['now_s']+OFFSETS,forecast,n['orders'],n['now_s']); selected = choose(scores)
                        truth = n['true_yields']; value = truth[selected]
                        row = dict(settings,**n,method=method,selected_order=selected,
                            model_predicted_yields=[s['service_yield'] for s in scores],
                            selected_true_yield=value, baseline_true_yield=truth[n['baseline_order']],
                            continue_true_yield=truth[n['initial_order']],
                            realized_delta=value-truth[n['baseline_order']],
                            delta_continue=value-truth[n['initial_order']],selected_regret=max(truth)-value,
                            evidence_ids=[f'{eid}/{n["now_s"]}/{target}' for target in n['targets']],
                            main_outcome=n['node_index'] == 0)
                        dump_line(decisions,row); episodes[n['case_id'],method].append(row)
                        if n['node_index'] == 0:
                            key=(split,n['family'],n['condition'],method)
                            initial_groups[key].append(row)
                            phase_groups[key+(n['response_period'],)].append(row)
                            if split == 'test' and method in METHODS:
                                original=old_index[eid,n['layout'],n['now_s'],method];checked+=1
                                if any(row[k] != original[k] for k in ('selected_order','baseline_order','true_yields','realized_delta')):
                                    mismatches.append(dict(event_id=eid,layout=n['layout'],now_s=n['now_s'],method=method))
                for (cid,method),rows in episodes.items():
                    first=rows[0];last=first['last_correctable_s']
                    good=[r['now_s'] for r in rows if r['delta_continue']>0]
                    capture=bool(good and last is not None and min(good)<=last)
                    record=dict(settings,case_id=cid,split=split,family=first['family'],condition=first['condition'],method=method,
                        initial_time_s=first['initial_time_s'],correctable=last is not None,
                        last_correctable_s=last,first_gain_node_s=min(good) if good else None,
                        captured=capture,initial_gain=first['delta_continue']>0,
                        late_only_gain=capture and first['delta_continue']<=0,
                        sampled_nodes=len(rows),potential_only_not_closed_loop=True)
                    dump_line(windows,record);window_groups[split,first['family'],first['condition'],method].append(record)
    for (split,family,condition,method),rows in initial_groups.items():
        aggregates['task'].append(dict(settings,split=split,family=family,condition=condition,method=method,
            cases=len(rows),mean_yield=float(np.mean([r['selected_true_yield'] for r in rows])),
            mean_delta_nominal=float(np.mean([r['realized_delta'] for r in rows])),
            mean_regret=float(np.mean([r['selected_regret'] for r in rows])),
            mean_baseline_regret=float(np.mean([r['baseline_regret'] for r in rows])),
            positive=sum(r['realized_delta']>0 for r in rows),negative=sum(r['realized_delta']<0 for r in rows),
            changed=sum(r['selected_order']!=r['baseline_order'] for r in rows)))
    for (split,family,condition,method),rows in window_groups.items():
        aggregates['windows'].append(dict(settings,split=split,family=family,condition=condition,method=method,
            cases=len(rows),correctable_cases=sum(r['correctable'] for r in rows),captured_cases=sum(r['captured'] for r in rows),
            initial_gain_cases=sum(r['initial_gain'] for r in rows),late_only_gain_cases=sum(r['late_only_gain'] for r in rows)))
    for (split,family,condition,method,period),rows in phase_groups.items():
        aggregates['strata'].append(dict(settings,split=split,family=family,condition=condition,method=method,
            response_period=period,cases=len(rows),mean_delta_nominal=float(np.mean([r['realized_delta'] for r in rows]))))
    reproduction=dict(schedule=name,checked=checked,old_rows=len(old_rows),mismatches=mismatches,
                      passed=checked==len(old_rows) and not mismatches)
    write_json(out/'preliminary_reproduction.json',reproduction)
    if not reproduction['passed']:raise ValueError(f'Preliminary reproduction failed: {name}')
    return reproduction


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--skip-timing',action='store_true');args=parser.parse_args()
    events=load_events(EVENT_RUN/'manifest.json');graph=read(EVENT_RUN/'physical_graph.json')['graph']
    settings=dict(protocol='V1_with_completion_addendum',manifest=str(EVENT_RUN/'manifest.json'),
        methods=RUN_METHODS,schedules=schedules(),families=12,internal_diagnostic_families=3,
        prior_results_seen=True,selected_readers=84,regularization_candidates=[.1,1,10],
        no_qmix=True,no_new_formal_test=True)
    run=new_run('residual_decision_complete',settings);print('RUN',run,flush=True)
    for doc in ('DECISION_VALUE_DIAGNOSTIC_V1.md','DECISION_VALUE_COMPLETION_ADDENDUM.md'):
        shutil.copyfile(ROOT/'docs'/doc,run/doc)
    sources=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    for source in sources:
        dest=run/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    inputs=[EVENT_RUN/'manifest.json',EVENT_RUN/'physical_graph.json',TIMING_RUN/'direction_audit.json']
    inputs+=list(EVENT_RUN.glob('lead_*.npz'))+list(EVENT_RUN.glob('lead_*.json'))
    hashes={str(p):digest(p) for p in inputs};write_json(run/'input_fingerprints.json',hashes)
    layouts=make_layouts();write_json(run/'layouts.json',[{k:v for k,v in l.items() if k!='paths'} for l in layouts])
    cache={};contexts=build_contexts(events,layouts,run,cache)
    print('Geometry and general reproduction complete',flush=True)
    aggregates=dict(task=[],information=[],windows=[],strata=[]);reproductions=[]
    for setting in schedules():
        reproductions.append(evaluate_schedule(events,graph,contexts,run,setting,cache,aggregates))
        for label,key in [('family_task_scores','task'),('family_information_scores','information'),('window_family_scores','windows'),('response_strata_scores','strata')]:
            write_json(run/f'{label}.json',aggregates[key])
        print('Completed',setting['schedule'],flush=True)
    if not args.skip_timing:
        from scan_decision_timing import scan
        scan(events,graph,run)
    audit=dict(input_files_unchanged=all(digest(p)==h for p,h in hashes.items()),
        preliminary_reproductions=reproductions,preliminary_all_reproduced=all(r['passed'] for r in reproductions),
        original_general_reproduced=read(run/'general_reproduction.json')['passed'],
        unique_initial_internal_cases=len({n['case_id'] for e in events['test'] for n in contexts[e['metadata']['event_id']] if n['node_index']==0}),
        accepted_calibrated_layouts=sorted({n['layout'] for seq in contexts.values() for n in seq if n['node_index']==0}),
        candidate_calibrated_layouts=32,independent_internal_families=3,
        full_timing_scan=not args.skip_timing,truth_only_for_labels_scoring_and_offline_strata=True)
    write_json(run/'implementation_audit.json',audit)
    from finalize_decision_value import finalize
    finalize(run)
    write_json(run/'completion.json',dict(completed=True,exploratory=True,protocol_completed=not args.skip_timing,
        formal_test_opened=False,qmix_trained=False,old_information_gate_passed=False))
    write_json(ROOT/'results/latest_decision_value.json',dict(run=str(run),report=str(run/'REPORT.md'),status='completed',formal_test_opened=False,qmix_trained=False))
    print(run,flush=True)


if __name__=='__main__':main()
