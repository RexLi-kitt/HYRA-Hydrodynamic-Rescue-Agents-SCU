"""Audit frozen P3 caches against arrived records and actual fitted forests.

No fitting, selection, scoring or source mutation. Historical node features are
rebuilt directly from observation records rather than the production builder.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyra_uav.core import FloodProcess
from hyra_uav.relations import weighted_relations
from hyra_uav.experiments import write_json

GROUPS = ("B1", "B2", "B3", "A1", "A2")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def nodes_from_records(records, now, latest=False):
    result = np.zeros((len(records), 19), dtype=np.float32)
    for cell, history in enumerate(records):
        if not history:
            continue
        use = history[-1:] if latest else history
        for k, (observed, arrived, label, quality) in enumerate(reversed(use)):
            assert 0 <= observed <= arrived <= now
            result[cell, 3*k:3*k+3] = label, 1, (now-observed)/600
        observed, _, label, quality = use[-1]
        result[cell, 12:15] = label, (now-observed)/600, quality
        transitions = [(a, b) for a, b in zip(use, use[1:]) if a[2] != b[2]]
        if transitions:
            a, b = transitions[-1]
            result[cell, 15:] = b[2]-a[2], (now-b[0])/600, (b[0]-a[0])/600, 1
    return result


def expected_common(nodes, cells, graph, bed):
    pool = graph['pool'][cells]
    static = np.stack((graph['distance'][cells]/5,
                       bed.ravel()[pool]-bed.ravel()[cells, None],
                       graph['terrain_factor'][cells]), axis=-1)
    return np.concatenate((nodes[cells], nodes[pool].reshape(len(cells), pool.shape[1]*19),
                           static.reshape(len(cells), pool.shape[1]*3)), axis=1).astype(np.float32)


def audit_entry(root, entry, plan, process, graph):
    for kind in ('replay', 'cache'):
        assert digest(root/entry[kind]) == entry[kind+'_sha256'], entry['run_id']
    replay = read(root/entry['replay'])
    n = int(np.prod(process.shape))
    records = [[] for _ in range(n)]
    observed = np.full(n, -1.)
    labels = np.full(n, -1, dtype=np.int8)
    cursor = offset = decisions = changed_history_rows = 0
    rng = np.random.default_rng(entry['sampling_seed'])
    strict = plan.get('strict_change_ablation', False)
    with np.load(root/entry['cache'], allow_pickle=False) as packed:
        # Materialize each array once; npz indexing otherwise decompresses the
        # entire feature matrix again for every decision and every check.
        data = {key: packed[key] for key in packed.files}
        count = len(data['y'])
        assert data['common'].shape == (count, 1427)
        for key in data:
            assert np.isfinite(data[key]).all(), key
        assert count == entry['rows'] and int(data['y'].sum()) == entry['positives']
        assert set(np.unique(data['y'])).issubset({0, 1})
        assert len(np.unique(np.column_stack((data['time_s'], data['cells'])), axis=0)) == count
        assert ('common_A1' in data) == strict
        for group in GROUPS:
            assert data[group].shape == (count, 21)
        for snapshot, truth in zip(replay['snapshots'], replay['truth']):
            now = snapshot['time_s']
            refreshed = np.zeros(n, dtype=bool)
            while cursor < len(replay['observations']) and replay['observations'][cursor]['arrived_s'] <= now:
                observation = replay['observations'][cursor]
                timestamp, arrival = observation['observed_s'], observation['arrived_s']
                assert 0 <= timestamp <= arrival <= now
                for cell, label in zip(observation['cells'], observation['labels']):
                    if timestamp > observed[cell]:
                        observed[cell], labels[cell] = timestamp, label
                        records[cell].append((timestamp, arrival, label, observation.get('quality', 1.)))
                        records[cell] = records[cell][-4:]
                if arrival == now:
                    refreshed[observation['cells']] = True
                cursor += 1
            if not 0 < now < replay['config']['duration_s'] or now % 10:
                continue
            cells = np.flatnonzero((labels >= 0) & ~refreshed)
            if entry['cap'] is not None and len(cells) > entry['cap']:
                cells = np.sort(rng.choice(cells, entry['cap'], replace=False))
            section = slice(offset, offset+len(cells))
            np.testing.assert_array_equal(data['cells'][section], cells)
            np.testing.assert_array_equal(data['time_s'][section], now)
            np.testing.assert_array_equal(data['age_s'][section], now-observed[cells])
            assert np.all(now-observed[cells] > 0)
            actual_truth = process.labels[snapshot['frame_index']].ravel()
            np.testing.assert_array_equal(truth, actual_truth)
            np.testing.assert_array_equal(data['y'][section], labels[cells] != actual_truth[cells])
            nodes = nodes_from_records(records, now)
            common = expected_common(nodes, cells, graph, process.bed)
            np.testing.assert_allclose(data['common'][section], common, rtol=1e-6, atol=1e-7)
            latest_nodes = nodes_from_records(records, now, latest=True) if strict else nodes
            if strict:
                latest_common = expected_common(latest_nodes, cells, graph, process.bed)
                np.testing.assert_allclose(data['common_A1'][section], latest_common, rtol=1e-6, atol=1e-7)
                changed_history_rows += int(np.any(common != latest_common, axis=1).sum())
                history = data['common_A1'][section, :65*19].reshape(-1, 65, 19)
                assert not np.any(history[:, :, 3:12]) and not np.any(history[:, :, 15:])
            pool = graph['pool'][cells]
            for group in GROUPS:
                source = latest_nodes if group == 'A1' else nodes
                weights = np.ones(pool.shape) if group == 'B1' else graph[
                    {'B2': 'B2_weight', 'A2': 'A2_weight'}.get(group, 'B3_weight')][cells]
                weights = weights/np.maximum(weights.sum(axis=1, keepdims=True), 1e-20)
                aggregate = np.einsum('nk,nkf->nf', weights, source[pool]).astype(np.float32)
                if group == 'A1':
                    aggregate[:, 15:] = 0
                tail = np.concatenate((aggregate, aggregate[:, [15, 18]]*source[cells, 13:14]), axis=1)
                np.testing.assert_allclose(data[group][section], tail, rtol=1e-6, atol=1e-7)
            offset += len(cells)
            decisions += 1
        assert offset == count and decisions == 59
        np.testing.assert_array_equal(data['decision_times_s'], np.arange(10, 600, 10))
    return dict(run_id=entry['run_id'], split=entry['split'], rows=count,
                positives=entry['positives'], decisions=decisions, strict_A1=strict,
                raw_histories_removed_rows=changed_history_rows,
                checks_passed=True, cache_sha256=entry['cache_sha256'])


def model_audit(root, plan, entries):
    local = ROOT/'.runtime/research-packages'
    sys.path.append(str(local))
    import joblib
    freeze = read(root/'model-freeze.json')
    assert freeze['plan_sha256'] == digest(root/'plan.json')
    trials, fits = read(root/'selection.json'), read(root/'fit-index.json')
    assert len(trials) == 10 and len(fits) == 15
    assert {(r['group'], r['candidate']) for r in trials} == {(g, c) for g in GROUPS for c in (0, 1)}
    assert {(r['group'], r['seed']) for r in fits} == {(g, s) for g in GROUPS for s in plan['seeds']}
    training = [e for e in entries if e['split'] == 'fit']
    assert len(training) == 18
    rows, positives = sum(e['rows'] for e in training), sum(e['positives'] for e in training)
    result = []
    for group in GROUPS:
        candidates = [r for r in trials if r['group'] == group]
        best = max(candidates, key=lambda r: (r['selection_mean_precision'], -r['candidate']))
        choice = next(r for r in freeze['choices'] if r['group'] == group)
        assert choice['candidate'] == best['candidate'] and choice['params'] == best['params']
        for row in (r for r in fits if r['group'] == group):
            assert row['fit_rows'] == rows and row['positives'] == positives
            assert digest(root/row['path']) == row['sha256']
            assert next(r for r in freeze['models'] if r['group'] == group and r['seed'] == row['seed']) == row
            model = joblib.load(root/row['path'])
            params = model.get_params()
            assert model.n_features_in_ == row['feature_dimension'] == 1448
            assert params['random_state'] == row['seed'] and params['n_estimators'] == 128
            assert params['max_depth'] == 18 and params['n_jobs'] == 2
            for key, value in choice['params'].items():
                assert params[key] == value
            nodes = sum(t.tree_.node_count for t in model.estimators_)
            leaves = sum(int(np.sum(t.tree_.children_left < 0))*2 for t in model.estimators_)
            assert nodes == row['total_nodes'] and leaves == row['leaf_parameters']
            result.append(dict(group=group, seed=row['seed'], fit_rows=rows, positives=positives,
                               nodes=nodes, stored_leaf_class_values=leaves, checks_passed=True))
            del model
    frozen_inputs = {e['run_id']: e for e in freeze['datasets']}
    assert set(frozen_inputs) == {e['run_id'] for e in entries if e['split'] != 'holdout'}
    for entry in entries:
        if entry['split'] != 'holdout':
            assert all(frozen_inputs[entry['run_id']][key] == entry[key]
                       for key in ('cache_sha256', 'replay_sha256'))
    return result


def main(root, output, expected, models, splits=None):
    root = root.resolve()
    plan, entries = read(root/'plan.json'), read(root/'index.json')
    if splits:
        entries = [e for e in entries if e['split'] in splits]
    assert len(entries) == expected
    graphs, results = {}, []
    for entry in entries:
        if entry['id'] not in graphs:
            process = FloodProcess(root/'scenes'/entry['id']/'map-result.zip')
            graph = weighted_relations(process, pool_size=plan['pool_size'], terrain_cost_scale_m=plan['cost_scale_m'])
            graphs[entry['id']] = process, graph
        results.append(audit_entry(root, entry, plan, *graphs[entry['id']]))
        print('audited', entry['run_id'], flush=True)
    family_sets = {s: sorted({e['family'] for e in entries if e['split'] == s})
                   for s in ('fit', 'selection', 'holdout')}
    assert all(not set(family_sets[a]) & set(family_sets[b]) for a, b in
               (('fit', 'selection'), ('fit', 'holdout'), ('selection', 'holdout')))
    budget = model_audit(root, plan, entries) if models else None
    write_json(output, dict(status='passed', round=plan['round'], replay_count=expected,
                           cache_rows=sum(r['rows'] for r in results), runs=results,
                           family_sets=family_sets, actual_models=budget,
                           auditor_sha256=digest(Path(__file__)),
                           checks=['frozen input hashes', 'candidate eligibility and deterministic cap',
                                   'actual-frame labels created after feature records', 'all raw history columns',
                                   'all five aggregate feature sets', 'strict latest-only A1 where registered',
                                   'all 59 decision times including empty sets', 'family split separation'],
                           limitation='Graph construction reused from frozen implementation; replay engineering audit and relation perturbation audit are separate evidence. Model budget checked only when actual_models is present.'))
    print('PASSED', expected, 'replays;', sum(r['rows'] for r in results), 'rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected', type=int, required=True)
    parser.add_argument('--models', action='store_true')
    parser.add_argument('--splits', nargs='+', choices=('fit', 'selection', 'holdout'))
    args = parser.parse_args()
    main(args.study, args.output, args.expected, args.models, args.splits)
