"""Independent saved-record / scoring audit; does not select or train models."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.local_service import (task_table, schedules, build_batch, event_foundation,
    predict_candidates, interval_max, OFFSETS)
from hyra_validation.local_service_data import load_family, family_events
from hyra_validation.decision_value import order_scores
from hyra_validation.lead import MONITORS


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array(zipped, prefix, name):
    with zipped.open(prefix+'/'+name+'.npy') as f: return np.load(f, allow_pickle=False)


def run_audit(run, confirmation=False):
    config = read(run/'config.json'); cases = task_table(read(run/'geometry.json'))
    models = read(run/'models.json'); selected = read(run/'selection.json')['selected']
    frozen = read(run/'freeze_manifest.json')
    for relative, checksum in frozen['source_hashes'].items(): assert digest(ROOT/relative) == checksum, relative
    manifests = read(run/'development_family_manifest.json')
    if confirmation:
        assert read(run/'status.json')['stage'] == 'complete'
        manifests += read(run/'confirmation_family_manifest.json')
    ids = [m['family_id'] for m in manifests]
    assert len(ids) == len(set(ids))
    summaries = {}
    for path in run.glob('evaluation_*/*.json'):
        if path.name.endswith('.audit.json'): continue
        result = read(path)
        if result.get('family_id') in ids: summaries[result['family_id']] = result
    audit_totals = dict(families=0, saved_batches=0, actual_input_references=0,
                       scalar_order_checks=0, paired_observation_difference_checks=0)
    derived = []; family_truth_ceiling = []
    for info in manifests:
        if info['split'] == 'train':
            for name, checksum in info['sha256'].items(): assert digest(run/'families'/name) == checksum
            continue
        result = summaries[info['family_id']]
        assert digest(result['raw_results']) == result['raw_results_sha256']
        arrays, meta = load_family(run/'families', info)
        names = result['candidate_names']; group_ids = {g: names.index(v['candidate']) for g, v in selected.items()}
        truth_event_by_key = {(e, m): event for e, m, event in family_events(arrays, meta)}
        observations = read(Path(result['raw_results']).parent/f"{meta['family_id']}.audit.json")
        lookup_audit = {(r['condition'], r['mode'], i % 32): r for i, r in enumerate(observations)}
        weighted_cells = defaultdict(list); ceiling_cells = defaultdict(list)
        ff_cells = defaultdict(list); fi_cells = defaultdict(list); margin_cells = defaultdict(list)
        before = {}; normal = list(schedules(config))
        with zipfile.ZipFile(result['raw_results']) as z:
            for condition in ('E0', 'E1'):
                for mode in ('M0', 'M1', 'M2', 'M3'):
                    event = truth_event_by_key[condition, mode]
                    foundation = event_foundation(event)
                    row_times = foundation['row_times']; row_regions = foundation['row_regions']
                    row_lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(row_times, row_regions))}
                    for si, setting in enumerate(normal):
                        prefix = f'{condition}_{mode}_s{si:02}'
                        actual = array(z, prefix, 'actual'); choices = array(z, prefix, 'chosen')
                        true_yields = array(z, prefix, 'true_yields'); oracle = array(z, prefix, 'oracle')
                        truth_maxima = array(z, prefix, 'truth_maxima')
                        valid = array(z, prefix, 'observations_valid_s'); arrivals = array(z, prefix, 'observations_arrival_s')
                        values = array(z, prefix, 'observations_values_m'); prior = array(z, prefix, 'observations_prior_m')
                        issued = array(z, prefix, 'observations_issue_index'); lead = array(z, prefix, 'observations_actual_lead_s')
                        used = array(z, prefix, 'actual_input_record_indices')
                        expected_issue = np.searchsorted(event['release_s'], valid-20, side='right')-1
                        np.testing.assert_array_equal(issued, expected_issue)
                        np.testing.assert_allclose(lead, valid-event['release_s'][issued], atol=1e-12)
                        assert np.all((lead >= 20) & (lead < 30))
                        rr, bb = np.nonzero(used >= 0); references = used[rr, bb]
                        ages = row_times[rr]-valid[references]
                        assert np.all(arrivals[references] <= row_times[rr])
                        assert np.all((ages >= 0) & (ages < 60))
                        np.testing.assert_array_equal(ages.astype(int)//10, bb)
                        hash_value = hashlib.sha256()
                        for value in (valid, arrivals, values, prior, issued): hash_value.update(np.ascontiguousarray(value).tobytes())
                        assert hash_value.hexdigest() == lookup_audit[condition, mode, si]['observation_sha256']
                        if condition == 'E0': before[mode, si] = values
                        else:
                            expected_delta = np.column_stack([np.interp(valid, arrays['time_s'], arrays['E1_truth_depth_m'][:, r]-arrays['E0_truth_depth_m'][:, r]) for r in MONITORS])
                            np.testing.assert_allclose(values-before[mode, si], expected_delta, atol=1e-12)
                            audit_totals['paired_observation_difference_checks'] += values.size
                        expected_y = np.take_along_axis(np.broadcast_to(true_yields, (*choices.shape, 2)), choices[..., None], axis=-1)[..., 0]
                        np.testing.assert_array_equal(actual, expected_y)
                        np.testing.assert_array_equal(oracle, true_yields.max(axis=-1))
                        cell = (condition, mode, setting['noise_std_m'], setting['delay_s'])
                        weighted_cells[cell].append(actual.mean(axis=1))
                        ceiling_cells[cell].append(float(oracle.mean()))
                        # Full-window classifications at each method's predicted starts.
                        pred_max = array(z, prefix, 'predicted_maxima'); pred_starts = array(z, prefix, 'predicted_starts')
                        true_at_predicted = np.empty_like(pred_max)
                        for oi in range(2):
                            for node in range(2):
                                row_indices = [row_lookup[c['now_s'], c['orders'][oi][0][node]] for c in cases]
                                curve = np.broadcast_to(foundation['truth_future'][row_indices], (len(names), len(cases), 26))
                                true_at_predicted[:, :, oi, node] = interval_max(curve, pred_starts[:, :, oi, node], 12)
                        chosen_idx = np.broadcast_to(choices[:, :, None, None], (len(names), len(cases), 1, 2))
                        pchosen = np.take_along_axis(pred_max, chosen_idx, axis=2)[:, :, 0]
                        tchosen = np.take_along_axis(true_at_predicted, chosen_idx, axis=2)[:, :, 0]
                        ff = ((pchosen <= .8) & (tchosen > .8)).mean(axis=(1, 2))
                        fi = ((pchosen > .8) & (tchosen <= .8)).mean(axis=(1, 2))
                        ff_cells[cell].append(ff); fi_cells[cell].append(fi)
                        gap = actual[group_ids['A4']]-actual[group_ids['A2']]
                        absolute_margin = np.min(abs(.8-truth_maxima), axis=(1, 2))
                        margin_group = np.digitize(absolute_margin, [.005, .02], right=True)
                        margin_cells[cell].append([float(np.mean(gap*(margin_group == b))) for b in range(3)])
                        if si in (0, 31):
                            batch = build_batch(event, setting, meta['seed_identity'], config['master_seed'], foundation)
                            rebuilt_names, errors, _ = predict_candidates(batch, models, config, names)
                            assert rebuilt_names == names
                            for g in ('A0', 'A1', 'A2', 'A4'):
                                mi = group_ids[g]
                                for ci in (0, 31, 64, 127):
                                    case = cases[ci]; depth = np.zeros((26, 100))
                                    for target in case['targets']:
                                        row = row_lookup[case['now_s'], target]
                                        depth[:, target] = np.maximum(0, batch['base'][row]+errors[mi, row])
                                    ss = order_scores(case['now_s']+OFFSETS, depth, case['orders'], case['now_s'])
                                    predicted_counts = [s['service_yield'] for s in ss]
                                    saved_counts = array(z, prefix, 'predicted_counts')[mi, ci]
                                    np.testing.assert_array_equal(predicted_counts, saved_counts)
                                    audit_totals['scalar_order_checks'] += 2
                        audit_totals['saved_batches'] += 1
                        audit_totals['actual_input_references'] += len(references)
        means = np.mean([np.mean(v, axis=0) for v in weighted_cells.values()], axis=0)
        np.testing.assert_allclose(means, result['aggregate']['yield'], atol=1e-12)
        ff_mean = np.mean([np.mean(v, axis=0) for v in ff_cells.values()], axis=0)
        fi_mean = np.mean([np.mean(v, axis=0) for v in fi_cells.values()], axis=0)
        contributions = np.mean([np.mean(v, axis=0) for v in margin_cells.values()], axis=0)
        np.testing.assert_allclose(sum(contributions), means[group_ids['A4']]-means[group_ids['A2']], atol=1e-12)
        ceiling = float(np.mean([np.mean(v) for v in ceiling_cells.values()]))
        family_truth_ceiling.append(dict(family=meta['family_id'], split=meta['split'], finite_oracle_mean=ceiling,
                                         remaining_gap_over_A2=ceiling-means[group_ids['A2']]))
        derived.append(dict(family=meta['family_id'], split=meta['split'],
            full_window_false_feasible={g: float(ff_mean[i]) for g, i in group_ids.items()},
            full_window_false_infeasible={g: float(fi_mean[i]) for g, i in group_ids.items()},
            A4_minus_A2_gain_contribution_by_truth_margin_bins=contributions))
        audit_totals['families'] += 1
        print('AUDITED '+meta['family_id'], flush=True)
    result = dict(passed=True, include_confirmation=confirmation, **audit_totals,
                  frozen_source_hashes_verified=True, family_split_isolation=True,
                  weighted_reduction_independently_reproduced=True,
                  false_window_label_denominator='two planned service intervals at predicted starts; separate from actual execution failure',
                  margin_bins_m=[.005, .02], derived=derived, finite_oracle=family_truth_ceiling,
                  auditor_source_sha256=digest(__file__))
    write_path = run/('final_independent_audit.json' if confirmation else 'development_independent_audit.json')
    write_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=lambda x: x.tolist()), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('derived', 'finite_oracle')}, ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path); parser.add_argument('--include-confirmation', action='store_true')
    args = parser.parse_args(); run_audit(args.run.resolve(), args.include_confirmation)
