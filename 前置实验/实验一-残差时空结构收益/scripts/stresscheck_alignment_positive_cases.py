"""Exploratory grid review of positive TRAINING cases after the fixed initial check.

This is a numerical sensitivity analysis, not a new efficacy test or data split.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('alignment_campaign_for_grid', HERE/'calibrate_pulse_service_alignment.py')
campaign = importlib.util.module_from_spec(spec); spec.loader.exec_module(campaign)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve(); parent = Path(campaign.read(run/'config.json')['parent'])
    results = [campaign.read(p) for p in sorted((run/'families').glob('local_v1_train_???.json'))]
    selected = [r['family'] for r in results if any(a['critical_paired_response_and_oracle_times_s']
                for a in r['actions'] if a['delay_s'] == 0)]
    entries = campaign.read(parent/'development_family_manifest.json')
    unique = {}
    for row in campaign.task_table(campaign.read(parent/'geometry.json')): unique.setdefault(row['layout'], row)
    layouts = list(unique.values()); comparisons = []
    for family in selected:
        entry = next(e for e in entries if e['family_id'] == family)
        campaign.verify_family(parent, entry)
        arrays, meta = campaign.load_family(parent/'families', entry)
        event = dict(time_s=np.arange(0, 301, 2), release_s=arrays['release_s'],
            forecast_valid_s=arrays['forecast_valid_s'], forecast_depth_m=arrays['M0_forecast_depth_m'])
        result = next(r for r in results if r['family'] == family)
        pairs = sorted({(r['anchor_s'], r['center_y_m']) for r in result['actions']
                        if r['critical_paired_response_and_oracle_times_s']})
        bases = []; nominal_bases = []
        for mesh in (10, 20):
            tt, base, _ = campaign.trajectory(run, meta, mesh, 'truth_base')
            _, nb, _ = campaign.trajectory(run, meta, mesh, 'nominal_base')
            bases.append(base); nominal_bases.append(nb)
        for anchor, center in pairs:
            cases, keep, _ = campaign.anchor_selection(event, layouts, anchor)
            chosen, _ = campaign.nominal_actions(event, cases)
            conditions = []
            for mesh, base, nb in zip((10, 20), bases, nominal_bases):
                _, pulse, _ = campaign.trajectory(run, meta, mesh, 'truth_pulse', anchor, center)
                _, nominal_pulse, _ = campaign.trajectory(run, meta, mesh, 'nominal_pulse', anchor, center)
                scores = campaign.evaluate_trajectory(event, pulse, cases)
                gap = scores['count'].max(axis=1)-scores['count'][np.arange(len(cases)), chosen]
                visible = {}; sets = {}
                for delay in (0, 10):
                    hits = []
                    for pi, phase in enumerate((0, .25, .5, .75)):
                        rec = campaign.sampled_records(event, base, pulse, phase, 0, 0,
                            [meta['index'], anchor, center, pi])
                        for j, case in enumerate(cases):
                            if keep[j] and campaign.paired_response_available(rec, case['now_s'], delay,
                                    campaign.gained_targets(case, scores, j, int(chosen[j]))):
                                hits.append([j, pi])
                    visible[str(delay)] = len(hits); sets[str(delay)] = hits
                conditions.append(dict(mesh=mesh, gap_sum=int(gap[keep].sum()),
                    critical_cases=int(keep.sum()), visible_oracle_decisions=visible,
                    visible_oracle_indices=sets, true_counts=scores['count'],
                    nominal_edges=campaign.propagation_edges(tt, nominal_pulse-nb, pulse-base, family, anchor, center)))
            coarse, fine = conditions
            changes = [abs(a['kernel_lag_s']-b['kernel_lag_s']) for a, b in zip(coarse['nominal_edges'], fine['nominal_edges'])
                       if a['kernel_lag_s'] is not None and b['kernel_lag_s'] is not None]
            comparisons.append(dict(family=family, anchor_s=anchor, center_y_m=center,
                critical_cases=int(keep.sum()), coarse_gap_sum=coarse['gap_sum'], fine_gap_sum=fine['gap_sum'],
                coarse_visible_oracle_decisions=coarse['visible_oracle_decisions'],
                fine_visible_oracle_decisions=fine['visible_oracle_decisions'],
                coarse_visible_oracle_indices=coarse['visible_oracle_indices'],
                fine_visible_oracle_indices=fine['visible_oracle_indices'],
                order_yields_changed=int(np.count_nonzero(coarse['true_counts'] != fine['true_counts'])),
                max_nominal_lag_change_s=max(changes) if changes else None,
                critical_opportunity_existence_reversed=(coarse['gap_sum'] > 0) != (fine['gap_sum'] > 0),
                critical_visible_existence_reversed=(coarse['visible_oracle_decisions']['0'] > 0) != (fine['visible_oracle_decisions']['0'] > 0)))
    out = dict(exploratory_posthoc_training_numerical_review=True,
        selection='All training families and anchor-center pairs with positive critical D0 visible-oracle bound',
        selected_families=selected, comparisons=comparisons,
        fixed_coarse_nominal_tasks_and_actions=True,
        core_existence_reversals=sum(r['critical_opportunity_existence_reversed'] or r['critical_visible_existence_reversed'] for r in comparisons),
        coarse_visible_total=sum(r['coarse_visible_oracle_decisions']['0'] for r in comparisons),
        fine_visible_total_on_selected_pairs=sum(r['fine_visible_oracle_decisions']['0'] for r in comparisons),
        not_a_full_fine_grid_population_estimate=True)
    campaign.write_json(run/'positive_case_grid_review.json', out)
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
