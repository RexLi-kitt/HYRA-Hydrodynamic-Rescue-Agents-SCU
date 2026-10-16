"""Nominal-only kernel identifiability at the refined mesh; no learned predictions."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

from diagnose_numerical_stability import read, load, sha, nominal_lags, csv_write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.artifacts import write_json
from hyra_validation.hidden_target_inputs import SourceRecord, causal_history, aligned_summary, SOURCES


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve(); config = read(run/'config.json')
    registry = read(run/'field_registry.json'); design = config['design']; rows = []
    completed_families = [f for f in design['families'] if (run/(f+'.json')).exists()]
    for family in completed_families:
        base_info = registry[f'{family}_n80_dt2_nominal_base_a0_y0']
        assert sha(base_info['projected_path']) == base_info['projected_sha256']
        base = load(base_info['projected_path'])
        for anchor in design['anchors_s']:
            for center in design['physical_centers_y_m']:
                probe_info = registry[f'{family}_n80_dt2_nominal_pulse_a{anchor}_y{center}']
                assert sha(probe_info['projected_path']) == probe_info['projected_sha256']
                probe = load(probe_info['projected_path'])
                lags = nominal_lags(base['time_s'], probe['linear']-base['linear'], center)
                targets = sorted({target for source, target in lags})
                for decision_offset in (0, 6, 12):
                    now = anchor+decision_offset
                    for phase_index, phase in enumerate((0, .25, .5, .75)):
                        for delay in (0, 10):
                            records = [SourceRecord(j*6+i, source, float(t), float(t+delay),
                                       float(np.floor((t-20)/10)*10), 0., 0.)
                                       for j, t in enumerate(np.arange(20+5*phase, 301, 5))
                                       for i, source in enumerate(SOURCES)]
                            history = causal_history(records, now)
                            for target in targets:
                                links = [(source, 1., max(lag, 0)) for (source, dest), lag in lags.items()
                                         if dest == target and lag is not None]
                                horizons = [6, 12, 20, 30]
                                physical = aligned_summary(history, links, horizons)
                                recency = aligned_summary(history, links, horizons, mode='recency')
                                for j, horizon in enumerate(horizons):
                                    distance = float(np.sum(abs(physical['normalized_weights'][j]-recency['normalized_weights'][j])))
                                    rows.append(dict(family=family, anchor_s=anchor, center_y_m=center,
                                        decision_offset_s=decision_offset, target=target, phase_index=phase_index,
                                        delay_s=delay, horizon_s=horizon, available_edges=len(links),
                                        kernel_L1=distance, nondegenerate=distance >= .1))
    groups = defaultdict(list)
    for row in rows: groups[row['delay_s'], row['horizon_s']].append(row)
    summary = [dict(delay_s=d, horizon_s=h, rows=len(items),
                    fraction_L1_ge_point1=float(np.mean([r['nondegenerate'] for r in items])),
                    mean_kernel_L1=float(np.mean([r['kernel_L1'] for r in items])))
               for (d, h), items in sorted(groups.items())]
    csv_write(run/'refined_kernel_support.csv', rows)
    write_json(run/'refined_kernel_support.json', dict(completed_training_families=completed_families,
        all_families_complete=len(completed_families) == len(design['families']), summary=summary,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__),
            ROOT/'src/hyra_validation/hidden_target_inputs.py', ROOT/'scripts/diagnose_numerical_stability.py']},
        interpretation='Equal-weight existing same-bank candidate links, nominal 10%-peak lags, time availability only. Necessary input-difference diagnostic, not predictive or action efficacy. No response detectability claim.'))
    print(summary)


if __name__ == '__main__': main()
