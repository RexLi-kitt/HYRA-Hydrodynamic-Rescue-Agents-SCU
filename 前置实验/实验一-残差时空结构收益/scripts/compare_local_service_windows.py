"""Compare frozen models on common nominally scheduled full service intervals."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from hyra_validation.local_service import (task_table, schedules, build_batch, event_foundation,
    predict_candidates, interval_max)
from hyra_validation.local_service_data import load_family, family_events


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('run', type=Path); args = parser.parse_args()
    run = args.run.resolve(); assert read(run/'status.json')['stage'] == 'complete'
    config = read(run/'config.json'); cases = task_table(read(run/'geometry.json'))
    models = read(run/'models.json'); selected = read(run/'selection.json')['selected']
    wanted = list(dict.fromkeys(['nominal']+[v['candidate'] for v in selected.values()]))
    evaluations = {}
    for path in run.glob('evaluation_*/*.json'):
        if not path.name.endswith('.audit.json'):
            result = read(path)
            if result.get('split') == 'confirmation': evaluations[result['family_id']] = result
    all_rows = []
    for info in read(run/'confirmation_family_manifest.json'):
        arrays, meta = load_family(run/'families', info)
        summary = evaluations[meta['family_id']]
        ff = defaultdict(list); fi = defaultdict(list); support = 0
        with zipfile.ZipFile(summary['raw_results']) as z:
            for condition, mode, event in family_events(arrays, meta):
                foundation = event_foundation(event)
                lookup = {(int(t), int(r)): i for i, (t, r) in enumerate(zip(foundation['row_times'], foundation['row_regions']))}
                for si, setting in enumerate(schedules(config)):
                    prefix = f'{condition}_{mode}_s{si:02}/'
                    with z.open(prefix+'chosen.npy') as f: choices = np.load(f, allow_pickle=False)
                    with z.open(prefix+'predicted_starts.npy') as f: starts = np.load(f, allow_pickle=False)
                    nominal_id = summary['candidate_names'].index('nominal')
                    nominal_choice = choices[nominal_id]
                    fixed_starts = starts[nominal_id, np.arange(len(cases)), nominal_choice]
                    batch = build_batch(event, setting, meta['seed_identity'], config['master_seed'], foundation)
                    names, errors, _ = predict_candidates(batch, models, config, wanted)
                    predicted = np.maximum(0, batch['base'][None]+errors)
                    pmax = []; tmax = []
                    for node in range(2):
                        indices = [lookup[c['now_s'], c['orders'][int(o)][0][node]] for c, o in zip(cases, nominal_choice)]
                        begin = np.broadcast_to(fixed_starts[:, node], (len(names), len(cases)))
                        pmax.append(interval_max(predicted[:, indices], begin, 12))
                        tmax.append(interval_max(foundation['truth_future'][indices][None], fixed_starts[None, :, node], 12)[0])
                    pp = np.stack(pmax, axis=-1); tt = np.stack(tmax, axis=-1)
                    cell = (condition, mode, setting['noise_std_m'], setting['delay_s'])
                    ff[cell].append(((pp <= .8) & (tt[None] > .8)).mean(axis=(1, 2)))
                    fi[cell].append(((pp > .8) & (tt[None] <= .8)).mean(axis=(1, 2)))
                    support += tt.size
        fmean = np.mean([np.mean(v, axis=0) for v in ff.values()], axis=0)
        imean = np.mean([np.mean(v, axis=0) for v in fi.values()], axis=0)
        all_rows.append(dict(family=meta['family_id'], interval_checks_per_model=support,
            false_feasible={g: float(fmean[names.index(v['candidate'])]) for g, v in selected.items()},
            false_infeasible={g: float(imean[names.index(v['candidate'])]) for g, v in selected.items()}))
        print('COMMON_WINDOWS '+meta['family_id'], flush=True)
    means = {g: {metric: float(np.mean([r[metric][g] for r in all_rows]))
                  for metric in ('false_feasible', 'false_infeasible')} for g in selected}
    result = dict(definition='same two complete intervals of A0 chosen order and predicted starts, shared by every model',
        interval_selection_uses_truth=False, model_selection_changed=False, confirmation_families=len(all_rows),
        denominator='two shared nominal full service intervals per case', means=means, family_values=all_rows,
        secondary_diagnostic_no_new_primary_hypothesis=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'common_window_diagnostic.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'families': len(all_rows), 'means': means}))


if __name__ == '__main__': main()
