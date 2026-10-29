"""Reexecute recorded actions and verify chosen policies before publication.

This does not select or train models. The production entry point refuses to
read holdout replays until all36 training jobs and all252 evaluations finish.
"""
import argparse
import gzip
import json
import math
import time
from pathlib import Path
from scale100_train import torch, np, config, ScaleEnv, FloodCase, digest, write
from scale100_backend import make_actor
from scale100_formal import validate_plan, check_selected, selection_key
from scale100_evaluate_sealed import seal_models, aggregate, replay_bytes, RULES, rule_sources
from scale100_run_audit import audit, require
from scale100_development import legal_rule
from scale100_rules import water_aware
from sur_rescue.environment import ACTIONS
from sur_rescue.learning import masked_q


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def normalized(value):
    return json.loads(replay_bytes(value))


def verify_replay(payload, row, case, actor=None):
    require(case.digest == row['case_sha256'], 'Hydraulic case changed')
    for key, expected in dict(schema='hyra-scale100-evaluation-replay-v1',
            split=row['split'], task=row['task'], method=row['method'],
            scenario_seed=row['scenario_seed'], training_seed=row['training_seed'],
            actor_sha256=row['actor_sha256'], source_sha256=case.digest,
            grid=case.process['grid']).items():
        require(payload.get(key) == expected, 'Replay attribution differs: ' + key)
    method = row['method'] if actor is not None else 'B3'
    env = ScaleEnv(case, config(row['task'], method, row['scenario_seed']))
    require(payload['config'] == normalized(env.config.__dict__), 'Replay environment changed')
    require(len(payload['actions']) == 180, 'Replay needs all180 joint decisions')
    view = env.reset()
    images = [view.images.copy()]
    total = 0.
    if actor is not None:
        actor.eval()
        hidden = actor.hidden(2)
        previous = torch.zeros(2, 8)
    for step, entry in enumerate(payload['actions']):
        require(entry['time_s'] == step * 5 and len(entry['actions']) == 2,
                'Action timestamp/fleet size changed')
        require(all(name in ACTIONS for name in entry['actions']), 'Unknown recorded action')
        actions = np.array([ACTIONS.index(name) for name in entry['actions']])
        if actor is not None:
            with torch.no_grad():
                q, hidden, _ = actor(torch.from_numpy(view.images.copy()), previous, step, hidden)
            expected = masked_q(q, torch.from_numpy(view.available.copy())).argmax(-1).numpy()
            previous = torch.nn.functional.one_hot(torch.as_tensor(actions), 8).float()
        else:
            require(row['method'] in RULES, 'Unknown evaluation rule')
            expected = water_aware(env) if row['method'] == 'water_aware' else legal_rule(env, row['method'])
        require(np.array_equal(actions, expected), 'Recorded action does not match its policy')
        view, reward, done = env.step(actions)
        total += reward
        images.append(view.images.copy())
        require(done == (step == 179), 'Unexpected episode endpoint')
    require(math.isclose(total, row['return_'], abs_tol=1e-9, rel_tol=0), 'Return cannot be reproduced')
    reproduced = dict(snapshots=env.snapshots, observations=env.observations,
                      trajectory=env.trajectory, actions=env.actions, events=env.events,
                      scenario_truth=dict(obstacles=env._blocked.astype(int).ravel().tolist(),
                          targets=[list(p) for p in env._targets],
                          scope='offline evaluator/replay only; not actor input'))
    for key, expected in normalized(reproduced).items():
        require(payload[key] == expected, 'Recorded replay cannot be reproduced: ' + key)
    metrics = audit(env, np.asarray(images))
    require(metrics == row['metrics'], 'Independent full replay metrics differ')
    return dict(audit_passed=True, policy_actions_reproduced=180,
                snapshots_reproduced=181, records_reproduced=362,
                actor_features_checked=True, return_reproduced=True)


def training_integrity(formal, plan):
    rows = []
    initial_reference = {}
    for job in plan['jobs']:
        folder = formal / job['id']
        meta = check_selected(folder, job, plan, digest(formal / 'plan.json'))
        history = read(folder / 'history.json')
        require(len(history) == plan['episodes_per_job']
                and [r['episode'] for r in history] == list(range(1, plan['episodes_per_job'] + 1)),
                'Formal training history is incomplete')
        updates = [r['update'] for r in history if r['update'] is not None]
        require(len(updates) == plan['updates_per_job']
                and [r['updates'] for r in updates] == list(range(1, plan['updates_per_job'] + 1)),
                'Formal optimizer budgets differ')
        require(all(math.isfinite(r['return_']) for r in history), 'Nonfinite formal return')
        require(all(math.isfinite(float(v)) for u in updates for k, v in u.items()
                    if k != 'shared_gradient_audit'), 'Nonfinite formal optimization value')
        initial = torch.load(folder / 'initial-actor.pt', weights_only=True, map_location='cpu')
        key = (job['task'], job['seed'])
        if key in initial_reference:
            reference = initial_reference[key]
            require(initial.keys() == reference.keys()
                    and all(torch.equal(initial[k], reference[k]) for k in initial),
                    'Methods do not share the same initial actor tensors')
        else:
            initial_reference[key] = initial
        selected = torch.load(folder / 'selected/actor.pt', weights_only=True, map_location='cpu')
        for prefix in ('conv.', 'gru.'):
            require(any(not torch.equal(initial[k], selected[k]) for k in initial if k.startswith(prefix)),
                    'Selected actor did not learn in ' + prefix)
        reports = [read(folder / f'development-ep{ep:04d}.json') for ep in plan['development_checkpoints']]
        chosen = max(reports, key=lambda r: selection_key(r, job['task']))
        require(chosen['episode'] == meta['selected_episode']
                and chosen['actor_sha256'] == meta['actor_sha256'], 'Frozen selection rule differs')
        gradients = [u['shared_gradient_audit'] for u in updates if u.get('shared_gradient_audit')]
        for gradient in gradients:
            require(all(math.isfinite(v) and v >= 0 for v in gradient.values()), 'Nonfinite shared gradient')
        final_gradient = gradients[-1] if gradients else None
        # Preserve failures as scientific findings, never hide them by resampling.
        shared_live = None
        if job['method'] in ('D', 'P'):
            shared_live = bool(final_gradient and final_gradient['data_shared_norm'] > 0
                              and (job['method'] != 'P' or final_gradient['physics_shared_norm'] > 0))
        rows.append(dict(id=job['id'], episodes=len(history), updates=len(updates),
                         selected_episode=meta['selected_episode'], actor_sha256=meta['actor_sha256'],
                         metadata_sha256=digest(folder / 'selected/actor.json'),
                         history_sha256=digest(folder / 'history.json'),
                         initial_actor_sha256=digest(folder / 'initial-actor.pt'),
                         shared_auxiliary_live_at_last_audited_update=shared_live,
                         last_shared_gradient=final_gradient))
    return rows


def execute(formal, evaluation):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(read(formal / 'queue.json')['status'] == 'completed', 'All36 training jobs must complete')
    require(read(evaluation / 'status.json')['status'] == 'completed', 'All252 evaluations must complete')
    plan = read(formal / 'plan.json')
    validate_plan(plan)
    seal = seal_models(formal, plan, evaluation)
    results = read(evaluation / 'results.json')
    require(results['complete'] is True and results['learned_trials'] == 216
            and results['rule_trials'] == 36 and results['all_audits_passed'] is True,
            'Evaluation report is incomplete')
    require(results['sealed_models_sha256'] == digest(evaluation / 'sealed-models.json')
            and results['formal_plan_sha256'] == digest(formal / 'plan.json')
            and results['evaluation_source_sha256'] == seal['evaluation_source_sha256'],
            'Evaluation is not linked to the original seal')
    training = training_integrity(formal, plan)
    models = {m['id']: m for m in seal['models']}
    cases = {r['parent_terrain']: FloodCase(r['path']) for r in plan['inputs'] if r['split'] == 'holdout'}
    expected = {}
    for identity, model in models.items():
        for parent in cases:
            for seed in plan['evaluation_scenario_seeds']:
                expected[f'{identity}__{parent}__s{seed}'] = (parent, model['task'], model['method'], seed,
                                                            model['seed'], model['actor_sha256'])
    for task in ('E1', 'E2'):
        for rule in RULES:
            for parent in cases:
                for seed in plan['evaluation_scenario_seeds']:
                    expected[f'{task}-{rule}__{parent}__s{seed}'] = (parent, task, rule, seed, None, None)
    rows = results['rows']
    require(len(rows) == len(expected) == 252 and {r['trial_id'] for r in rows} == set(expected),
            'Trial matrix is incomplete/duplicated')
    progress = dict(status='running', completed=0, expected=252, pid=__import__('os').getpid())
    write(evaluation / 'release-audit-status.json', progress)
    verified = []
    actor = None
    current_identity = None
    for row in rows:
        identity = row['trial_id']
        require(tuple(row[k] for k in ('parent', 'task', 'method', 'scenario_seed', 'training_seed', 'actor_sha256'))
                == expected[identity] and row['split'] == 'holdout', 'Trial attribution differs from sealed matrix')
        require(read(evaluation / (identity + '.json')) == row, 'Individual and aggregate trial reports differ')
        require(row['replay_path'] == identity + '.json.gz', 'Trial replay path differs')
        path = evaluation / row['replay_path']
        require(digest(path) == row['replay_sha256'], 'Raw replay checksum changed')
        require(row['evaluation_source_sha256'] == seal['evaluation_source_sha256']
                and row['rule_sources'] == rule_sources()
                and row['audit_source_sha256'] == plan['audit_source_sha256'], 'Evaluator source changed')
        model_id = f"{row['task']}-{row['method']}-s{row['training_seed']}"
        if row['training_seed'] is not None:
            if model_id != current_identity:
                actor = make_actor(plan['training_backend'])
                actor.load_state_dict(torch.load(Path(models[model_id]['folder']) / 'actor.pt',
                                                 weights_only=True, map_location='cpu'))
                actor.eval()
                current_identity = model_id
        else:
            actor, current_identity = None, None
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        proof = verify_replay(payload, row, cases[row['parent']], actor)
        verified.append(dict(trial_id=identity, replay_sha256=digest(path), **proof))
        progress.update(completed=len(verified), current=identity)
        write(evaluation / 'release-audit-status.json', progress)
    recomputed = aggregate(rows)
    require(all(results[k] == value for k, value in recomputed.items()), 'Parent-weighted aggregation changed')
    output = dict(schema='hyra-scale100-release-audit-v1', complete=True, all_replays_reaudited=True,
                  trials=252, learned_trials=216, rule_trials=36,
                  evaluation_results_sha256=digest(evaluation / 'results.json'),
                  model_seal_sha256=digest(evaluation / 'sealed-models.json'),
                  formal_plan_sha256=digest(formal / 'plan.json'),
                  release_auditor_sha256=digest(__file__), training=training, replay_checks=verified,
                  completed_at=time.time(), scope='Full deterministic policy/action/measurement/service reproduction; no guarantee of convergence or positive scientific effect')
    write(evaluation / 'release-audit.json', output)
    progress.update(status='completed', current=None)
    write(evaluation / 'release-audit-status.json', progress)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal-root', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    execute(args.formal_root, args.evaluation)
