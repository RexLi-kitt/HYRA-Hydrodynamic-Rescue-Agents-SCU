"""Post-selection auxiliary diagnostics; never train, select, or plan actions.

The production entry refuses holdout access until all252 policy trials have
finished and their full replay audit has passed. D/P selected checkpoints may
differ; all have the same1024-episode maximum training and selection budget.
"""
import argparse
import hashlib
import json
from pathlib import Path
from scale100_train import torch, np, F, FloodCase, ScaleEnv, config, digest, write
from scale100_formal import validate_plan, check_selected
from scale100_backend import make_actor
from scale100_run_audit import require
from scale100_evaluate_sealed import seal_models
from scale100_resilient_forecast import score
from scale100_rules import water_aware

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / 'generated/scale100-formal-20261026'
EVALUATION = ROOT / 'generated/scale100-heldout-20261026'
PROTOCOL = ROOT / 'research/scale100-formal-forecast-protocol-20261026.json'
BASELINES = ('legal_memory_persistence', 'fit_mean_constant', 'truth_current_persistence_oracle')
SAMPLE_INDICES = list(range(0, 181, 10))
SOURCE_FILES = ('scale100_formal_forecasts.py', 'scale100_resilient_forecast.py',
                'scale100_rules.py', 'scale100_development.py')
ATTRIBUTION = {'task', 'parent', 'scenario_seed', 'training_seed', 'method',
               'time_s', 'actual_microstep_s'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare():
    plan = read(FORMAL / 'plan.json')
    value = dict(schema='hyra-scale100-formal-forecast-protocol-v1',
        formal_plan_sha256=digest(FORMAL / 'plan.json'),
        source_sha256={name: digest(ROOT / name) for name in SOURCE_FILES},
        tasks=['E1', 'E2'], methods=['D', 'P'], training_seeds=plan['training_seeds'],
        scenario_seeds=plan['evaluation_scenario_seeds'], decision_indices=SAMPLE_INDICES,
        baselines=list(BASELINES), reference_policy='water_aware',
        max_training_episodes=1024, models='all sealed task-selected D/P models; no forecast selection',
        aggregation='equal parent means after within-parent time/scenario/training-seed averaging',
        scoring_horizon='actual accepted internal solver microstep; no long-horizon forecast',
        guard='all36 selected and all252 policy trials fully reaudited before holdout diagnostics',
        new_training_or_policy_selection=False)
    if PROTOCOL.exists():
        require(read(PROTOCOL) == value, 'Forecast diagnostic protocol changed')
    else:
        write(PROTOCOL, value)
    return value


def fit_prior(plan):
    cases = [FloodCase(row['path']) for row in plan['inputs'] if row['split'] == 'fit']
    return np.mean([case.probes['before'].mean(axis=(0, 2, 3)) for case in cases], axis=0)


def reference(case, task, scenario_seed, actors, prior):
    """actors: {(D/P,training_seed): actor}; same legal inputs for every actor."""
    env = ScaleEnv(case, config(task, 'D', scenario_seed))
    view = env.reset()
    hidden = {key: actor.hidden(2) for key, actor in actors.items()}
    previous = torch.zeros(2, 8)
    points = []
    stream = hashlib.sha256()
    bed = torch.tensor(case.bed, dtype=torch.float32)[None].expand(2, -1, -1)
    for step in range(181):
        stream.update(view.images.tobytes())
        stream.update(view.available.tobytes())
        with torch.no_grad():
            for key, actor in actors.items():
                actor.eval()
                _, hidden[key], _ = actor(torch.from_numpy(view.images.copy()), previous, step, hidden[key], False)
            if step in SAMPLE_INDICES:
                before, after = case.probes['before'][step], case.probes['after'][step]
                dt, source = float(case.probes['dt'][step]), case.probes['source'][step]
                require(dt > 0 and np.isfinite(dt), 'Missing actual microstep label')
                common = dict(task=task, parent=case.case['parent_terrain'], scenario_seed=scenario_seed,
                              time_s=step * 5., actual_microstep_s=dt)
                for (method, seed), actor in actors.items():
                    prediction = actor.predict(hidden[(method, seed)], bed)
                    points.append(dict(**common, method=method, training_seed=seed,
                        **score(prediction, before, after, case.bed, dt, source, view.explored)))
                h, u, v = env.water
                memory = np.stack((h, h*u, h*v))
                constant = np.broadcast_to(prior[:, None, None], (3, 20, 20)).copy()
                for name, state in ((BASELINES[0], memory), (BASELINES[1], constant), (BASELINES[2], before)):
                    prediction = torch.tensor(np.stack((state, state)), dtype=torch.float32)[None].expand(2, -1, -1, -1, -1)
                    points.append(dict(**common, method=name, training_seed=None,
                        **score(prediction, before, after, case.bed, dt, source, view.explored)))
        if step < 180:
            actions = water_aware(env)
            stream.update(np.asarray(actions, dtype='<i8').tobytes())
            view, _, _ = env.step(actions)
            previous = F.one_hot(torch.as_tensor(actions), 8).float()
    require(all(p.grad is None for actor in actors.values() for p in actor.parameters()),
            'Auxiliary diagnostic unexpectedly created training gradients')
    return points, stream.hexdigest()


def mean_values(rows, keys):
    return {key: float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in keys}


def aggregate(points):
    keys = [key for key in points[0] if key not in ATTRIBUTION]
    parents = sorted({row['parent'] for row in points})
    per_parent, summary, contrasts = [], [], []
    for task in sorted({row['task'] for row in points}):
        for method in ('D', 'P', *BASELINES):
            rows = [row for row in points if row['task'] == task and row['method'] == method]
            if not rows:
                continue
            values = []
            for parent in parents:
                selected = [row for row in rows if row['parent'] == parent]
                if not selected:
                    continue
                means = mean_values(selected, keys)
                per_parent.append(dict(task=task, method=method, parent=parent, samples=len(selected),
                    available_samples={key: sum(row[key] is not None for row in selected) for key in keys}, **means))
                values.append(means)
            summary.append(dict(task=task, method=method, parents=len(values), **mean_values(values, keys)))
        index = {(row['parent'], row['training_seed'], row['scenario_seed'], row['time_s'], row['method']): row
                 for row in points if row['task'] == task and row['method'] in ('D', 'P')}
        differences = []
        for parent, training_seed, scenario_seed, time_s in sorted({key[:4] for key in index}):
            require(all((parent, training_seed, scenario_seed, time_s, method) in index for method in ('D', 'P')),
                    'Matched D/P diagnostic point is missing')
            data, physics = (index[(parent, training_seed, scenario_seed, time_s, method)] for method in ('D', 'P'))
            differences.append(dict(parent=parent, **{key: physics[key] - data[key]
                if physics[key] is not None and data[key] is not None else None for key in keys}))
        values = [dict(parent=parent, **mean_values([r for r in differences if r['parent'] == parent], keys))
                  for parent in parents if any(r['parent'] == parent for r in differences)]
        contrasts.append(dict(task=task, name='P_minus_D_auxiliary',
                              parent_differences=values, means=mean_values(values, keys)))
    return dict(parent_tables=per_parent, summary=summary, contrasts=contrasts)


def execute(output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not output.exists(), 'Refusing to overwrite auxiliary results')
    require(read(FORMAL / 'queue.json')['status'] == 'completed', 'All36 training jobs must finish')
    require(read(EVALUATION / 'status.json')['status'] == 'completed', 'All252 policy evaluations must finish')
    proof = read(EVALUATION / 'release-audit.json')
    require(proof['complete'] and proof['all_replays_reaudited'] and proof['trials'] == 252
            and proof['evaluation_results_sha256'] == digest(EVALUATION / 'results.json')
            and proof['release_auditor_sha256'] == digest(ROOT / 'scale100_release_audit.py'),
            'The full policy replay audit must pass before auxiliary holdout scoring')
    protocol = prepare()
    plan = read(FORMAL / 'plan.json')
    validate_plan(plan)
    seal = seal_models(FORMAL, plan, EVALUATION)
    prior = fit_prior(plan)
    points, reference_streams, model_rows = [], [], []
    for task in ('E1', 'E2'):
        actors = {}
        for model in seal['models']:
            if model['task'] != task or model['method'] not in ('D', 'P'):
                continue
            folder = FORMAL / model['id']
            meta = check_selected(folder, model, plan, digest(FORMAL / 'plan.json'))
            require(np.array_equal(prior, np.asarray(meta['fit_only_prior_U'])), 'Fit-only constant baseline changed')
            actor = make_actor(plan['training_backend'])
            actor.load_state_dict(torch.load(folder / 'selected/actor.pt', weights_only=True, map_location='cpu'))
            actor.eval()
            actors[(model['method'], model['seed'])] = actor
            model_rows.append(dict(id=model['id'], selected_episode=meta['selected_episode'],
                                   actor_sha256=meta['actor_sha256'], total_episodes=1024))
        require(set(actors) == {(m, s) for m in ('D', 'P') for s in (11, 29, 47)}, 'Incomplete D/P matrix')
        for row in plan['inputs']:
            if row['split'] != 'holdout':
                continue
            case = FloodCase(row['path'])
            for scenario_seed in plan['evaluation_scenario_seeds']:
                values, stream = reference(case, task, scenario_seed, actors, prior)
                points.extend(values)
                reference_streams.append(dict(task=task, parent=row['parent_terrain'],
                    scenario_seed=scenario_seed, legal_input_action_stream_sha256=stream))
    require(len(points) == 2052 and len(reference_streams) == 12 and len(model_rows) == 12,
            'Auxiliary diagnostic matrix incomplete')
    result = dict(schema='hyra-scale100-formal-auxiliary-diagnostic-v1', complete=True,
        formal_plan_sha256=digest(FORMAL / 'plan.json'), model_seal_sha256=digest(EVALUATION / 'sealed-models.json'),
        policy_results_sha256=digest(EVALUATION / 'results.json'), release_audit_sha256=digest(EVALUATION / 'release-audit.json'),
        protocol_sha256=digest(PROTOCOL), protocol=protocol, fit_only_prior_U=prior.tolist(),
        models=model_rows, reference_streams=reference_streams, rows=points,
        min_horizon_s=min(r['actual_microstep_s'] for r in points),
        max_horizon_s=max(r['actual_microstep_s'] for r in points), **aggregate(points),
        limitations=['Only two synthetic holdout parents; cells/times/agents/seeds are correlated, no p-values or confidence claims',
            'The selected D/P episodes can differ under the frozen task selection rule; no auxiliary-accuracy selection',
            'A solver microstep reconstruction diagnostic, not a long-horizon world model or planning method',
            'Legal memory persistence keeps raw pixel records, more detail than pooled actor tensors',
            'Truth-current persistence is a privileged offline temporal yardstick, never an executable method',
            'The terminal900s microstep label is offline-only and never enters a policy',
            'Lower residual or field error alone is not evidence of better exploration or rescue'])
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(json.dumps(dict(complete=True, rows=len(points), parents=2, horizons_s=[result['min_horizon_s'], result['max_horizon_s']])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print(str(PROTOCOL))
    else:
        require(args.output is not None, 'Provide --output')
        execute(args.output)
