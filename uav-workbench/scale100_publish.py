"""Register audited sealed weights without retraining, ranking, or copying them."""
import argparse
import json
import uuid
from pathlib import Path
import console_scale100_contract as contract
from scale100_formal import torch, validate_plan, check_selected, digest
from scale100_evaluate_sealed import seal_models
from scale100_run_audit import require

ROOT = Path(__file__).resolve().parent
FORMAL = contract.FORMAL
EVALUATION = contract.HELDOUT


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validated_bundle():
    require(read(FORMAL / 'queue.json')['status'] == 'completed', 'All36 training jobs must complete')
    require(read(EVALUATION / 'status.json')['status'] == 'completed', 'All252 policy trials must complete')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    plan = read(FORMAL / 'plan.json')
    validate_plan(plan)
    seal = seal_models(FORMAL, plan, EVALUATION)
    proof = read(EVALUATION / 'release-audit.json')
    results = read(EVALUATION / 'results.json')
    require(proof['complete'] is True and proof['all_replays_reaudited'] is True
            and proof['trials'] == 252 and proof['learned_trials'] == 216 and proof['rule_trials'] == 36,
            'Full release audit has not passed')
    require(proof['formal_plan_sha256'] == digest(FORMAL / 'plan.json')
            and proof['model_seal_sha256'] == digest(EVALUATION / 'sealed-models.json')
            and proof['evaluation_results_sha256'] == digest(EVALUATION / 'results.json')
            and proof['release_auditor_sha256'] == digest(ROOT / 'scale100_release_audit.py'),
            'Release audit provenance changed')
    require(results['complete'] is True and results['all_audits_passed'] is True
            and results['sealed_models_sha256'] == digest(EVALUATION / 'sealed-models.json')
            and len(results['rows']) == 252, 'Policy evaluation is incomplete or belongs to another seal')
    checks = proof['replay_checks']
    expected_ids = {r['trial_id'] for r in results['rows']}
    require(len(expected_ids) == len(checks) == 252
            and {r['trial_id'] for r in checks} == expected_ids, 'Replay proof matrix is incomplete/duplicated')
    by_trial = {r['trial_id']: r for r in checks}
    for row in results['rows']:
        saved = by_trial[row['trial_id']]
        require(saved['audit_passed'] and saved['policy_actions_reproduced'] == 180
                and saved['snapshots_reproduced'] == 181 and saved['records_reproduced'] == 362
                and saved['actor_features_checked'] and saved['return_reproduced']
                and saved['replay_sha256'] == row['replay_sha256'], 'Trial audit proof is incomplete')
        require(row['replay_path'] == row['trial_id'] + '.json.gz'
                and digest(EVALUATION / row['replay_path']) == row['replay_sha256']
                and read(EVALUATION / (row['trial_id'] + '.json')) == row,
                'Audited trial artifact changed')
    audits = proof['training']
    require(len(audits) == 36 and {r['id'] for r in audits} == {j['id'] for j in plan['jobs']},
            'Training audit matrix incomplete/duplicated')
    by_job = {r['id']: r for r in audits}
    models = []
    for job in plan['jobs']:
        folder = FORMAL / job['id']
        meta = check_selected(folder, job, plan, digest(FORMAL / 'plan.json'))
        audited = by_job[job['id']]
        require(audited['episodes'] == 1024 and audited['updates'] == 1021
                and audited['selected_episode'] == meta['selected_episode']
                and audited['actor_sha256'] == meta['actor_sha256']
                and audited['metadata_sha256'] == digest(folder / 'selected/actor.json')
                and audited['history_sha256'] == digest(folder / 'history.json')
                and audited['initial_actor_sha256'] == digest(folder / 'initial-actor.pt'),
                'Audited training or selected model changed')
        models.append(dict(**job, actor_sha256=meta['actor_sha256'],
                           metadata_sha256=digest(folder / 'selected/actor.json'),
                           selected_episode=meta['selected_episode']))
    return plan, seal, proof, results, models


def publish():
    plan, _, proof, _, models = validated_bundle()
    value = dict(schema='hyra-scale100-workbench-release-v1', complete=True,
        formal_plan_sha256=digest(FORMAL / 'plan.json'), model_seal_sha256=digest(EVALUATION / 'sealed-models.json'),
        evaluation_results_sha256=digest(EVALUATION / 'results.json'),
        release_audit_sha256=digest(EVALUATION / 'release-audit.json'),
        publisher_source_sha256=digest(__file__), training_backend=plan['training_backend'],
        grid=dict(shape=[20, 20], dx_m=5.), duration_s=900, n_agents=2,
        actions=8, observation_channels=21, default_training_seed=11,
        training_seeds=[11, 29, 47], models=models,
        scientific_scope='Two synthetic holdout parents; audited execution is not evidence of superiority, convergence, original-paper reproduction, or large-map transfer',
        auxiliary_gradient_flags={r['id']: r['shared_auxiliary_live_at_last_audited_update']
                                  for r in proof['training'] if '-D-' in r['id'] or '-P-' in r['id']})
    if contract.RELEASE.exists():
        require(read(contract.RELEASE) == value, 'A different release already exists; preserve it for review')
        contract.release_record()
        return value
    candidate = contract.RELEASE.with_name(contract.RELEASE.stem + '.candidate-' + uuid.uuid4().hex + '.json')
    candidate.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    actual = contract.RELEASE
    try:
        # Exercise the production lookup for every task/group/seed BEFORE install.
        contract.RELEASE = candidate
        for model in models:
            task = 'exploration' if model['task'] == 'E1' else 'exploration_rescue'
            contract.model_record(task, model['method'], model['seed'])
        contract.RELEASE = actual
        require(not actual.exists(), 'Release appeared concurrently; preserve both candidates')
        # On the Windows host rename refuses to overwrite an existing target.
        candidate.rename(actual)
    finally:
        contract.RELEASE = actual
    contract.release_record()
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        validated_bundle()
        print('All publication evidence verified; no registry written')
    else:
        result = publish()
        print(json.dumps(dict(published=True, models=len(result['models']),
                              default_training_seed=result['default_training_seed'])))
