"""Actual production-worker checks for all36 released task/group/seed choices."""
import argparse
import json
from pathlib import Path
from console_scale100_contract import ROOT, release_record, default_config, scene_records, digest
from console_scale100_runner import execute_job
from hyra_uav.experiments import save_run


def require(condition, message):
    if not condition:
        raise ValueError(message)


def execute(output):
    release = release_record()
    require(not output.exists(), 'Refusing to overwrite model acceptance evidence')
    scene = next(row for row in scene_records() if row['split'] == 'development')
    output.mkdir(parents=True)
    rows = []
    for model in release['models']:
        task = 'exploration' if model['task'] == 'E1' else 'exploration_rescue'
        job = dict(scene=scene['id'], task=task, group=model['method'], mode='model',
                   training_seed=model['seed'], config=default_config(task, model['method'], seed=11))
        result = execute_job(job)
        execution = result['execution']
        require(result['task'] == task and result['group'] == model['method']
                and execution['actual_group'] == model['method'] and execution['mode'] == 'model'
                and execution['training_seed'] == model['seed'] and execution['scenario_seed'] == 11
                and execution['actor_sha256'] == model['actor_sha256']
                and execution['selected_episode'] == model['selected_episode']
                and execution['model_formal'] is True and execution['total_training_episodes_executed'] == 1024,
                'Production model provenance differs from requested sealed model')
        require(result['audit']['audit_passed'] and len(result['snapshots']) == 181
                and len(result['observations']) == 362 and len(result['actions']) == 180,
                'Production model replay is incomplete')
        target = output / model['id']
        save_run(result, target)
        saved = json.loads((target / 'replay.json').read_text(encoding='utf-8'))
        require(saved == result, 'Saved production replay differs from audited object')
        rows.append(dict(id=model['id'], actor_sha256=execution['actor_sha256'],
                         replay_sha256=digest(target / 'replay.json'), audit=result['audit']))
        print(json.dumps(dict(id=model['id'], audit_passed=True)), flush=True)
    require(len(rows) == 36 and len({r['id'] for r in rows}) == 36, 'Incomplete model acceptance matrix')
    value = dict(schema='hyra-scale100-production-model-acceptance-v1', complete=True,
                 development_only=True, official_policy_results_changed=False,
                 release_registry_sha256=digest(ROOT / 'research/scale100-workbench-release-20261026.json'),
                 source_sha256={name:digest(ROOT / name) for name in ('scale100_model_acceptance.py',
                     'console_scale100_runner.py', 'console_scale100_contract.py')}, rows=rows,
                 browser_acceptance_pending=True)
    (output / 'acceptance.json').write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.output)
