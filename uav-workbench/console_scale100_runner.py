"""Separate learning-runtime worker for the exact frozen 100m environment.

Development validation is a CLI-only check of four existing prototype models;
it cannot publish a model, use holdout parents, or write web history.
"""
import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from console_scale100_contract import (ROOT, FORMAL, BACKEND, TASKS, LABELS,
    default_config, digest, read, require, validate, scene_records, model_record)
from scale100_train import torch, np, FloodCase, ScaleEnv, config, rollout
from scale100_backend import make_actor, validate_meta
from scale100_development import legal_rule
from scale100_rules import water_aware
from scale100_run_audit import audit
from hyra_uav.experiments import save_run

TASK_LABELS = {'exploration': '自主探索', 'exploration_rescue': '探索—搜救'}


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Unsupported replay value {type(value).__name__}')


def assemble(checked, actor=None, meta=None, development=False, *, environment=None, method_label=None,
             contract_id='scale100-900s-v1'):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = environment.source if environment is not None else FloodCase(checked['scene']['path'])
    require(source.digest == checked['source_sha256'], '洪水过程摘要已变化')
    code = TASKS[checked['task']]
    # Rules use the legal dual-GMRF view; a requested raw group never changes
    # rule behavior or turns its score into an observation from that group.
    effective = checked['group'] if actor is not None else 'B3'
    env = environment if environment is not None else ScaleEnv(source, config(code, effective, checked['config']['seed']))
    started = time.perf_counter()
    if actor is not None:
        total, data, _ = rollout(actor, env, np.random.default_rng(env.config.seed), 0., True)
        images = data['images'].numpy()
    else:
        view = env.reset()
        images = [view.images.copy()]
        total, done = 0., False
        while not done:
            view, reward, done = env.step(water_aware(env))
            total += reward
            images.append(view.images.copy())
        images = np.asarray(images)
    report = audit(env, images)
    summary = dict(time_average_correct_coverage=report['balanced_current_correct_coverage_auc'],
                   time_average_first_coverage=report['first_coverage_auc'],
                   first_90_percent_coverage_s=report['first_90_s'],
                   undiscovered_change_fraction=report['undiscovered_change_fraction'],
                   distance_m=env.distance_m, turn_s=0., collision_events=env.collision_events,
                   final=env.snapshots[-1]['metrics'], rescued=report['rescued'],
                   total_targets=report['targets'], rescue_fraction=report['rescue_fraction'],
                   explored_fraction=report['final_first_coverage'],
                   discovered_targets=report['discovered'])
    payload = dict(schema='hyra-eight-workbench-v1', contract=contract_id,
        task=checked['task'], task_label=TASK_LABELS[checked['task']],
        group=checked['group'], requested_group=checked['requested_group'],
        scene={k: v for k, v in checked['scene'].items() if k != 'path'},
        method=(method_label or LABELS[checked['group']]) if actor is not None else '八向水情规则核查',
        status='completed', elapsed_s=time.perf_counter()-started,
        created_at=datetime.now(timezone.utc).isoformat(), n_agents=2,
        grid=source.process['grid'], source_sha256=source.digest,
        config=dict(duration_s=900., range_m=10., fov_deg=360., action_s=5., sample_s=5.),
        rescue_config=asdict(env.config), requested_config=checked['config'], summary=summary,
        data_split='development_integration_only' if development else 'interactive_not_formal_results',
        independent_event_claim=False, snapshots=env.snapshots, trajectory=env.trajectory,
        actions=env.actions, observations=env.observations, changes=[], events=env.events,
        truth=[source.labels(s['time_s']).astype(int).ravel().tolist() for s in env.snapshots],
        scenario_truth=dict(obstacles=env._blocked.astype(int).ravel().tolist(),
                            targets=[list(p) for p in env._targets],
                            scope='synthetic evaluator/replay overlay; never actor input'), audit=report,
        execution=dict(mode=checked['mode'], kind='scale100_checkpoint' if actor is not None else 'scale100_rule',
            actual_group=checked['group'], actor_sha256=meta['actor_sha256'] if meta else None,
            training_backend=BACKEND if meta else None,
            training_seed=meta['seed'] if meta else None, scenario_seed=env.config.seed,
            selected_episode=meta.get('selected_episode') if meta else None,
            trained_episodes=meta['trained_episodes'] if meta else None,
            total_training_episodes_executed=meta.get('total_training_episodes_executed') if meta else None,
            model_formal=meta['formal'] if meta else False, development_integration_only=development,
            return_=total, rule='water_aware' if actor is None else None,
            scope='交互回放不计入正式评价；水深/流速施救阈值仅为合成任务条件'))
    # Rescue event cells contain NumPy scalars; the existing writer is strict.
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=scalar))


def load_model(folder, task, group, seed, formal):
    meta = read(folder / 'actor.json')
    validate_meta(meta, BACKEND)
    require((meta['task'], meta['method'], meta['seed'], meta['formal'])
            == (TASKS[task], group, seed, formal), '检查点任务、组别、种子或训练阶段不匹配')
    require(meta['config'] == default_config(task, group, seed)
            and meta['observation_channels'] == 21 and meta['actions'] == 8,
            '检查点不属于21通道八动作100米环境')
    require(digest(folder / 'actor.pt') == meta['actor_sha256'], '模型权重摘要不匹配')
    actor = make_actor(BACKEND)
    actor.load_state_dict(torch.load(folder / 'actor.pt', weights_only=True, map_location='cpu'))
    actor.eval()
    return actor, meta


def execute_job(job):
    checked = validate(job)
    actor = meta = None
    if checked['mode'] == 'model':
        folder, _ = model_record(checked['task'], checked['group'], checked['training_seed'])
        actor, meta = load_model(folder, checked['task'], checked['group'], checked['training_seed'], True)
    return assemble(checked, actor, meta)


def validate_development(output):
    require(not output.exists(), '拒绝覆盖已有接入验证证据')
    output.mkdir(parents=True)
    scene = next(s for s in scene_records() if s['split'] == 'development')
    rows = []
    for task in TASKS:
        for group in (None, 'D', 'P'):
            requested = group or 'B3'
            checked = validate(dict(scene=scene['id'], task=task, group=requested, mode='rule'))
            actor = meta = None
            if group:
                # Explicitly CLI-only: no formal/model release gate bypass in API.
                folder = ROOT / 'generated/scale100-aux-resilient-20261026' / f'{TASKS[task]}-{group}-s11'
                actor, meta = load_model(folder, task, group, 11, False)
                checked.update(group=group, mode='model')
            result = assemble(checked, actor, meta, development=True)
            identity = TASKS[task] + '-' + (group or 'rule')
            save_run(result, output / identity)
            # Verify the exact artifact consumed by the UI, not just memory.
            saved = read(output / identity / 'replay.json')
            require(saved == result, '保存后的回放与审计对象不同')
            require(len(saved['snapshots']) == 181 and len(saved['observations']) == 362
                    and len(saved['actions']) == 180 and saved['audit']['audit_passed'],
                    '100米工作台回放不完整')
            rows.append(dict(id=identity, task=task, actual_group=result['group'],
                             replay_sha256=digest(output / identity / 'replay.json'),
                             actor_sha256=result['execution']['actor_sha256'],
                             audit=result['audit']))
            print(json.dumps(dict(id=identity, audit_passed=True,
                                  rescued=result['summary']['rescued'])), flush=True)
    evidence = dict(schema='hyra-scale100-console-validation-v1', complete=True,
                    formal_results=False, holdout_used=False, trials=rows,
                    adapter_sources={n: digest(ROOT / n) for n in
                        ('console_scale100_contract.py', 'console_scale100_runner.py')})
    (output / 'validation.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--development-validation', type=Path)
    modes.add_argument('--job', type=Path)
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.development_validation is not None:
        # Restrict diagnostic output to a distinct directory outside web history.
        target = args.development_validation.resolve()
        base = (ROOT / 'generated/scale100-console-validation').resolve()
        require(base in target.parents, '开发验证只能写入独立验证目录')
        validate_development(target)
    else:
        require(isinstance(args.run_id, str) and len(args.run_id) == 12
                and all(c in '0123456789abcdef' for c in args.run_id), '无效运行编号')
        result = execute_job(read(args.job))
        save_run(result, ROOT / 'generated/web' / args.run_id)
        print(json.dumps(dict(run_id=args.run_id, task=result['task'], group=result['group'],
                              mode=result['execution']['mode'], summary=result['summary'])))
