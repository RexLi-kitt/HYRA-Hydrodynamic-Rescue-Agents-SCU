"""Lightweight workbench contract. Safe to import in the Python3.12 server.

Formal models remain unavailable until a separately audited release exists.
There are no Torch, NumPy, training, or evaluator imports in this module.
"""
from runtime_assets import runtime_path

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMAL = runtime_path(ROOT / 'generated/scale100-formal-20261026')
HELDOUT = runtime_path(ROOT / 'generated/scale100-heldout-20261026')
RELEASE = ROOT / 'research/scale100-workbench-release-20261026.json'
TASKS = {'exploration': 'E1', 'exploration_rescue': 'E2'}
LABELS = {'B0': 'QMIX', 'B1': 'GMRF + QMIX', 'B2': 'Sur-QMIX',
          'B3': 'GMRF + Sur-QMIX', 'D': 'GMRF + Sur-QMIX + Data',
          'P': 'GMRF + Sur-QMIX + Physics'}
ACTIVE_GROUPS = ('B3', 'D', 'P')
ACTIONS = ('west', 'east', 'north', 'south', 'northwest', 'southwest',
           'northeast', 'southeast')
BACKEND = 'aux_resilient_v4'
TRAINING_SEEDS = (11, 29, 47)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def task_code(task):
    require(task in TASKS, '请选择自主探索或探索—搜救')
    return TASKS[task]


def default_config(task, group, seed=11):
    code = task_code(task)
    require(group in LABELS, '请选择六个已登记组别之一')
    require(type(seed) is int and 0 <= seed < 2**32, '场景种子必须是有效整数')
    return dict(step_s=5., duration_s=900., sense_width=5, service_s=10.,
                n_targets=0 if code == 'E1' else 5, obstacle_fraction=.1,
                seed=seed, n_agents=2, task=code,
                representation='raw' if group in ('B0', 'B2') else 'gmrf',
                service_depth_limit_m=.08, service_speed_limit_m_s=.3)


def scene_records():
    # Never expose the formal holdout parents as interactive development data.
    plan = read(FORMAL / 'plan.json')
    return [dict(id='scale100:' + row['parent_terrain'],
                 name=('开发验证' if row['split'] == 'development' else '训练场景')
                      + ' / ' + row['parent_terrain'],
                 path=str(runtime_path(row['path'])), source_sha256=row['sha256'],
                 split=row['split'], parent=row['parent_terrain'],
                 grid=dict(shape=[20, 20], dx_m=5., axis_order=['z', 'x']),
                 duration_s=900., actual_frame_count=181,
                 kind='synthetic_scale100_swe')
            for row in plan['inputs'] if row['split'] in ('fit', 'development')]


def release_record():
    require(RELEASE.is_file(), '100米正式模型仍在训练或评价，尚未发布')
    release = read(RELEASE)
    require(release.get('schema') == 'hyra-scale100-workbench-release-v1'
            and release.get('complete') is True, '正式发布记录不完整')
    expected = {f'{t}-{g}-s{s}' for t in ('E1', 'E2') for g in LABELS
                for s in TRAINING_SEEDS}
    models = release.get('models', [])
    require(len(models) == 36 and {m['id'] for m in models} == expected,
            '正式发布必须保留全部36项选模结果')
    for key, path in [('formal_plan_sha256', FORMAL / 'plan.json'),
                      ('model_seal_sha256', HELDOUT / 'sealed-models.json'),
                      ('evaluation_results_sha256', HELDOUT / 'results.json'),
                      ('release_audit_sha256', HELDOUT / 'release-audit.json')]:
        require(path.is_file() and release.get(key) == digest(path),
                '正式模型或评价证据摘要不匹配：' + path.name)
    require(read(FORMAL / 'queue.json')['status'] == 'completed'
            and read(HELDOUT / 'status.json')['status'] == 'completed',
            '正式训练及留出评价必须全部完成')
    report = read(HELDOUT / 'results.json')
    proof = read(HELDOUT / 'release-audit.json')
    require(report.get('complete') is True and report.get('learned_trials') == 216
            and report.get('rule_trials') == 36
            and report.get('all_audits_passed') is True
            and proof.get('complete') is True
            and proof.get('all_replays_reaudited') is True
            and proof.get('trials') == 252
            and proof.get('evaluation_results_sha256') == digest(HELDOUT / 'results.json'),
            '252项留出运行及独立回放复核尚未完成')
    return release


def model_record(task, group, training_seed=11):
    code = task_code(task)
    require(group in LABELS and type(training_seed) is int
            and training_seed in TRAINING_SEEDS, '模型组别或训练种子无效')
    release = release_record()
    identity = f'{code}-{group}-s{training_seed}'
    row = next(m for m in release['models'] if m['id'] == identity)
    folder = FORMAL / identity / 'selected'
    meta = read(folder / 'actor.json')
    require(digest(folder / 'actor.pt') == row['actor_sha256'] == meta['actor_sha256'],
            '正式模型权重摘要不匹配')
    require(digest(folder / 'actor.json') == row['metadata_sha256'],
            '正式模型元数据已变化')
    require((meta['task'], meta['method'], meta['seed'], meta['formal'])
            == (code, group, training_seed, True)
            and meta['formal_plan_sha256'] == release['formal_plan_sha256'],
            '模型不属于所选任务、组别、训练种子或正式方案')
    return folder, meta


def validate(data):
    require(isinstance(data, dict), '任务参数必须是对象')
    require(not set(data) - {'scene', 'task', 'group', 'mode', 'config', 'training_seed'},
            '包含未知100米工作台参数')
    task, group, mode = (data.get(k) for k in ('task', 'group', 'mode'))
    task_code(task)
    require(group in ACTIVE_GROUPS and mode in ('rule', 'model'), '组别或执行方式无效')
    values = data.get('config', {})
    require(isinstance(values, dict), '任务配置必须是对象')
    expected = default_config(task, group, values.get('seed', 11))
    require(not set(values) - set(expected), '包含未知环境参数')
    require(all(type(v) in (int, float, str) and v == expected[k]
                for k, v in values.items()),
            '正式100米环境固定为两机、900秒、5米格距及登记感知/施救配置；仅场景种子可变')
    scene = next((s for s in scene_records() if s['id'] == data.get('scene')), None)
    require(scene is not None, '请选择已登记的100米训练或开发场景')
    require(digest(scene['path']) == scene['source_sha256'], '洪水过程摘要已变化')
    training_seed = data.get('training_seed', 11)
    require(type(training_seed) is int and training_seed in TRAINING_SEEDS,
            '训练种子必须是11、29或47，与场景种子分别记录')
    if mode == 'model':
        model_record(task, group, training_seed)
    return dict(scene=scene, task=task, requested_group=group,
                group=group if mode == 'model' else None, mode=mode,
                config=expected, training_seed=training_seed,
                source_sha256=scene['source_sha256'])


def catalog():
    try:
        release_record()
        ready, reason = True, '已完成统一预算训练、开发集选模和封存后的留出评价'
    except (OSError, ValueError, KeyError, TypeError) as error:
        ready, reason = False, str(error)
    queue = read(FORMAL / 'queue.json')
    progress = dict(completed=len(queue['completed']), expected=36,
                    recorded_status=queue['status'])
    return dict(version='hyra-scale100-contract-v1', released=ready, reason=reason,
                actions=list(ACTIONS), grid=dict(shape=[20, 20], dx_m=5.),
                profile='scale100', default_task='exploration', default_group='B3',
                default_mode='rule', training_progress=progress,
                training_seeds=list(TRAINING_SEEDS), default_training_seed=11,
                defaults={t: default_config(t, 'B3') for t in TASKS},
                groups=[dict(id=g, label=label, checkpoint=None,
                             representation='raw' if g in ('B0', 'B2') else 'dual_gmrf',
                             surprise=g in ('B2', 'B3', 'D', 'P'),
                             auxiliary='data' if g == 'D' else 'physics' if g == 'P' else 'none',
                             tasks={t: dict(ready=ready, reason=reason)
                                                      for t in TASKS})
                        for g, label in LABELS.items() if g in ACTIVE_GROUPS])
