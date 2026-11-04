"""Replay sealed actions or infer on development fields; never update a model."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from .six_study import read, digest, validate, dev_cases


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError('来源校验失败：' + str(path))


def image_codes(depth, known=None):
    # Display classes only; these bins never enter the controller or evaluation.
    bins = np.digitize(depth, [.001, .05, .1, .3, 1., 3.]) + 1
    if known is not None:
        bins = np.where(known, bins, 0)
    return ''.join(bins.astype('U1').ravel())


def frame(env):
    return dict(t=int(env.t * 5), positions=env.pos.tolist(), headings=env.heading.tolist(),
                people=[dict(id=int(i), xy=env.people[i].tolist(), state=int(env.status[i]),
                             progress=float(env.progress[i])) for i in np.flatnonzero(env.status)],
                coverage=float(env.known.mean()), rescued=int((env.status == 2).sum()),
                conflicts=int(env.conflicts), observed=image_codes(env.measured[0], env.known),
                truth=image_codes(env.water[env.t, 0]))


def reconstruct(Env, water, people, actions, trace, expected):
    env = Env(water, people=people)
    frames = [frame(env)]
    for i, action in enumerate(actions):
        _, _, done, info = env.step(action)
        if not np.allclose(env.pos, trace['position'][i], atol=1e-9, rtol=0):
            raise ValueError('动作重放位置与封存轨迹不符')
        actual = [info[k] for k in ('new_cells', 'found', 'rescued', 'conflicts', 'boundary')]
        if not np.array_equal(actual, trace['info'][i]):
            raise ValueError('动作重放事件与封存轨迹不符')
        if done and i != len(actions) - 1:
            raise ValueError('任务完成后存在额外动作')
        frames.append(frame(env))
    if not np.array_equal(env.known, trace['known']) or not np.array_equal(env.status, trace['target_status']):
        raise ValueError('终态与封存轨迹不符')
    for key, value in env.metrics().items():
        if expected.get(key) != value:
            raise ValueError('正式指标重算不符：' + key)
    return frames


def run(job):
    payload = read(job)
    cfg = payload['config']
    root = Path(cfg['study'])
    # Local installation pins every manifest and runtime source before imports.
    for path, sha in cfg['seals'].items():
        require_hash(path, sha)
    frozen = read(root / 'frozen-candidates.json')['candidates']
    rows = [json.loads(line) for line in (root / 'results.jsonl').read_text(encoding='utf-8').splitlines() if line]
    dev = dev_cases(cfg)
    request = validate(payload['request'], rows, frozen, dev)
    for path, sha in read(root / 'registration.json')['source_hashes'].items():
        require_hash(path, sha)
    spec = importlib.util.spec_from_file_location('six_frozen_adapter', cfg['adapter'])
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    adapter.bootstrap()
    engine = adapter.e
    if request['mode'] == 'record':
        row = rows[request['index']]
        case = next(c for c in read(root / 'scenario-manifest.json') if c['id'] == row['case'])
        water_path = Path(case['path'])
        require_hash(water_path, case['sha256'])
        trace_path = root / 'traces' / Path(row['trace']).name
        require_hash(trace_path, cfg['trace_hashes'][trace_path.name])
        water = np.load(water_path, mmap_mode='r', allow_pickle=False)
        with np.load(trace_path, allow_pickle=False) as trace:
            if not np.array_equal(trace['people'], case['people']):
                raise ValueError('人员配置不符')
            frames = reconstruct(engine.Env, water, np.asarray(case['people']), trace['action'], trace, row)
        metrics = {k: v for k, v in row.items() if k != 'trace'}
        title = row['candidate'].upper() + ' / ' + row['case']
        evidence = dict(kind='record', trace_sha256=digest(trace_path), field_sha256=case['sha256'],
                        record_index=request['index'], verified=True)
    else:
        candidate = next(c for c in frozen if c['id'] == request['candidate'])
        case = dev[request['case']]
        checkpoint = Path(candidate['path'])
        require_hash(checkpoint, candidate['sha256'])
        require_hash(case['path'], case['sha256'])
        water = np.load(case['path'], mmap_mode='r', allow_pickle=False)
        # Only the pinned, locally produced checkpoint is deserialized.
        models, _ = engine.restore(adapter.torch.load(checkpoint, map_location='cpu', weights_only=False))
        roles = (0, 0, 0, 0) if candidate['family'] == 'single' else ((0, 1, 1, 0), (1, 0, 0, 1))[request['assignment']]
        trace_path = Path(job).parent / 'demo-trace.npz'
        layout = 770003  # fixed interactive layout; never appended to formal scores
        with adapter.torch.inference_mode():
            metrics, _ = engine.rollout(models, water, layout, roles, deterministic=True,
                                       coverage=True, trace_path=trace_path)
        require_hash(checkpoint, candidate['sha256'])
        with np.load(trace_path, allow_pickle=False) as trace:
            frames = reconstruct(engine.Env, water, trace['people'], trace['action'], trace, metrics)
        title = candidate['id'].upper() + ' / ' + case['parent']
        evidence = dict(kind='demo', checkpoint_sha256=candidate['sha256'], episode=candidate['episode'],
                        field_sha256=case['sha256'], assignment=request['assignment'], roles=roles,
                        layout_seed=layout, weights_unchanged=True, contributes_to_formal_scores=False)
    result = dict(title=title, evidence=evidence, metrics=metrics, frames=frames, side_m=1000,
                  grid=100, frame_seconds=5,
                  note='已完成服务是仿真内停留任务，不代表现实人员转运。真值层仅供回放核查，不作为策略输入。')
    (Path(job).parent / 'replay.json').write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status='completed', frames=len(frames), evidence=evidence)))


if __name__ == '__main__':
    run(Path(sys.argv[1]))
