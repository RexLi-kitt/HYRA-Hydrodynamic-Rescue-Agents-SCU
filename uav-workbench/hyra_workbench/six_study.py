"""Read-only study catalogue and isolated, non-training replay jobs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid

from .paths import ROOT
from .jobs import JobBusy, learning_python


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def local_config():
    path = Path(os.environ.get('HYRA_SIX_CONFIG', ROOT / 'generated/six-study/local.json'))
    return read(path) if path.is_file() else None


def study_root():
    cfg = local_config()
    return Path(cfg['study']) if cfg else ROOT / 'docs/strict-eval-20261101'


def records():
    with (study_root() / 'results.jsonl').open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def validate(request, rows, candidates, dev):
    if not isinstance(request, dict):
        raise ValueError('任务必须是对象')
    mode = request.get('mode')
    if mode == 'record':
        index = request.get('index')
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError('请选择已登记的正式记录')
        return dict(mode=mode, index=index)
    if mode == 'demo':
        cid = request.get('candidate')
        if cid not in {c['id'] for c in candidates}:
            raise ValueError('模型不在冻结登记表内')
        case = request.get('case')
        if type(case) is not int or not 0 <= case < len(dev):
            raise ValueError('演示只允许开发场景')
        family = next(c['family'] for c in candidates if c['id'] == cid)
        assignment = request.get('assignment', 0)
        if type(assignment) is not int or assignment not in ((0, 1) if family == 'pair' else (0,)):
            raise ValueError('无效的模型分工')
        return dict(mode=mode, candidate=cid, case=case, assignment=assignment)
    raise ValueError('未知运行方式')


def dev_cases(cfg):
    return [c for c in read(cfg['dataset'])['cases'] if c['split'] == 'validation']


def catalogue():
    root = study_root()
    frozen = read(root / 'frozen-candidates.json')
    cfg = local_config()
    rows = records()
    public = [{k: v for k, v in row.items() if k != 'trace'} | {'index': i} for i, row in enumerate(rows)]
    return dict(candidates=[{k: v for k, v in c.items() if k not in ('path', 'best_source')}
                            for c in frozen['candidates']],
                records=public, summary=read(root / 'strict-summary.json')['summary'],
                runtime_ready=bool(cfg),
                dev=[dict(index=i, name=c['parent']) for i, c in enumerate(dev_cases(cfg))] if cfg else [],
                scope='合成场景、单训练种子；完整方法包含覆盖规划及服务控制。正式成绩不等于真实救援成功率。')


def execute(request, jobs):
    cfg = local_config()
    if not cfg:
        raise ValueError('本机尚未配置水场、轨迹与权重；仍可查看正式统计。见六算法接入说明。')
    root = study_root()
    checked = validate(request, records(), read(root / 'frozen-candidates.json')['candidates'], dev_cases(cfg))
    if not jobs.lock.acquire(blocking=False):
        raise JobBusy('已有工作台任务正在运行，请稍候')
    try:
        identity = uuid.uuid4().hex
        folder = ROOT / 'generated/six-study/runs' / identity
        folder.mkdir(parents=True)
        job = folder / 'job.json'
        job.write_text(json.dumps(dict(request=checked, config=cfg), ensure_ascii=False), encoding='utf-8')
        result = subprocess.run([str(learning_python()), '-B', '-m', 'hyra_workbench.six_worker', str(job)],
                                cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                timeout=180, env=dict(os.environ, PYTHONUTF8='1'),
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        (folder / 'worker.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
        if result.returncode:
            raise ValueError('回放核验或运行失败；日志：generated/six-study/runs/' + identity + '/worker.log')
        replay = read(folder / 'replay.json')
        return dict(url='/six-study-runs/' + identity + '.json', summary=replay['metrics'])
    finally:
        jobs.lock.release()


def replay_file(name):
    if len(name) != 37 or not name.endswith('.json') or any(c not in '0123456789abcdef' for c in name[:-5]):
        raise ValueError('无效回放标识')
    return ROOT / 'generated/six-study/runs' / name[:-5] / 'replay.json'
