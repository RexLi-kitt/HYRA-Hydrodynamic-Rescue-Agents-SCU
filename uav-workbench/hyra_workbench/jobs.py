"""Validate and execute one interactive job in a separate learning process."""
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import uuid

from .paths import ROOT
from .profiles import select_profile


class JobBusy(RuntimeError):
    pass


def learning_python():
    """Honor an explicit runtime and retain the existing local installation."""
    configured = os.environ.get('HYRA_LEARNING_PYTHON')
    runtime = Path(configured) if configured else Path('D:/Scripts/python.exe')
    if not configured and not runtime.is_file():
        runtime = Path(sys.executable)
    if not runtime.is_file():
        raise ValueError('找不到配置的模型解释器，请检查 HYRA_LEARNING_PYTHON')
    return runtime


class JobRunner:
    def __init__(self, root=ROOT, timeout=180):
        self.root = Path(root)
        self.timeout = timeout
        self.lock = threading.Lock()

    def execute(self, request):
        profile = select_profile(request)
        checked = profile.contract.validate(request)
        if not self.lock.acquire(blocking=False):
            raise JobBusy('已有任务正在生成，请稍候')
        try:
            runtime = learning_python()
            identity = uuid.uuid4().hex[:12]
            folder = self.root / 'generated/workbench-jobs'
            folder.mkdir(parents=True, exist_ok=True)
            job = folder / (identity + '.json')
            data = profile.worker_payload(request, checked, identity)
            job.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            args = [str(runtime), '-B', '-u', str(self.root / profile.runner), '--job', str(job)]
            if profile.name != 'original_grid':
                args += ['--run-id', identity]
            result = subprocess.run(
                args, cwd=self.root, capture_output=True, text=True,
                encoding='utf-8', errors='replace', timeout=self.timeout,
                env=dict(os.environ, PYTHONUTF8='1'),
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
            )
            (folder / (identity + '.log')).write_text(
                result.stdout + '\n' + result.stderr, encoding='utf-8',
            )
            if result.returncode:
                raise RuntimeError('任务执行失败，日志：generated/workbench-jobs/' + identity + '.log')
            replay = json.loads(
                (self.root / 'generated/web' / identity / 'replay.json').read_text(encoding='utf-8')
            )
            return dict(run_id=identity, replay_url=f'/runs/{identity}/replay.json', summary=replay['summary'])
        finally:
            self.lock.release()
