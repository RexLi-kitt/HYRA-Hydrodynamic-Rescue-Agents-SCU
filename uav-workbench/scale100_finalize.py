"""One-shot analysis/publishing continuation after the pinned replay audit.

Waiting uses stdlib only. Never launches training, changes selected weights, or
marks the user's goal complete. Browser/result review remains with the agent.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scale100_posttraining import process_identity

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'generated/scale100-posttraining-20261026.json'
STATE = ROOT / 'generated/scale100-finalize-20261026.json'
PROTOCOL = ROOT / 'research/scale100-formal-forecast-protocol-20261026.json'
FORECAST = ROOT / 'generated/scale100-heldout-20261026/auxiliary-forecasts.json'
REPORT = ROOT / 'generated/scale100-report-20261026'
ACCEPTANCE = ROOT / 'generated/scale100-model-acceptance-20261026'
SOURCES = ('scale100_finalize.py', 'scale100_posttraining.py', 'scale100_formal_forecasts.py',
           'scale100_resilient_forecast.py', 'scale100_report.py', 'scale100_publish.py',
           'scale100_model_acceptance.py', 'console_scale100_runner.py', 'console_scale100_contract.py')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(state):
    state['observed_at'] = time.time()
    temporary = STATE.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(STATE)


def bindings():
    paths = [ROOT / name for name in SOURCES] + [PROTOCOL]
    return {str(path):sha(path) for path in paths}


def verify(expected):
    if any(sha(Path(path)) != value for path, value in expected.items()):
        raise ValueError('Pinned analysis/publishing code changed; preserve evidence and inspect')


def inspect():
    upstream = read(UPSTREAM)
    identity = process_identity(upstream['pid'])
    if upstream['status'] != 'completed' and identity is None:
        raise ValueError('Existing posttraining process is not verified live; no duplicate is launched')
    return dict(upstream_pid=upstream['pid'], upstream_identity=identity,
                upstream_status=upstream['status'], bindings=bindings())


def command(state, phase, args):
    state.update(status='running', phase=phase, child_pid=None)
    write(state)
    with (ROOT / f'generated/scale100-finalize-{phase}.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', *args], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        state['child_pid'] = child.pid
        write(state)
        code = child.wait()
    if code:
        raise RuntimeError(f'{phase} exited {code}; keep all records and inspect before resuming')
    state['child_pid'] = None
    write(state)


def execute():
    import msvcrt
    with STATE.with_suffix('.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if STATE.exists():
            raise ValueError('A finalization attempt already exists; inspect its actual process and artifacts')
        current = inspect()
        state = dict(schema='hyra-scale100-finalize-v1', status='waiting',
                     phase='policy_evaluation_and_replay_audit', pid=os.getpid(),
                     started_at=time.time(), child_pid=None, **current)
        write(state)
        missing = 0
        try:
            while True:
                verify(state['bindings'])
                upstream = read(UPSTREAM)
                if upstream['status'] == 'completed':
                    break
                if upstream['status'] == 'failed':
                    raise RuntimeError('Policy evaluation/audit continuation failed; no replacement runs launched')
                try:
                    identity = process_identity(upstream['pid'])
                except OSError as error:
                    state.update(status='observation_uncertain', observation_error=repr(error))
                    write(state)
                    time.sleep(30)
                    continue
                if identity != state['upstream_identity']:
                    missing += 1
                    if missing >= 3:
                        raise RuntimeError('Pinned upstream process is no longer live; inspect before any resume')
                else:
                    missing = 0
                state.pop('observation_error', None)
                state.update(status='waiting', upstream_status=upstream['status'], upstream_live=missing == 0)
                write(state)
                time.sleep(30)
            command(state, 'auxiliary', ['scale100_formal_forecasts.py', '--output', str(FORECAST)])
            verify(state['bindings'])
            command(state, 'report', ['scale100_report.py', '--output', str(REPORT), '--forecasts', str(FORECAST)])
            verify(state['bindings'])
            command(state, 'publish', ['scale100_publish.py'])
            verify(state['bindings'])
            command(state, 'model_acceptance', ['scale100_model_acceptance.py', '--output', str(ACCEPTANCE)])
            state.update(status='completed', phase='awaiting_agent_result_and_browser_acceptance',
                         child_pid=None, finished_at=time.time(),
                         report_manifest_sha256=sha(REPORT / 'manifest.json'),
                         model_acceptance_sha256=sha(ACCEPTANCE / 'acceptance.json'))
            write(state)
        except Exception as error:
            state.update(status='failed', error=repr(error), finished_at=time.time())
            write(state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        print(json.dumps(inspect(), ensure_ascii=False, indent=2))
    else:
        execute()
