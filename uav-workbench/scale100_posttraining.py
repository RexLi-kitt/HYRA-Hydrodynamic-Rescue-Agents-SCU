"""One-shot continuation: wait for the existing run, evaluate, then audit.

No training launch/restart, scheduling, model selection changes, or publishing.
The waiting process uses only stdlib and pins the current host process identity.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / 'generated/scale100-formal-20261026'
EVALUATION = ROOT / 'generated/scale100-heldout-20261026'
STATE = ROOT / 'generated/scale100-posttraining-20261026.json'
PINNED = ('scale100_posttraining.py', 'scale100_evaluate_sealed.py',
          'scale100_release_audit.py', 'scale100_rules.py', 'scale100_development.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(state):
    state['observed_at'] = time.time()
    temp = STATE.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(STATE)


def process_identity(pid):
    """None means verified missing/terminated; access failures are exceptions."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        if ctypes.get_last_error() == 87:
            return None
        raise OSError(ctypes.get_last_error(), 'Cannot observe process; no restart is attempted')
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        if code.value != 259:
            return None
        created, ended, kernel_time, user_time = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(ended),
                                      ctypes.byref(kernel_time), ctypes.byref(user_time)):
            raise ctypes.WinError(ctypes.get_last_error())
        return dict(pid=int(pid), created_filetime=(created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)


def bindings():
    return {str(FORMAL / 'plan.json'): sha(FORMAL / 'plan.json'),
            **{str(ROOT / name): sha(ROOT / name) for name in PINNED}}


def verify_bindings(expected):
    if any(sha(path) != value for path, value in expected.items()):
        raise ValueError('Pinned posttraining protocol changed; retain evidence and inspect')


def inspect():
    queue = read(FORMAL / 'queue.json')
    identity = process_identity(queue['pid'])
    if queue['status'] != 'completed' and not identity:
        raise ValueError('Existing training is not verified live; this tool never restarts it')
    return dict(controller=identity, queue_status=queue['status'],
                completed=len(queue['completed']), active_jobs=list(queue['active_jobs']),
                bindings=bindings())


def command(state, phase, args):
    state.update(status='running', phase=phase, child_pid=None)
    write(state)
    with (ROOT / f'generated/scale100-posttraining-{phase}.log').open('a', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-B', *args], cwd=ROOT, stdout=log,
                                 stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        state['child_pid'] = child.pid
        write(state)
        code = child.wait()
    if code:
        raise RuntimeError(f'{phase} exited {code}; retain logs; no automatic retraining or retuning')
    state['child_pid'] = None
    write(state)


def execute():
    import msvcrt
    # OS lock survives stale lock files and releases when this process exits.
    with STATE.with_suffix('.lock').open('a+b') as lock:
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        previous = read(STATE) if STATE.exists() else None
        if previous and previous['status'] == 'completed':
            raise ValueError('Posttraining evaluation/audit already completed')
        current = inspect()
        pinned = previous['bindings'] if previous else current['bindings']
        verify_bindings(pinned)
        controller = previous['controller'] if previous else current['controller']
        state = dict(schema='hyra-scale100-posttraining-v1', status='waiting',
                     phase='formal_training', pid=os.getpid(), controller=controller,
                     started_at=time.time(), bindings=pinned, child_pid=None,
                     completed_formal_jobs=current['completed'], formal_jobs=36)
        write(state)
        missing_observations = 0
        try:
            while True:
                verify_bindings(pinned)
                queue = read(FORMAL / 'queue.json')
                if queue['status'] == 'completed':
                    if len(queue['completed']) != 36 or queue['failed_jobs']:
                        raise ValueError('Completed queue has an incomplete formal matrix')
                    state['completed_formal_jobs'] = 36
                    break
                if queue['status'] == 'failed':
                    raise RuntimeError('Formal training reported failure; no duplicate training is launched')
                try:
                    identity = process_identity(queue['pid'])
                except OSError as error:
                    state.update(observation_error=repr(error), status='observation_uncertain')
                    write(state)
                    time.sleep(30)
                    continue
                if identity != controller:
                    missing_observations += 1
                    if missing_observations >= 3:
                        raise RuntimeError('Original training process is no longer live; inspect before resuming')
                else:
                    missing_observations = 0
                state.pop('observation_error', None)
                state.update(status='waiting', completed_formal_jobs=len(queue['completed']),
                             controller_live=identity == controller, active_jobs=queue['active_jobs'])
                write(state)
                time.sleep(30)
            command(state, 'evaluation', ['scale100_evaluate_sealed.py', '--formal-root',
                str(FORMAL), '--output', str(EVALUATION)])
            verify_bindings(pinned)
            command(state, 'audit', ['scale100_release_audit.py', '--formal-root',
                str(FORMAL), '--evaluation', str(EVALUATION)])
            state.update(status='completed', phase='awaiting_report_and_workbench_delivery',
                         finished_at=time.time(), child_pid=None,
                         release_audit_sha256=sha(EVALUATION / 'release-audit.json'))
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
