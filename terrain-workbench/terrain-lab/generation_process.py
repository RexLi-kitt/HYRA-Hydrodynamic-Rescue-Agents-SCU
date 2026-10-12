"""Supervise a generator without blocking on unterminated download output."""
import os
from pathlib import Path
import signal
import subprocess
import time


def stop_tree(process):
    if process.poll() is not None: return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=20)
    else:
        try: os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError: pass
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != 'nt':
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else: process.kill()
        process.wait(timeout=5)


def run(command, cwd, logfile, report, cancelled, *, idle_timeout=600, total_timeout=1800):
    start = last = time.monotonic()
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
    options = dict(creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else dict(start_new_session=True)
    logfile = Path(logfile)
    with logfile.open('wb', buffering=0) as output:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, **options)
        try:
            with logfile.open('r', encoding='utf-8', errors='replace') as reader:
                while True:
                    text = reader.read()
                    now = time.monotonic()
                    if text: last = now
                    report(text, now-start, now-last)
                    if cancelled(): raise InterruptedError('生成已取消，原地图保留')
                    if process.poll() is not None:
                        # Capture bytes written between the read and process exit.
                        report(reader.read(), now-start, 0)
                        return process.returncode
                    if now-last > idle_timeout:
                        raise TimeoutError(f'生成器连续 {idle_timeout} 秒无新日志，已停止本次任务；请查看生成记录中的依赖或网络错误后重试')
                    if now-start > total_timeout:
                        raise TimeoutError(f'生成超过 {total_timeout} 秒，已停止本次任务；可缩小地图范围后重试')
                    time.sleep(.25)
        finally:
            stop_tree(process)
