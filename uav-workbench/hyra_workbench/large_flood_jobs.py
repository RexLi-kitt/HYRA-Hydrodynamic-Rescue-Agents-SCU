"""Run the large-flood prototype using the shared workbench job lock."""
import json
from pathlib import Path
import subprocess
import sys
import uuid

from .paths import ROOT


def run_large_flood(handler, jobs):
    origin = handler.headers.get('Origin')
    if origin and origin != 'http://' + handler.headers.get('Host', ''):
        return handler.reply(dict(error='拒绝跨来源请求'), 403)
    locked = False
    try:
        size = int(handler.headers.get('Content-Length', '0'))
        if not 0 < size <= 65536:
            raise ValueError('任务参数为空或过大')
        data = json.loads(handler.rfile.read(size))
        if not isinstance(data, dict) or set(data) - {'input', 'agents', 'speed', 'footprint'}:
            raise ValueError('任务参数无效')
        source = Path(data.get('input') or ROOT / 'generated/large-flood/development-1200m.zip').resolve()
        if not source.is_file() or source.suffix.lower() != '.zip':
            raise ValueError('洪水过程ZIP不存在')
        agents, speed, footprint = data.get('agents', 8), data.get('speed', 8), data.get('footprint', 25)
        if type(agents) is not int or agents not in (4,8,12) or type(speed) not in (int,float) or speed not in (5,8,12) or type(footprint) not in (int,float) or footprint not in (25,50,100):
            raise ValueError('机数、速度或感知宽度不在登记选项')
        locked = jobs.lock.acquire(blocking=False)
        if not locked:
            return handler.reply(dict(error='已有任务正在生成，请稍候'), 409)
        identity = uuid.uuid4().hex[:12]
        output = ROOT / 'generated/large-flood/runs' / (identity + '.json')
        proc = subprocess.run([sys.executable, '-m', 'large_flood.simulate', '--input', str(source),
                               '--output', str(output), '--agents', str(agents), '--speed', str(speed),
                               '--footprint', str(footprint)], cwd=ROOT, capture_output=True, text=True,
                               encoding='utf-8', errors='replace', timeout=300)
        if proc.returncode:
            return handler.reply(dict(error='运行失败：' + proc.stderr[-2000:]), 400)
        return handler.reply(dict(replay_url='/large-flood-runs/' + identity + '.json', summary=json.loads(proc.stdout)))
    except (ValueError, TypeError, OSError) as error:
        return handler.reply(dict(error=str(error)), 400)
    except subprocess.TimeoutExpired:
        return handler.reply(dict(error='运行超时'), 504)
    finally:
        if locked:
            jobs.lock.release()

