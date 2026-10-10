"""Open the terrain controls once the local generator service is ready."""
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'http://127.0.0.1:8766'

def ready():
    try:
        with urllib.request.urlopen(URL + '/api/config', timeout=1) as response:
            return response.status == 200
    except OSError:
        return False

if __name__ == '__main__':
    owned = None
    if not ready():
        owned = subprocess.Popen([sys.executable, str(ROOT / 'server.py'), '--port', '8766'], cwd=ROOT)
        for _ in range(80):
            if ready(): break
            if owned.poll() is not None: raise SystemExit('页面服务未能启动')
            time.sleep(0.1)
    if not ready(): raise SystemExit('页面服务没有响应')
    webbrowser.open(URL)
    if owned:
        print('页面已打开。关闭此窗口将停止本次启动的服务。')
        try: owned.wait()
        except KeyboardInterrupt: owned.terminate()
