"""Loopback HTTP host for the current three-profile, eight-direction workbench."""
import argparse
from http.server import ThreadingHTTPServer
import json
import subprocess
from urllib.parse import urlparse

from hyra_uav.web import Handler as ArtifactHandler
from . import profiles
from .jobs import JobBusy, JobRunner
from .paths import STATIC

JOBS = JobRunner()
VERSION = 'hyra-eight-workbench-v5'
CATALOG_ROUTES = {
    '/api/workbench/catalog': 'original_grid',
    '/api/workbench/scale100/catalog': 'scale100',
    '/api/workbench/signal-routes/catalog': 'signal_routes',
}
STATIC_ROUTES = {
    '/': ('workbench-scale100.html', 'text/html; charset=utf-8'),
    '/rescue.html': ('workbench-scale100.html', 'text/html; charset=utf-8'),
    '/app.js': ('workbench-scale100.js', 'text/javascript; charset=utf-8'),
    '/rescue.js': ('workbench-scale100.js', 'text/javascript; charset=utf-8'),
    '/style.css': ('workbench.css', 'text/css; charset=utf-8'),
    '/rescue.css': ('workbench.css', 'text/css; charset=utf-8'),
}


class Handler(ArtifactHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        try:
            if path == '/api/console-version':
                return self.reply(dict(version=VERSION, action_count=8,
                                       profiles=[p.name for p in profiles.PROFILES]))
            if path in CATALOG_ROUTES:
                return self.reply(profiles.catalog(CATALOG_ROUTES[path]))
            if path == '/api/workbench/scenes':
                return self.reply(profiles.scenes())
            if path == '/api/workbench/history':
                return self.reply(profiles.history())
            if path.startswith('/runs/'):
                return super().do_GET()
            if path.startswith('/api/'):
                return self.reply(dict(error='旧接口已停用，请刷新八方向工作台', url='/'), 410)
            if path in STATIC_ROUTES:
                name, mime = STATIC_ROUTES[path]
                return self.reply((STATIC / name).read_bytes(), content_type=mime)
            return self.reply(dict(error='页面不存在'), 404)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return self.reply(dict(error=str(error)), 400)

    def do_POST(self):
        if urlparse(self.path).path != '/api/workbench/run':
            return self.reply(dict(error='旧运行入口已停用，请使用八方向工作台'), 410)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.reply(dict(error='拒绝跨来源请求'), 403)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 65536:
                raise ValueError('任务参数为空或过大')
            return self.reply(JOBS.execute(json.loads(self.rfile.read(size))))
        except JobBusy as error:
            return self.reply(dict(error=str(error)), 409)
        except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
            return self.reply(dict(error=str(error)), 400)
        except subprocess.TimeoutExpired:
            return self.reply(dict(error='回放生成超时，未得到完整结果'), 504)
        except Exception as error:
            return self.reply(dict(error=str(error)), 500)


def serve(port=8767):
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    print(f'HyRA eight-direction workbench http://127.0.0.1:{server.server_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description='HyRA 八方向无人机工作台')
    parser.add_argument('--port', type=int, default=8767)
    args = parser.parse_args()
    serve(args.port)


if __name__ == '__main__':
    main()
