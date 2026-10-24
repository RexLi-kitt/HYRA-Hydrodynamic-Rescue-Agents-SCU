"""Independent read-only dataset workbench; does not change the terrain lab's state."""
import argparse
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote
from dataset import replay, grid_plan

ROOT = Path(__file__).resolve().parent
SHARED = ROOT.parent / 'terrain-workbench' / 'terrain-lab'


@lru_cache(maxsize=1)
def load_event():
    folder = ROOT / 'data' / 'harvey-2017'
    return (json.loads((folder / 'manifest.json').read_text(encoding='utf-8')),
        json.loads((folder / 'observations.json').read_text(encoding='utf-8')))


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        try:
            if parsed.path == '/api/event':
                return self.json_response(load_event()[0])
            if parsed.path == '/api/observations':
                args = parse_qs(parsed.query)
                event, records = load_event()
                policy = event['replay_policy']
                return self.json_response(replay(records, args['at'][0],
                    assumed_latency_s=float(args.get('latency', [policy['assumed_latency_s']])[0]),
                    max_age_s=policy['max_age_s']))
            if parsed.path == '/api/grid-plan':
                args = parse_qs(parsed.query)
                return self.json_response(grid_plan(float(args['width'][0]), float(args['cell'][0])))
        except FileNotFoundError:
            return self.json_response(dict(error='事件尚未导入，请先运行 import_harvey.py --terrain'), 503)
        except (ValueError, KeyError, OverflowError):
            return self.json_response(dict(error='时间、延迟或网格参数不正确'), 400)
        path = unquote(parsed.path)
        # Only two pure frontend assets are shared. Never expose the old runtime or current pointer.
        if path in ('/shared/style.css', '/shared/map-arrays.mjs'):
            target = SHARED / path.rsplit('/', 1)[1]
        elif path in ('/', '/index.html', '/app.mjs', '/style.css'):
            target = ROOT / ('index.html' if path == '/' else path[1:])
        elif path.startswith('/data/') and path.endswith(('.f32', '.u8')):
            target = (ROOT / path[1:]).resolve()
            if ROOT / 'data' not in target.parents:
                return self.send_error(404)
        else:
            return self.send_error(404)
        if not target.is_file():
            return self.send_error(404)
        body = target.read_bytes()
        self.send_response(200)
        content_type = {'.mjs': 'text/javascript', '.css': 'text/css', '.html': 'text/html; charset=utf-8'}.get(target.suffix, 'application/octet-stream')
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    print(f'真实洪水数据工作台 http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
