"""Serve the read-only local two-UAV replay viewer; no scientific mutations."""
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--example',type=Path,default=ROOT/'generated/sciml-round1/engineering-deployment/fit-cli-replay.json')
    args=parser.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path=urlsplit(self.path).path
            files={'/':('index.html','text/html; charset=utf-8'),'/app.js':('app.js','text/javascript; charset=utf-8'),'/style.css':('style.css','text/css; charset=utf-8')}
            if path in files:
                name,kind=files[path];body=(ROOT/'team-viewer'/name).read_bytes()
            elif path=='/example.json' and args.example.is_file():
                if args.example.stat().st_size>30*2**20:self.send_error(413);return
                body=args.example.read_bytes();value=json.loads(body)
                if value.get('replay',value).get('schema')!='hyra-team-run-v1':self.send_error(422);return
                kind='application/json; charset=utf-8'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Security-Policy',"default-src 'self';script-src 'self';style-src 'self';connect-src 'self';img-src 'self' data:;object-src 'none';base-uri 'none';frame-ancestors 'none'")
            self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'HyRA two-UAV viewer http://127.0.0.1:{args.port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
