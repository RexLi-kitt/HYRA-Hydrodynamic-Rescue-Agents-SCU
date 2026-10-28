"""Loopback-only workbench. Only registered process bundles and generated runs."""
from dataclasses import fields
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from urllib.parse import urlparse
import uuid

from .core import Config, FloodProcess, Pose
from .experiments import ROOT, OUTPUT, available_scenes, run_episode, save_run

STATIC = ROOT / "static"


class Handler(BaseHTTPRequestHandler):
    def reply(self, data, status=200, content_type="application/json; charset=utf-8"):
        raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/scenes":
            result = []
            for scene in available_scenes():
                process = FloodProcess(scene["path"])
                result.append(dict(**{k: v for k, v in scene.items() if k != "path"},
                                   grid=process.process["grid"], duration_s=float(process.times[-1]),
                                   sample_s=float(process.times[1]-process.times[0])))
            return self.reply(result)
        if path.startswith("/runs/"):
            parts = path.split("/")
            if len(parts) == 4 and len(parts[2]) == 12 and all(c in "0123456789abcdef" for c in parts[2]) and parts[3] in ("replay.json", "manifest.json", "observations.json", "trajectory.json", "actions.json", "audit.json"):
                file = OUTPUT / "web" / parts[2] / parts[3]
                if file.is_file():
                    return self.reply(file.read_bytes())
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in files:
            name, mime = files[path]
            return self.reply((STATIC / name).read_bytes(), content_type=mime)
        self.reply(dict(error="页面或运行不存在"), 404)

    def do_POST(self):
        try:
            # Browser cross-origin pages may not initiate local mutations.
            origin = self.headers.get("Origin")
            if origin and origin != "http://"+self.headers.get("Host", ""):
                return self.reply(dict(error="拒绝跨来源请求"), 403)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 32*1024*1024:
                raise ValueError("请求为空或超过32MB限制")
            path = urlparse(self.path).path
            raw = self.rfile.read(size)
            if path == "/api/import":
                digest = sha256(raw).hexdigest()[:12]
                folder = OUTPUT / "imports" / digest
                folder.mkdir(parents=True, exist_ok=True)
                file = folder / "map-result.zip"
                file.write_bytes(raw)
                try:
                    process = FloodProcess(file)
                    if max(process.shape) > 64 or len(process.times) > 241:
                        raise ValueError("网页MVP限制64×64、241帧；更大过程请先离线评估内存")
                except Exception:
                    file.unlink(missing_ok=True)
                    raise
                return self.reply(dict(id="import-"+digest, audit=process.audit()))
            if path != "/api/run":
                return self.reply(dict(error="接口不存在"), 404)
            data = json.loads(raw)
            scene = next((s for s in available_scenes() if s["id"] == data["scene"]), None)
            if scene is None:
                raise ValueError("场景不存在")
            allowed = {f.name for f in fields(Config)}
            values = data.get("config", {})
            if not isinstance(values, dict) or set(values)-allowed:
                raise ValueError("包含未知参数")
            config = Config(**values)
            pose = Pose(**data.get("pose", dict(x_m=0, z_m=0, heading_deg=0)))
            result = run_episode(scene, data.get("method", "R0"), config, pose, data.get("actions"))
            run_id = uuid.uuid4().hex[:12]
            save_run(result, OUTPUT / "web" / run_id)
            self.reply(dict(run_id=run_id, replay_url=f"/runs/{run_id}/replay.json", summary=result["summary"]))
        except (ValueError, KeyError, TypeError, OSError, OverflowError) as error:
            self.reply(dict(error=str(error)), 400)
        except Exception as error:
            self.log_error("run error: %s", error)
            self.reply(dict(error="运行失败，详见服务日志："+str(error)), 500)


def serve(port=8767):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"HyRA 无人机观测：http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
