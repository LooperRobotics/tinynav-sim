"""Switch between provisioned TinyNav scene containers."""
import json
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SCENES = json.loads((ROOT / "deployment.json").read_text())
LOCK = threading.Lock()


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True, timeout=60).strip()


def activate(scene):
    runtime = Path(scene["runtime"])
    baseline = {name: (runtime / name).stat().st_mtime_ns
                if (runtime / name).exists() else 0
                for name in ("front.jpg", "sensors.bin")}
    docker("start", scene["container"])
    deadline = time.monotonic() + 100
    while time.monotonic() < deadline:
        if docker("inspect", "--format", "{{.State.Running}}", scene["container"]) != "true":
            raise RuntimeError("场景容器启动后退出")
        try:
            with urlopen(scene["health"], timeout=2) as response:
                if json.load(response).get("topics") and all(
                    (runtime / name).stat().st_mtime_ns > previous
                    for name, previous in baseline.items()
                ):
                    return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError("TinyNav 传感器接口未就绪")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def reply(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/scenes.json":
            return self.reply(200, [
                {"name": scene["name"], "url": scene["url"], "id": key}
                for key, scene in SCENES.items()
            ])
        if self.path in ("/", "/index.html"):
            return super().do_GET()
        self.send_error(404)

    def do_POST(self):
        if self.headers.get("Origin") != f"http://{BIND}:8766":
            return self.reply(403, {"error": "请求来源不匹配"})
        scene = SCENES.get(self.path.removeprefix("/activate/"))
        if not self.path.startswith("/activate/") or scene is None:
            return self.reply(404, {"error": "场景未配置"})
        with LOCK:
            running = set(docker("ps", "--format", "{{.Names}}").splitlines())
            previous = [
                item for item in SCENES.values()
                if item != scene and item["container"] in running
            ]
            try:
                for item in previous:
                    docker("stop", item["container"])
                activate(scene)
            except (RuntimeError, subprocess.SubprocessError) as error:
                try:
                    if scene["container"] not in running:
                        docker("stop", scene["container"])
                    for item in previous:
                        activate(item)
                except (RuntimeError, subprocess.SubprocessError):
                    return self.reply(503, {"error": "切换失败，原场景恢复失败"})
                return self.reply(503, {"error": str(error)})
            self.reply(200, {"url": scene["url"]})


if __name__ == "__main__":
    import sys
    BIND = sys.argv[1]
    ThreadingHTTPServer((BIND, 8766), Handler).serve_forever()
