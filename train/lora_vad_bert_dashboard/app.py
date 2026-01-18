import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HOST = "127.0.0.1"
PORT = 8022

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]
SCRIPT_PATH = REPO_DIR / "train" / "lora_vad_bert.py"

LOSS_RE = re.compile(
    r"^\[step (?P<step>\d+)\] loss=(?P<loss>[-+0-9.eE]+) "
    r"zh=(?P<zh>[-+0-9.eE]+) align=(?P<align>[-+0-9.eE]+) "
    r"retain=(?P<retain>[-+0-9.eE]+)"
)
EVAL_RE = re.compile(
    r"^\[step (?P<step>\d+)\] E_old=(?P<e_old>[-+0-9.eE]+) "
    r"E_zh=(?P<e_zh>[-+0-9.eE]+)"
)

process_lock = threading.Lock()
events_lock = threading.Lock()
event_counter = 0
events = deque(maxlen=2000)

state = {
    "running": False,
    "pid": None,
    "cmd": "",
    "started_at": None,
    "last_metrics": {},
}
process = None


def emit_event(payload):
    global event_counter
    with events_lock:
        event_counter += 1
        payload["id"] = event_counter
        events.append(payload)


def update_state(metrics=None, running=None, cmd=None, pid=None):
    if running is not None:
        state["running"] = running
    if cmd is not None:
        state["cmd"] = cmd
    if pid is not None:
        state["pid"] = pid
    if metrics:
        state["last_metrics"].update(metrics)


def parse_metrics(line):
    match = LOSS_RE.match(line)
    if match:
        return {
            "step": int(match.group("step")),
            "loss": float(match.group("loss")),
            "zh": float(match.group("zh")),
            "align": float(match.group("align")),
            "retain": float(match.group("retain")),
        }
    match = EVAL_RE.match(line)
    if match:
        return {
            "step": int(match.group("step")),
            "E_old": float(match.group("e_old")),
            "E_zh": float(match.group("e_zh")),
        }
    return None


def read_loop(proc):
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        metrics = parse_metrics(line)
        if metrics:
            update_state(metrics=metrics)
            emit_event({"type": "metrics", "metrics": metrics})
        emit_event({"type": "log", "line": line})
    code = proc.poll()
    update_state(running=False, pid=None)
    emit_event({"type": "exit", "code": code})


def start_process(args):
    global process
    with process_lock:
        if state["running"]:
            return False, "already_running"
        cmd = [sys.executable, "-u", str(SCRIPT_PATH)] + args
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        update_state(running=True, pid=proc.pid, cmd=" ".join(cmd))
        state["started_at"] = time.time()
        emit_event({"type": "state", "state": state})
        thread = threading.Thread(target=read_loop, args=(proc,), daemon=True)
        thread.start()
        process = proc
        return True, None


def stop_process():
    global process
    with process_lock:
        proc = process
        if not proc or proc.poll() is not None:
            return False, "not_running"
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        process = None
        return True, None


def make_json_response(handler, payload, status=200):
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def read_json(handler):
    length = int(handler.headers.get("Content-Length", 0))
    raw = handler.rfile.read(length) if length else b"{}"
    return json.loads(raw.decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/stream":
            query = parse_qs(parsed.query)
            cursor = int((query.get("cursor") or ["0"])[0])

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            while True:
                with events_lock:
                    pending = [e for e in events if e["id"] > cursor]
                if pending:
                    for event in pending:
                        data = json.dumps(event, ensure_ascii=True).encode("utf-8")
                        self.wfile.write(b"data: " + data + b"\n\n")
                        self.wfile.flush()
                        cursor = event["id"]
                else:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                time.sleep(0.5)

        if parsed.path == "/api/state":
            return make_json_response(self, {"state": state})

        if self.path == "/":
            data = (BASE_DIR / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/app.js":
            data = (BASE_DIR / "app.js").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/styles.css":
            data = (BASE_DIR / "styles.css").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/css")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/start":
            body = read_json(self)
            args_text = (body.get("args") or "").strip()
            try:
                args = shlex.split(args_text, posix=False) if args_text else []
            except ValueError:
                return make_json_response(self, {"error": "bad_args"}, status=400)
            ok, err = start_process(args)
            if not ok:
                return make_json_response(self, {"error": err}, status=400)
            return make_json_response(self, {"status": "started", "state": state})

        if self.path == "/api/stop":
            ok, err = stop_process()
            if not ok:
                return make_json_response(self, {"error": err}, status=400)
            return make_json_response(self, {"status": "stopped"})

        self.send_response(404)
        self.end_headers()


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
