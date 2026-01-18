import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

HOST = "127.0.0.1"
PORT = 8010
MODEL_NAME = "Qwen/Qwen3-0.6B"
END_THINK_TOKEN_ID = 151668

BASE_DIR = Path(__file__).resolve().parent

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype="auto",
    device_map="auto",
)
model.eval()

model_lock = threading.Lock()
sessions = {}


def strip_think_tags(text):
    if not text:
        return text
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text


class SessionState:
    def __init__(self, prompt, system_prompt=None, allow_end_think=True, allow_eos=True):
        self.messages = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        self.messages.append({"role": "user", "content": prompt})
        self.base_prompt = tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        self.prompt_has_think = self.base_prompt.rstrip().endswith("<think>")
        inputs = tokenizer([self.base_prompt], return_tensors="pt").to(model.device)
        self.input_ids = inputs.input_ids
        self.attention_mask = inputs.attention_mask

        self.past_key_values = None
        self.next_input_ids = None
        self.first_step = True
        self.paused = False
        self.finished = False
        self.allow_end_think = allow_end_think
        self.allow_eos = allow_eos

        self.thinking_text = ""
        self.reply_preview = ""

    def step(self):
        if self.finished or self.paused:
            return None

        with torch.no_grad():
            if self.first_step:
                outputs = model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    use_cache=True,
                )
                self.past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                if not self.allow_end_think:
                    logits[:, END_THINK_TOKEN_ID] = float("-inf")
                if not self.allow_eos and tokenizer.eos_token_id is not None:
                    logits[:, tokenizer.eos_token_id] = float("-inf")
                next_token = torch.argmax(logits, dim=-1)
                self.first_step = False
            else:
                outputs = model(
                    input_ids=self.next_input_ids,
                    use_cache=True,
                    past_key_values=self.past_key_values,
                )
                self.past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                if not self.allow_end_think:
                    logits[:, END_THINK_TOKEN_ID] = float("-inf")
                if not self.allow_eos and tokenizer.eos_token_id is not None:
                    logits[:, tokenizer.eos_token_id] = float("-inf")
                next_token = torch.argmax(logits, dim=-1)

        token_id = next_token.item()
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            if self.allow_eos:
                self.finished = True
                return None

        self.next_input_ids = next_token.unsqueeze(0)
        raw_piece = tokenizer.decode(next_token, skip_special_tokens=True)

        if token_id == END_THINK_TOKEN_ID or "</think>" in raw_piece:
            if self.allow_end_think:
                piece = strip_think_tags(raw_piece)
                if piece:
                    self.thinking_text += piece
                self.paused = True
                self.reply_preview = self.preview_reply()
                return piece or ""
            return None

        piece = strip_think_tags(raw_piece)
        if piece:
            self.thinking_text += piece
        return piece or ""

    def think_until_pause(self, max_steps):
        for _ in range(max_steps):
            if self.paused or self.finished:
                break
            self.step()

    def apply_thinking(self, edited_think):
        edited = (edited_think or "").replace("<think>", "").replace("</think>", "")
        if self.prompt_has_think:
            prefix = edited
        else:
            prefix = "<think>" + edited

        full_text = self.base_prompt + prefix
        inputs = tokenizer([full_text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                use_cache=True,
            )

        self.past_key_values = outputs.past_key_values
        self.next_input_ids = inputs.input_ids[:, -1:]
        self.first_step = False
        self.paused = False
        self.finished = False
        self.thinking_text = edited
        self.reply_preview = ""

    def preview_reply(self, max_new_tokens=512):
        if not self.thinking_text:
            return ""
        if self.prompt_has_think:
            prefix = self.thinking_text
        else:
            prefix = "<think>" + self.thinking_text
        full_text = self.base_prompt + prefix + "</think>"
        inputs = tokenizer([full_text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
            )
        output_ids = generated_ids[0][inputs.input_ids.shape[-1] :]
        return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def set_allow_end_think(self, allow_end_think):
        self.allow_end_think = bool(allow_end_think)

    def set_allow_eos(self, allow_eos):
        self.allow_eos = bool(allow_eos)


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
        if parsed.path == "/api/think_stream":
            query = parse_qs(parsed.query)
            session_id = (query.get("session_id") or [""])[0]
            session = sessions.get(session_id)
            if not session:
                self.send_response(404)
                self.end_headers()
                return
            max_steps = int((query.get("max_steps") or [512])[0])
            max_steps = max(1, min(max_steps, 4096))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            def send_event(payload):
                data = json.dumps(payload).encode("utf-8")
                self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()

            send_event(
                {
                    "type": "init",
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                }
            )

            with model_lock:
                for _ in range(max_steps):
                    if session.paused or session.finished:
                        break
                    piece = session.step()
                    if piece:
                        send_event({"type": "token", "text": piece})
                    if session.paused or session.finished:
                        break

            send_event(
                {
                    "type": "state",
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                }
            )
            return

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
            prompt = (body.get("prompt") or "").strip()
            system_prompt = (body.get("system_prompt") or "").strip()
            allow_end_think = bool(body.get("allow_end_think", True))
            allow_eos = bool(body.get("allow_eos", True))
            if not prompt:
                return make_json_response(self, {"error": "prompt_required"}, status=400)
            session_id = uuid.uuid4().hex
            sessions[session_id] = SessionState(prompt, system_prompt, allow_end_think, allow_eos)
            return make_json_response(
                self,
                {
                    "session_id": session_id,
                    "thinking": "",
                    "reply": "",
                    "paused": False,
                    "finished": False,
                },
            )

        if self.path == "/api/step":
            body = read_json(self)
            session_id = body.get("session_id")
            session = sessions.get(session_id)
            if not session:
                return make_json_response(self, {"error": "invalid_session"}, status=404)

            with model_lock:
                session.step()

            return make_json_response(
                self,
                {
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                },
            )

        if self.path == "/api/think":
            body = read_json(self)
            session_id = body.get("session_id")
            session = sessions.get(session_id)
            if not session:
                return make_json_response(self, {"error": "invalid_session"}, status=404)
            max_steps = int(body.get("max_steps") or 512)
            max_steps = max(1, min(max_steps, 4096))

            with model_lock:
                session.think_until_pause(max_steps)

            return make_json_response(
                self,
                {
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                },
            )

        if self.path == "/api/config":
            body = read_json(self)
            session_id = body.get("session_id")
            session = sessions.get(session_id)
            if not session:
                return make_json_response(self, {"error": "invalid_session"}, status=404)
            allow_end_think = bool(body.get("allow_end_think", True))
            allow_eos = bool(body.get("allow_eos", True))
            session.set_allow_end_think(allow_end_think)
            session.set_allow_eos(allow_eos)
            return make_json_response(
                self,
                {
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                },
            )

        if self.path == "/api/apply":
            body = read_json(self)
            session_id = body.get("session_id")
            session = sessions.get(session_id)
            if not session:
                return make_json_response(self, {"error": "invalid_session"}, status=404)
            edited_think = body.get("thinking", "")

            with model_lock:
                session.apply_thinking(edited_think)

            return make_json_response(
                self,
                {
                    "thinking": session.thinking_text,
                    "reply": session.reply_preview,
                    "paused": session.paused,
                    "finished": session.finished,
                },
            )

        self.send_response(404)
        self.end_headers()


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Server running at http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    main()
