import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer

HOST = "127.0.0.1"
PORT = 8000
MODEL_NAME = "Qwen/Qwen3-0.6B"
END_THINK_TOKEN_ID = 151668

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
    def __init__(self, prompt):
        self.messages = [{"role": "user", "content": prompt}]
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
        self.in_think = True
        self.finished = False

        self.generated_text = ""
        self.thinking_text = ""
        self.content_text = ""

    def step(self):
        if self.finished:
            return

        with torch.no_grad():
            if self.first_step:
                outputs = model(
                    input_ids=self.input_ids,
                    attention_mask=self.attention_mask,
                    use_cache=True,
                )
                self.past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                self.first_step = False
            else:
                outputs = model(
                    input_ids=self.next_input_ids,
                    use_cache=True,
                    past_key_values=self.past_key_values,
                )
                self.past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)

        token_id = next_token.item()
        if tokenizer.eos_token_id is not None and token_id == tokenizer.eos_token_id:
            self.finished = True
            return

        self.next_input_ids = next_token.unsqueeze(0)
        piece = tokenizer.decode(next_token, skip_special_tokens=True)

        if token_id == END_THINK_TOKEN_ID or "</think>" in piece:
            self.in_think = False
            piece = strip_think_tags(piece)
            if piece:
                self.content_text += piece
            return

        piece = strip_think_tags(piece)
        self.generated_text += piece
        if self.in_think:
            self.thinking_text += piece
        else:
            self.content_text += piece

    def apply_thinking(self, edited_think, close_think):
        edited = (edited_think or "").replace("<think>", "").replace("</think>", "")
        if self.prompt_has_think:
            prefix = edited
        else:
            prefix = "<think>" + edited
        if close_think:
            prefix += "</think>"

        self.generated_text = prefix
        full_text = self.base_prompt + self.generated_text
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
        self.in_think = not close_think
        self.finished = False
        self.thinking_text = edited
        self.content_text = ""


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
        if self.path == "/":
            with open("web/index.html", "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/app.js":
            with open("web/app.js", "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if self.path == "/styles.css":
            with open("web/styles.css", "rb") as f:
                data = f.read()
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
            if not prompt:
                return make_json_response(self, {"error": "prompt_required"}, status=400)
            session_id = uuid.uuid4().hex
            sessions[session_id] = SessionState(prompt)
            return make_json_response(
                self,
                {
                    "session_id": session_id,
                    "thinking": "",
                    "content": "",
                    "in_think": True,
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
                    "content": session.content_text,
                    "in_think": session.in_think,
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
            close_think = bool(body.get("close_think"))

            with model_lock:
                session.apply_thinking(edited_think, close_think)

            return make_json_response(
                self,
                {
                    "thinking": session.thinking_text,
                    "content": session.content_text,
                    "in_think": session.in_think,
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
