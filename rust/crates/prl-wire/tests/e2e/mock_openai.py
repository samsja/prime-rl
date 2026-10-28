"""A minimal OpenAI-compatible chat endpoint (streaming and not) for prl-wire's e2e test."""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONTENT = "Let me think. The answer is \\boxed{18}."
USAGE = {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        base = {"id": "cmpl-1", "created": int(time.time()), "model": body.get("model", "m")}
        if not body.get("stream"):
            message = {"role": "assistant", "content": CONTENT}
            reply = {**base, "object": "chat.completion", "usage": USAGE,
                     "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        chunk = {**base, "object": "chat.completion.chunk"}
        events = [
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": CONTENT[:10]}, "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {"content": CONTENT[10:]}, "finish_reason": None}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**chunk, "choices": [], "usage": USAGE},
        ]
        data = b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = json.dumps({"object": "list", "data": [{"id": "m", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
