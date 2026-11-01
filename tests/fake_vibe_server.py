#!/usr/bin/env python3
"""Minimal stand-in for `vibe-server serve` used by the tests (same CLI handshake and routes)."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

state = {"model": None}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})
        if self.path == "/ready":
            if state["model"]:
                return self._send(200, {"status": "ready", "model": state["model"]})
            return self._send(503, {"status": "not_ready"})
        self._send(404, {})

    def do_POST(self):
        data = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/v1/models/load":
            state["model"] = json.loads(data)["path"]
            return self._send(200, {"status": "loaded", "model": state["model"]})
        if self.path == "/v1/audio/transcriptions":
            assert b'name="file"' in data and b'verbose_json' in data
            ru = b'name="language"\r\n\r\nru' in data
            segs = ([{"start": 0.0, "end": 2.5, "text": " Привет, мир.", "no_speech_prob": 0.0}] if ru else
                    [{"start": 0.0, "end": 2.5, "text": " Hello world.", "no_speech_prob": 0.0},
                     {"start": 2.5, "end": 3661.2, "text": " Second line.", "no_speech_prob": 0.0}])
            return self._send(200, {"text": "".join(s["text"] for s in segs).strip(), "segments": segs})
        self._send(404, {})


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 0
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(json.dumps({"status": "ready", "port": srv.server_address[1]}), flush=True)
    srv.serve_forever()
