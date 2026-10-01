"""Read-only dashboard server (stdlib only). Serves the SPA and /api/state, /healthz."""
from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..monitoring import check_health
from ..storage.db import connect
from .state import build_state

log = logging.getLogger("cryptoalgo.dashboard")
INDEX = Path(__file__).with_name("index.html")


def make_handler(db_path: str, initial_capital: float, health_path: str | None):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            try:
                if self.path in ("/", "/index.html"):
                    self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                elif self.path.startswith("/api/state"):
                    conn = connect(db_path, read_only=True)
                    try:
                        body = json.dumps(build_state(conn, initial_capital), default=str).encode()
                    finally:
                        conn.close()
                    self._send(200, body, "application/json")
                elif self.path == "/healthz":
                    ok, msg = check_health(health_path) if health_path else (True, "no health file configured")
                    self._send(200 if ok else 503, json.dumps({"healthy": ok, "message": msg}).encode(), "application/json")
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # never crash the server on a bad request / locked DB
                log.warning("dashboard error: %s", e)
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

        def log_message(self, fmt, *args):  # quiet
            pass

    return Handler


def serve(db_path: str, host="127.0.0.1", port=8080, initial_capital=100000.0, health_path=None,
          background: bool = False) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(db_path, initial_capital, health_path))
    if background:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    else:
        srv.serve_forever()
    return srv
