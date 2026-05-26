"""Local HTTP server that serves a transparent subtitle page for OBS Browser Source.

Architecture:
- GET /            → serves the HTML/CSS/JS page
- GET /events      → text/event-stream (SSE), one line per AI utterance
- main_window pushes new lines via push_subtitle(); each connected SSE client
  receives the same payload near-instantly.

Why SSE not WebSocket: SSE is built on plain HTTP, supports auto-reconnect in
the browser, and stdlib http.server can implement it in ~30 lines. WebSocket
would require an extra dep (websockets or aiohttp) for one tiny channel.

The server runs in a daemon thread; UI controls start/stop. OBS Browser Source
URL is just http://127.0.0.1:<port>/ — typical setup: 1920x1080, transparent,
30 fps, "Refresh browser when scene becomes active" off (we want it to stay
connected across scene switches).
"""
from __future__ import annotations

import html
import json
import queue
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_PORT = 7002


def lan_ip_guess() -> str:
    """Best-effort: open a UDP socket toward a public IP and read back the
    chosen source address. Doesn't actually send anything. Returns a string
    like ``192.168.x.x``, or ``127.0.0.1`` on failure.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


_OVERLAY_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Eri 字幕</title>
<style>
  html, body {
    margin: 0; padding: 0;
    background: transparent;
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
    overflow: hidden;
    width: 100vw; height: 100vh;
  }
  #subtitle {
    position: fixed;
    bottom: 8%;
    left: 50%;
    transform: translateX(-50%);
    max-width: 80vw;
    text-align: center;
    color: white;
    font-size: __FONT_SIZE__px;
    font-weight: 700;
    line-height: 1.3;
    /* 8-direction text-stroke for legibility on any background */
    text-shadow:
      -3px -3px 0 #000, 3px -3px 0 #000,
      -3px  3px 0 #000, 3px  3px 0 #000,
       0   -3px 0 #000, 0    3px 0 #000,
      -3px  0   0 #000, 3px  0   0 #000,
       2px  2px 6px rgba(0,0,0,0.6);
    transition: opacity 0.25s ease-out, transform 0.25s ease-out;
    opacity: 0;
    transform: translateX(-50%) translateY(8px);
  }
  #subtitle.visible {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }
  /* tiny debug strip — disable by removing 'debug' class on body */
  #status {
    position: fixed; top: 6px; right: 8px;
    color: rgba(255,255,255,0.4);
    font-size: 11px;
    text-shadow: 0 0 2px #000;
    display: none;
  }
  body.debug #status { display: block; }
</style>
</head><body>
<div id="subtitle"></div>
<div id="status">●</div>
<script>
  const el = document.getElementById("subtitle");
  const statusEl = document.getElementById("status");
  let hideTimer = null;
  const HOLD_MS = __HOLD_MS__;

  function showLine(text) {
    el.textContent = text;
    el.classList.add("visible");
    if (hideTimer) clearTimeout(hideTimer);
    if (HOLD_MS > 0) {
      hideTimer = setTimeout(() => el.classList.remove("visible"), HOLD_MS);
    }
  }

  function connect() {
    statusEl.style.color = "rgba(255,200,0,0.6)";
    const es = new EventSource("/events");
    es.onopen = () => { statusEl.style.color = "rgba(0,255,0,0.6)"; };
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data && typeof data.text === "string" && data.text) showLine(data.text);
        if (data && data.clear) {
          el.classList.remove("visible");
        }
      } catch (e) {}
    };
    es.onerror = () => {
      statusEl.style.color = "rgba(255,0,0,0.6)";
      es.close();
      setTimeout(connect, 2000);
    };
  }
  connect();

  // Toggle debug strip with ?debug=1
  if (location.search.includes("debug=1")) {
    document.body.classList.add("debug");
  }
</script>
</body></html>
"""


class OverlayServer:
    """Threading HTTP server with broadcast-to-all-clients SSE channel."""

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        *,
        bind: str = "0.0.0.0",
        default_font_size: int = 48,
        default_hold_ms: int = 6000,
    ) -> None:
        self.port = port
        self.bind = bind
        self.default_font_size = default_font_size
        self.default_hold_ms = default_hold_ms
        self._clients: set[queue.Queue] = set()
        self._clients_lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self._server is not None

    @property
    def url(self) -> str:
        """URL for clients on THIS machine — points at loopback."""
        return f"http://127.0.0.1:{self.port}/"

    @property
    def lan_url(self) -> str:
        """URL for clients on the LAN (e.g. OBS on the game PC). Returns
        ``http://127.0.0.1:<port>/`` if binding was set to loopback only."""
        if self.bind in ("127.0.0.1", "localhost"):
            return self.url
        return f"http://{lan_ip_guess()}:{self.port}/"

    def start(self) -> None:
        if self.is_running:
            return
        handler = _make_handler(self)
        self._server = ThreadingHTTPServer((self.bind, self.port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="livepartner-overlay-http",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        # Disconnect every active client by enqueueing a sentinel.
        with self._clients_lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(None)
                except Exception:
                    pass
            self._clients.clear()
        srv = self._server
        if srv is not None:
            try:
                srv.shutdown()
                srv.server_close()
            except Exception:
                pass
        self._server = None
        self._thread = None

    def push_subtitle(self, text: str, **extra: Any) -> None:
        """Broadcast a subtitle line to all connected SSE clients."""
        payload = json.dumps({"text": text, **extra}, ensure_ascii=False)
        self._broadcast(payload)

    def push_clear(self) -> None:
        self._broadcast(json.dumps({"clear": True}))

    def _broadcast(self, payload: str) -> None:
        with self._clients_lock:
            for q in list(self._clients):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    # Drop frame if client is too slow — better than blocking.
                    pass

    def _register_client(self, q: queue.Queue) -> None:
        with self._clients_lock:
            self._clients.add(q)

    def _unregister_client(self, q: queue.Queue) -> None:
        with self._clients_lock:
            self._clients.discard(q)

    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    def render_html(self, font_size: int | None = None, hold_ms: int | None = None) -> str:
        fs = font_size if font_size is not None else self.default_font_size
        hm = hold_ms if hold_ms is not None else self.default_hold_ms
        return (
            _OVERLAY_HTML
            .replace("__FONT_SIZE__", str(int(fs)))
            .replace("__HOLD_MS__", str(int(hm)))
        )


def _make_handler(server: OverlayServer):
    class _Handler(BaseHTTPRequestHandler):
        # Silence the default per-request stderr access log — daemon thread,
        # noisy, not useful for production use.
        def log_message(self, format, *args):
            return

        def do_GET(self) -> None:  # noqa: N802 — http.server signature
            path = self.path
            # Strip query string for routing.
            base = path.split("?", 1)[0]
            qs = path.split("?", 1)[1] if "?" in path else ""

            if base in ("/", "/overlay", "/overlay.html"):
                self._serve_html(qs)
            elif base == "/events":
                self._serve_sse()
            elif base == "/status":
                self._serve_status()
            else:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

        def _serve_html(self, qs: str) -> None:
            # Allow URL query overrides: ?size=64&hold=4
            font_size = server.default_font_size
            hold_ms = server.default_hold_ms
            for pair in qs.split("&") if qs else []:
                if "=" not in pair:
                    continue
                k, v = pair.split("=", 1)
                if k == "size":
                    try:
                        font_size = max(8, int(v))
                    except ValueError:
                        pass
                elif k == "hold":
                    try:
                        hold_ms = max(0, int(float(v) * 1000))
                    except ValueError:
                        pass
            body = server.render_html(font_size, hold_ms).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _serve_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            q: queue.Queue = queue.Queue(maxsize=20)
            server._register_client(q)
            try:
                # Tell the client they're connected (so onopen fires reliably
                # behind some proxies).
                self.wfile.write(b": connected\n\n")
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15.0)
                    except queue.Empty:
                        # SSE comment line keeps the connection from idle-closing
                        self.wfile.write(b": heartbeat\n\n")
                        self.wfile.flush()
                        continue
                    if msg is None:
                        break  # server shutdown sentinel
                    self.wfile.write(b"data: ")
                    self.wfile.write(msg.encode("utf-8"))
                    self.wfile.write(b"\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Browser navigated away / OBS swapped scenes.
                pass
            finally:
                server._unregister_client(q)

        def _serve_status(self) -> None:
            body = json.dumps({
                "running": server.is_running,
                "clients": server.client_count(),
                "url": server.url,
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return _Handler
