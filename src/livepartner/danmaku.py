"""Bilibili 弹幕 (live chat) source.

Bridges ``bilibili_api.live.LiveDanmaku`` (asyncio, websockets) into a
Qt-friendly thread-safe callback. Each chat message → ``on_danmaku(name, text)``
fires on the asyncio thread; the Qt UI relays via a signal.

Same threading pattern as ``vts_controller.VTSController``: background asyncio
loop on a daemon thread, no blocking on UI thread, all errors swallowed +
logged.

Optional dependency. If ``bilibili_api`` isn't installed, ``available()``
returns False and the UI tab can still load (informational only).
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Callable

try:
    from bilibili_api import live as _bili_live
    _BILI_AVAILABLE = True
    _BILI_IMPORT_ERR = ""
except Exception as _e:  # pragma: no cover
    _bili_live = None  # type: ignore
    _BILI_AVAILABLE = False
    _BILI_IMPORT_ERR = f"{type(_e).__name__}: {_e}"


def available() -> bool:
    return _BILI_AVAILABLE


def import_error() -> str:
    return _BILI_IMPORT_ERR


@dataclass
class DanmakuStatus:
    available: bool = True
    connected: bool = False
    room_id: int = 0
    last_error: str = ""


# Callback type: (viewer_display_name: str, message_text: str) -> None.
# The viewer_display_name is the chatter's username, message_text is the chat
# content. The callback fires on the asyncio thread — the Qt UI should bridge
# via a Signal to the main thread.
DanmakuCallback = Callable[[str, str], None]


class BilibiliDanmakuSource:
    """Connects to a Bilibili live room and feeds 弹幕 to a callback.

    Lifecycle::

        src = BilibiliDanmakuSource(on_message=lambda n, t: print(n, t),
                                    logger=print)
        src.start(room_id=12345)
        # ... chat messages stream in ...
        src.stop()

    Both ``start`` and ``stop`` are sync and return immediately. Status reads
    via ``status`` (thread-safe snapshot).
    """

    def __init__(
        self,
        *,
        on_message: DanmakuCallback,
        logger: Callable[[str], None] | None = None,
    ):
        self._on_message = on_message
        self._log = logger or (lambda _m: None)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._client: object | None = None
        self._lock = threading.Lock()
        self._status = DanmakuStatus(
            available=_BILI_AVAILABLE,
            last_error=_BILI_IMPORT_ERR,
        )

    @property
    def status(self) -> DanmakuStatus:
        with self._lock:
            return DanmakuStatus(
                available=self._status.available,
                connected=self._status.connected,
                room_id=self._status.room_id,
                last_error=self._status.last_error,
            )

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._status.connected

    def start(self, room_id: int) -> None:
        if not _BILI_AVAILABLE:
            with self._lock:
                self._status.last_error = (
                    "bilibili_api not installed: " + _BILI_IMPORT_ERR
                )
            self._log(
                f"[danmaku] bilibili_api unavailable ({_BILI_IMPORT_ERR}) — disabled"
            )
            return
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._log("[danmaku] already running — ignoring start()")
            return

        with self._lock:
            self._status.room_id = int(room_id)

        loop_ready = threading.Event()

        def _run_loop() -> None:
            loop = asyncio.new_event_loop()
            self._loop = loop
            asyncio.set_event_loop(loop)
            loop_ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        self._loop_thread = threading.Thread(
            target=_run_loop, name="danmaku-asyncio", daemon=True,
        )
        self._loop_thread.start()
        if not loop_ready.wait(2.0):
            self._log("[danmaku] asyncio thread failed to start")
            return

        self._submit(self._async_connect(int(room_id)))

    def stop(self) -> None:
        if self._loop is None:
            return

        async def _shutdown() -> None:
            current = asyncio.current_task()
            for t in asyncio.all_tasks():
                if t is not current:
                    t.cancel()
            if self._client is not None:
                try:
                    await self._client.disconnect()
                except Exception:
                    pass

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            fut.result(timeout=3.0)
        except Exception:
            pass
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)
        with self._lock:
            self._status.connected = False
        self._loop = None
        self._loop_thread = None
        self._client = None

    # ---------- internals ----------

    def _submit(self, coro) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            try:
                coro.close()
            except Exception:
                pass
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:
            self._log(f"[danmaku] submit error: {type(e).__name__}: {e}")

    async def _async_connect(self, room_id: int) -> None:
        assert _bili_live is not None
        try:
            client = _bili_live.LiveDanmaku(room_id)

            # Register the DANMU_MSG handler — Bilibili's event name for plain
            # text 弹幕. bilibili-api-python normalizes raw 弹幕 packets so the
            # name + text are easy to extract from event["data"]["info"].
            @client.on("DANMU_MSG")
            async def _on_danmu(event):
                try:
                    info = event.get("data", {}).get("info", [])
                    # info schema: info[1] = text, info[2][1] = sender name
                    text = ""
                    name = ""
                    if isinstance(info, list) and len(info) >= 3:
                        text = str(info[1]) if info[1] is not None else ""
                        sender = info[2]
                        if isinstance(sender, list) and len(sender) >= 2:
                            name = str(sender[1]) or ""
                    text = text.strip()
                    if not text or not name:
                        return
                    # Hand off to the consumer callback. The callback should
                    # marshal to the Qt UI thread via a Signal — we DON'T do
                    # Qt work here (this is the asyncio thread).
                    try:
                        self._on_message(name, text)
                    except Exception as e:
                        self._log(f"[danmaku] callback err: {type(e).__name__}: {e}")
                except Exception as e:
                    self._log(f"[danmaku] parse err: {type(e).__name__}: {e}")

            self._client = client
            with self._lock:
                self._status.connected = False
                self._status.last_error = ""
            self._log(f"[danmaku] connecting to room {room_id} …")
            # connect() is a long-running coroutine that pumps messages.
            # When it returns (disconnect / error) we mark connected=False.
            with self._lock:
                self._status.connected = True
            self._log(f"[danmaku] connected to room {room_id}")
            await client.connect()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            with self._lock:
                self._status.connected = False
                self._status.last_error = f"{type(e).__name__}: {e}"
            self._log(f"[danmaku] connect failed: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._status.connected = False
