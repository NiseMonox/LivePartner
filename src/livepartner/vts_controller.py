"""VTS controller — manages an async pyvts client in a background asyncio thread.

LivePartner runs inside PySide6's Qt event loop. pyvts is websockets-based asyncio,
which can't share Qt's loop. So we start a dedicated asyncio thread and the
public API uses ``run_coroutine_threadsafe`` to submit work — UI / worker threads
never block on VTS round trips.

All public set_* / trigger_* methods are fire-and-forget. They return immediately
and swallow errors (logged via the optional ``logger`` callback) so a transient
VTS disconnect can never crash the UI thread.

Status reads are thread-safe via a single lock. Treat ``status`` as a snapshot.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

try:
    from pyvts import vts as _pyvts_cls
    from pyvts.vts_request import VTSRequest
    _PYVTS_AVAILABLE = True
    _PYVTS_IMPORT_ERR = ""
except Exception as _e:  # pragma: no cover
    _pyvts_cls = None  # type: ignore
    VTSRequest = None  # type: ignore
    _PYVTS_AVAILABLE = False
    _PYVTS_IMPORT_ERR = f"{type(_e).__name__}: {_e}"


# Token persists at .memory/vts_token.txt so subsequent runs skip the VTS
# popup. .memory/ is already gitignored.
_DEFAULT_TOKEN_PATH = (
    Path(__file__).resolve().parent.parent.parent / ".memory" / "vts_token.txt"
)


def pyvts_available() -> bool:
    """True if ``pyvts`` imported cleanly. Callers should still check
    ``VTSController.status.connected`` before assuming the avatar is live."""
    return _PYVTS_AVAILABLE


def pyvts_import_error() -> str:
    return _PYVTS_IMPORT_ERR


@dataclass
class HotkeyInfo:
    id: str
    name: str
    type: str   # ToggleExpression / TriggerAnimation / MoveModel / TwitchClip / ...
    file: str   # for ToggleExpression: the .exp3.json filename (used to drive expr)


@dataclass
class VTSStatus:
    available: bool = True       # pyvts module importable
    connected: bool = False      # websocket open
    authed: bool = False         # plugin authenticated against VTS
    model_name: str = ""         # currently loaded model display name
    hotkeys: list[HotkeyInfo] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    mouth_param: str = ""        # auto-detected: 'MouthOpen' or 'ParamMouthOpenY'
    last_error: str = ""


class VTSController:
    """Threaded wrapper around pyvts.

    Lifecycle::

        ctrl = VTSController(logger=print)
        ctrl.start(host="localhost", port=8001)   # connects + auths in background
        # ... use set_expression / set_parameter / trigger_hotkey ...
        ctrl.stop()                                # graceful close

    All ``set_*`` / ``trigger_*`` methods are fire-and-forget — they queue work
    to the asyncio thread and return. The Qt UI never blocks. Read state via
    the ``status`` property or ``is_connected`` shortcut.
    """

    def __init__(
        self,
        *,
        plugin_name: str = "Eri Director",
        plugin_developer: str = "LivePartner",
        token_path: Path | str = _DEFAULT_TOKEN_PATH,
        logger: Callable[[str], None] | None = None,
        init_hotkey_name: str = "",
    ):
        self.plugin_name = plugin_name
        self.plugin_developer = plugin_developer
        self.token_path = Path(token_path)
        # Name (case-insensitive exact match) of a hotkey to trigger right
        # after authenticate + deactivate-all. Used to restore the user's
        # customized appearance preset which they save into a hotkey (e.g.
        # the Type-H3 model has all the slider customization saved into a
        # hotkey named "Eri"). Empty = no init trigger.
        self.init_hotkey_name = init_hotkey_name
        try:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self._log = logger or (lambda _msg: None)

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._vts: Any | None = None
        self._lock = threading.Lock()
        self._status = VTSStatus(
            available=_PYVTS_AVAILABLE,
            last_error=_PYVTS_IMPORT_ERR,
        )
        # Mouth-update throttle: VTS happily takes 60+ Hz but we only emit at
        # ~30 Hz (cap below) so a hot lip-sync loop doesn't saturate the WS.
        self._last_mouth_emit_ts: float = 0.0
        self._mouth_min_interval = 1.0 / 30.0
        # Track which expression file we last activated so the next swap can
        # deactivate it first. ToggleExpression hotkeys flip on/off — bad when
        # the director wants "set to X" semantics. We use ExpressionActivation
        # Request explicitly so the off→on transition is deterministic.
        self._active_expression_file: str = ""

    # ---------- status ----------

    @property
    def status(self) -> VTSStatus:
        """Snapshot of current state. Safe to read from any thread."""
        with self._lock:
            return VTSStatus(
                available=self._status.available,
                connected=self._status.connected,
                authed=self._status.authed,
                model_name=self._status.model_name,
                hotkeys=list(self._status.hotkeys),
                parameters=list(self._status.parameters),
                mouth_param=self._status.mouth_param,
                last_error=self._status.last_error,
            )

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._status.connected and self._status.authed

    # ---------- lifecycle ----------

    def start(self, host: str = "localhost", port: int = 8001) -> None:
        """Spawn the asyncio thread and submit a connect+authenticate task.

        Returns immediately. UI should poll ``status`` (e.g. via a 500 ms QTimer)
        to refresh its 'connected' label. Errors stay non-fatal — we just stay
        unconnected.
        """
        if not _PYVTS_AVAILABLE:
            with self._lock:
                self._status.last_error = (
                    "pyvts not installed: " + _PYVTS_IMPORT_ERR
                )
            self._log(f"[vts] pyvts unavailable ({_PYVTS_IMPORT_ERR}) — Live2D disabled")
            return
        if self._loop_thread is not None and self._loop_thread.is_alive():
            self._log("[vts] start() called while already running — ignoring")
            return

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
            target=_run_loop, name="vts-asyncio", daemon=True,
        )
        self._loop_thread.start()
        if not loop_ready.wait(2.0):
            self._log("[vts] asyncio thread failed to start within 2 s")
            return

        self._submit(self._async_connect(host, port))

    def stop(self) -> None:
        """Disconnect + tear down the asyncio thread. Safe to call multiple times."""
        if self._loop is None:
            return

        async def _shutdown() -> None:
            # Cancel any in-flight tasks (e.g. a connect() still hanging on
            # ConnectionRefusedError timeout) so they don't leak "Task was
            # destroyed but it is pending" warnings.
            current = asyncio.current_task()
            for t in asyncio.all_tasks():
                if t is not current:
                    t.cancel()
            # Then gracefully close the VTS websocket.
            if self._vts is not None:
                try:
                    await self._vts.close()
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
            self._status.authed = False
        self._loop = None
        self._loop_thread = None
        self._vts = None
        self._active_expression_file = ""

    def reconnect(self, host: str = "localhost", port: int = 8001) -> None:
        """Tear down + reconnect. Use after the user restarts VTS."""
        self.stop()
        self.start(host=host, port=port)

    # ---------- fire-and-forget public API ----------

    def set_parameter(self, param: str, value: float) -> None:
        """Push one Live2D parameter (e.g. MouthOpen) to its model. Throttled
        to ~30 Hz; calls in excess of that are dropped silently. ``param`` is
        the VTS parameter name (auto-detected for mouth via ``status.mouth_param``);
        ``value`` must already be in the parameter's range (0-1 for MouthOpen)."""
        # Throttle hot-path mouth driver
        import time
        now = time.monotonic()
        if now - self._last_mouth_emit_ts < self._mouth_min_interval:
            return
        self._last_mouth_emit_ts = now
        self._submit(self._async_set_parameter(param, value))

    def set_parameters_bulk(self, updates: list[tuple[str, float]]) -> None:
        """Push multiple Live2D parameters in a single VTS request.

        Used by IdleMotion (~20 Hz, 6+ params per tick) to avoid saturating
        the websocket with one round-trip per parameter. Pass a list of
        ``(param_name, value)`` tuples — order doesn't matter.

        No internal throttle here: the caller's tick rate IS the throttle.
        ``set_parameter`` (single-param mouth driver) has its own 30 Hz cap.
        """
        if not updates:
            return
        self._submit(self._async_set_parameters_bulk(updates))

    def set_expression_file(self, file: str) -> None:
        """Activate a Live2D expression by its source filename (e.g.
        'smile_big.exp3.json'). Uses ExpressionActivationRequest with explicit
        active=True (and deactivates the previously-set one first) so swapping
        expressions is deterministic — unlike triggering ToggleExpression hot-
        keys, which flip on/off and would mis-cancel if the director hit the
        same expression twice in a row."""
        if not file:
            return
        self._submit(self._async_activate_expression(file))

    def trigger_hotkey(self, hotkey_id: str) -> None:
        if not hotkey_id:
            return
        self._submit(self._async_trigger_hotkey(hotkey_id))

    def refresh_capabilities(self) -> None:
        """Re-discover hotkeys + parameters. Call after the user loads a
        different model in VTS without restarting the plugin connection."""
        self._submit(self._async_refresh_capabilities())

    # ---------- async impl ----------

    def _submit(self, coro) -> None:
        loop = self._loop
        if loop is None or not loop.is_running():
            # Drop silently; caller might be in shutdown path
            try:
                coro.close()
            except Exception:
                pass
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as e:
            self._log(f"[vts] submit error: {type(e).__name__}: {e}")

    async def _async_connect(self, host: str, port: int) -> None:
        try:
            assert _pyvts_cls is not None
            self._vts = _pyvts_cls(
                plugin_info={
                    "developer": self.plugin_developer,
                    "plugin_name": self.plugin_name,
                    "plugin_icon": None,
                    "authentication_token_path": str(self.token_path),
                },
                vts_api_info={
                    "version": "1.0",
                    "name": "VTubeStudioPublicAPI",
                    "host": host,
                    "port": port,
                },
            )
            await self._vts.connect()
            with self._lock:
                self._status.connected = True
                self._status.last_error = ""
            self._log(f"[vts] connected to {host}:{port}")

            # Try cached token first (silent), fall back to fresh auth (VTS popup).
            # NOTE: pyvts's read_token/write_token are async despite the type
            # signature in some versions reading like sync — they MUST be awaited
            # or you get RuntimeWarning + the token never actually loads/saves.
            try:
                await self._vts.read_token()
            except Exception:
                pass  # no token file yet — request_authenticate will return False
            authed = False
            try:
                authed = await self._vts.request_authenticate()
            except Exception as e:
                self._log(f"[vts] auth check failed: {type(e).__name__}: {e}")
            if not authed:
                self._log("[vts] no cached token — requesting new (check VTS popup)")
                try:
                    await self._vts.request_authenticate_token()
                    await self._vts.write_token()
                    authed = await self._vts.request_authenticate()
                except Exception as e:
                    self._log(f"[vts] new auth failed: {type(e).__name__}: {e}")
            with self._lock:
                self._status.authed = bool(authed)
            if not authed:
                self._log("[vts] authentication FAILED — Live2D disabled")
                return
            self._log("[vts] authenticated")

            await self._async_refresh_capabilities()
            # Clean slate: any expressions left active from a previous session
            # (or manual test) stack on top of new activations, producing a
            # mess of overlays. Deactivate everything before director takes over.
            await self._async_deactivate_all_expressions()
            # Restore user's appearance preset hotkey — Type-H3 (and similar
            # customizable models) saves slider state into a named hotkey.
            # Triggering it here applies the saved look automatically on every
            # connect, so the user doesn't need to manually click it in VTS.
            if self.init_hotkey_name:
                await self._async_trigger_hotkey_by_name(self.init_hotkey_name)
        except Exception as e:
            with self._lock:
                self._status.connected = False
                self._status.authed = False
                self._status.last_error = f"{type(e).__name__}: {e}"
            self._log(f"[vts] connect failed: {type(e).__name__}: {e}")

    async def _async_close(self) -> None:
        if self._vts is None:
            return
        try:
            await self._vts.close()
        except Exception:
            pass

    async def _async_refresh_capabilities(self) -> None:
        """Pull current model's hotkey + parameter lists. Cheap; idempotent."""
        if self._vts is None or VTSRequest is None:
            return
        r = VTSRequest()

        # Hotkeys (expressions + animations live here)
        try:
            resp = await self._vts.request(r.requestHotKeyList())
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            raw = data.get("availableHotkeys", []) or []
            hotkeys = [
                HotkeyInfo(
                    id=h.get("hotkeyID", ""),
                    name=h.get("name", ""),
                    type=h.get("type", ""),
                    file=h.get("file", ""),
                )
                for h in raw
            ]
            model_name = data.get("modelName", "")
            with self._lock:
                self._status.hotkeys = hotkeys
                self._status.model_name = model_name
            n_expr = sum(1 for h in hotkeys if h.type == "ToggleExpression")
            self._log(
                f"[vts] model={model_name!r}  {len(hotkeys)} hotkeys "
                f"({n_expr} expressions)"
            )
        except Exception as e:
            self._log(f"[vts] hotkey list failed: {type(e).__name__}: {e}")

        # Parameters (live + custom). We auto-detect the mouth param name —
        # different rigs use 'MouthOpen' (VTS-mapped) vs 'ParamMouthOpenY' (raw
        # Cubism param). Lip-sync prefers 'MouthOpen' if both exist.
        try:
            resp = await self._vts.request(r.requestTrackingParameterList())
            data = resp.get("data", {}) if isinstance(resp, dict) else {}
            param_raw = (
                (data.get("defaultParameters", []) or []) +
                (data.get("customParameters", []) or [])
            )
            params = [p.get("name", "") for p in param_raw if p.get("name")]
            mouth = ""
            for candidate in ("MouthOpen", "ParamMouthOpenY", "MouthOpenY"):
                if candidate in params:
                    mouth = candidate
                    break
            with self._lock:
                self._status.parameters = params
                self._status.mouth_param = mouth
            if mouth:
                self._log(f"[vts] mouth param auto-detected: {mouth}")
            else:
                self._log(
                    "[vts] WARNING: no MouthOpen / ParamMouthOpenY / MouthOpenY "
                    "in model — lip-sync will be skipped"
                )
        except Exception as e:
            self._log(f"[vts] parameter list failed: {type(e).__name__}: {e}")

    async def _async_set_parameter(self, param: str, value: float) -> None:
        if self._vts is None or VTSRequest is None or not param:
            return
        try:
            r = VTSRequest()
            await self._vts.request(r.requestSetParameterValue(param, float(value)))
        except Exception:
            # Mouth updates fire at ~30 Hz; spamming the log on transient errors
            # is worse than silent failure. The next successful call clears.
            pass

    async def _async_set_parameters_bulk(self, updates: list[tuple[str, float]]) -> None:
        if self._vts is None or VTSRequest is None:
            return
        params = [p for p, _ in updates if p]
        values = [float(v) for p, v in updates if p]
        if not params:
            return
        try:
            r = VTSRequest()
            await self._vts.request(
                r.requestSetMultiParameterValue(params, values)
            )
        except Exception:
            # Same as single-param: idle-motion loop is hot, silent failure is
            # better than log spam on transient WS hiccups.
            pass

    async def _async_deactivate_all_expressions(self) -> None:
        """Query current expression states and deactivate every one that's
        currently active. Run once after connect so prior-session leftovers
        (or manual-test leftovers) don't stack on top of director activations.
        """
        if self._vts is None or VTSRequest is None:
            return
        try:
            r = VTSRequest()
            msg = r.BaseRequest("ExpressionStateRequest", data={"details": False})
            resp = await self._vts.request(msg)
            if not isinstance(resp, dict):
                return
            exprs = (resp.get("data", {}) or {}).get("expressions", []) or []
            active_files = [e.get("file") for e in exprs if e.get("active") and e.get("file")]
            if not active_files:
                return
            self._log(f"[vts] clearing {len(active_files)} stale active expression(s)")
            for f in active_files:
                off = r.BaseRequest(
                    "ExpressionActivationRequest",
                    data={"expressionFile": f, "active": False, "fadeTime": 0.0},
                )
                try:
                    await self._vts.request(off)
                except Exception as e:
                    self._log(f"[vts] clear {f!r} failed: {type(e).__name__}: {e}")
            self._active_expression_file = ""
        except Exception as e:
            self._log(f"[vts] deactivate-all failed: {type(e).__name__}: {e}")

    async def _async_activate_expression(self, file: str) -> None:
        """Switch to the given expression file using ExpressionActivationRequest.

        Sequence: deactivate the previously active expression (if any) → activate
        the new one. This is explicit on/off (unlike ToggleExpression hotkeys
        which flip state, breaking 'set to X' semantics when X was already on).
        """
        if self._vts is None or VTSRequest is None or not file:
            return
        # Sanity check: file should actually exist on the loaded model (or we
        # log a warning so it's diagnosable).
        with self._lock:
            available_files = {
                h.file for h in self._status.hotkeys
                if h.type == "ToggleExpression" and h.file
            }
        if available_files and file not in available_files:
            self._log(f"[vts] expression file {file!r} not in loaded model")
            return

        r = VTSRequest()
        prev = self._active_expression_file
        # Step 1 — deactivate previous so VTS doesn't end up stacking two.
        # Skip if we'd just toggle off the same file (no-op switch).
        if prev and prev != file:
            try:
                msg = r.BaseRequest(
                    "ExpressionActivationRequest",
                    data={
                        "expressionFile": prev,
                        "active": False,
                        "fadeTime": 0.3,
                    },
                )
                resp_off = await self._vts.request(msg)
                err_id = (
                    resp_off.get("data", {}).get("errorID")
                    if isinstance(resp_off, dict) else None
                )
                if err_id:
                    err_msg = resp_off.get("data", {}).get("message", "")
                    self._log(f"[vts] deactivate {prev!r} errorID={err_id} msg={err_msg}")
            except Exception as e:
                self._log(f"[vts] deactivate {prev!r} failed: {type(e).__name__}: {e}")
        # Step 2 — activate the new one.
        try:
            msg = r.BaseRequest(
                "ExpressionActivationRequest",
                data={
                    "expressionFile": file,
                    "active": True,
                    "fadeTime": 0.3,
                },
            )
            resp = await self._vts.request(msg)
            # VTS returns {data: {}} on success, {data: {errorID, message}} on
            # failure. The empty-data success path is undocumented but
            # confirmed via direct probing (see commit history for the trace).
            if not isinstance(resp, dict):
                self._log(f"[vts] activate {file!r} got non-dict resp: {resp!r}")
                return
            data = resp.get("data", {}) or {}
            err_id = data.get("errorID")
            if err_id:
                err_msg = data.get("message", "")
                self._log(f"[vts] activate {file!r} errorID={err_id} msg={err_msg}")
                return
            # Empty data = success.
            self._log(f"[vts] activate {file!r} OK")
            self._active_expression_file = file
        except Exception as e:
            self._log(f"[vts] activate {file!r} failed: {type(e).__name__}: {e}")

    async def _async_trigger_hotkey(self, hotkey_id: str) -> None:
        if self._vts is None or VTSRequest is None:
            return
        try:
            r = VTSRequest()
            await self._vts.request(r.requestTriggerHotKey(hotkey_id))
        except Exception as e:
            self._log(f"[vts] trigger hotkey failed: {type(e).__name__}: {e}")

    async def _async_trigger_hotkey_by_name(self, name: str) -> bool:
        """Find a hotkey whose display name matches (case-insensitive exact)
        and trigger it. Returns True on found+triggered, False otherwise."""
        if not name or self._vts is None or VTSRequest is None:
            return False
        with self._lock:
            hotkeys = list(self._status.hotkeys)
        target_id = ""
        target_type = ""
        for h in hotkeys:
            if h.name.lower() == name.lower():
                target_id = h.id
                target_type = h.type
                break
        if not target_id:
            self._log(f"[vts] init hotkey {name!r} not found on this model")
            return False
        try:
            r = VTSRequest()
            await self._vts.request(r.requestTriggerHotKey(target_id))
            self._log(f"[vts] init hotkey triggered: {name!r} (type={target_type})")
            return True
        except Exception as e:
            self._log(f"[vts] init hotkey {name!r} failed: {type(e).__name__}: {e}")
            return False
