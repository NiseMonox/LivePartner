"""Thin wrapper around pymumble for LivePartner's Mumble integration.

M1 scope: connect, ensure channel exists, join it, TX PCM, expose a callback
hook for incoming user voice (so we can wire STT in M3).

Windows note: pymumble pulls opuslib which needs libopus on the system. We piggyback
on the opus.dll shipped by the `pyogg` package and prepend its dir to PATH before
importing pymumble.
"""
from __future__ import annotations

import os
import ssl as _ssl
import sys
from pathlib import Path

# Make sure libopus is findable BEFORE pymumble imports opuslib.
if sys.platform == "win32":
    try:
        import pyogg  # noqa: F401  — only need its install location
        _OPUS_DIR = str(Path(pyogg.__file__).parent)
        if _OPUS_DIR not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _OPUS_DIR + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(_OPUS_DIR)
    except Exception:
        pass

# pymumble 1.6 uses ssl.wrap_socket() which was removed in Python 3.12.
# Provide a compat shim that builds an SSLContext under the hood.
if not hasattr(_ssl, "wrap_socket"):
    if not hasattr(_ssl, "PROTOCOL_TLS"):
        _ssl.PROTOCOL_TLS = _ssl.PROTOCOL_TLS_CLIENT  # type: ignore[attr-defined]
    if not hasattr(_ssl, "PROTOCOL_TLSv1"):
        _ssl.PROTOCOL_TLSv1 = _ssl.PROTOCOL_TLS_CLIENT  # type: ignore[attr-defined]

    def _wrap_socket(sock, keyfile=None, certfile=None, server_side=False,
                     cert_reqs=_ssl.CERT_NONE, ssl_version=None,
                     ca_certs=None, do_handshake_on_connect=True,
                     suppress_ragged_eofs=True, ciphers=None, **_):
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        if certfile:
            ctx.load_cert_chain(certfile, keyfile)
        if ca_certs:
            ctx.load_verify_locations(ca_certs)
        if ciphers:
            ctx.set_ciphers(ciphers)
        return ctx.wrap_socket(sock, do_handshake_on_connect=do_handshake_on_connect,
                               suppress_ragged_eofs=suppress_ragged_eofs)

    _ssl.wrap_socket = _wrap_socket  # type: ignore[attr-defined]

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pymumble_py3 as pm
from pymumble_py3.callbacks import PYMUMBLE_CLBK_SOUNDRECEIVED

log = logging.getLogger(__name__)


@dataclass
class MumbleConfig:
    host: str = "127.0.0.1"
    port: int = 64738
    name: str = "LivePartner"
    password: str = ""
    channel: str = "LivePartner"
    reconnect: bool = True


# Per-chunk callback (user_dict, sound_chunk).
VoiceCallback = Callable[[dict[str, Any], Any], None]

# Per-utterance callback (user_dict, raw 48kHz mono int16 PCM bytes).
# Fires when the user has been silent for `utterance_silence_ms`.
UtteranceCallback = Callable[[dict[str, Any], bytes], None]


class MumbleBot:
    def __init__(
        self,
        cfg: MumbleConfig,
        on_user_voice: VoiceCallback | None = None,
        on_user_utterance: UtteranceCallback | None = None,
        *,
        utterance_silence_ms: int = 800,
        ignore_own_session: bool = True,
        channel_only: bool = True,
        voice_whitelist: set[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.on_user_voice = on_user_voice
        self.on_user_utterance = on_user_utterance
        self.utterance_silence_ms = utterance_silence_ms
        self.ignore_own_session = ignore_own_session
        self.channel_only = channel_only
        # None = no filter (all users). Otherwise a set of lowercased names.
        self._voice_whitelist: set[str] | None = (
            {n.strip().lower() for n in voice_whitelist if n.strip()}
            if voice_whitelist
            else None
        )
        self._client: pm.Mumble | None = None

        # Per-user utterance accumulation state.
        self._utter_buffers: dict[int, bytearray] = {}
        self._utter_timers: dict[int, threading.Timer] = {}
        self._utter_lock = threading.Lock()

    def set_voice_whitelist(self, names: set[str] | list[str] | None) -> None:
        """Live-update the whitelist. None or empty = no filter (all users)."""
        if not names:
            self._voice_whitelist = None
        else:
            self._voice_whitelist = {n.strip().lower() for n in names if n.strip()}
        # Drop any in-flight buffers from users that just got filtered out.
        with self._utter_lock:
            if self._voice_whitelist is not None and self._client is not None:
                for session in list(self._utter_buffers):
                    try:
                        u = self._client.users.get(session, {})
                        name = (u.get("name") or "").lower()
                    except Exception:
                        name = ""
                    if name not in self._voice_whitelist:
                        self._utter_buffers.pop(session, None)
                        t = self._utter_timers.pop(session, None)
                        if t is not None:
                            t.cancel()

    @property
    def voice_whitelist(self) -> set[str] | None:
        return set(self._voice_whitelist) if self._voice_whitelist else None

    def start(self, timeout: float = 10.0) -> None:
        c = pm.Mumble(
            host=self.cfg.host,
            user=self.cfg.name,
            port=self.cfg.port,
            password=self.cfg.password,
            reconnect=self.cfg.reconnect,
        )
        c.set_application_string("LivePartner/0.1")
        want_audio = self.on_user_voice is not None or self.on_user_utterance is not None
        c.set_receive_sound(want_audio)
        if want_audio:
            c.callbacks.set_callback(PYMUMBLE_CLBK_SOUNDRECEIVED, self._on_sound)
        c.start()
        c.is_ready()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if c.connected == pm.constants.PYMUMBLE_CONN_STATE_CONNECTED:
                break
            time.sleep(0.05)
        else:
            c.stop()
            raise TimeoutError(f"Mumble connect timed out after {timeout}s")
        self._client = c
        log.info("mumble: connected to %s:%d as %r", self.cfg.host, self.cfg.port, self.cfg.name)
        self._enter_channel(self.cfg.channel)

    def _enter_channel(self, name: str) -> None:
        """Find channel by name, create it if absent, then move into it.

        pymumble's find_by_name raises UnknownChannelError instead of returning None,
        so the lookups are wrapped accordingly.
        """
        assert self._client is not None

        def _safe_find():
            try:
                return self._client.channels.find_by_name(name)
            except Exception:
                return None

        ch = _safe_find()
        if ch is None:
            try:
                self._client.channels.new_channel(0, name)
            except Exception as e:
                log.warning("mumble: could not create channel %r (%s)", name, e)
            # Server is async — poll briefly for the new channel to appear.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                ch = _safe_find()
                if ch is not None:
                    break
                time.sleep(0.1)
        if ch is None:
            log.warning("mumble: channel %r not found; staying in root", name)
            return
        ch.move_in()
        log.info("mumble: entered channel %r", name)

    # ---------- audio RX ----------
    def _on_sound(self, user: dict[str, Any], sound_chunk: Any) -> None:
        """Single callback registered with pymumble. Dispatches to per-chunk and
        per-utterance handlers, applies self-filter + channel filter."""
        if self._client is None:
            return
        session = user.get("session") if isinstance(user, dict) else None
        if session is None:
            try:
                session = getattr(user, "session", None)
            except Exception:
                session = None

        # Self-filter: never re-ingest the bot's own audio.
        if self.ignore_own_session:
            try:
                me = self._client.users.myself
                if me is not None and session == me.get("session"):
                    return
            except Exception:
                pass

        # Channel-only filter: skip users in a different channel.
        if self.channel_only:
            try:
                my_ch = self._client.my_channel()
                user_ch_id = user.get("channel_id") if isinstance(user, dict) else None
                if my_ch is not None and user_ch_id is not None and user_ch_id != my_ch.get("channel_id"):
                    return
            except Exception:
                pass

        # Whitelist filter: only listen to specific user names.
        if self._voice_whitelist is not None:
            name = ""
            if isinstance(user, dict):
                name = (user.get("name") or "").lower()
            if name not in self._voice_whitelist:
                return

        # 1) per-chunk hook.
        if self.on_user_voice is not None:
            try:
                self.on_user_voice(user, sound_chunk)
            except Exception:
                log.exception("on_user_voice handler raised")

        # 2) per-utterance accumulator.
        if self.on_user_utterance is not None and session is not None:
            pcm = getattr(sound_chunk, "pcm", None)
            if pcm:
                self._accumulate(session, pcm, user)

    def _accumulate(self, session: int, pcm: bytes, user: dict[str, Any]) -> None:
        with self._utter_lock:
            buf = self._utter_buffers.get(session)
            if buf is None:
                buf = bytearray()
                self._utter_buffers[session] = buf
            buf.extend(pcm)

            old = self._utter_timers.get(session)
            if old is not None:
                old.cancel()
            t = threading.Timer(
                self.utterance_silence_ms / 1000.0,
                self._flush_utterance,
                args=(session,),
            )
            t.daemon = True
            self._utter_timers[session] = t
            t.start()

    def _flush_utterance(self, session: int) -> None:
        with self._utter_lock:
            buf = self._utter_buffers.pop(session, None)
            self._utter_timers.pop(session, None)
        if not buf or self.on_user_utterance is None:
            return
        # Look up the user dict by session for the callback's payload.
        user: dict[str, Any] = {"session": session}
        try:
            if self._client is not None:
                for s, u in self._client.users.items():
                    if s == session:
                        user = u
                        break
        except Exception:
            pass
        try:
            self.on_user_utterance(user, bytes(buf))
        except Exception:
            log.exception("on_user_utterance handler raised")

    def send_pcm(self, pcm_bytes: bytes) -> None:
        """Send 48 kHz mono int16 PCM bytes. pymumble handles Opus framing."""
        if self._client is None:
            raise RuntimeError("MumbleBot not started")
        self._client.sound_output.add_sound(pcm_bytes)

    def wait_until_silent(self, max_wait: float = 30.0, poll: float = 0.1) -> None:
        """Block until the TX queue is empty or max_wait elapses."""
        if self._client is None:
            return
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            # pymumble exposes get_buffer_size() — bytes remaining to send
            try:
                remaining = self._client.sound_output.get_buffer_size()
            except Exception:
                remaining = 0
            if remaining <= 0:
                return
            time.sleep(poll)

    @property
    def current_channel_name(self) -> str | None:
        if self._client is None:
            return None
        try:
            ch = self._client.my_channel()
            return ch.get("name") if ch else None
        except Exception:
            return None

    def stop(self) -> None:
        # Cancel any pending utterance timers so they don't fire post-shutdown.
        with self._utter_lock:
            for t in self._utter_timers.values():
                try:
                    t.cancel()
                except Exception:
                    pass
            self._utter_timers.clear()
            self._utter_buffers.clear()
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
            log.info("mumble: stopped")
