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


# Callback signature: (user_dict, sound_chunk) → None
# user_dict has session id, name, etc. sound_chunk is pymumble's SoundChunk with .pcm (int16 mono 48k).
VoiceCallback = Callable[[dict[str, Any], Any], None]


class MumbleBot:
    def __init__(self, cfg: MumbleConfig, on_user_voice: VoiceCallback | None = None) -> None:
        self.cfg = cfg
        self.on_user_voice = on_user_voice
        self._client: pm.Mumble | None = None

    def start(self, timeout: float = 10.0) -> None:
        c = pm.Mumble(
            host=self.cfg.host,
            user=self.cfg.name,
            port=self.cfg.port,
            password=self.cfg.password,
            reconnect=self.cfg.reconnect,
        )
        c.set_application_string("LivePartner/0.1")
        c.set_receive_sound(self.on_user_voice is not None)
        if self.on_user_voice is not None:
            c.callbacks.set_callback(PYMUMBLE_CLBK_SOUNDRECEIVED, self.on_user_voice)
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
        if self._client is not None:
            try:
                self._client.stop()
            except Exception:
                pass
            self._client = None
            log.info("mumble: stopped")
