"""Lip-sync — drives Live 2D MouthOpen from streaming TTS PCM.

Why Python-side and not VTS's built-in audio driver:
- VTS Audio Input listens to a system output device. Our TTS goes straight to
  the Mumble TX queue, never plays locally. Playing through PyAudio just so VTS
  can hear it = doubled latency + audio drift.

# Why this is more than just "RMS in, mouth out"

Qwen3-TTS generates audio FASTER than real-time (RTF ~1.5: ~3.5 s of audio
arrives in ~2.3 s of wall time). If we naively push mouth values as chunks
arrive, lip animation completes 1+ seconds before audio finishes playing
through Mumble — the avatar closes its mouth mid-sentence.

Fix: schedule each chunk's mouth value at AUDIO realtime. First chunk applies
when the first chunk arrives; chunk N applies at ``t_first + sum(durations
of chunks 0..N-1)``. Background ``threading.Timer``s sleep for the right
delta; the calling TTS thread stays free to push more chunks to Mumble at
full speed (so Mumble's TX buffer stays healthy).

Each ~83 ms chunk (48 kHz int16, ~4000 samples) → RMS → nonlinear stretch +
asymmetric exponential smoothing → ``vts.set_parameter("MouthOpen", value)``.

Smoothing keeps motion natural:
- **attack** (rise): high — mouth opens quickly when a syllable starts
- **release** (fall): low — mouth lingers open between syllables instead of
  flapping shut every consonant
"""
from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .vts_controller import VTSController


# Tuned defaults. ``scale`` compensates for typical TTS RMS (0.03-0.20) being
# much smaller than full-scale. ``gamma`` < 1 stretches the small range up.
# ``attack`` 0.6 / ``release`` 0.15 = mouth opens fast, closes slowly (~5×
# slower) — mimics natural speech where the jaw lingers.
DEFAULT_ATTACK = 0.6
DEFAULT_RELEASE = 0.15
DEFAULT_SCALE = 1.5
DEFAULT_GAMMA = 0.6

# Sample rate of the PCM chunks the Qwen3-TTS server emits. Used to convert
# byte count → audio duration so we know when each chunk "should" play.
_PCM_SAMPLE_RATE = 48000
_PCM_BYTES_PER_SAMPLE = 2  # int16


class LipSyncDriver:
    """Feed PCM chunks in, mouth animates at audio realtime.

    Lifecycle (one instance per TTS stream):

        lip = LipSyncDriver(vts)
        for chunk in tts_stream:
            bot.send_pcm(chunk)
            lip.feed_pcm_chunk(chunk)
        lip.silence()           # schedules mouth-close at audio end

        # On interrupt:
        # lip.silence_now()     # cancel queued, mouth shut immediately

    Each ``feed_pcm_chunk`` returns immediately; the actual VTS parameter set
    happens later, on a ``threading.Timer``, at the correct audio time. Even
    if TTS arrives 2× faster than realtime, the avatar's mouth tracks what
    the listener actually hears.
    """

    def __init__(
        self,
        vts: "VTSController | None",
        *,
        param_name: str = "",
        attack: float = DEFAULT_ATTACK,
        release: float = DEFAULT_RELEASE,
        scale: float = DEFAULT_SCALE,
        gamma: float = DEFAULT_GAMMA,
    ):
        self.vts = vts
        # If param_name is empty, ``_resolve_param`` falls back to whatever
        # vts.status.mouth_param auto-detected (MouthOpen / ParamMouthOpenY).
        self.param_name = param_name
        self.attack = attack
        self.release = release
        self.scale = scale
        self.gamma = gamma

        # Smoothing state — mutated by _apply on the timer thread. Single
        # writer (timers fire serially because they share a sleep-then-apply
        # lambda; no cross-timer contention on small TTS loads), so no lock.
        self._value: float = 0.0

        # Audio timeline anchor + cumulative duration of chunks seen so far.
        # _t_first is set on the first chunk and never reset within one
        # LipSyncDriver lifetime (one instance per utterance).
        self._t_first: float | None = None
        self._audio_consumed_s: float = 0.0
        # Track pending timers so silence_now() can cancel them on interrupt.
        self._timers: list[threading.Timer] = []

    def _resolve_param(self) -> str:
        if self.param_name:
            return self.param_name
        if self.vts is not None:
            return self.vts.status.mouth_param
        return ""

    def feed_pcm_chunk(self, chunk: bytes) -> None:
        """Schedule one mouth update at the audio-realtime moment this chunk
        will be playing through Mumble.

        Returns immediately. The chunk's RMS is computed now (so the closure
        captures a number, not the bytes), and a daemon ``Timer`` fires after
        the appropriate delay to push the value to VTS.

        No-op if VTS isn't connected or no mouth parameter is detected.
        """
        if self.vts is None or not self.vts.is_connected:
            return
        param = self._resolve_param()
        if not param:
            return
        if len(chunk) < 2:
            return

        arr = np.frombuffer(chunk, dtype=np.int16)
        if arr.size == 0:
            return
        # RMS normalized to [0, 1].
        rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2))) / 32768.0
        # Nonlinear stretch: speech RMS clusters at the low end of [0,1].
        x = min(1.0, rms * self.scale)
        x_target = math.pow(x, self.gamma) if x > 0 else 0.0

        chunk_audio_s = (
            len(chunk) / (_PCM_SAMPLE_RATE * _PCM_BYTES_PER_SAMPLE)
        )

        now = time.monotonic()
        if self._t_first is None:
            self._t_first = now
        # Audio time when THIS chunk should actually be heard. Sum of all
        # prior chunks' audio durations, relative to t_first. The first chunk
        # fires immediately (delta 0); subsequent ones space out by their
        # predecessors' durations.
        scheduled_at = self._t_first + self._audio_consumed_s
        delay = max(0.0, scheduled_at - now)

        timer = threading.Timer(
            delay,
            self._apply_value,
            args=(param, x_target),
        )
        timer.daemon = True
        timer.start()
        self._timers.append(timer)

        self._audio_consumed_s += chunk_audio_s

    def _apply_value(self, param: str, x_target: float) -> None:
        # Asymmetric smoothing. Single timer thread fires this at a time; no
        # races on _value because timers don't overlap (each pop runs to
        # completion before its successor's delay elapses, modulo Python's
        # threading scheduling — fine for audio-rate updates).
        if x_target > self._value:
            self._value = (
                self.attack * x_target + (1.0 - self.attack) * self._value
            )
        else:
            self._value = (
                self.release * x_target + (1.0 - self.release) * self._value
            )
        self._value = max(0.0, min(1.0, self._value))
        if self.vts is None or not self.vts.is_connected:
            return
        self.vts.set_parameter(param, self._value)

    def silence(self) -> None:
        """Schedule mouth-close at the END of the queued audio.

        Called by the TTS loop when the stream completes naturally. Because
        chunks were queued at audio realtime, the silencer fires after all of
        them — about ``_audio_consumed_s`` seconds after the first chunk.
        """
        param = self._resolve_param()
        if not param or self.vts is None or not self.vts.is_connected:
            return
        if self._t_first is None:
            # No chunks were ever fed (empty stream) — push silence now.
            self._do_silence(param)
            return
        # Tiny tail (50 ms) past the last chunk so the release-smoothed value
        # has time to settle to baseline before we hard-close it.
        scheduled_at = self._t_first + self._audio_consumed_s + 0.05
        delay = max(0.0, scheduled_at - time.monotonic())
        timer = threading.Timer(delay, self._do_silence, args=(param,))
        timer.daemon = True
        timer.start()
        self._timers.append(timer)

    def silence_now(self) -> None:
        """Cancel all queued mouth updates and slam the mouth shut immediately.

        Use on interrupt — when ``bot.flush_tx()`` drops queued audio, the
        listener stops hearing it instantly, so the mouth should match.
        """
        for t in self._timers:
            try:
                t.cancel()
            except Exception:
                pass
        self._timers.clear()
        param = self._resolve_param()
        if param and self.vts is not None and self.vts.is_connected:
            self._do_silence(param)

    def _do_silence(self, param: str) -> None:
        self._value = 0.0
        if self.vts is not None and self.vts.is_connected:
            self.vts.set_parameter(param, 0.0)
