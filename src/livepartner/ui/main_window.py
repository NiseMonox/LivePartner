"""LivePartner main window — tabbed layout.

Tabs:
  - 运行    : persona pick, event input, trigger, log, AI last line
  - 采集卡  : capture device selection + live preview (~10 fps)
  - 语音    : TTS engine + Mumble connection + STT settings
"""
from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio_codec import mp3_to_pcm48k, pcm48k_duration_seconds
from .._tts_server_proc import TtsServerProcess
from ..overlay_server import OverlayServer
from .subtitle_overlay import SubtitleOverlay
from ..capture import CaptureSource, FrameSnapshot, list_video_devices
from ..decision import describe_scene, extract_fact, gate, generate, summarize_session
from ..director import DirectorIntent, decide_intent, fallback_from_line
from ..memory import MemoryStore
from ..mumble_bot import MumbleBot, MumbleConfig
from ..persona import Persona, list_personas, load_persona
from ..tts import synthesize_for_persona
from ..tts_qwen3 import (
    DEFAULT_BASE_URL as QWEN3_DEFAULT_URL,
    Qwen3TtsConfig,
    is_alive as qwen3_is_alive,
    list_voices as qwen3_list_voices,
    stream_pcm as qwen3_stream_pcm,
)
from ..vts_controller import VTSController, pyvts_available, pyvts_import_error
from ..lip_sync import LipSyncDriver
from ..idle_motion import IdleMotion
from ..appearance import describe_self
from ..window_capture import capture_window_by_title
from ..danmaku import (
    BilibiliDanmakuSource,
    available as danmaku_available,
    import_error as danmaku_import_error,
)
# stt is imported lazily by STTLoadWorker


# Substrings (lowercase, any-match) used to auto-pick a sensible default capture
# device on this machine. Edit if you swap cards.
_PREFERRED_CAPTURE_DEVICE_HINTS = ("live gamer ultra", "elgato hd60", "avermedia")


def _synth_you_died_frame(size: tuple[int, int] = (1280, 720)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), color=(40, 6, 6))
    draw = ImageDraw.Draw(img)
    text = "YOU DIED"
    font = None
    for cand in [
        "C:/Windows/Fonts/seguibl.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/Arial.ttf",
    ]:
        if Path(cand).exists():
            try:
                font = ImageFont.truetype(cand, size=120)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2 - 20), text, fill=(180, 20, 20), font=font)
    return img


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------- background workers ----------


class MumbleConnectWorker(QThread):
    log = Signal(str)
    connected = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg: MumbleConfig, on_user_utterance=None,
                 voice_whitelist: set[str] | None = None):
        super().__init__()
        self.cfg = cfg
        self.on_user_utterance = on_user_utterance
        self.voice_whitelist = voice_whitelist

    def run(self) -> None:
        try:
            bot = MumbleBot(
                self.cfg,
                on_user_utterance=self.on_user_utterance,
                voice_whitelist=self.voice_whitelist,
            )
            self.log.emit(f"[mumble] connecting to {self.cfg.host}:{self.cfg.port} …")
            bot.start(timeout=8.0)
            self.log.emit(f"[mumble] connected as {self.cfg.name!r}, channel "
                          f"{bot.current_channel_name!r}")
            self.connected.emit(bot)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class STTLoadWorker(QThread):
    log = Signal(str)
    loaded = Signal(object)
    failed = Signal(str)

    def __init__(self, model_size: str, language: str | None):
        super().__init__()
        self.model_size = model_size
        self.language = language

    def run(self) -> None:
        try:
            from ..stt import STT
            self.log.emit(f"[stt] loading whisper-{self.model_size} on cuda …")
            t0 = time.perf_counter()
            stt = STT(model_size=self.model_size, language=self.language)
            stt.warm_load()
            self.log.emit(f"[stt] loaded in {(time.perf_counter()-t0)*1000:.0f} ms")
            self.loaded.emit(stt)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class STTTranscribeWorker(QThread):
    log = Signal(str)
    transcribed = Signal(str, dict, object)
    failed = Signal(str)

    def __init__(self, stt, user: dict, pcm_bytes: bytes):
        super().__init__()
        self.stt = stt
        self.user = user
        self.pcm = pcm_bytes

    def run(self) -> None:
        try:
            r = self.stt.transcribe_pcm(self.pcm, sample_rate=48000)
            self.transcribed.emit(r.text, dict(self.user), r)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class Qwen3ProbeWorker(QThread):
    done = Signal(bool, list, str)

    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def run(self) -> None:
        try:
            alive = qwen3_is_alive(self.url, timeout=1.5)
            voices = qwen3_list_voices(self.url) if alive else []
            self.done.emit(alive, voices, self.url)
        except Exception as e:
            self.done.emit(False, [], f"{type(e).__name__}: {e}")


class SessionSummaryWorker(QThread):
    """Flash-summarize today's observed.md off the UI thread.

    Emits done(summary_md). Empty string on failure / no events.
    """
    done = Signal(str)
    log = Signal(str)

    def __init__(self, events_text: str):
        super().__init__()
        self.events_text = events_text

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            summary = summarize_session(self.events_text)
            dt = (time.perf_counter() - t0) * 1000
            if summary:
                self.log.emit(f"[session-summary] +{dt:.0f}ms · {len(summary)} chars")
                self.done.emit(summary)
            else:
                self.log.emit(f"[session-summary] +{dt:.0f}ms · empty")
                self.done.emit("")
        except Exception as e:
            self.log.emit(f"[session-summary err] {type(e).__name__}: {e}")
            self.done.emit("")


class SceneDescriberWorker(QThread):
    """Background flash call: 1-line description of the current game frame.

    If a ``last_description`` is passed in, the model can return SAME (mapped
    to empty string) — caller treats empty as "no meaningful change, skip
    appending to memory".
    """
    done = Signal(str)
    log = Signal(str)

    def __init__(self, frame_b64: str, mime: str, last_description: str | None = None):
        super().__init__()
        # Do NOT name attributes ``event``/``timer``/etc. — those shadow Qt
        # QObject methods and cause cryptic dispatch-time TypeErrors.
        self.frame_b64 = frame_b64
        self.mime = mime
        self.last_description = last_description

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            desc = describe_scene(
                self.frame_b64,
                mime=self.mime,
                last_description=self.last_description,
            )
            dt = (time.perf_counter() - t0) * 1000
            if desc:
                self.log.emit(f"[scene] +{dt:.0f}ms — {desc}")
                self.done.emit(desc)
            else:
                # Either explicit SAME, or empty response — both treated as
                # "no change". Skip noise unless the diff prompt was used
                # (which is the common case, so just log SAME).
                tag = "SAME" if self.last_description else "empty"
                self.log.emit(f"[scene] +{dt:.0f}ms — {tag}")
                self.done.emit("")
        except Exception as e:
            self.log.emit(f"[scene err] {type(e).__name__}: {e}")
            self.done.emit("")


class SelfDescriberWorker(QThread):
    """Background flash-vision call: describe Eri's current VTS-window appearance.

    Triggered by a 90 s timer (and on-demand). Captures the VTube Studio
    window via OS screen-grab, sends to flash-vision for a short Chinese
    description, emits the result. Caller updates the MemoryStore which
    handles change-detection + 'you just changed' marker.
    """
    done = Signal(str)
    log = Signal(str)

    def __init__(self, vts_window_title: str = "VTube Studio"):
        super().__init__()
        # Avoid the QObject.event() shadow trap — keep custom attrs distinct.
        self.window_title = vts_window_title

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            img = capture_window_by_title(
                self.window_title,
                logger=lambda m: self.log.emit(m),
            )
            if img is None:
                self.log.emit("[self] VTS window not found / not capturable")
                self.done.emit("")
                return
            desc = describe_self(img)
            dt = (time.perf_counter() - t0) * 1000
            if desc:
                self.log.emit(f"[self] +{dt:.0f}ms — {desc}")
            else:
                self.log.emit(f"[self] +{dt:.0f}ms — (空)")
            self.done.emit(desc)
        except Exception as e:
            self.log.emit(f"[self err] {type(e).__name__}: {e}")
            self.done.emit("")


class FactExtractorWorker(QThread):
    """Background flash call: pull a durable fact (if any) out of one turn.

    Always emits ``done(line)`` — empty string if nothing extracted, otherwise
    a markdown bullet starting with ``- ``. Runs after the main decision
    pipeline finishes so it doesn't add to user-perceived latency.
    """
    done = Signal(str)
    log = Signal(str)

    def __init__(self, event: str, ai_text: str):
        super().__init__()
        # IMPORTANT: do NOT name this attribute ``self.event`` — that shadows
        # QObject.event(), and Qt's event dispatcher will then try to call a
        # string as a method → ``TypeError: 'str' object is not callable``
        # spam in the terminal every time this thread receives a QEvent.
        self.event_text = event
        self.ai_text = ai_text

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            fact = extract_fact(self.event_text, self.ai_text)
            dt = (time.perf_counter() - t0) * 1000
            if fact:
                self.log.emit(f"[fact] +{dt:.0f}ms — {fact}")
                self.done.emit(fact)
            else:
                self.log.emit(f"[fact] +{dt:.0f}ms — NONE")
                self.done.emit("")
        except Exception as e:
            self.log.emit(f"[fact err] {type(e).__name__}: {e}")
            self.done.emit("")


class DirectorWorker(QThread):
    """Background flash call: pick the avatar's expression / hotkey for one
    line. Runs in parallel with TTS (kicked off from ``text_ready`` signal)
    so the avatar usually reacts before the first audio chunk plays.

    Always emits ``intent_ready`` (with a fallback intent on any error) — never
    leaves the avatar without a decision.
    """
    intent_ready = Signal(object)  # DirectorIntent
    log = Signal(str)

    def __init__(
        self,
        line: str,
        context: list[str],
        expr_palette: list[str],
        hotkey_palette: list[str],
        persona: Persona,
    ):
        super().__init__()
        # Same QObject.event() shadow trap as FactExtractorWorker — keep
        # attribute names away from the ``event`` / ``done`` reserved set.
        self.line_text = line
        self.ctx_lines = list(context)
        self.expr_palette = list(expr_palette)
        self.hotkey_palette = list(hotkey_palette)
        self.persona = persona

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            intent = decide_intent(
                self.line_text, self.ctx_lines,
                self.expr_palette, self.hotkey_palette, self.persona,
            )
            dt = (time.perf_counter() - t0) * 1000
            self.log.emit(
                f"[director] +{dt:.0f}ms — expr={intent.expr!r} "
                f"intensity={intent.intensity:.2f} trigger={intent.trigger!r}"
            )
        except Exception as e:
            self.log.emit(f"[director err] {type(e).__name__}: {e}")
            intent = fallback_from_line(self.line_text, self.expr_palette)
        self.intent_ready.emit(intent)


@dataclass
class DecisionRequest:
    persona: Persona
    event: str
    force_speak: bool
    save_mp3: bool
    bot: MumbleBot | None
    engine: str
    qwen3_url: str
    synthesize_frame: bool = True
    is_conversation: bool = False
    captured_frame: FrameSnapshot | None = None
    memory_text: str = ""
    # Last N AI-spoken lines, passed to generate() for output-side repeat
    # detection. Empty list = first turn, no check.
    recent_ai_lines: list[str] = field(default_factory=list)
    # Optional Live2D controller; if connected, the TTS chunk loop drives
    # MouthOpen via LipSyncDriver. None = avatar layer disabled, all skip.
    vts: VTSController | None = None


class DecisionWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)
    # Emitted when the worker bailed early because cancel() was called.
    # Carries no payload — UI uses it to log "interrupted" + reset state.
    interrupted = Signal()
    # Fires the moment the LLM returns text, BEFORE TTS+Mumble. Lets the UI
    # show the subtitle (and push to OBS browser source) without waiting for
    # the 3-5 second TTS+drain. ~1s earlier than finished_ok in typical flow.
    # Carries just the line text; the director infers expression from line
    # content alone (EMO was removed from the LLM output schema).
    text_ready = Signal(str)

    def __init__(self, req: DecisionRequest):
        super().__init__()
        self.req = req
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Mark for cancellation. Also flushes bot TX so any already-queued
        Opus packets stop playing on the receiving end.

        Cooperative — `run()` checks the flag at gate/generate/TTS-chunk
        boundaries. The currently in-flight HTTP call (LLM or TTS) finishes
        first, then the next checkpoint bails.
        """
        self._cancel.set()
        if self.req.bot is not None:
            try:
                self.req.bot.flush_tx()
            except Exception:
                pass

    def _cancelled(self) -> bool:
        if self._cancel.is_set():
            self.log.emit("[cancel] worker stopping at checkpoint")
            self.interrupted.emit()
            return True
        return False

    def run(self) -> None:
        try:
            frame_mime = "image/png"
            if self.req.captured_frame is not None:
                snap = self.req.captured_frame
                h, w = snap.frame.shape[:2]
                # Generate sees the rich frame — bump from 768/75 to 1024/82
                # so the model can actually read on-screen text (UI numbers,
                # video titles, menu options). ~2x image tokens but worth it
                # because this is the call that has to ground Eri's reply in
                # specifics.
                frame_b64, frame_mime = snap.to_vlm_b64(
                    max_side=1024, quality=82, prefer="webp",
                )
                thumb_b64 = snap.thumbnail_png_b64(max_side=256)
                self.log.emit(
                    f"frame from capture {w}x{h} → "
                    f"{len(frame_b64)*3//4//1024} KB {frame_mime.split('/')[-1]}  "
                    f"age={time.monotonic()-snap.timestamp:.2f}s"
                )
            elif self.req.synthesize_frame:
                img = _synth_you_died_frame()
                frame_b64 = _pil_to_b64(img)
                thumb = img.copy()
                thumb.thumbnail((256, 256))
                thumb_b64 = _pil_to_b64(thumb)
                self.log.emit(f"frame synthesized {img.size[0]}x{img.size[1]} (no capture)")
            else:
                frame_b64 = None
                thumb_b64 = None
                self.log.emit("conversation mode — no game frame")

            if self.req.force_speak:
                self.log.emit("[gate] skipped (force_speak)")
                speak = True
                dt_gate = 0.0
            elif thumb_b64 is not None:
                t0 = time.perf_counter()
                speak = gate(self.req.event, thumb_b64, self.req.persona)
                dt_gate = (time.perf_counter() - t0) * 1000
                self.log.emit(f"[gate] speak={speak}  ({dt_gate:.0f} ms)")
            else:
                self.log.emit("[gate] no frame + not forced → silent")
                self.finished_ok.emit("")
                return

            if self._cancelled():
                return

            if not speak:
                self.log.emit("AI 选择沉默。")
                self.finished_ok.emit("")
                return

            t0 = time.perf_counter()
            if self.req.memory_text:
                self.log.emit(f"[memory] {len(self.req.memory_text)} chars injected")
            reply = generate(
                self.req.event, frame_b64, self.req.persona,
                memory=self.req.memory_text,
                mime=frame_mime,
                is_conversation=self.req.is_conversation,
                recent_ai_lines=self.req.recent_ai_lines,
            )
            dt_gen = (time.perf_counter() - t0) * 1000
            tag = ""
            if reply.rerolled == 1:
                tag = " [reroll: repeat caught, replaced]"
            elif reply.rerolled == 2:
                tag = " [reroll: still repeat → silence]"
            self.log.emit(f"[generate] {dt_gen:.0f} ms{tag}")

            if self._cancelled():
                return

            self.log.emit(f"AI: {reply.text}" if reply.text else "AI: (沉默)")
            # tts_instruct is fixed (mechanical tone) — not worth logging every turn.
            self.log.emit(f"total LLM: {dt_gate + dt_gen:.0f} ms")

            # Push subtitle text to the UI / overlays NOW, before TTS+drain.
            # finished_ok fires later (after TX queue drains) and still does
            # the memory write / fact extraction.
            if reply.text:
                self.text_ready.emit(reply.text)

            if not reply.tts_text:
                self.finished_ok.emit("")
                return

            if self.req.engine == "qwen3":
                self._tts_qwen3_streaming(reply.tts_text, instruct=reply.tts_instruct)
            else:
                self._tts_edge(reply.tts_text)

            self.finished_ok.emit(reply.text)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _tts_edge(self, text: str) -> None:
        t0 = time.perf_counter()
        tts = synthesize_for_persona(text, self.req.persona)
        dt_tts = (time.perf_counter() - t0) * 1000
        self.log.emit(f"[edge-tts] {dt_tts:.0f} ms  voice={tts.voice_id}  "
                      f"{len(tts.mp3)//1024} KB mp3")
        if self.req.save_mp3:
            Path("demo_tts.mp3").write_bytes(tts.mp3)
        if self.req.bot is not None:
            t0 = time.perf_counter()
            pcm = mp3_to_pcm48k(tts.mp3)
            dur = pcm48k_duration_seconds(pcm)
            self.log.emit(f"[pcm decode] {(time.perf_counter()-t0)*1000:.0f} ms  "
                          f"{len(pcm)//1024} KB  {dur:.2f}s")
            self.req.bot.send_pcm(pcm)
            self.log.emit(f"[mumble] streaming {dur:.2f}s into "
                          f"{self.req.bot.current_channel_name!r}, "
                          f"waiting for buffer to drain…")
            # Block until Mumble has actually finished sending — so the worker
            # accurately represents "AI still speaking".
            self.req.bot.wait_until_silent(max_wait=max(dur + 3.0, 5.0))

    def _tts_qwen3_streaming(self, text: str, *, instruct: str = "") -> None:
        # Mixed Chinese + occasional katakana: LINE is mostly Chinese, with
        # Eri's own name written as エリ (katakana) so TTS reads it with Japanese
        # phonetics. Player's name 似曾 reads as natural Chinese. "Auto" lets the
        # model handle the code-switch.
        cfg = Qwen3TtsConfig(base_url=self.req.qwen3_url, language="Auto")
        # Fresh LipSync per stream — internal smoothing state resets so each
        # utterance starts with mouth closed. No-op if VTS not connected.
        lip = LipSyncDriver(self.req.vts) if self.req.vts is not None else None
        t0 = time.perf_counter()
        total = 0
        first_ms = None
        for chunk in qwen3_stream_pcm(text, self.req.persona, cfg=cfg, instruct=instruct or None):
            if self._cancel.is_set():
                # Player interrupted (or shutdown) — stop reading TTS chunks
                # AND drop anything queued. The for-loop's break closes the
                # httpx stream context manager.
                self.log.emit("[cancel] mid-TTS stop")
                if self.req.bot is not None:
                    try:
                        self.req.bot.flush_tx()
                    except Exception:
                        pass
                if lip is not None:
                    # Cancel queued mouth updates + slam shut immediately —
                    # Mumble TX was just flushed, audio stopped, mouth matches.
                    lip.silence_now()
                self.interrupted.emit()
                return
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
                if self.req.bot is not None:
                    target = f"Mumble 频道 {self.req.bot.current_channel_name!r}"
                else:
                    target = "(丢弃 — 没连 Mumble 或'AI 通过 Mumble 说话'未勾)"
                self.log.emit(f"[qwen3] TTFB {first_ms:.0f} ms → streaming to {target}")
            total += len(chunk)
            if self.req.bot is not None:
                self.req.bot.send_pcm(chunk)
            if lip is not None:
                lip.feed_pcm_chunk(chunk)
        # Stream ended naturally — close mouth.
        if lip is not None:
            lip.silence()
        dt = (time.perf_counter() - t0) * 1000
        dur_s = total / (48000 * 2)
        self.log.emit(f"[qwen3] gen+queue {dt:.0f} ms  {total//1024} KB PCM  {dur_s:.2f}s "
                      f"(RTF {dur_s / (dt/1000):.2f})")
        # Wait for the Mumble TX queue to actually drain — otherwise the worker
        # "finishes" while audio is still playing, and a new utterance racing
        # in would think the AI is free. Poll the cancel flag during the wait
        # so an interrupt cuts the tail short.
        if self.req.bot is not None:
            deadline = time.monotonic() + max(dur_s + 3.0, 5.0)
            while time.monotonic() < deadline:
                if self._cancel.is_set():
                    self.req.bot.flush_tx()
                    if lip is not None:
                        lip.silence_now()
                    self.log.emit("[cancel] tail-drain interrupted")
                    self.interrupted.emit()
                    return
                try:
                    if self.req.bot._client.sound_output.get_buffer_size() <= 0:
                        break
                except Exception:
                    break
                time.sleep(0.1)
            self.log.emit("[mumble] TX queue drained")


# ---------- main window ----------


class MainWindow(QMainWindow):
    DEFAULT_EVENT = (
        "玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。BGM 转为低沉死亡音乐。"
    )

    utterance_signal = Signal(dict, bytes)
    # Bridge from danmaku asyncio thread → Qt thread. Bilibili WSS callback
    # fires here; UI thread picks up via the connected slot.
    danmaku_signal = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LivePartner (M1)")
        self.resize(1100, 800)

        # Mutable state holders — must exist before tab builders reference them.
        self.decision_worker: DecisionWorker | None = None
        self.connect_worker: MumbleConnectWorker | None = None
        self.probe_worker: Qwen3ProbeWorker | None = None
        self.bot: MumbleBot | None = None
        self.stt = None
        self.stt_load_worker: STTLoadWorker | None = None
        self.stt_workers: list[STTTranscribeWorker] = []
        self.capture: CaptureSource | None = None
        # If a new utterance arrives while DecisionWorker is busy, we stash the
        # latest one here (replacing any older pending), and pick it up the
        # instant the current pipeline finishes. Per-utterance not per-stream:
        # this is the raw PCM, STT happens only when we're ready to act on it.
        self.pending_utterance: tuple[dict, bytes] | None = None
        # Second-stage queue: text that finished transcribing while AI was
        # still mid-sentence. We can't go back to PCM at this point (transcribe
        # already consumed it), so we stash the result and drain on cleanup.
        # Latest text wins (older gets replaced) — same policy as pending_utterance.
        self.pending_transcribed: tuple[dict, str] | None = None
        # Danmaku queue: at most 1 pending viewer message awaiting AI dispatch.
        # Newer replaces older — high-volume chat won't backlog.
        self.pending_danmaku: tuple[str, str] | None = None
        # Live-chat cooldown so a noisy room doesn't ping Eri every 弹幕.
        self._last_danmaku_dispatch_ts: float = 0.0

        # Per-game memory store. game_id="default" until we add multi-game UI.
        self.memory = MemoryStore(game_id="default")

        # Ambient loop state. last_ai_spoke_at = monotonic clock the LAST time
        # an AI line played; we won't auto-fire again until cooldown elapses.
        self.last_ai_spoke_at: float = 0.0
        self.fact_workers: list[FactExtractorWorker] = []
        self.director_workers: list[DirectorWorker] = []

        # Scene-describer state. _last_scene_fingerprint is a 16x16 grayscale
        # snapshot of the last frame we described; we compare new frames to
        # it and skip describing if the diff is below threshold.
        self.scene_workers: list[SceneDescriberWorker] = []
        self._last_scene_fingerprint = None  # np.ndarray | None

        # Quickstart orchestrator state. Phases: capture → tts → stt → mumble.
        # Hooks into _on_probe_done / _on_stt_loaded / _on_mumble_connected
        # to advance, and into _on_*_failed / probe timeout to abort.
        self._quickstart_active = False

        # Subtitle overlay window (created on demand, lives independent of tabs).
        # _on_decision_done pushes the AI's ZH line into it when enabled.
        self.subtitle_overlay: SubtitleOverlay | None = None
        # OBS browser-source HTTP server. Created on demand when the user
        # toggles "OBS 浏览器源" on. Lives until UI close.
        self.web_overlay: OverlayServer | None = None
        # Event string the current/last decision worker was responding to.
        # Used by the fact extractor (in _on_decision_done) to bind the AI
        # reply back to the trigger.
        self._last_decision_event: str = ""

        # Explicit busy flag — set THE MOMENT we decide to spawn a
        # DecisionWorker, cleared in _cleanup_decision. We can't rely on
        # `decision_worker.isRunning()` because QThread.isRunning() returns
        # False in the brief gap between start() and the thread actually
        # being scheduled. That gap lets two workers spawn in parallel,
        # which then race on the Qwen3-TTS server and crash the UI.
        self._decision_busy: bool = False

        # TTS server subprocess. Started/stopped via the 语音 tab button; the
        # closeEvent below tears it down so we don't leak GPU memory.
        self.tts_server = TtsServerProcess()

        # Live2D / VTS — created lazily by the tab's Connect button. We never
        # auto-start so first-launch UX is "open app → see red status → click
        # connect → approve VTS popup" rather than a surprise dialog.
        self.vts: VTSController | None = None
        # IdleMotion drives the avatar's eye blinks / head sway / breath at
        # 20 Hz once VTS is up. Created together with vts in _on_vts_connect.
        self.idle_motion: IdleMotion | None = None
        # Tracks whether IdleMotion has been started for the current VTS
        # connection — flipped by the poll timer when status crosses ready.
        self._idle_motion_started: bool = False
        # Visual self-awareness — periodic 90 s tick captures the VTS window,
        # describes Eri's appearance via flash vision, stores in memory.
        # Diff with prev description marks "你的形象刚刚变了" → next AI turn
        # naturally comments on the change.
        self._self_describer_workers: list = []
        self._self_describer_timer = QTimer(self)
        self._self_describer_timer.setInterval(90000)
        self._self_describer_timer.timeout.connect(self._on_self_describer_tick)
        # Bilibili 弹幕 source — created lazily by the "直播" tab connect button.
        self.danmaku: BilibiliDanmakuSource | None = None
        # Idle decay: 8 s after the last director intent we drop the avatar
        # back to a neutral expression so it doesn't sit on whatever the last
        # line happened to make it (e.g. permanent smug after one quip).
        self._idle_decay_timer = QTimer(self)
        self._idle_decay_timer.setInterval(8000)
        self._idle_decay_timer.setSingleShot(True)
        self._idle_decay_timer.timeout.connect(self._on_director_idle_decay)

        tabs = QTabWidget()
        tabs.addTab(self._build_run_tab(), "运行")
        tabs.addTab(self._build_capture_tab(), "采集卡")
        tabs.addTab(self._build_audio_tab(), "语音 / TTS / Mumble")
        tabs.addTab(self._build_memory_tab(), "记忆")
        tabs.addTab(self._build_vts_tab(), "Live2D")
        tabs.addTab(self._build_danmaku_tab(), "直播")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("就绪")

        # Cross-thread bridge: pymumble callback → Qt signal → UI thread slot.
        self.utterance_signal.connect(self._on_utterance_arrived)
        # Same pattern for danmaku — Bilibili WSS asyncio thread → Qt slot.
        self.danmaku_signal.connect(self._on_danmaku_arrived)

        # Preview timer (~10 fps).
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._update_preview)
        self.preview_timer.start()

        # Ambient tick timer — created stopped, started by checkbox.
        self.ambient_timer = QTimer(self)
        self.ambient_timer.setInterval(self.ambient_interval_spin.value() * 1000)
        self.ambient_timer.timeout.connect(self._on_ambient_tick)

        # Scene-describer tick timer — separate cadence from ambient.
        self.scene_timer = QTimer(self)
        self.scene_timer.setInterval(self.scene_interval_spin.value() * 1000)
        self.scene_timer.timeout.connect(self._on_scene_tick)

        # Default-on: ambient mode + scene memory + OBS browser-source overlay.
        # Done AFTER timers are constructed because setChecked(True) fires the
        # toggled signal, whose handlers access self.ambient_timer/scene_timer/
        # web_overlay. All handlers skip gracefully if dependencies aren't
        # ready (no capture / port taken / no Mumble), so this is safe even at
        # cold-start before the user hits the quickstart button.
        self.ambient_check.setChecked(True)
        self.scene_check.setChecked(True)
        self.web_overlay_check.setChecked(True)

        self._on_engine_changed()

    # ---------- tab builders ----------
    def _build_run_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # ── one-tap startup ─────────────────────────────────────────
        qs_row = QHBoxLayout()
        self.quickstart_btn = QPushButton("一键启动 (采集卡 → TTS → STT → Mumble)")
        self.quickstart_btn.setMinimumHeight(36)
        f = QFont()
        f.setBold(True)
        self.quickstart_btn.setFont(f)
        self.quickstart_btn.clicked.connect(self._on_quickstart)
        qs_row.addWidget(self.quickstart_btn, stretch=1)
        self.quickstart_status_lbl = QLabel("(空闲)")
        self.quickstart_status_lbl.setStyleSheet("color: #888;")
        qs_row.addWidget(self.quickstart_status_lbl)
        layout.addLayout(qs_row)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("人格:"))
        self.persona_combo = QComboBox()
        for pid in list_personas():
            try:
                p = load_persona(pid)
                self.persona_combo.addItem(f"{p.display_name} ({pid})", userData=pid)
            except Exception:
                self.persona_combo.addItem(f"{pid} (load failed)", userData=pid)
        row1.addWidget(self.persona_combo, stretch=1)
        self.force_check = QCheckBox("force_speak (跳过 gate)")
        self.force_check.setChecked(False)
        row1.addWidget(self.force_check)
        self.save_mp3_check = QCheckBox("保存 demo_tts.mp3 (Edge)")
        row1.addWidget(self.save_mp3_check)
        layout.addLayout(row1)

        layout.addWidget(QLabel("事件描述："))
        self.event_edit = QLineEdit(self.DEFAULT_EVENT)
        layout.addWidget(self.event_edit)
        self.trigger_btn = QPushButton("触发一次 AI 反应")
        self.trigger_btn.clicked.connect(self._on_trigger)
        layout.addWidget(self.trigger_btn)

        ambient_box = QGroupBox("环境模式 (AI 主动开口)")
        ambient_outer = QVBoxLayout(ambient_box)
        amb_row = QHBoxLayout()
        self.ambient_check = QCheckBox("启用环境模式")
        self.ambient_check.toggled.connect(self._on_ambient_toggled)
        amb_row.addWidget(self.ambient_check)
        amb_row.addWidget(QLabel("Tick 间隔(秒):"))
        self.ambient_interval_spin = QSpinBox()
        self.ambient_interval_spin.setRange(3, 120)
        self.ambient_interval_spin.setValue(5)
        self.ambient_interval_spin.valueChanged.connect(self._on_ambient_interval_changed)
        amb_row.addWidget(self.ambient_interval_spin)
        amb_row.addWidget(QLabel("说完后冷却(秒):"))
        self.ambient_cooldown_spin = QSpinBox()
        self.ambient_cooldown_spin.setRange(0, 600)
        self.ambient_cooldown_spin.setValue(4)
        amb_row.addWidget(self.ambient_cooldown_spin)
        amb_row.addStretch()
        self.ambient_status_lbl = QLabel("(关闭)")
        self.ambient_status_lbl.setStyleSheet("color: #888;")
        amb_row.addWidget(self.ambient_status_lbl)
        ambient_outer.addLayout(amb_row)
        ambient_outer.addWidget(QLabel(
            "勾上后，每隔 tick 间隔 AI 看一眼画面，gate 自己决定要不要开口。"
            "需要采集卡在跑 + 离上一句间隔超过冷却时间。"
        ))
        layout.addWidget(ambient_box)

        scene_box = QGroupBox("画面记忆 (Eri 能记住刚才看到的画面)")
        scene_outer = QVBoxLayout(scene_box)
        scene_row = QHBoxLayout()
        self.scene_check = QCheckBox("启用画面记忆")
        self.scene_check.toggled.connect(self._on_scene_toggled)
        scene_row.addWidget(self.scene_check)
        scene_row.addWidget(QLabel("描述间隔(秒):"))
        self.scene_interval_spin = QSpinBox()
        self.scene_interval_spin.setRange(3, 60)
        self.scene_interval_spin.setValue(5)
        self.scene_interval_spin.valueChanged.connect(self._on_scene_interval_changed)
        scene_row.addWidget(self.scene_interval_spin)
        scene_row.addWidget(QLabel("保留时长(秒):"))
        self.scene_retention_spin = QSpinBox()
        self.scene_retention_spin.setRange(15, 600)
        self.scene_retention_spin.setValue(90)
        scene_row.addWidget(self.scene_retention_spin)
        scene_row.addStretch()
        self.scene_status_lbl = QLabel("(关闭)")
        self.scene_status_lbl.setStyleSheet("color: #888;")
        scene_row.addWidget(self.scene_status_lbl)
        scene_outer.addLayout(scene_row)
        scene_outer.addWidget(QLabel(
            "勾上后，每隔间隔 flash 描述一次画面(只在画面有变化时跑)，"
            "结果带时间戳存进记忆，Eri 能引用 \"刚才\" / \"X 分钟前\"。"
        ))
        layout.addWidget(scene_box)

        overlay_box = QGroupBox("字幕浮窗 (OBS 直播友好)")
        overlay_outer = QVBoxLayout(overlay_box)
        ov_row = QHBoxLayout()
        self.overlay_check = QCheckBox("启用字幕浮窗")
        self.overlay_check.toggled.connect(self._on_overlay_toggled)
        ov_row.addWidget(self.overlay_check)
        ov_row.addWidget(QLabel("字号:"))
        self.overlay_font_spin = QSpinBox()
        self.overlay_font_spin.setRange(12, 96)
        self.overlay_font_spin.setValue(32)
        self.overlay_font_spin.valueChanged.connect(self._on_overlay_font_changed)
        ov_row.addWidget(self.overlay_font_spin)
        ov_row.addWidget(QLabel("不透明度(%):"))
        self.overlay_opacity_spin = QSpinBox()
        self.overlay_opacity_spin.setRange(20, 100)
        self.overlay_opacity_spin.setValue(100)
        self.overlay_opacity_spin.valueChanged.connect(self._on_overlay_opacity_changed)
        ov_row.addWidget(self.overlay_opacity_spin)
        ov_row.addWidget(QLabel("保持(秒):"))
        self.overlay_hold_spin = QSpinBox()
        self.overlay_hold_spin.setRange(0, 60)
        self.overlay_hold_spin.setValue(6)
        self.overlay_hold_spin.setToolTip("0 = 一直显示直到下一条")
        self.overlay_hold_spin.valueChanged.connect(self._on_overlay_hold_changed)
        ov_row.addWidget(self.overlay_hold_spin)
        ov_row.addStretch()
        overlay_outer.addLayout(ov_row)
        overlay_outer.addWidget(QLabel(
            "底部居中的透明置顶窗,Eri 每说一句字幕会推到这里。可以鼠标拖动改位置。"
            "OBS 加「窗口采集 / WGC」抓 \"Eri 字幕\" 即可叠到直播画面上。"
        ))
        layout.addWidget(overlay_box)

        web_box = QGroupBox("OBS 浏览器源 (本地 HTTP)")
        web_outer = QVBoxLayout(web_box)
        web_row = QHBoxLayout()
        self.web_overlay_check = QCheckBox("启用")
        self.web_overlay_check.toggled.connect(self._on_web_overlay_toggled)
        web_row.addWidget(self.web_overlay_check)
        web_row.addWidget(QLabel("端口:"))
        self.web_overlay_port_spin = QSpinBox()
        self.web_overlay_port_spin.setRange(1024, 65535)
        self.web_overlay_port_spin.setValue(7002)
        web_row.addWidget(self.web_overlay_port_spin)
        web_row.addWidget(QLabel("字号:"))
        self.web_overlay_font_spin = QSpinBox()
        self.web_overlay_font_spin.setRange(12, 200)
        self.web_overlay_font_spin.setValue(48)
        web_row.addWidget(self.web_overlay_font_spin)
        web_row.addWidget(QLabel("保持(秒):"))
        self.web_overlay_hold_spin = QSpinBox()
        self.web_overlay_hold_spin.setRange(0, 60)
        self.web_overlay_hold_spin.setValue(6)
        web_row.addWidget(self.web_overlay_hold_spin)
        web_row.addStretch()
        self.web_overlay_status_lbl = QLabel("(未启用)")
        self.web_overlay_status_lbl.setStyleSheet("color: #888;")
        web_row.addWidget(self.web_overlay_status_lbl)
        web_outer.addLayout(web_row)
        url_row = QHBoxLayout()
        url_row.addWidget(QLabel("局域网 URL (填 OBS):"))
        self.web_overlay_url_edit = QLineEdit("http://127.0.0.1:7002/")
        self.web_overlay_url_edit.setReadOnly(True)
        # Bigger font so it's easy to read across the room.
        f = self.web_overlay_url_edit.font()
        f.setBold(True)
        self.web_overlay_url_edit.setFont(f)
        url_row.addWidget(self.web_overlay_url_edit, stretch=1)
        web_outer.addLayout(url_row)
        web_outer.addWidget(QLabel(
            "游戏 PC OBS → 来源 → 浏览器(Browser),URL 填上面那个,1920x1080,勾「透明背景」。"
            "笔记本和游戏 PC 必须在同一局域网。第一次启动 Windows 会弹防火墙允许窗口,勾「专用网络」放行。"
        ))
        layout.addWidget(web_box)

        layout.addWidget(QLabel("日志："))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log_view, stretch=1)

        layout.addWidget(QLabel("AI 最近一句："))
        self.last_line_lbl = QLabel("(尚未生成)")
        self.last_line_lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.last_line_lbl.setWordWrap(True)
        self.last_line_lbl.setStyleSheet("color: #c33; padding: 8px;")
        layout.addWidget(self.last_line_lbl)
        return page

    def _build_capture_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        row = QHBoxLayout()
        row.addWidget(QLabel("设备:"))
        self.cap_device_combo = QComboBox()
        for i, name in enumerate(list_video_devices()):
            self.cap_device_combo.addItem(f"[{i}] {name}", userData=i)
        self._select_preferred_capture_device()
        row.addWidget(self.cap_device_combo, stretch=1)
        row.addWidget(QLabel("分辨率:"))
        self.cap_res_combo = QComboBox()
        for w, h in [(1920, 1080), (1280, 720), (640, 360)]:
            self.cap_res_combo.addItem(f"{w}x{h}", userData=(w, h))
        row.addWidget(self.cap_res_combo)
        self.cap_start_btn = QPushButton("启动")
        self.cap_start_btn.clicked.connect(self._on_capture_start)
        row.addWidget(self.cap_start_btn)
        self.cap_stop_btn = QPushButton("停止")
        self.cap_stop_btn.clicked.connect(self._on_capture_stop)
        self.cap_stop_btn.setEnabled(False)
        row.addWidget(self.cap_stop_btn)
        self.cap_refresh_btn = QPushButton("刷新设备")
        self.cap_refresh_btn.clicked.connect(self._on_capture_refresh_devices)
        row.addWidget(self.cap_refresh_btn)
        layout.addLayout(row)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("状态:"))
        self.cap_status = QLabel("(未启动)")
        self.cap_status.setStyleSheet("color: #888;")
        status_row.addWidget(self.cap_status, stretch=1)
        self.cap_info_lbl = QLabel("")
        self.cap_info_lbl.setStyleSheet("color: #888;")
        status_row.addWidget(self.cap_info_lbl)
        layout.addLayout(status_row)

        # Big preview area.
        self.preview_label = QLabel("(未启动)")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background: #111; color: #888; border: 1px solid #444;")
        self.preview_label.setMinimumSize(640, 360)
        layout.addWidget(self.preview_label, stretch=1)

        return page

    def _build_audio_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # --- TTS ---
        tts_box = QGroupBox("TTS 引擎")
        tts_outer = QVBoxLayout(tts_box)
        tts_row1 = QHBoxLayout()
        tts_row1.addWidget(QLabel("引擎:"))
        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItem("Edge TTS (云,免费)", userData="edge")
        self.tts_engine_combo.addItem("Qwen3-TTS (本地,克隆)", userData="qwen3")
        self.tts_engine_combo.setCurrentIndex(1)
        self.tts_engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        tts_row1.addWidget(self.tts_engine_combo)
        # Mono-Chinese pipeline: subtitle and TTS share the same Chinese line.
        # No language picker needed.
        tts_row1.addStretch()
        tts_outer.addLayout(tts_row1)
        tts_row2 = QHBoxLayout()
        tts_row2.addWidget(QLabel("Qwen3 URL:"))
        self.qwen3_url_edit = QLineEdit(QWEN3_DEFAULT_URL)
        self.qwen3_url_edit.setMaximumWidth(260)
        tts_row2.addWidget(self.qwen3_url_edit)
        self.tts_server_btn = QPushButton("启动 TTS 服务")
        self.tts_server_btn.setToolTip(
            "本地起一个 Qwen3-TTS 子进程（独立 venv）。首次加载需 ~20s。"
            "关闭 UI 时自动停止。"
        )
        self.tts_server_btn.clicked.connect(self._on_tts_server_toggle)
        tts_row2.addWidget(self.tts_server_btn)
        self.qwen3_probe_btn = QPushButton("检测")
        self.qwen3_probe_btn.clicked.connect(self._on_probe_qwen3)
        tts_row2.addWidget(self.qwen3_probe_btn)
        self.qwen3_status = QLabel("(未检测)")
        self.qwen3_status.setStyleSheet("color: #888;")
        tts_row2.addWidget(self.qwen3_status, stretch=1)
        tts_outer.addLayout(tts_row2)
        layout.addWidget(tts_box)

        # --- Mumble ---
        mumble_box = QGroupBox("Mumble")
        mumble_layout = QVBoxLayout(mumble_box)
        mrow1 = QHBoxLayout()
        mrow1.addWidget(QLabel("host:"))
        self.mumble_host = QLineEdit("127.0.0.1")
        self.mumble_host.setMaximumWidth(160)
        mrow1.addWidget(self.mumble_host)
        mrow1.addWidget(QLabel("port:"))
        self.mumble_port = QSpinBox()
        self.mumble_port.setRange(1, 65535)
        self.mumble_port.setValue(64738)
        mrow1.addWidget(self.mumble_port)
        mrow1.addWidget(QLabel("name:"))
        self.mumble_name = QLineEdit("Eri")
        self.mumble_name.setMaximumWidth(160)
        mrow1.addWidget(self.mumble_name)
        mrow1.addWidget(QLabel("channel:"))
        self.mumble_channel = QLineEdit("LivePartner")
        self.mumble_channel.setMaximumWidth(160)
        mrow1.addWidget(self.mumble_channel)
        mrow1.addStretch()
        mumble_layout.addLayout(mrow1)
        mrow_pwd = QHBoxLayout()
        mrow_pwd.addWidget(QLabel("password:"))
        self.mumble_password = QLineEdit()
        self.mumble_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.mumble_password.setPlaceholderText("(留空 = 无密码)")
        self.mumble_password.setMaximumWidth(220)
        mrow_pwd.addWidget(self.mumble_password)
        self.mumble_no_create_check = QCheckBox("不要创建频道 (公网服务器)")
        self.mumble_no_create_check.setToolTip(
            "公网 Mumble 服务器通常不允许客户端建频道。勾上后,如果指定的 "
            "channel 不存在,bot 待在 Root 不再尝试创建。"
        )
        mrow_pwd.addWidget(self.mumble_no_create_check)
        mrow_pwd.addStretch()
        mumble_layout.addLayout(mrow_pwd)
        mrow2 = QHBoxLayout()
        self.mumble_connect_btn = QPushButton("连接 Mumble")
        self.mumble_connect_btn.clicked.connect(self._on_connect_mumble)
        mrow2.addWidget(self.mumble_connect_btn)
        self.mumble_disconnect_btn = QPushButton("断开")
        self.mumble_disconnect_btn.clicked.connect(self._on_disconnect_mumble)
        self.mumble_disconnect_btn.setEnabled(False)
        mrow2.addWidget(self.mumble_disconnect_btn)
        self.mumble_status = QLabel("未连接")
        self.mumble_status.setStyleSheet("color: #888;")
        mrow2.addWidget(self.mumble_status, stretch=1)
        self.mumble_speak_check = QCheckBox("AI 通过 Mumble 说话")
        self.mumble_speak_check.setChecked(True)
        self.mumble_speak_check.setEnabled(False)
        mrow2.addWidget(self.mumble_speak_check)
        mumble_layout.addLayout(mrow2)
        layout.addWidget(mumble_box)

        # --- STT ---
        stt_box = QGroupBox("STT 监听玩家")
        stt_outer = QVBoxLayout(stt_box)
        stt_row1 = QHBoxLayout()
        stt_row1.addWidget(QLabel("模型:"))
        self.stt_size_combo = QComboBox()
        for s in [
            "tiny", "base", "small", "medium",
            "large-v3", "large-v3-turbo", "distil-large-v3",
        ]:
            self.stt_size_combo.addItem(s)
        # turbo = 80% quality of large-v3, ~3× faster, much less hallucination
        # in JA/ZH/EN. First load triggers a ~1.5 GB download from HF.
        self.stt_size_combo.setCurrentText("large-v3-turbo")
        stt_row1.addWidget(self.stt_size_combo)
        stt_row1.addWidget(QLabel("语种:"))
        self.stt_lang_combo = QComboBox()
        for label, code in [("自动", None), ("中文", "zh"), ("日文", "ja"), ("英文", "en")]:
            self.stt_lang_combo.addItem(label, userData=code)
        stt_row1.addWidget(self.stt_lang_combo)
        self.stt_load_btn = QPushButton("加载模型")
        self.stt_load_btn.clicked.connect(self._on_stt_load)
        stt_row1.addWidget(self.stt_load_btn)
        self.stt_listen_check = QCheckBox("监听频道")
        self.stt_listen_check.setEnabled(False)
        self.stt_listen_check.toggled.connect(self._on_listen_toggled)
        stt_row1.addWidget(self.stt_listen_check)
        self.stt_status = QLabel("(未加载)")
        self.stt_status.setStyleSheet("color: #888;")
        stt_row1.addWidget(self.stt_status, stretch=1)
        stt_outer.addLayout(stt_row1)
        stt_row2 = QHBoxLayout()
        stt_row2.addWidget(QLabel("白名单 (逗号分隔,留空=全监听):"))
        self.stt_whitelist_edit = QLineEdit("NiseMono")
        self.stt_whitelist_edit.setPlaceholderText("e.g. NiseMono, 朋友A")
        self.stt_whitelist_edit.editingFinished.connect(self._on_whitelist_changed)
        stt_row2.addWidget(self.stt_whitelist_edit, stretch=1)
        self.stt_whitelist_apply_btn = QPushButton("应用")
        self.stt_whitelist_apply_btn.clicked.connect(self._on_whitelist_changed)
        stt_row2.addWidget(self.stt_whitelist_apply_btn)
        stt_outer.addLayout(stt_row2)
        layout.addWidget(stt_box)

        layout.addStretch()
        return page

    def _build_memory_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        header.addWidget(QLabel("游戏:"))
        self.memory_game_lbl = QLabel(self.memory.game_id)
        self.memory_game_lbl.setStyleSheet("color: #888;")
        header.addWidget(self.memory_game_lbl)
        path_lbl = QLabel(f"路径: {self.memory.game_dir}")
        path_lbl.setStyleSheet("color: #888;")
        header.addWidget(path_lbl)
        header.addStretch()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh_memory_tab)
        header.addWidget(refresh_btn)
        layout.addLayout(header)

        layout.addWidget(QLabel("本次会话事件流 (写入 observed.md):"))
        self.memory_timeline = QTextEdit()
        self.memory_timeline.setReadOnly(True)
        self.memory_timeline.setFont(QFont("Consolas", 10))
        self.memory_timeline.setMaximumHeight(220)
        layout.addWidget(self.memory_timeline)

        layout.addWidget(QLabel("长期记忆 (markdown — 编辑后点保存):"))
        self.memory_inner_tabs = QTabWidget()
        self._memory_editors: dict[str, tuple[QTextEdit, Path]] = {}
        for label, path in [
            ("玩家档案 (跨游戏)", self.memory.player_path),
            ("本游戏设定 identity", self.memory.identity_path),
            ("梗 / 笑话 jokes", self.memory.jokes_path),
            ("剧情进度 progress", self.memory.progress_path),
            ("自动抽取 auto", self.memory.auto_path),
        ]:
            editor = QTextEdit()
            editor.setFont(QFont("Consolas", 10))
            self.memory_inner_tabs.addTab(editor, label)
            self._memory_editors[label] = (editor, path)
        layout.addWidget(self.memory_inner_tabs, stretch=1)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.session_summary_btn = QPushButton("总结今日会话")
        self.session_summary_btn.setToolTip(
            "flash 总结今天的 observed.md → sessions/<日期>.md。\n"
            "想关电脑前点一下,后续启动 Eri 能引用今天发生的事。"
        )
        self.session_summary_btn.clicked.connect(self._on_summarize_today)
        save_row.addWidget(self.session_summary_btn)
        save_btn = QPushButton("保存当前页")
        save_btn.clicked.connect(self._on_memory_save_current)
        save_row.addWidget(save_btn)
        save_all_btn = QPushButton("保存全部")
        save_all_btn.clicked.connect(self._on_memory_save_all)
        save_row.addWidget(save_all_btn)
        layout.addLayout(save_row)

        # Curate-auto.md row: only meaningful when the 自动抽取 tab is active.
        # Buttons work on the editor's CURRENT-CURSOR line; user clicks once
        # to select the line, then picks a target.
        curate_row = QHBoxLayout()
        curate_row.addWidget(QLabel("当前行 →"))
        for label, target_attr in [
            ("玩家档案", "player_path"),
            ("梗", "jokes_path"),
            ("进度", "progress_path"),
        ]:
            btn = QPushButton(f"推到 {label}")
            btn.clicked.connect(
                lambda _checked=False, attr=target_attr, lbl=label:
                    self._on_curate_auto_promote(attr, lbl)
            )
            curate_row.addWidget(btn)
        del_btn = QPushButton("删除此行")
        del_btn.clicked.connect(self._on_curate_auto_delete)
        curate_row.addWidget(del_btn)
        dedup_btn = QPushButton("清理重复")
        dedup_btn.setToolTip(
            "对 auto.md 做一次去重 (Jaccard 相似度 + 字符集重叠)。"
            "删除内容跟前面某条几乎一样的 bullet。"
        )
        dedup_btn.clicked.connect(self._on_curate_auto_dedup)
        curate_row.addWidget(dedup_btn)
        curate_row.addStretch()
        curate_hint = QLabel("(只对「自动抽取 auto」页生效;点光标先放到要操作的那行)")
        curate_hint.setStyleSheet("color: #888;")
        curate_row.addWidget(curate_hint)
        layout.addLayout(curate_row)

        self._refresh_memory_tab()
        return page

    # ---------- Live2D / VTS tab ----------
    def _build_vts_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Header: pyvts install status. If pyvts didn't import, the tab is
        # informational only (Connect button still visible but greyed).
        avail_box = QHBoxLayout()
        avail_lbl = QLabel("pyvts:")
        avail_box.addWidget(avail_lbl)
        if pyvts_available():
            self.vts_pyvts_lbl = QLabel("已安装")
            self.vts_pyvts_lbl.setStyleSheet("color: #2a2;")
        else:
            self.vts_pyvts_lbl = QLabel(f"未安装 — {pyvts_import_error()[:80]}")
            self.vts_pyvts_lbl.setStyleSheet("color: #c33;")
        avail_box.addWidget(self.vts_pyvts_lbl, stretch=1)
        layout.addLayout(avail_box)

        # Connection group
        conn_box = QGroupBox("VTube Studio 连接")
        conn_outer = QVBoxLayout(conn_box)
        conn_row1 = QHBoxLayout()
        conn_row1.addWidget(QLabel("Host:"))
        self.vts_host_edit = QLineEdit("localhost")
        self.vts_host_edit.setMaximumWidth(140)
        conn_row1.addWidget(self.vts_host_edit)
        conn_row1.addWidget(QLabel("端口:"))
        self.vts_port_spin = QSpinBox()
        self.vts_port_spin.setRange(1, 65535)
        self.vts_port_spin.setValue(8001)
        conn_row1.addWidget(self.vts_port_spin)
        self.vts_connect_btn = QPushButton("连接")
        self.vts_connect_btn.clicked.connect(self._on_vts_connect)
        self.vts_connect_btn.setEnabled(pyvts_available())
        conn_row1.addWidget(self.vts_connect_btn)
        self.vts_reconnect_btn = QPushButton("重连")
        self.vts_reconnect_btn.clicked.connect(self._on_vts_reconnect)
        self.vts_reconnect_btn.setEnabled(False)
        conn_row1.addWidget(self.vts_reconnect_btn)
        self.vts_disconnect_btn = QPushButton("断开")
        self.vts_disconnect_btn.clicked.connect(self._on_vts_disconnect)
        self.vts_disconnect_btn.setEnabled(False)
        conn_row1.addWidget(self.vts_disconnect_btn)
        conn_row1.addStretch()
        conn_outer.addLayout(conn_row1)

        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("状态:"))
        self.vts_status_lbl = QLabel("(未连接)")
        self.vts_status_lbl.setStyleSheet("color: #c33;")
        status_row.addWidget(self.vts_status_lbl, stretch=1)
        conn_outer.addLayout(status_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型:"))
        self.vts_model_lbl = QLabel("(无)")
        self.vts_model_lbl.setStyleSheet("color: #888;")
        model_row.addWidget(self.vts_model_lbl, stretch=1)
        self.vts_refresh_btn = QPushButton("刷新模型能力")
        self.vts_refresh_btn.clicked.connect(self._on_vts_refresh_caps)
        self.vts_refresh_btn.setEnabled(False)
        model_row.addWidget(self.vts_refresh_btn)
        conn_outer.addLayout(model_row)

        init_row = QHBoxLayout()
        init_row.addWidget(QLabel("外观预设 hotkey:"))
        self.vts_init_hotkey_edit = QLineEdit("Eri")
        self.vts_init_hotkey_edit.setMaximumWidth(160)
        self.vts_init_hotkey_edit.setToolTip(
            "连接 VTS 后自动触发的 hotkey 名 (大小写不敏感的精确匹配)。\n"
            "把你在 VTS 里调好的外观滑块保存到一个 hotkey 里(默认起名 'Eri'),"
            "LivePartner 启动时会自动应用,免去每次手动点。"
        )
        init_row.addWidget(self.vts_init_hotkey_edit)
        init_row.addStretch()
        conn_outer.addLayout(init_row)

        layout.addWidget(conn_box)

        # Manual expression test
        test_box = QGroupBox("手动测试")
        test_outer = QVBoxLayout(test_box)
        test_row = QHBoxLayout()
        test_row.addWidget(QLabel("表情:"))
        self.vts_expr_combo = QComboBox()
        self.vts_expr_combo.setMinimumWidth(220)
        test_row.addWidget(self.vts_expr_combo)
        self.vts_expr_trigger_btn = QPushButton("触发")
        self.vts_expr_trigger_btn.clicked.connect(self._on_vts_test_expression)
        self.vts_expr_trigger_btn.setEnabled(False)
        test_row.addWidget(self.vts_expr_trigger_btn)
        test_row.addStretch()
        test_outer.addLayout(test_row)

        hotkey_row = QHBoxLayout()
        hotkey_row.addWidget(QLabel("Hotkey:"))
        self.vts_hotkey_combo = QComboBox()
        self.vts_hotkey_combo.setMinimumWidth(220)
        hotkey_row.addWidget(self.vts_hotkey_combo)
        self.vts_hotkey_trigger_btn = QPushButton("触发")
        self.vts_hotkey_trigger_btn.clicked.connect(self._on_vts_test_hotkey)
        self.vts_hotkey_trigger_btn.setEnabled(False)
        hotkey_row.addWidget(self.vts_hotkey_trigger_btn)
        hotkey_row.addStretch()
        test_outer.addLayout(hotkey_row)
        layout.addWidget(test_box)

        # Idle-motion toggles (auto blink / head sway / breath). All on by
        # default — the avatar reads as alive even between turns. Drive the
        # IdleMotion instance directly via the toggles; changes take effect
        # on the next 50 ms tick.
        idle_box = QGroupBox("Idle 动作 (角色'活'的感觉, 跟表情独立)")
        idle_outer = QHBoxLayout(idle_box)
        self.vts_blink_check = QCheckBox("自动眨眼")
        self.vts_blink_check.setChecked(True)
        self.vts_blink_check.setToolTip("随机 2-5s 自动眨一次,驱动 EyeOpenLeft/Right 参数")
        self.vts_blink_check.toggled.connect(self._on_vts_idle_toggle)
        idle_outer.addWidget(self.vts_blink_check)
        self.vts_sway_check = QCheckBox("头部摆动")
        self.vts_sway_check.setChecked(True)
        self.vts_sway_check.setToolTip("FaceAngleX/Y/Z + BodyAngleX 慢 sinewave (±3-5°)")
        self.vts_sway_check.toggled.connect(self._on_vts_idle_toggle)
        idle_outer.addWidget(self.vts_sway_check)
        self.vts_breath_check = QCheckBox("呼吸")
        self.vts_breath_check.setChecked(True)
        self.vts_breath_check.setToolTip("ParamBreath 4s 周期 0↔1,胸口起伏")
        self.vts_breath_check.toggled.connect(self._on_vts_idle_toggle)
        idle_outer.addWidget(self.vts_breath_check)
        idle_outer.addStretch()
        layout.addWidget(idle_box)

        # Tips block
        tips = QTextEdit()
        tips.setReadOnly(True)
        tips.setMaximumHeight(170)
        tips.setStyleSheet("background-color: #f8f8f8; color: #444;")
        tips.setPlainText(
            "首次连接:\n"
            "  1. Steam 装 VTube Studio,启动它,设置→API→\"Start API\" 打开 (默认端口 8001)\n"
            "  2. 加载一个 Live 2D 模型(VTS 自带 Hiyori 或从 Booth.pm 下免费模型)\n"
            "  3. 点上面 [连接] 按钮 — VTS 窗口里会弹\"是否允许插件 Eri Director\",点 Allow\n"
            "  4. 授权 token 自动保存到 .memory/vts_token.txt, 下次自动重连\n"
            "\n"
            "直播模式:VTS 普通窗口 → OBS 用 Window Capture 抓\n"
            "桌宠模式:VTS 设置 → 启用透明背景 + 总在最前, 把窗口拖到屏幕角落即可\n"
            "\n"
            "模型加载在 VTS 里换 → 这边点 [刷新模型能力] 重新枚举 hotkey / 表情"
        )
        layout.addWidget(tips)

        layout.addStretch()

        # Status poll timer (refresh label + dropdowns when connection state
        # or capabilities change). Lightweight: just a status snapshot read.
        self._vts_poll_timer = QTimer(self)
        self._vts_poll_timer.setInterval(500)
        self._vts_poll_timer.timeout.connect(self._refresh_vts_status)
        self._vts_poll_timer.start()

        return page

    def _on_vts_connect(self) -> None:
        if not pyvts_available():
            QMessageBox.warning(self, "pyvts 未安装",
                                "请先 `uv pip install pyvts>=0.3.3`")
            return
        if self.vts is None:
            init_hotkey = self.vts_init_hotkey_edit.text().strip()
            self.vts = VTSController(
                logger=self._log,
                init_hotkey_name=init_hotkey,
            )
        host = self.vts_host_edit.text().strip() or "localhost"
        port = self.vts_port_spin.value()
        self.vts.start(host=host, port=port)
        self.statusBar().showMessage(f"连接 VTS @ {host}:{port} …")
        self.vts_connect_btn.setEnabled(False)
        self.vts_reconnect_btn.setEnabled(True)
        self.vts_disconnect_btn.setEnabled(True)
        self.vts_refresh_btn.setEnabled(True)

    def _on_vts_reconnect(self) -> None:
        if self.vts is None:
            return
        host = self.vts_host_edit.text().strip() or "localhost"
        port = self.vts_port_spin.value()
        self.vts.reconnect(host=host, port=port)
        self.statusBar().showMessage(f"重连 VTS @ {host}:{port} …")

    def _on_vts_disconnect(self) -> None:
        if self.idle_motion is not None:
            try:
                self.idle_motion.stop()
            except Exception:
                pass
            self.idle_motion = None
            self._idle_motion_started = False
        if self.vts is None:
            return
        self.vts.stop()
        self.vts = None
        self.vts_connect_btn.setEnabled(True)
        self.vts_reconnect_btn.setEnabled(False)
        self.vts_disconnect_btn.setEnabled(False)
        self.vts_refresh_btn.setEnabled(False)
        self.vts_expr_trigger_btn.setEnabled(False)
        self.vts_hotkey_trigger_btn.setEnabled(False)
        self.statusBar().showMessage("VTS 已断开")

    def _on_vts_idle_toggle(self) -> None:
        """Push the 3 idle-motion checkbox states into the IdleMotion instance.
        Safe to call even before IdleMotion exists (skips silently)."""
        if self.idle_motion is None:
            return
        self.idle_motion.set_auto_blink(self.vts_blink_check.isChecked())
        self.idle_motion.set_idle_sway(self.vts_sway_check.isChecked())
        self.idle_motion.set_breath(self.vts_breath_check.isChecked())

    def _on_self_describer_tick(self) -> None:
        """Periodic 90 s tick: spawn a worker that captures the VTS window
        and asks flash-vision for a short description. Result drops into
        memory on completion."""
        if self.vts is None or not self.vts.is_connected:
            return
        w = SelfDescriberWorker()
        w.log.connect(self._log)
        w.done.connect(lambda desc, w=w: self._on_self_describer_done(desc, w))
        w.finished.connect(lambda w=w: self._cleanup_self_describer_worker(w))
        self._self_describer_workers.append(w)
        w.start()

    def _on_self_describer_done(self, description: str, w: "SelfDescriberWorker") -> None:
        if not description:
            return
        changed = self.memory.update_self_appearance(description)
        if changed:
            self._log("[self] 形象有变化 — 下次 AI 开口会主动评论")

    def _cleanup_self_describer_worker(self, w: "SelfDescriberWorker") -> None:
        try:
            self._self_describer_workers.remove(w)
        except ValueError:
            pass

    def _on_vts_refresh_caps(self) -> None:
        if self.vts is None:
            return
        self.vts.refresh_capabilities()

    def _on_vts_test_expression(self) -> None:
        if self.vts is None:
            return
        file = self.vts_expr_combo.currentData()
        if file:
            self.vts.set_expression_file(file)
            self._log(f"[vts] manual trigger expression: {file}")

    def _on_vts_test_hotkey(self) -> None:
        if self.vts is None:
            return
        hotkey_id = self.vts_hotkey_combo.currentData()
        if hotkey_id:
            self.vts.trigger_hotkey(hotkey_id)
            self._log(f"[vts] manual trigger hotkey: {hotkey_id}")

    def _refresh_vts_status(self) -> None:
        """500 ms tick — refresh status label + populate dropdowns when caps change."""
        if not hasattr(self, "vts_status_lbl"):
            return
        if self.vts is None:
            self.vts_status_lbl.setText("(未连接)")
            self.vts_status_lbl.setStyleSheet("color: #c33;")
            self.vts_model_lbl.setText("(无)")
            self.vts_model_lbl.setStyleSheet("color: #888;")
            return
        st = self.vts.status
        if not st.available:
            self.vts_status_lbl.setText(f"pyvts 不可用 — {st.last_error[:60]}")
            self.vts_status_lbl.setStyleSheet("color: #c33;")
            return
        if not st.connected:
            txt = f"WS 未连接 — {st.last_error}" if st.last_error else "WS 连接中…"
            self.vts_status_lbl.setText(txt)
            self.vts_status_lbl.setStyleSheet("color: #c80;")
            return
        if not st.authed:
            self.vts_status_lbl.setText("已连接, 等待授权 (检查 VTS 弹窗)")
            self.vts_status_lbl.setStyleSheet("color: #c80;")
            return
        n_expr = sum(1 for h in st.hotkeys if h.type == "ToggleExpression")
        self.vts_status_lbl.setText(
            f"● 已就绪  ·  {len(st.hotkeys)} hotkeys ({n_expr} expressions)"
            + (f"  ·  mouth={st.mouth_param}" if st.mouth_param else "  ·  无嘴参数!")
        )
        self.vts_status_lbl.setStyleSheet("color: #2a2;")
        self.vts_model_lbl.setText(st.model_name or "(无模型加载)")
        self.vts_model_lbl.setStyleSheet("color: #444;")

        # Lazy-start IdleMotion the first time we reach "ready" on a fresh
        # connection. Params are now populated in vts.status so resolve will
        # work. Disconnect / stop()-then-reconnect will clear the flag.
        if not self._idle_motion_started and st.parameters:
            self.idle_motion = IdleMotion(self.vts, logger=self._log)
            self.idle_motion.start()
            # Apply current UI toggle state immediately.
            self._on_vts_idle_toggle()
            self._idle_motion_started = True
            # Spin up visual self-awareness — take a baseline snapshot now
            # (won't fire 'changed' marker on first time, just sets current),
            # then poll every 90 s.
            self._on_self_describer_tick()
            self._self_describer_timer.start()

        # Repopulate dropdowns iff the hotkey set changed since last refresh.
        # (Hashing the list-of-ids is enough — model swaps always change ids.)
        new_hotkey_sig = tuple(h.id for h in st.hotkeys)
        if getattr(self, "_vts_last_hotkey_sig", None) != new_hotkey_sig:
            self._vts_last_hotkey_sig = new_hotkey_sig
            # Expressions
            self.vts_expr_combo.clear()
            for h in st.hotkeys:
                if h.type == "ToggleExpression" and h.file:
                    label = h.name or h.file
                    self.vts_expr_combo.addItem(f"{label}  [{h.file}]", userData=h.file)
            self.vts_expr_trigger_btn.setEnabled(self.vts_expr_combo.count() > 0)
            # All hotkeys (incl. animations, model moves)
            self.vts_hotkey_combo.clear()
            for h in st.hotkeys:
                self.vts_hotkey_combo.addItem(
                    f"[{h.type}] {h.name}", userData=h.id,
                )
            self.vts_hotkey_trigger_btn.setEnabled(self.vts_hotkey_combo.count() > 0)

    # ---------- 直播 / 弹幕 tab ----------
    def _build_danmaku_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        # Header — bilibili-api-python install status.
        avail_box = QHBoxLayout()
        avail_box.addWidget(QLabel("bilibili-api-python:"))
        if danmaku_available():
            lbl = QLabel("已安装")
            lbl.setStyleSheet("color: #2a2;")
        else:
            lbl = QLabel(f"未安装 — {danmaku_import_error()[:80]}")
            lbl.setStyleSheet("color: #c33;")
        avail_box.addWidget(lbl, stretch=1)
        layout.addLayout(avail_box)

        # Connection
        conn_box = QGroupBox("Bilibili 直播间")
        conn_outer = QVBoxLayout(conn_box)
        row = QHBoxLayout()
        row.addWidget(QLabel("房间号:"))
        self.danmaku_room_edit = QLineEdit("")
        self.danmaku_room_edit.setPlaceholderText("e.g. 22637261")
        self.danmaku_room_edit.setMaximumWidth(140)
        row.addWidget(self.danmaku_room_edit)
        self.danmaku_connect_btn = QPushButton("连接")
        self.danmaku_connect_btn.clicked.connect(self._on_danmaku_connect)
        self.danmaku_connect_btn.setEnabled(danmaku_available())
        row.addWidget(self.danmaku_connect_btn)
        self.danmaku_disconnect_btn = QPushButton("断开")
        self.danmaku_disconnect_btn.clicked.connect(self._on_danmaku_disconnect)
        self.danmaku_disconnect_btn.setEnabled(False)
        row.addWidget(self.danmaku_disconnect_btn)
        row.addStretch()
        conn_outer.addLayout(row)
        st_row = QHBoxLayout()
        st_row.addWidget(QLabel("状态:"))
        self.danmaku_status_lbl = QLabel("(未连接)")
        self.danmaku_status_lbl.setStyleSheet("color: #c33;")
        st_row.addWidget(self.danmaku_status_lbl, stretch=1)
        conn_outer.addLayout(st_row)
        layout.addWidget(conn_box)

        # AI response policy
        ai_box = QGroupBox("AI 回应策略")
        ai_outer = QHBoxLayout(ai_box)
        self.danmaku_respond_check = QCheckBox("让 Eri 回应弹幕")
        self.danmaku_respond_check.setChecked(False)
        self.danmaku_respond_check.setToolTip(
            "勾上后,合格的弹幕(过了冷却 + 长度足够) 会喂给 AI 触发 Eri 回应。"
            "未勾时弹幕只显示在下面日志,不消耗 LLM token。"
        )
        ai_outer.addWidget(self.danmaku_respond_check)
        ai_outer.addWidget(QLabel("冷却(秒):"))
        self.danmaku_cooldown_spin = QSpinBox()
        self.danmaku_cooldown_spin.setRange(5, 600)
        self.danmaku_cooldown_spin.setValue(30)
        self.danmaku_cooldown_spin.setToolTip("两次回应弹幕间最短间隔,避免高流量房间 token 烧光")
        ai_outer.addWidget(self.danmaku_cooldown_spin)
        ai_outer.addWidget(QLabel("最短长度:"))
        self.danmaku_min_len_spin = QSpinBox()
        self.danmaku_min_len_spin.setRange(1, 50)
        self.danmaku_min_len_spin.setValue(4)
        self.danmaku_min_len_spin.setToolTip("低于此长度的弹幕(如 666 / 表情)不触发 AI")
        ai_outer.addWidget(self.danmaku_min_len_spin)
        ai_outer.addStretch()
        layout.addWidget(ai_box)

        # Recent danmaku log
        layout.addWidget(QLabel("最近弹幕 (最新在上):"))
        self.danmaku_log = QTextEdit()
        self.danmaku_log.setReadOnly(True)
        self.danmaku_log.setFont(QFont("Consolas", 9))
        layout.addWidget(self.danmaku_log, stretch=1)

        return page

    def _on_danmaku_connect(self) -> None:
        if not danmaku_available():
            QMessageBox.warning(self, "提示", "bilibili-api-python 未安装")
            return
        try:
            room_id = int(self.danmaku_room_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "提示", "房间号必须是数字")
            return
        if self.danmaku is None:
            self.danmaku = BilibiliDanmakuSource(
                # The callback fires on the danmaku asyncio thread — emit
                # via Qt signal so the slot runs on the UI thread.
                on_message=lambda name, text: self.danmaku_signal.emit(name, text),
                logger=self._log,
            )
        self.danmaku.start(room_id)
        self.danmaku_status_lbl.setText(f"连接中 room={room_id}…")
        self.danmaku_status_lbl.setStyleSheet("color: #c80;")
        self.danmaku_connect_btn.setEnabled(False)
        self.danmaku_disconnect_btn.setEnabled(True)
        # Drive the status label off the same VTS poll timer (no new timer).
        QTimer.singleShot(800, self._refresh_danmaku_status)

    def _on_danmaku_disconnect(self) -> None:
        if self.danmaku is not None:
            try:
                self.danmaku.stop()
            except Exception:
                pass
            self.danmaku = None
        self.danmaku_status_lbl.setText("(未连接)")
        self.danmaku_status_lbl.setStyleSheet("color: #c33;")
        self.danmaku_connect_btn.setEnabled(True)
        self.danmaku_disconnect_btn.setEnabled(False)

    def _refresh_danmaku_status(self) -> None:
        if self.danmaku is None:
            return
        st = self.danmaku.status
        if st.connected:
            self.danmaku_status_lbl.setText(f"● 已连接 · room={st.room_id}")
            self.danmaku_status_lbl.setStyleSheet("color: #2a2;")
        elif st.last_error:
            self.danmaku_status_lbl.setText(f"未连接 — {st.last_error[:80]}")
            self.danmaku_status_lbl.setStyleSheet("color: #c33;")
        else:
            # Still trying — re-poll
            QTimer.singleShot(800, self._refresh_danmaku_status)

    def _on_danmaku_arrived(self, name: str, text: str) -> None:
        """Slot fired by danmaku_signal — runs on UI thread. Always logs;
        optionally dispatches to AI subject to cooldown + min-length + busy."""
        # Tail-append to the log (newest on top by inserting at start).
        cursor = self.danmaku_log.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        cursor.insertText(f"  {name}: {text}\n")
        # Cap to ~50 lines so the widget doesn't grow unbounded.
        if self.danmaku_log.document().lineCount() > 50:
            cursor = self.danmaku_log.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            while self.danmaku_log.document().lineCount() > 50:
                cursor.movePosition(cursor.MoveOperation.StartOfLine,
                                    cursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()

        # AI response gating
        if not self.danmaku_respond_check.isChecked():
            return
        if len(text) < self.danmaku_min_len_spin.value():
            return  # too short — likely emote / spam
        cooldown_s = self.danmaku_cooldown_spin.value()
        now = time.monotonic()
        if (now - self._last_danmaku_dispatch_ts) < cooldown_s:
            # In cooldown — stash latest, drop older. AI will pick this up
            # when the cooldown timer + AI free coincide. We don't auto-fire
            # on cooldown expiry alone; next danmaku after cooldown does it.
            self.pending_danmaku = (name, text)
            return
        if self._decision_busy:
            self.pending_danmaku = (name, text)
            return
        self._dispatch_danmaku(name, text)

    def _dispatch_danmaku(self, name: str, text: str) -> None:
        """Spawn a decision for one (viewer_name, message) — same path as STT
        voice except the event prefix tells Eri this is a viewer, not the player."""
        self._last_danmaku_dispatch_ts = time.monotonic()
        # Fake-user dict matches what STT passes — _handle_voice_text reads
        # name from this. Prefix "观众·" makes Eri's prompt aware it's chat.
        user = {"name": f"观众·{name}"}
        self._log(f"[弹幕] 处理 {name!r}: {text!r}")
        self._handle_voice_text(user, text)

    # ---------- memory tab helpers ----------
    def _refresh_memory_tab(self) -> None:
        self._refresh_memory_timeline()
        self._refresh_memory_editors()

    def _refresh_memory_timeline(self) -> None:
        if not hasattr(self, "memory_timeline"):
            return
        entries = self.memory.recent_entries(n=80)
        if not entries:
            self.memory_timeline.setPlainText("(空 — 本次会话尚无事件)")
            return
        lines = []
        for e in entries:
            tstr = datetime.fromtimestamp(e.ts).strftime("%H:%M:%S")
            lines.append(f"{tstr}  [{e.kind:6s}] {e.text}")
        self.memory_timeline.setPlainText("\n".join(lines))
        sb = self.memory_timeline.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_memory_editors(self) -> None:
        if not hasattr(self, "_memory_editors"):
            return
        for label, (editor, path) in self._memory_editors.items():
            try:
                text = path.read_text(encoding="utf-8") if path.exists() else ""
            except OSError:
                text = ""
            editor.setPlainText(text)

    def _on_memory_save_current(self) -> None:
        idx = self.memory_inner_tabs.currentIndex()
        label = self.memory_inner_tabs.tabText(idx)
        editor, path = self._memory_editors[label]
        self._save_memory_editor(label, editor, path)

    def _on_memory_save_all(self) -> None:
        for label, (editor, path) in self._memory_editors.items():
            self._save_memory_editor(label, editor, path)

    def _save_memory_editor(self, label: str, editor: QTextEdit, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(editor.toPlainText(), encoding="utf-8")
            self._log(f"[memory] saved {label} → {path} ({path.stat().st_size} B)")
            self.statusBar().showMessage(f"已保存 {label}")
        except OSError as e:
            QMessageBox.critical(self, "保存失败", f"{label}: {e}")

    # ---------- session summary ----------
    def _on_summarize_today(self) -> None:
        if getattr(self, "_session_summary_worker", None) is not None \
                and self._session_summary_worker.isRunning():
            return
        events_text = self.memory.read_observed_today()
        if not events_text.strip():
            QMessageBox.information(
                self, "提示", "今天的 observed.md 还没内容,先玩一会儿再来总结。"
            )
            return
        self.session_summary_btn.setEnabled(False)
        self.session_summary_btn.setText("总结中…")
        self._log(f"[session-summary] start · {len(events_text)} chars input")
        self._session_summary_worker = SessionSummaryWorker(events_text)
        self._session_summary_worker.log.connect(self._log)
        self._session_summary_worker.done.connect(self._on_session_summary_done)
        self._session_summary_worker.finished.connect(self._on_session_summary_cleanup)
        self._session_summary_worker.start()

    def _on_session_summary_done(self, summary: str) -> None:
        if not summary:
            QMessageBox.warning(self, "总结失败", "flash 返回空,看日志找原因。")
            return
        try:
            path = self.memory.write_session_summary(summary)
            self._log(f"[session-summary] 写入 {path}")
            QMessageBox.information(
                self, "完成",
                f"今日会话总结已写入:\n{path}\n\n预览:\n\n{summary[:300]}…"
            )
        except OSError as e:
            QMessageBox.critical(self, "写入失败", str(e))

    def _on_session_summary_cleanup(self) -> None:
        self.session_summary_btn.setEnabled(True)
        self.session_summary_btn.setText("总结今日会话")
        self._session_summary_worker = None

    # ---------- auto.md curate buttons ----------
    def _auto_editor_and_line(self) -> tuple[QTextEdit, str, int] | None:
        """Return (editor, current_line_text, line_number) IF the auto tab is
        active and the cursor is on a non-empty line. Otherwise pops a hint
        and returns None.
        """
        # Find the auto editor (path == self.memory.auto_path).
        target = None
        for label, (editor, path) in self._memory_editors.items():
            if path == self.memory.auto_path:
                target = (label, editor)
                break
        if target is None:
            QMessageBox.information(self, "提示", "找不到自动抽取页")
            return None
        label, editor = target
        # Switch to that tab so the user sees what happened.
        for i in range(self.memory_inner_tabs.count()):
            if self.memory_inner_tabs.tabText(i) == label:
                self.memory_inner_tabs.setCurrentIndex(i)
                break
        cursor = editor.textCursor()
        block = cursor.block()
        line = block.text()
        line_no = block.blockNumber()
        if not line.strip() or not line.lstrip().startswith("-"):
            QMessageBox.information(
                self, "提示",
                "请把光标放到要操作的那一行(以 - 起头的事实行)上,再点按钮"
            )
            return None
        return editor, line, line_no

    def _on_curate_auto_promote(self, target_path_attr: str, target_label: str) -> None:
        got = self._auto_editor_and_line()
        if got is None:
            return
        editor, line, line_no = got
        target_path: Path = getattr(self.memory, target_path_attr)

        # Append to target file.
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            new = existing.rstrip() + ("\n" if existing.strip() else "") + line.strip() + "\n"
            target_path.write_text(new, encoding="utf-8")
        except OSError as e:
            QMessageBox.critical(self, "推送失败", str(e))
            return

        self._delete_line_from_editor(editor, line_no)
        # Save the auto editor (removes the line from auto.md too).
        self._save_memory_editor("自动抽取 auto", editor, self.memory.auto_path)
        # Refresh the target editor in the inner tabs so the user can see it.
        for lbl, (ed, p) in self._memory_editors.items():
            if p == target_path:
                try:
                    ed.setPlainText(target_path.read_text(encoding="utf-8"))
                except OSError:
                    pass
                break
        self._log(f"[memory] 推到 {target_label}: {line.strip()}")
        self.statusBar().showMessage(f"已推到 {target_label}")

    def _on_curate_auto_delete(self) -> None:
        got = self._auto_editor_and_line()
        if got is None:
            return
        editor, line, line_no = got
        self._delete_line_from_editor(editor, line_no)
        self._save_memory_editor("自动抽取 auto", editor, self.memory.auto_path)
        self._log(f"[memory] 删除: {line.strip()}")
        self.statusBar().showMessage("已删除")

    def _on_curate_auto_dedup(self) -> None:
        dropped = self.memory.dedup_auto_facts()
        # Refresh the auto editor so the user sees the cleaned content.
        for label, (editor, path) in self._memory_editors.items():
            if path == self.memory.auto_path:
                try:
                    editor.setPlainText(
                        path.read_text(encoding="utf-8") if path.exists() else ""
                    )
                except OSError:
                    pass
                break
        if dropped > 0:
            self._log(f"[memory] auto.md 去重: 删除 {dropped} 条重复")
            self.statusBar().showMessage(f"已去重 {dropped} 条")
        else:
            self._log("[memory] auto.md 没有重复要清理")
            self.statusBar().showMessage("没有重复")

    @staticmethod
    def _delete_line_from_editor(editor: QTextEdit, line_no: int) -> None:
        """Delete the entire line ``line_no`` (0-indexed) from a QTextEdit."""
        doc = editor.document()
        block = doc.findBlockByNumber(line_no)
        if not block.isValid():
            return
        cursor = editor.textCursor()
        cursor.setPosition(block.position())
        cursor.movePosition(
            cursor.MoveOperation.EndOfBlock, cursor.MoveMode.KeepAnchor
        )
        cursor.removeSelectedText()
        # Also remove the trailing newline so we don't leave a blank line.
        cursor.deleteChar()

    # ---------- logging ----------
    def _log(self, msg: str) -> None:
        self.log_view.append(msg)
        # Mirror to .memory/ui.log so the user can hand me the file after
        # a session and I can diagnose silently-failed turns (e.g. parse
        # misses, errors below the visible scroll).
        try:
            if not hasattr(self, "_ui_log_fp") or self._ui_log_fp is None:
                from .._tts_server_proc import LOG_PATH as _TTS_LOG
                ui_log_path = _TTS_LOG.parent / "ui.log"
                ui_log_path.parent.mkdir(parents=True, exist_ok=True)
                self._ui_log_fp = ui_log_path.open("a", encoding="utf-8", buffering=1)
                self._ui_log_fp.write(
                    f"\n\n===== UI session {datetime.now().isoformat()} =====\n"
                )
            self._ui_log_fp.write(
                f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
            )
        except OSError:
            pass

    # ---------- TTS engine ----------
    def _on_engine_changed(self) -> None:
        is_qwen3 = self.tts_engine_combo.currentData() == "qwen3"
        self.qwen3_url_edit.setEnabled(is_qwen3)
        self.qwen3_probe_btn.setEnabled(is_qwen3)

    def _on_probe_qwen3(self) -> None:
        if self.probe_worker is not None and self.probe_worker.isRunning():
            return
        url = self.qwen3_url_edit.text().strip().rstrip("/")
        self.qwen3_status.setText("检测中…")
        self.qwen3_status.setStyleSheet("color: #c80;")
        self.probe_worker = Qwen3ProbeWorker(url)
        self.probe_worker.done.connect(self._on_probe_done)
        self.probe_worker.finished.connect(self._cleanup_probe)
        self.probe_worker.start()

    def _on_probe_done(self, alive: bool, voices: list, info: str) -> None:
        if alive:
            self.qwen3_status.setText(f"在线 · voices: {', '.join(voices) or '(空)'}")
            self.qwen3_status.setStyleSheet("color: #2a2;")
            # Chain to next quickstart phase if a quickstart is in progress.
            if self._quickstart_active:
                self._quickstart_step("stt")
        else:
            # Suppress 不可达 flicker during startup poll — the timer's elapsed
            # counter is more useful there. Show 不可达 only for standalone
            # 检测 button presses.
            startup_polling = (
                hasattr(self, "_tts_poll_timer") and self._tts_poll_timer.isActive()
            )
            if not startup_polling:
                self.qwen3_status.setText(f"不可达 ({info})")
                self.qwen3_status.setStyleSheet("color: #c33;")

    def _cleanup_probe(self) -> None:
        self.probe_worker = None

    # ---------- one-tap startup ----------
    def _on_quickstart(self) -> None:
        if self._quickstart_active:
            return
        self._quickstart_active = True
        self.quickstart_btn.setEnabled(False)
        self._log("[quickstart] === 启动序列 begin ===")
        self.statusBar().showMessage("一键启动中…")
        self._quickstart_step("capture")

    def _quickstart_step(self, phase: str) -> None:
        """Drive the state machine. Each phase either:
        - already-ready  → log + chain to next immediately
        - kick-off async → return and let the existing completion signal call
          us again with the next phase name.
        """
        if not self._quickstart_active:
            return

        if phase == "capture":
            self.quickstart_status_lbl.setText("1/4 采集卡…")
            self.quickstart_status_lbl.setStyleSheet("color: #c80;")
            if self.capture is not None and self.capture.is_running:
                self._log("[quickstart] capture 已就绪")
                self._quickstart_step("tts")
                return
            self._log("[quickstart] 启动 capture")
            try:
                self._on_capture_start()
            except Exception as e:
                self._quickstart_abort(f"capture failed: {e}")
                return
            # Capture is synchronous — check + advance after a short tick.
            QTimer.singleShot(300, lambda: self._quickstart_step("tts"))

        elif phase == "tts":
            self.quickstart_status_lbl.setText("2/4 TTS 服务…")
            if self.qwen3_status.text().startswith("在线"):
                self._log("[quickstart] TTS 已在线")
                self._quickstart_step("stt")
                return
            if not self.tts_server.is_running:
                self._log("[quickstart] 启动 TTS server (~80s 首次加载)")
                self._on_tts_server_toggle()
            else:
                self._log("[quickstart] TTS 正在启动,等待 probe")
            # Advance is triggered by _on_probe_done when alive=True
            return

        elif phase == "stt":
            self.quickstart_status_lbl.setText("3/4 STT 模型…")
            if self.stt is not None:
                self._log("[quickstart] STT 已就绪")
                self._quickstart_step("mumble")
                return
            self._log("[quickstart] 加载 STT (large-v3-turbo)")
            self._on_stt_load()
            # Advance triggered by _on_stt_loaded

        elif phase == "mumble":
            self.quickstart_status_lbl.setText("4/4 Mumble…")
            if self.bot is not None:
                self._log("[quickstart] Mumble 已连接")
                self._quickstart_step("done")
                return
            self._log("[quickstart] 连接 Mumble")
            self._on_connect_mumble()
            # Advance triggered by _on_mumble_connected

        elif phase == "done":
            self._log("[quickstart] ✓ 全部就绪 — capture + TTS + STT + Mumble")
            self.statusBar().showMessage("一键启动完成")
            self.quickstart_status_lbl.setText("● 完成")
            self.quickstart_status_lbl.setStyleSheet("color: #2a2;")
            self._quickstart_active = False
            self.quickstart_btn.setEnabled(True)

    def _quickstart_abort(self, reason: str) -> None:
        if not self._quickstart_active:
            return
        self._log(f"[quickstart] ✗ 中止: {reason}")
        self.statusBar().showMessage(f"一键启动中止: {reason}")
        self.quickstart_status_lbl.setText("✗ 中止")
        self.quickstart_status_lbl.setStyleSheet("color: #c33;")
        self._quickstart_active = False
        self.quickstart_btn.setEnabled(True)

    # ---------- TTS server subprocess ----------
    def _on_tts_server_toggle(self) -> None:
        if self.tts_server.is_running:
            self._log(f"[tts-server] stopping pid={self.tts_server.pid}…")
            self.tts_server_btn.setEnabled(False)
            self.tts_server.stop()
            self._log("[tts-server] stopped")
            self.tts_server_btn.setText("启动 TTS 服务")
            self.tts_server_btn.setEnabled(True)
            self.qwen3_status.setText("(已停止)")
            self.qwen3_status.setStyleSheet("color: #888;")
            if hasattr(self, "_tts_poll_timer") and self._tts_poll_timer.isActive():
                self._tts_poll_timer.stop()
            return
        try:
            pid = self.tts_server.start()
        except FileNotFoundError as e:
            self._log(f"[tts-server] cannot start: {e}")
            QMessageBox.critical(self, "启动失败", str(e))
            return
        except Exception as e:
            self._log(f"[tts-server] start error: {type(e).__name__}: {e}")
            QMessageBox.critical(self, "启动失败", f"{type(e).__name__}: {e}")
            return
        self._log(
            f"[tts-server] spawned pid={pid} · 首次加载约 60-90s "
            f"(模型 ~60s + CUDA graph 暖机 ~20s)·"
            f"log={self.tts_server.log_path}"
        )
        self.tts_server_btn.setText("停止 TTS 服务")
        self.qwen3_status.setText("启动中… 0s")
        self.qwen3_status.setStyleSheet("color: #c80;")

        # Recurring poller: probe every 5s for up to 3 min, show elapsed time
        # so user knows it's not frozen. Stops itself on success or timeout.
        import time as _time
        self._tts_poll_t0 = _time.monotonic()
        if not hasattr(self, "_tts_poll_timer"):
            self._tts_poll_timer = QTimer(self)
            self._tts_poll_timer.timeout.connect(self._tts_server_probe_tick)
        self._tts_poll_timer.setInterval(5000)
        self._tts_poll_timer.start()
        # Also probe immediately at t=5s — most of the time the user is
        # restarting and the model file is in OS cache, so it could be ready
        # in just a few seconds.
        QTimer.singleShot(5000, self._tts_server_probe_tick)

    def _tts_server_probe_tick(self) -> None:
        import time as _time
        # If the child died, surface that loudly instead of silently probing.
        if not self.tts_server.is_running:
            self._log("[tts-server] 子进程已退出 — 查看 .memory/tts_server.log")
            self.qwen3_status.setText("子进程退出")
            self.qwen3_status.setStyleSheet("color: #c33;")
            self.tts_server_btn.setText("启动 TTS 服务")
            if hasattr(self, "_tts_poll_timer"):
                self._tts_poll_timer.stop()
            if self._quickstart_active:
                self._quickstart_abort("TTS 子进程退出")
            return

        if self.qwen3_status.text().startswith("在线"):
            # Confirmed alive — stop polling.
            if hasattr(self, "_tts_poll_timer"):
                self._tts_poll_timer.stop()
            return

        elapsed = int(_time.monotonic() - self._tts_poll_t0)
        if elapsed > 180:
            self._log("[tts-server] 启动超时 (>3 分钟) — 查看 .memory/tts_server.log")
            self.qwen3_status.setText("启动超时")
            self.qwen3_status.setStyleSheet("color: #c33;")
            if hasattr(self, "_tts_poll_timer"):
                self._tts_poll_timer.stop()
            if self._quickstart_active:
                self._quickstart_abort("TTS 启动超时")
            return

        # Update elapsed counter even while we wait for the probe response.
        if elapsed < 60:
            phase = "加载模型"
        elif elapsed < 90:
            phase = "暖机 CUDA graph"
        else:
            phase = "还在等…"
        self.qwen3_status.setText(f"启动中…{elapsed}s · {phase}")
        self._on_probe_qwen3()

    # ---------- Mumble ----------
    def _on_connect_mumble(self) -> None:
        if self.connect_worker is not None and self.connect_worker.isRunning():
            return
        if self.bot is not None:
            self._log("[mumble] already connected")
            return
        cfg = MumbleConfig(
            host=self.mumble_host.text().strip(),
            port=self.mumble_port.value(),
            name=self.mumble_name.text().strip() or "Eri",
            password=self.mumble_password.text(),
            channel=self.mumble_channel.text().strip() or "LivePartner",
            create_channel_if_missing=not self.mumble_no_create_check.isChecked(),
        )
        self.mumble_connect_btn.setEnabled(False)
        self.mumble_status.setText("连接中…")
        self.mumble_status.setStyleSheet("color: #c80;")
        self.connect_worker = MumbleConnectWorker(
            cfg,
            on_user_utterance=self._bot_utterance_cb,
            voice_whitelist=self._effective_whitelist(),
        )
        self.connect_worker.log.connect(self._log)
        self.connect_worker.connected.connect(self._on_mumble_connected)
        self.connect_worker.failed.connect(self._on_mumble_failed)
        self.connect_worker.finished.connect(self._cleanup_connect_worker)
        self.connect_worker.start()

    def _on_mumble_connected(self, bot: object) -> None:
        assert isinstance(bot, MumbleBot)
        self.bot = bot
        ch = bot.current_channel_name or "(unknown)"
        self.mumble_status.setText(f"已连接 · 频道 {ch!r}")
        self.mumble_status.setStyleSheet("color: #2a2;")
        self.mumble_disconnect_btn.setEnabled(True)
        self.mumble_speak_check.setEnabled(True)
        self.mumble_speak_check.setChecked(True)
        self._apply_listen_state()
        # Chain to quickstart "done" if we're in a startup sequence.
        if self._quickstart_active:
            self._quickstart_step("done")

    def _on_mumble_failed(self, msg: str) -> None:
        self._log(f"[mumble] connect FAILED: {msg}")
        self.mumble_status.setText("连接失败")
        self.mumble_status.setStyleSheet("color: #c33;")
        self.mumble_connect_btn.setEnabled(True)
        if self._quickstart_active:
            self._quickstart_abort(f"Mumble 连接失败: {msg[:60]}")

    def _cleanup_connect_worker(self) -> None:
        self.connect_worker = None
        if self.bot is None:
            self.mumble_connect_btn.setEnabled(True)

    def _on_disconnect_mumble(self) -> None:
        if self.bot is not None:
            self._log("[mumble] disconnecting")
            try:
                self.bot.stop()
            except Exception as e:
                self._log(f"[mumble] stop error: {e}")
            self.bot = None
        self.mumble_status.setText("未连接")
        self.mumble_status.setStyleSheet("color: #888;")
        self.mumble_connect_btn.setEnabled(True)
        self.mumble_disconnect_btn.setEnabled(False)
        self.mumble_speak_check.setChecked(False)
        self.mumble_speak_check.setEnabled(False)

    def _parse_whitelist(self) -> set[str] | None:
        raw = self.stt_whitelist_edit.text().strip()
        if not raw:
            return None
        return {n.strip() for n in raw.split(",") if n.strip()}

    def _effective_whitelist(self) -> set[str] | None:
        """What we actually push to the bot.

        - listen OFF        → ``set()`` (block all — bot drops every utterance)
        - listen ON, empty  → ``None`` (no filter, all users pass)
        - listen ON, names  → parsed set
        """
        if not self.stt_listen_check.isChecked():
            return set()
        return self._parse_whitelist()

    def _apply_listen_state(self) -> None:
        """Sync bot whitelist + status label to checkbox + edit text."""
        wl = self._effective_whitelist()
        if self.bot is not None:
            self.bot.set_voice_whitelist(wl)
        if wl is None:
            self._log("[stt] listen: ON, whitelist=(全监听)")
        elif not wl:
            self._log("[stt] listen: OFF — bot 不再接收频道音频")
        else:
            self._log(f"[stt] listen: ON, whitelist={sorted(wl)}")
        self._update_stt_status_label()

    def _on_listen_toggled(self, checked: bool) -> None:
        self._apply_listen_state()

    def _on_whitelist_changed(self) -> None:
        self._apply_listen_state()

    def _update_stt_status_label(self) -> None:
        if self.stt is None:
            return
        size = self.stt_size_combo.currentText()
        if self.stt_listen_check.isChecked():
            self.stt_status.setText(f"已加载 · whisper-{size} · ● 监听中")
            self.stt_status.setStyleSheet("color: #2a2;")
        else:
            self.stt_status.setText(f"已加载 · whisper-{size} · ○ 已暂停")
            self.stt_status.setStyleSheet("color: #888;")

    # ---------- STT ----------
    def _bot_utterance_cb(self, user: dict, pcm_bytes: bytes) -> None:
        self.utterance_signal.emit(dict(user), pcm_bytes)

    def _on_utterance_arrived(self, user: dict, pcm_bytes: bytes) -> None:
        name = user.get("name", "?")
        size_kb = len(pcm_bytes) // 1024
        dur = len(pcm_bytes) / (48000 * 2)
        if self.stt is None:
            self._log(f"[mumble] heard {name!r} ({size_kb} KB, {dur:.2f}s) — STT not loaded, ignoring")
            return
        if not self.stt_listen_check.isChecked():
            self._log(f"[mumble] heard {name!r} ({dur:.2f}s) — listen toggle off, ignoring")
            return

        # AI is mid-pipeline (gen+TTS+Mumble drain). Eri always finishes her
        # current line — we queue this utterance and process it on _cleanup_decision.
        # If a stale pending_utterance is already there, the newer one replaces
        # it (later turn supersedes earlier turn the player abandoned).
        if self._decision_busy:
            replaced = self.pending_utterance is not None
            self.pending_utterance = (user, pcm_bytes)
            # Newer PCM supersedes any stale text from an earlier turn the player
            # never followed up on — don't let cleanup process them BOTH.
            self.pending_transcribed = None
            tag = "替换待处理" if replaced else "排队"
            self._log(f"[stt] AI 在说,{tag}: {name!r} ({dur:.2f}s) — 等 AI 说完再处理")
            self.statusBar().showMessage(f"AI 在说话, 说完后会处理 {name!r} 这句")
            return

        self._dispatch_transcribe(user, pcm_bytes)

    def _dispatch_transcribe(self, user: dict, pcm_bytes: bytes) -> None:
        name = user.get("name", "?")
        dur = len(pcm_bytes) / (48000 * 2)
        self._log(f"[stt] {name!r} spoke {dur:.2f}s, transcribing…")
        w = STTTranscribeWorker(self.stt, user, pcm_bytes)
        w.log.connect(self._log)
        w.transcribed.connect(self._on_transcribed)
        w.failed.connect(lambda m: self._log(f"[stt err] {m}"))
        w.finished.connect(lambda: self._cleanup_stt_worker(w))
        self.stt_workers.append(w)
        w.start()

    def _cleanup_stt_worker(self, w: STTTranscribeWorker) -> None:
        try:
            self.stt_workers.remove(w)
        except ValueError:
            pass

    def _on_transcribed(self, text: str, user: dict, result) -> None:
        name = user.get("name", "?")
        self._log(f"[stt] {name!r} ({result.language} {result.language_probability:.2f}, "
                  f"nsp={result.no_speech_prob:.2f}, "
                  f"{result.inference_ms:.0f}ms): {text!r}")
        if not text:
            # Show what got dropped + why, otherwise silent drops look like the
            # bot ignoring the player.
            if getattr(result, "dropped_reason", ""):
                raw = getattr(result, "raw_text", "") or "(empty)"
                self._log(f"[stt] DROPPED — {result.dropped_reason} · raw={raw!r}")
            return
        if self._decision_busy:
            # Race: ambient tick (or another path) grabbed _decision_busy while
            # we were busy transcribing. Don't throw the player's words away —
            # stash them for _cleanup_decision to drain. Newer text replaces
            # older (same policy as pending_utterance: latest wins).
            replaced = self.pending_transcribed is not None
            self.pending_transcribed = (user, text)
            tag = "替换待处理文本" if replaced else "排队文本"
            self._log(
                f"[stt] AI 在说,{tag}: {name!r}: {text!r} — 等 AI 说完再处理"
            )
            self.statusBar().showMessage(f"AI 在说,排队 {name!r}: {text!r}")
            return
        self._handle_voice_text(user, text)

    def _handle_voice_text(self, user: dict, text: str) -> None:
        """Post-transcribe: spawn a decision for one (user, text) pair. Shared
        between the live STT path (``_on_transcribed`` when AI is idle) and the
        cleanup-drain path (``_cleanup_decision`` when an old utterance was
        queued behind an in-flight AI turn)."""
        name = user.get("name", "?")
        persona_id = self.persona_combo.currentData()
        if not persona_id:
            return
        try:
            persona = load_persona(persona_id)
        except Exception as e:
            self._log(f"[stt] persona load failed: {e}")
            return
        event = f'{name}: 「{text}」'

        # If capture is running with a fresh frame, send it along — lets the AI
        # actually answer "what do you see?"-style voice questions.
        captured = None
        if self.capture is not None and self.capture.is_running:
            captured = self.capture.latest_frame(max_age_sec=1.5)

        self._log(f"=== {persona.display_name} ({persona_id}) — 对话{' + 画面' if captured else ''} ===")
        self._log(f"事件: {event}")
        self.memory.add_player_voice(text)
        self._refresh_memory_timeline()

        self._dispatch_decision(
            persona=persona,
            event=event,
            force_speak=True,
            is_conversation=True,
            captured_frame=captured,
            synthesize_frame=False,
        )

    def _on_stt_load(self) -> None:
        if self.stt_load_worker is not None and self.stt_load_worker.isRunning():
            return
        size = self.stt_size_combo.currentText()
        lang = self.stt_lang_combo.currentData()
        self.stt_load_btn.setEnabled(False)
        self.stt_status.setText("加载中…")
        self.stt_status.setStyleSheet("color: #c80;")
        self.stt_load_worker = STTLoadWorker(size, lang)
        self.stt_load_worker.log.connect(self._log)
        self.stt_load_worker.loaded.connect(self._on_stt_loaded)
        self.stt_load_worker.failed.connect(self._on_stt_failed)
        self.stt_load_worker.finished.connect(self._cleanup_stt_load_worker)
        self.stt_load_worker.start()

    def _on_stt_loaded(self, stt_obj: object) -> None:
        self.stt = stt_obj
        self.stt_listen_check.setEnabled(True)
        # setChecked emits toggled → _on_listen_toggled → _apply_listen_state
        # which pushes whitelist to bot AND refreshes the status label.
        self.stt_listen_check.setChecked(True)
        # If we were already checked (re-load), toggled doesn't fire — force sync.
        self._apply_listen_state()
        # Chain to next quickstart phase.
        if self._quickstart_active:
            self._quickstart_step("mumble")

    def _on_stt_failed(self, msg: str) -> None:
        self._log(f"[stt] load FAILED: {msg}")
        self.stt_status.setText("加载失败")
        self.stt_status.setStyleSheet("color: #c33;")
        self.stt_load_btn.setEnabled(True)
        if self._quickstart_active:
            self._quickstart_abort(f"STT load failed: {msg[:60]}")

    def _cleanup_stt_load_worker(self) -> None:
        self.stt_load_worker = None
        if self.stt is None:
            self.stt_load_btn.setEnabled(True)

    # ---------- capture ----------
    def _on_capture_start(self) -> None:
        if self.capture is not None and self.capture.is_running:
            return
        dev = self.cap_device_combo.currentData()
        if dev is None:
            QMessageBox.warning(self, "提示", "没有可用视频设备")
            return
        w, h = self.cap_res_combo.currentData() or (1280, 720)
        try:
            cap = CaptureSource(device_index=int(dev), width=w, height=h, fps=30)
            cap.start()
        except Exception as e:
            self.cap_status.setText("启动失败")
            self.cap_status.setStyleSheet("color: #c33;")
            self._log(f"[capture] {type(e).__name__}: {e}")
            return
        self.capture = cap
        size = cap.negotiated_size or (w, h)
        self.cap_status.setText(f"运行中 · {size[0]}x{size[1]}")
        self.cap_status.setStyleSheet("color: #2a2;")
        self.cap_start_btn.setEnabled(False)
        self.cap_stop_btn.setEnabled(True)
        self._log(f"[capture] started  device={dev}  {size[0]}x{size[1]}")

    def _on_capture_stop(self) -> None:
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.cap_status.setText("(未启动)")
        self.cap_status.setStyleSheet("color: #888;")
        self.cap_info_lbl.setText("")
        self.cap_start_btn.setEnabled(True)
        self.cap_stop_btn.setEnabled(False)
        self._log("[capture] stopped")

    def _on_capture_refresh_devices(self) -> None:
        prev = self.cap_device_combo.currentData()
        self.cap_device_combo.clear()
        for i, name in enumerate(list_video_devices()):
            self.cap_device_combo.addItem(f"[{i}] {name}", userData=i)
        # Try to keep previous selection; otherwise re-apply default heuristic.
        if prev is not None:
            for idx in range(self.cap_device_combo.count()):
                if self.cap_device_combo.itemData(idx) == prev:
                    self.cap_device_combo.setCurrentIndex(idx)
                    return
        self._select_preferred_capture_device()

    def _select_preferred_capture_device(self) -> None:
        """Bias selection toward a real capture card (e.g. Live Gamer Ultra)
        instead of whatever virtual cam happens to be at index 0."""
        for idx in range(self.cap_device_combo.count()):
            label = (self.cap_device_combo.itemText(idx) or "").lower()
            if any(hint in label for hint in _PREFERRED_CAPTURE_DEVICE_HINTS):
                self.cap_device_combo.setCurrentIndex(idx)
                return

    def _update_preview(self) -> None:
        if self.capture is None or not self.capture.is_running:
            if self.preview_label.pixmap() and not self.preview_label.pixmap().isNull():
                self.preview_label.setPixmap(QPixmap())
            self.preview_label.setText("(未启动)")
            return
        snap = self.capture.latest_frame(max_age_sec=2.0)
        if snap is None:
            self.preview_label.setText("(等待信号 / 帧太老)")
            self.preview_label.setPixmap(QPixmap())
            return
        h, w = snap.frame.shape[:2]
        # BGR → RGB → QImage. copy() because numpy buffer would be reused.
        rgb = cv2.cvtColor(snap.frame, cv2.COLOR_BGR2RGB)
        rgb = rgb.copy()
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        target_w = max(320, self.preview_label.width() - 4)
        target_h = max(180, self.preview_label.height() - 4)
        pix = QPixmap.fromImage(qimg).scaled(
            target_w, target_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(pix)
        self.cap_info_lbl.setText(
            f"{w}x{h}  ·  mean brightness {snap.frame.mean():.0f}  ·  age {time.monotonic() - snap.timestamp:.2f}s"
        )

    # ---------- Decision ----------
    def _dispatch_decision(
        self,
        *,
        persona: Persona,
        event: str,
        force_speak: bool,
        is_conversation: bool,
        captured_frame: "FrameSnapshot | None",
        synthesize_frame: bool = False,
        save_mp3: bool = False,
    ) -> None:
        """Single spawn site for DecisionWorker.

        All three trigger sources (manual button / ambient tick / STT voice)
        funnel through here. Each source keeps its own precheck (busy guard,
        cooldown, capture availability) — this method assumes those passed
        and just handles the boilerplate:
        - resource lookup (engine, qwen3_url, bot, memory_text, recent_ai_lines)
        - DecisionRequest assembly
        - signal wiring (text_ready / finished_ok / failed / interrupted / cleanup)
        - _decision_busy = True

        The cleanup callback `_cleanup_decision` clears _decision_busy AND
        drains pending_utterance, so once this fires the bus is unified.
        """
        engine = self.tts_engine_combo.currentData() or "edge"
        qwen3_url = self.qwen3_url_edit.text().strip().rstrip("/")
        bot_for_speech = self.bot if (
            self.bot is not None and self.mumble_speak_check.isChecked()
        ) else None
        memory_text = self.memory.render_for_prompt()
        recent_ai = self.memory.recent_ai_lines(n=5)

        self._last_decision_event = event
        self._decision_busy = True
        self.decision_worker = DecisionWorker(DecisionRequest(
            persona=persona,
            event=event,
            force_speak=force_speak,
            save_mp3=save_mp3,
            bot=bot_for_speech,
            engine=engine,
            qwen3_url=qwen3_url,
            synthesize_frame=synthesize_frame,
            is_conversation=is_conversation,
            captured_frame=captured_frame,
            memory_text=memory_text,
            recent_ai_lines=recent_ai,
            vts=self.vts,
        ))
        self.decision_worker.log.connect(self._log)
        self.decision_worker.text_ready.connect(self._on_decision_text_ready)
        self.decision_worker.finished_ok.connect(self._on_decision_done)
        self.decision_worker.failed.connect(self._on_decision_failed)
        self.decision_worker.interrupted.connect(self._on_decision_interrupted)
        self.decision_worker.finished.connect(self._cleanup_decision)
        self.decision_worker.start()

    def _on_trigger(self) -> None:
        if self._decision_busy:
            return
        persona_id = self.persona_combo.currentData()
        if not persona_id:
            QMessageBox.warning(self, "提示", "没有可用人格")
            return
        try:
            persona = load_persona(persona_id)
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"{type(e).__name__}: {e}")
            return

        captured = None
        if self.capture is not None and self.capture.is_running:
            captured = self.capture.latest_frame(max_age_sec=1.5)
            if captured is None:
                self._log("[capture] 启动但没有最近帧, 回落到合成帧")

        engine = self.tts_engine_combo.currentData() or "edge"
        qwen3_url = self.qwen3_url_edit.text().strip().rstrip("/")
        bot_for_speech = self.bot if (
            self.bot is not None and self.mumble_speak_check.isChecked()
        ) else None

        self._log(f"\n=== {persona.display_name} ({persona_id}) ===")
        self._log(f"事件: {self.event_edit.text()}")
        self._log(f"TTS 引擎: {engine}{' @ ' + qwen3_url if engine == 'qwen3' else ''}  (中文)")
        if bot_for_speech is not None:
            self._log(f"(通过 Mumble 频道 {bot_for_speech.current_channel_name!r} 播放)")
        self.trigger_btn.setEnabled(False)
        self.statusBar().showMessage("运行中…")

        self.memory.add_event(self.event_edit.text())
        self._refresh_memory_timeline()

        self._dispatch_decision(
            persona=persona,
            event=self.event_edit.text(),
            force_speak=self.force_check.isChecked(),
            is_conversation=False,
            captured_frame=captured,
            synthesize_frame=True,
            save_mp3=self.save_mp3_check.isChecked(),
        )

    def _on_decision_text_ready(self, ai_text: str) -> None:
        """Fires the moment the LLM returns text — BEFORE TTS+drain. Push
        subtitle to overlays + UI immediately so they're in sync with audio
        rather than 3-5s behind. Also spawn the Live2D Director worker in
        parallel so the avatar's expression usually beats the first TTS chunk."""
        if not ai_text:
            return
        self.last_line_lbl.setText(ai_text)
        if self.subtitle_overlay is not None and self.overlay_check.isChecked():
            self.subtitle_overlay.show_line(ai_text)
        if self.web_overlay is not None and self.web_overlay.is_running:
            self.web_overlay.push_subtitle(ai_text)
        # Director — only spawn if VTS is actually live; otherwise the flash
        # call would be a no-op anyway.
        if self.vts is None or not self.vts.is_connected:
            return
        expr_palette, _semantic_to_file, hotkey_palette = self._build_director_palette()
        if not expr_palette:
            return  # no usable expressions on this model — skip director
        persona_id = self.persona_combo.currentData()
        if not persona_id:
            return
        try:
            persona = load_persona(persona_id)
        except Exception:
            return
        ctx = self.memory.recent_context_lines(n=3)
        w = DirectorWorker(
            ai_text, ctx, expr_palette, hotkey_palette, persona,
        )
        w.log.connect(self._log)
        w.intent_ready.connect(self._on_director_intent_ready)
        w.finished.connect(lambda w=w: self._cleanup_director_worker(w))
        self.director_workers.append(w)
        w.start()

    def _build_director_palette(self) -> tuple[list[str], dict[str, str], list[str]]:
        """Resolve the expression palette to hand to the director, plus the
        semantic→file map and the hotkey-id palette.

        Logic:
        - If ``persona.vts.expression_map`` (semantic name → .exp3.json file)
          has entries whose value matches a currently loaded VTS expression,
          use those semantic names as the palette.
        - Otherwise fall back to "raw mode": the palette is the bare expression
          filenames from VTS, and ``semantic_to_file`` is identity.
        Hotkey palette: all non-expression hotkey IDs (animations / model moves).
        """
        if self.vts is None or not self.vts.is_connected:
            return [], {}, []
        st = self.vts.status
        # Files VTS actually has — the source of truth
        avail_files = {h.file for h in st.hotkeys if h.type == "ToggleExpression" and h.file}
        if not avail_files:
            return [], {}, []
        # Hotkey palette: animation / movement etc, NOT expressions (those are
        # the director's expr field instead)
        hotkey_palette = [
            h.id for h in st.hotkeys
            if h.type and h.type != "ToggleExpression" and h.id
        ]
        # Read persona's expression_map if available
        persona_id = self.persona_combo.currentData()
        expr_map: dict[str, str] = {}
        if persona_id:
            try:
                p = load_persona(persona_id)
                raw_map = (p.vts or {}).get("expression_map", {}) or {}
                if isinstance(raw_map, dict):
                    expr_map = {
                        str(k): str(v) for k, v in raw_map.items()
                        if isinstance(v, str) and v in avail_files
                    }
            except Exception:
                expr_map = {}
        if expr_map:
            return list(expr_map.keys()), expr_map, hotkey_palette
        # Raw mode: palette = filenames (strip .exp3.json suffix for readability)
        files = sorted(avail_files)
        # Identity map but with friendlier display: 'smile_a.exp3.json' kept as-is
        return files, {f: f for f in files}, hotkey_palette

    def _on_director_intent_ready(self, intent: object) -> None:
        """Push the director's chosen expression + face params + trigger to VTS.

        Three independent visual layers fire here:
        1. Expression sticker  (decorative overlay, e.g. sweat drop / star)
        2. Face parameter offsets (BrowLeftY / BrowRightY / MouthSmile —
           the actual face change, much more visible than stickers)
        3. Optional hotkey trigger (animation, if rig has it)
        """
        from .director import DirectorIntent as _Intent  # local — circular hint
        if not isinstance(intent, _Intent) or self.vts is None:
            return
        if not intent.expr and intent.brow_l == 0 and intent.brow_r == 0 and intent.smile == 0:
            return

        # Layer 1: sticker expression
        if intent.expr:
            _palette, semantic_to_file, _ = self._build_director_palette()
            file = semantic_to_file.get(intent.expr, intent.expr)
            self.vts.set_expression_file(file)

        # Layer 2: face parameters — only push those the loaded model exposes.
        # Type-H3 (and most VTS rigs) maps:
        #   brow_l → "BrowLeftY", brow_r → "BrowRightY", smile → "MouthSmile"
        # We honor the same auto-detect pattern as IdleMotion (try VTS-mapped
        # names first, fall back to raw Cubism if needed).
        if self.vts.is_connected:
            avail = set(self.vts.status.parameters)
            face_updates: list[tuple[str, float]] = []
            for name_candidates, value in [
                (["BrowLeftY", "ParamBrowLY"], intent.brow_l),
                (["BrowRightY", "ParamBrowRY"], intent.brow_r),
                (["MouthSmile", "ParamMouthForm"], intent.smile),
            ]:
                for nm in name_candidates:
                    if nm in avail:
                        face_updates.append((nm, value))
                        break
            if face_updates:
                self.vts.set_parameters_bulk(face_updates)

        # Layer 3: optional hotkey
        if intent.trigger:
            self.vts.trigger_hotkey(intent.trigger)

        # Restart idle decay — 8 s after the LAST intent we'll auto-relax.
        self._idle_decay_timer.start()

    def _on_director_idle_decay(self) -> None:
        """Avatar has been holding the last director state for 8 s — relax it.

        Resets BOTH the sticker expression AND the face parameter offsets so
        Eri returns to a neutral resting face between turns instead of getting
        stuck mid-smug-eyebrow-up.
        """
        if self.vts is None or not self.vts.is_connected:
            return

        # Reset face params to neutral 0.0 — same VTS bindings we drove above.
        avail = set(self.vts.status.parameters)
        face_resets: list[tuple[str, float]] = []
        for candidates in (
            ["BrowLeftY", "ParamBrowLY"],
            ["BrowRightY", "ParamBrowRY"],
            ["MouthSmile", "ParamMouthForm"],
        ):
            for nm in candidates:
                if nm in avail:
                    face_resets.append((nm, 0.0))
                    break
        if face_resets:
            self.vts.set_parameters_bulk(face_resets)

        # Sticker: prefer 'neutral' / 'default' name; fall back to first palette entry.
        palette, semantic_to_file, _ = self._build_director_palette()
        for candidate in ("neutral", "idle_neutral", "default", "idle"):
            if candidate in palette:
                self.vts.set_expression_file(semantic_to_file.get(candidate, candidate))
                return
        if palette:
            first = palette[0]
            self.vts.set_expression_file(semantic_to_file.get(first, first))

    def _cleanup_director_worker(self, w: "DirectorWorker") -> None:
        try:
            self.director_workers.remove(w)
        except ValueError:
            pass

    def _on_decision_done(self, ai_text: str) -> None:
        # By the time we get here, text_ready has already updated the UI.
        # This handler is just for: memory write, fact extraction, cooldown
        # bookkeeping, and ambient status — i.e. things that should happen
        # after the AI line has finished playing.
        if ai_text:
            self.memory.add_ai_response(ai_text)
            # Also pin this line onto the current scene entry so the next
            # prompt shows "在这个画面下你已经说过 N 次" — the structural
            # anti-repetition signal that makes the model notice it's looping.
            self.memory.attach_ai_to_current_scene(ai_text)
            # Eri has had a chance to comment on any pending appearance
            # change → clear it so the next turn doesn't re-mention it.
            self.memory.consume_self_appearance_change()
            self.statusBar().showMessage("完成")
            self.last_ai_spoke_at = time.monotonic()
            self._refresh_memory_timeline()
            self._spawn_fact_extractor(self._last_decision_event, ai_text)
            self._update_ambient_status()
        else:
            self.statusBar().showMessage("AI 沉默")

    def _on_decision_interrupted(self) -> None:
        # Only fires now if cancel() was called from closeEvent — the player-
        # interrupt path was removed, so STT no longer cancels the running
        # worker. Kept as a primitive for clean UI shutdown.
        self._log("[cancel] worker cancelled (likely shutdown)")
        self.statusBar().showMessage("AI 被取消")

    def _on_decision_failed(self, msg: str) -> None:
        self._log(f"[ERR] {msg}")
        self.statusBar().showMessage("出错")

    def _cleanup_decision(self) -> None:
        self.trigger_btn.setEnabled(True)
        self.decision_worker = None
        # Cleared LAST so the dispatch-site set + cleanup pair is symmetric.
        # Any check between dispatch and cleanup (ambient tick / STT) sees True.
        self._decision_busy = False
        # Drain order: raw PCM first (still needs transcribe), then already-
        # transcribed text. Only one is typically set; if both somehow are
        # (player spoke twice while AI was busy), PCM is newer → transcribe it
        # and discard the older stale text.
        pending_pcm = self.pending_utterance
        pending_text = self.pending_transcribed
        self.pending_utterance = None
        self.pending_transcribed = None
        if pending_pcm is not None:
            user, pcm = pending_pcm
            name = user.get("name", "?")
            self._log(f"[stt] 处理排队的 PCM: {name!r}")
            self.statusBar().showMessage(f"处理排队的 {name!r}")
            self._dispatch_transcribe(user, pcm)
        elif pending_text is not None:
            user, text = pending_text
            name = user.get("name", "?")
            self._log(f"[stt] 处理排队的文本: {name!r}: {text!r}")
            self.statusBar().showMessage(f"处理排队的 {name!r}")
            self._handle_voice_text(user, text)
        elif self.pending_danmaku is not None:
            # Lowest priority — player utterances win over viewer chat.
            # Also still subject to cooldown so a fresh cleanup doesn't re-fire
            # immediately after a recent danmaku dispatch.
            cooldown_s = (
                self.danmaku_cooldown_spin.value()
                if hasattr(self, "danmaku_cooldown_spin") else 30
            )
            if (time.monotonic() - self._last_danmaku_dispatch_ts) >= cooldown_s:
                name, text = self.pending_danmaku
                self.pending_danmaku = None
                self._dispatch_danmaku(name, text)

    # ---------- fact extractor ----------
    def _spawn_fact_extractor(self, event: str, ai_text: str) -> None:
        if not event or not ai_text:
            return
        w = FactExtractorWorker(event, ai_text)
        w.log.connect(self._log)
        w.done.connect(lambda line, w=w: self._on_fact_extracted(line, w))
        w.finished.connect(lambda w=w: self._cleanup_fact_worker(w))
        self.fact_workers.append(w)
        w.start()

    def _on_fact_extracted(self, line: str, w: "FactExtractorWorker") -> None:
        if not line:
            return
        self.memory.append_auto_fact(line)
        self._refresh_memory_timeline()
        # If memory tab is open, refresh the auto.md editor too.
        if hasattr(self, "_memory_editors"):
            for label, (editor, path) in self._memory_editors.items():
                if path == self.memory.auto_path:
                    try:
                        editor.setPlainText(path.read_text(encoding="utf-8"))
                    except OSError:
                        pass
                    break

    def _cleanup_fact_worker(self, w: "FactExtractorWorker") -> None:
        try:
            self.fact_workers.remove(w)
        except ValueError:
            pass

    # ---------- ambient ----------
    def _on_ambient_toggled(self, checked: bool) -> None:
        if checked:
            self.ambient_timer.start()
            self._log(
                f"[ambient] ON · 每 {self.ambient_interval_spin.value()}s tick · "
                f"冷却 {self.ambient_cooldown_spin.value()}s"
            )
        else:
            self.ambient_timer.stop()
            self._log("[ambient] OFF")
        self._update_ambient_status()

    def _on_ambient_interval_changed(self, v: int) -> None:
        self.ambient_timer.setInterval(v * 1000)
        self._update_ambient_status()

    def _update_ambient_status(self) -> None:
        if not hasattr(self, "ambient_status_lbl"):
            return
        if not self.ambient_check.isChecked():
            self.ambient_status_lbl.setText("(关闭)")
            self.ambient_status_lbl.setStyleSheet("color: #888;")
            return
        cooldown = self.ambient_cooldown_spin.value()
        elapsed = time.monotonic() - self.last_ai_spoke_at if self.last_ai_spoke_at else cooldown + 1
        if elapsed < cooldown:
            self.ambient_status_lbl.setText(f"冷却中… 剩 {cooldown - int(elapsed)}s")
            self.ambient_status_lbl.setStyleSheet("color: #c80;")
        else:
            self.ambient_status_lbl.setText("● 待 tick")
            self.ambient_status_lbl.setStyleSheet("color: #2a2;")

    def _on_ambient_tick(self) -> None:
        # Status indicator always refreshes (even when we skip).
        self._update_ambient_status()
        if not self.ambient_check.isChecked():
            return
        if self._decision_busy:
            return  # AI is mid-pipeline already
        cooldown = self.ambient_cooldown_spin.value()
        if self.last_ai_spoke_at and (time.monotonic() - self.last_ai_spoke_at) < cooldown:
            return
        # Need capture for ambient — no point firing gate with no frame.
        if self.capture is None or not self.capture.is_running:
            return
        captured = self.capture.latest_frame(max_age_sec=2.0)
        if captured is None:
            return
        persona_id = self.persona_combo.currentData()
        if not persona_id:
            return
        try:
            persona = load_persona(persona_id)
        except Exception:
            return

        event = "(环境扫描) 你正在看着当前画面,自己决定要不要开口"
        self._log("[ambient] tick — gate 看画面")
        self._dispatch_decision(
            persona=persona,
            event=event,
            force_speak=False,           # gate decides
            is_conversation=False,
            captured_frame=captured,
            synthesize_frame=False,
        )

    # ---------- subtitle overlay ----------
    def _ensure_overlay(self) -> SubtitleOverlay:
        if self.subtitle_overlay is None:
            self.subtitle_overlay = SubtitleOverlay(
                font_size=self.overlay_font_spin.value(),
                hold_ms=self.overlay_hold_spin.value() * 1000,
                opacity=self.overlay_opacity_spin.value() / 100.0,
            )
            self.subtitle_overlay.setWindowTitle("Eri 字幕")
        return self.subtitle_overlay

    def _on_overlay_toggled(self, checked: bool) -> None:
        ov = self._ensure_overlay()
        if checked:
            # Show "字幕浮窗" placeholder so user sees + can move the window.
            ov.set_hold_ms(0)  # disable auto-hide for the placeholder
            ov.show_line("Eri 字幕  (拖动定位 · 取消勾选关闭)")
            QTimer.singleShot(2500, lambda: ov.set_hold_ms(
                self.overlay_hold_spin.value() * 1000
            ))
            self._log("[overlay] ON")
        else:
            ov.clear()
            self._log("[overlay] OFF")

    def _on_overlay_font_changed(self, pt: int) -> None:
        if self.subtitle_overlay is not None:
            self.subtitle_overlay.set_font_size(pt)

    def _on_overlay_opacity_changed(self, pct: int) -> None:
        if self.subtitle_overlay is not None:
            self.subtitle_overlay.set_opacity_pct(pct)

    def _on_overlay_hold_changed(self, sec: int) -> None:
        if self.subtitle_overlay is not None:
            self.subtitle_overlay.set_hold_ms(sec * 1000)

    # ---------- OBS browser-source HTTP overlay ----------
    def _on_web_overlay_toggled(self, checked: bool) -> None:
        if checked:
            port = self.web_overlay_port_spin.value()
            font_size = self.web_overlay_font_spin.value()
            hold_ms = self.web_overlay_hold_spin.value() * 1000
            if self.web_overlay is not None and self.web_overlay.is_running:
                self.web_overlay.stop()  # restart with current settings
            self.web_overlay = OverlayServer(
                port=port,
                default_font_size=font_size,
                default_hold_ms=hold_ms,
            )
            try:
                self.web_overlay.start()
            except OSError as e:
                self._log(f"[overlay-web] 端口 {port} 启动失败: {e}")
                QMessageBox.critical(
                    self, "端口被占用",
                    f"端口 {port} 启动失败:\n{e}\n\n"
                    "换一个端口再启用,或关掉占着这个端口的程序。"
                )
                self.web_overlay = None
                self.web_overlay_check.blockSignals(True)
                self.web_overlay_check.setChecked(False)
                self.web_overlay_check.blockSignals(False)
                return
            # Show the LAN URL (e.g. http://192.168.1.42:7002/) because the
            # whole point of this overlay is OBS on a *different* PC reaching
            # the laptop. Loopback URL would mislead the user.
            self.web_overlay_url_edit.setText(self.web_overlay.lan_url)
            self.web_overlay_status_lbl.setText("● 已启动 (0 客户端)")
            self.web_overlay_status_lbl.setStyleSheet("color: #2a2;")
            self._log(
                f"[overlay-web] ON @ {self.web_overlay.lan_url} "
                f"(loopback: {self.web_overlay.url})"
            )
        else:
            if self.web_overlay is not None:
                self.web_overlay.stop()
                self.web_overlay = None
            self.web_overlay_status_lbl.setText("(未启用)")
            self.web_overlay_status_lbl.setStyleSheet("color: #888;")
            self._log("[overlay-web] OFF")

    # ---------- scene describer ----------
    def _on_scene_toggled(self, checked: bool) -> None:
        if checked:
            self.scene_timer.start()
            self._log(
                f"[scene] ON · 描述间隔 {self.scene_interval_spin.value()}s · "
                f"保留 {self.scene_retention_spin.value()}s"
            )
            self.scene_status_lbl.setText("● 已启用")
            self.scene_status_lbl.setStyleSheet("color: #2a2;")
        else:
            self.scene_timer.stop()
            self._log("[scene] OFF")
            self.scene_status_lbl.setText("(关闭)")
            self.scene_status_lbl.setStyleSheet("color: #888;")

    def _on_scene_interval_changed(self, v: int) -> None:
        self.scene_timer.setInterval(v * 1000)

    def _on_scene_tick(self) -> None:
        if not self.scene_check.isChecked():
            return
        if self.capture is None or not self.capture.is_running:
            return
        snap = self.capture.latest_frame(max_age_sec=2.0)
        if snap is None:
            return

        # Frame-change check: skip the LLM call if nothing visible changed
        # since last description.
        import cv2
        try:
            gray = cv2.cvtColor(snap.frame, cv2.COLOR_BGR2GRAY)
            fp = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype("float32")
        except Exception:
            return
        if self._last_scene_fingerprint is not None:
            diff = float(abs(fp - self._last_scene_fingerprint).mean()) / 255.0
            if diff < 0.05:  # < 5% mean abs change — basically static
                return

        # Encode to webp and dispatch worker.
        # Describer reads on-screen text (titles, menu options, UI numbers) —
        # 512/70 was too small + lossy to OCR them, model just said "网页界面"
        # generically. 1024/82 lets it pick out specific titles.
        try:
            frame_b64, mime = snap.to_vlm_b64(
                max_side=1024, quality=82, prefer="webp",
            )
        except Exception as e:
            self._log(f"[scene] encode failed: {e}")
            return

        self._last_scene_fingerprint = fp

        # Pass the most recent stored scene description so the describer can
        # short-circuit with SAME when nothing meaningful changed.
        scenes = self.memory.scene_entries()
        last_desc = scenes[-1].text if scenes else None

        w = SceneDescriberWorker(frame_b64, mime, last_description=last_desc)
        w.log.connect(self._log)
        w.done.connect(self._on_scene_described)
        w.finished.connect(lambda w=w: self._cleanup_scene_worker(w))
        self.scene_workers.append(w)
        w.start()

    def _on_scene_described(self, desc: str) -> None:
        retention = float(self.scene_retention_spin.value())
        if not desc:
            # Model returned SAME — bump the existing entry's "last seen" so
            # the "(持续 X 分钟)" hint grows, without adding a redundant row.
            scenes = self.memory.scene_entries()
            if scenes:
                self.memory.add_scene_description(
                    scenes[-1].text, retention_sec=retention
                )
        else:
            self.memory.add_scene_description(desc, retention_sec=retention)
        # Status label always refreshed so user sees what's in memory now.
        if hasattr(self, "scene_status_lbl") and self.scene_check.isChecked():
            n = len(self.memory.scene_entries(retention_sec=retention))
            self.scene_status_lbl.setText(f"● 已记录 {n} 条")
            self.scene_status_lbl.setStyleSheet("color: #2a2;")

    def _cleanup_scene_worker(self, w: "SceneDescriberWorker") -> None:
        try:
            self.scene_workers.remove(w)
        except ValueError:
            pass

    # ---------- shutdown ----------
    def closeEvent(self, ev) -> None:
        # Stop periodic timers first so they can't fire mid-teardown.
        if self.preview_timer.isActive():
            self.preview_timer.stop()
        if hasattr(self, "ambient_timer") and self.ambient_timer.isActive():
            self.ambient_timer.stop()
        if hasattr(self, "scene_timer") and self.scene_timer.isActive():
            self.scene_timer.stop()

        # Cancel any in-flight decision so its child HTTP calls return + the
        # Mumble TX queue gets flushed.
        if self.decision_worker is not None and self.decision_worker.isRunning():
            try:
                self.decision_worker.cancel()
                self.decision_worker.wait(2000)
            except Exception:
                pass

        # Stop the capture polling thread.
        if self.capture is not None:
            try:
                self.capture.stop()
            except Exception:
                pass

        # Disconnect from Mumble (sends Bye + closes SSL).
        if self.bot is not None:
            try:
                self.bot.stop()
            except Exception:
                pass

        # Tear down our spawned TTS server subprocess — biggest win, releases
        # the loaded model + GPU memory.
        try:
            if self.tts_server.is_running:
                self.tts_server.stop()
        except Exception:
            pass

        # Wait briefly for in-flight director workers (~500 ms each at worst).
        # Without this we can race with vts.stop() and lose the last expression
        # update, or leak QThread destruction warnings.
        for w in list(self.director_workers):
            try:
                w.wait(800)
            except Exception:
                pass

        # Stop idle decay + VTS poll timers BEFORE vts.stop so they can't try
        # to submit work to a dying asyncio loop.
        if hasattr(self, "_idle_decay_timer") and self._idle_decay_timer.isActive():
            self._idle_decay_timer.stop()
        if hasattr(self, "_vts_poll_timer") and self._vts_poll_timer.isActive():
            self._vts_poll_timer.stop()

        # Stop self-describer timer + any in-flight workers.
        if hasattr(self, "_self_describer_timer") and self._self_describer_timer.isActive():
            self._self_describer_timer.stop()
        for w in list(getattr(self, "_self_describer_workers", [])):
            try:
                w.wait(500)
            except Exception:
                pass

        # Stop IdleMotion ticker before tearing down VTS — otherwise its tick
        # could submit a coroutine to an event loop being torn down.
        if self.idle_motion is not None:
            try:
                self.idle_motion.stop()
            except Exception:
                pass

        # VTS asyncio thread + websocket close. Safe to call even if never
        # connected. ~2 s timeout inside; doesn't block close indefinitely.
        if self.vts is not None:
            try:
                self.vts.stop()
            except Exception:
                pass

        # Bilibili 弹幕 WSS close — independent asyncio thread.
        if self.danmaku is not None:
            try:
                self.danmaku.stop()
            except Exception:
                pass

        # Close the subtitle overlay window so it doesn't outlive the UI.
        if self.subtitle_overlay is not None:
            try:
                self.subtitle_overlay.close()
            except Exception:
                pass

        # Stop the OBS browser-source HTTP server (frees the port).
        if self.web_overlay is not None and self.web_overlay.is_running:
            try:
                self.web_overlay.stop()
            except Exception:
                pass

        # Flush + close the UI log mirror.
        fp = getattr(self, "_ui_log_fp", None)
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass
            self._ui_log_fp = None

        super().closeEvent(ev)
