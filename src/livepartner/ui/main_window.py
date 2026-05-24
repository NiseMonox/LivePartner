"""Minimal M1 main window per SPEC §13.

Provides:
  - persona dropdown
  - editable event description (manual trigger source)
  - TTS engine selection: Edge TTS (cloud) or Qwen3 (local server)
  - Mumble connect/disconnect with status, so the bot stays present in a channel
    and AI lines play live through Mumble while connected
  - "Trigger" button → runs gate→generate→TTS on a synthesized YOU DIED frame
  - live log of pipeline timings + AI output
"""
from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..audio_codec import mp3_to_pcm48k, pcm48k_duration_seconds
from ..decision import gate, generate
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
# stt import is deferred to first use because faster-whisper + ctranslate2 take
# ~1s on import (CUDA DLL preload) — keeps the UI snappy on cold start.


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
    loaded = Signal(object)  # STT instance
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
    transcribed = Signal(str, dict, object)  # text, user_dict, STTResult
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


@dataclass
class DecisionRequest:
    persona: Persona
    event: str
    force_speak: bool
    save_mp3: bool          # save Edge TTS mp3 to demo_tts.mp3
    bot: MumbleBot | None   # if set, also stream PCM through it
    engine: str             # "edge" or "qwen3"
    qwen3_url: str          # base url for qwen3 server when engine=qwen3
    tts_language: str       # voice language, e.g. "日语" / "中文" / "英语"
    subtitle_language: str  # display language, e.g. "中文"
    synthesize_frame: bool = True   # game-event trigger: feed a YOU DIED frame; voice chat: no frame
    is_conversation: bool = False   # voice chat — tells the LLM to drop "stay silent" rules


class DecisionWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, req: DecisionRequest):
        super().__init__()
        self.req = req

    def run(self) -> None:
        try:
            if self.req.synthesize_frame:
                img = _synth_you_died_frame()
                frame_b64 = _pil_to_b64(img)
                thumb = img.copy()
                thumb.thumbnail((256, 256))
                thumb_b64 = _pil_to_b64(thumb)
                self.log.emit(f"frame {img.size[0]}x{img.size[1]}, thumb {thumb.size[0]}x{thumb.size[1]}")
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
                # No frame, no force — refuse to invent intent for the AI.
                self.log.emit("[gate] no frame + not forced → silent")
                self.finished_ok.emit("")
                return

            if not speak:
                self.log.emit("AI 选择沉默。")
                self.finished_ok.emit("")
                return

            t0 = time.perf_counter()
            reply = generate(
                self.req.event, frame_b64, self.req.persona,
                bilingual=(self.req.tts_language != self.req.subtitle_language),
                tts_language=self.req.tts_language,
                subtitle_language=self.req.subtitle_language,
                is_conversation=self.req.is_conversation,
            )
            dt_gen = (time.perf_counter() - t0) * 1000
            self.log.emit(f"[generate] {dt_gen:.0f} ms")
            self.log.emit(f"AI 字幕({self.req.subtitle_language}): {reply.text}")
            if reply.tts_text != reply.text:
                self.log.emit(f"AI 配音({self.req.tts_language}): {reply.tts_text}")
            self.log.emit(f"total LLM: {dt_gate + dt_gen:.0f} ms")

            if not reply.tts_text:
                self.finished_ok.emit("")
                return

            if self.req.engine == "qwen3":
                self._tts_qwen3_streaming(reply.tts_text)
            else:
                self._tts_edge(reply.tts_text)

            # UI displays the subtitle (Chinese), so emit reply.text not tts_text.
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
                          f"{self.req.bot.current_channel_name!r}")

    def _tts_qwen3_streaming(self, text: str) -> None:
        cfg = Qwen3TtsConfig(base_url=self.req.qwen3_url, language=self.req.tts_language)
        t0 = time.perf_counter()
        total = 0
        first_ms = None
        for chunk in qwen3_stream_pcm(text, self.req.persona, cfg=cfg):
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
        dt = (time.perf_counter() - t0) * 1000
        dur_s = total / (48000 * 2)
        self.log.emit(f"[qwen3] done {dt:.0f} ms  {total//1024} KB PCM  {dur_s:.2f}s "
                      f"(RTF {dur_s / (dt/1000):.2f})")


class Qwen3ProbeWorker(QThread):
    """Quick async check of qwen3 server: alive + voice list."""
    done = Signal(bool, list, str)  # alive, voices, error_or_url

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


# ---------- main window ----------


class MainWindow(QMainWindow):
    DEFAULT_EVENT = (
        "玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。BGM 转为低沉死亡音乐。"
    )

    # Cross-thread bridge: pymumble's network thread emits this; the slot runs on the UI thread.
    utterance_signal = Signal(dict, bytes)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LivePartner (M1)")
        self.resize(960, 820)

        central = QWidget()
        root = QVBoxLayout(central)

        # --- persona + options row ---
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
        self.force_check.setChecked(True)
        row1.addWidget(self.force_check)
        self.save_mp3_check = QCheckBox("保存 demo_tts.mp3 (Edge)")
        self.save_mp3_check.setChecked(False)
        row1.addWidget(self.save_mp3_check)
        root.addLayout(row1)

        # --- TTS panel ---
        tts_box = QGroupBox("TTS 引擎")
        tts_outer = QVBoxLayout(tts_box)

        tts_row1 = QHBoxLayout()
        tts_row1.addWidget(QLabel("引擎:"))
        self.tts_engine_combo = QComboBox()
        self.tts_engine_combo.addItem("Edge TTS (云,免费)", userData="edge")
        self.tts_engine_combo.addItem("Qwen3-TTS (本地,克隆)", userData="qwen3")
        self.tts_engine_combo.setCurrentIndex(1)  # default to Qwen3 now that voices are trained
        self.tts_engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        tts_row1.addWidget(self.tts_engine_combo)

        # Qwen3-TTS only accepts these English codes (lowercased server-side):
        # chinese, english, german, italian, portuguese, spanish, japanese, korean, french, russian, auto
        tts_row1.addWidget(QLabel("配音语种:"))
        self.tts_lang_combo = QComboBox()
        for label, code in [("日语", "Japanese"), ("中文", "Chinese"), ("英语", "English"),
                            ("韩语", "Korean"), ("法语", "French"),
                            ("德语", "German"), ("自动", "Auto")]:
            self.tts_lang_combo.addItem(label, userData=code)
        tts_row1.addWidget(self.tts_lang_combo)

        tts_row1.addWidget(QLabel("字幕语种:"))
        self.sub_lang_combo = QComboBox()
        for label, code in [("中文", "Chinese"), ("日语", "Japanese"), ("英语", "English")]:
            self.sub_lang_combo.addItem(label, userData=code)
        tts_row1.addWidget(self.sub_lang_combo)
        tts_row1.addStretch()
        tts_outer.addLayout(tts_row1)

        tts_row2 = QHBoxLayout()
        tts_row2.addWidget(QLabel("Qwen3 URL:"))
        self.qwen3_url_edit = QLineEdit(QWEN3_DEFAULT_URL)
        self.qwen3_url_edit.setMaximumWidth(240)
        tts_row2.addWidget(self.qwen3_url_edit)
        self.qwen3_probe_btn = QPushButton("检测")
        self.qwen3_probe_btn.clicked.connect(self._on_probe_qwen3)
        tts_row2.addWidget(self.qwen3_probe_btn)
        self.qwen3_status = QLabel("(未检测)")
        self.qwen3_status.setStyleSheet("color: #888;")
        tts_row2.addWidget(self.qwen3_status, stretch=1)
        tts_outer.addLayout(tts_row2)

        root.addWidget(tts_box)

        # --- Mumble panel ---
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
        root.addWidget(mumble_box)

        # --- STT panel ---
        stt_box = QGroupBox("STT 监听玩家")
        stt_outer = QVBoxLayout(stt_box)

        stt_row1 = QHBoxLayout()
        stt_row1.addWidget(QLabel("模型:"))
        self.stt_size_combo = QComboBox()
        for s in ["tiny", "base", "small", "medium", "large-v3"]:
            self.stt_size_combo.addItem(s)
        self.stt_size_combo.setCurrentText("medium")
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
        stt_row1.addWidget(self.stt_listen_check)
        self.stt_status = QLabel("(未加载)")
        self.stt_status.setStyleSheet("color: #888;")
        stt_row1.addWidget(self.stt_status, stretch=1)
        stt_outer.addLayout(stt_row1)

        stt_row2 = QHBoxLayout()
        stt_row2.addWidget(QLabel("白名单 (逗号分隔,留空=全监听):"))
        self.stt_whitelist_edit = QLineEdit()
        self.stt_whitelist_edit.setPlaceholderText("e.g. NiseMono, 朋友A")
        self.stt_whitelist_edit.editingFinished.connect(self._on_whitelist_changed)
        stt_row2.addWidget(self.stt_whitelist_edit, stretch=1)
        self.stt_whitelist_apply_btn = QPushButton("应用")
        self.stt_whitelist_apply_btn.clicked.connect(self._on_whitelist_changed)
        stt_row2.addWidget(self.stt_whitelist_apply_btn)
        stt_outer.addLayout(stt_row2)

        root.addWidget(stt_box)

        # --- event input + trigger ---
        root.addWidget(QLabel("事件描述："))
        self.event_edit = QLineEdit(self.DEFAULT_EVENT)
        root.addWidget(self.event_edit)
        self.trigger_btn = QPushButton("触发一次 AI 反应（合成 YOU DIED 帧）")
        self.trigger_btn.clicked.connect(self._on_trigger)
        root.addWidget(self.trigger_btn)

        # --- log + last line ---
        root.addWidget(QLabel("日志："))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        root.addWidget(self.log_view, stretch=1)

        last_lbl = QLabel("AI 最近一句：")
        self.last_line_lbl = QLabel("(尚未生成)")
        self.last_line_lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.last_line_lbl.setWordWrap(True)
        self.last_line_lbl.setStyleSheet("color: #c33; padding: 8px;")
        root.addWidget(last_lbl)
        root.addWidget(self.last_line_lbl)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

        self.decision_worker: DecisionWorker | None = None
        self.connect_worker: MumbleConnectWorker | None = None
        self.probe_worker: Qwen3ProbeWorker | None = None
        self.bot: MumbleBot | None = None
        self.stt = None  # populated by STTLoadWorker
        self.stt_load_worker: STTLoadWorker | None = None
        self.stt_workers: list[STTTranscribeWorker] = []

        # Bridge: bot callback (network thread) → Qt signal → UI-thread slot
        self.utterance_signal.connect(self._on_utterance_arrived)

        self._on_engine_changed()  # set initial state of qwen3 controls

    # ---------- logging ----------
    def _log(self, msg: str) -> None:
        self.log_view.append(msg)

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
        else:
            self.qwen3_status.setText(f"不可达 ({info})")
            self.qwen3_status.setStyleSheet("color: #c33;")

    def _cleanup_probe(self) -> None:
        self.probe_worker = None

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
            channel=self.mumble_channel.text().strip() or "LivePartner",
        )
        self.mumble_connect_btn.setEnabled(False)
        self.mumble_status.setText("连接中…")
        self.mumble_status.setStyleSheet("color: #c80;")
        # Pass a thread-safe forwarder: pymumble callback emits signal that
        # marshals to the UI thread.
        self.connect_worker = MumbleConnectWorker(
            cfg,
            on_user_utterance=self._bot_utterance_cb,
            voice_whitelist=self._parse_whitelist(),
        )
        self.connect_worker.log.connect(self._log)
        self.connect_worker.connected.connect(self._on_mumble_connected)
        self.connect_worker.failed.connect(self._on_mumble_failed)
        self.connect_worker.finished.connect(self._cleanup_connect_worker)
        self.connect_worker.start()

    def _parse_whitelist(self) -> set[str] | None:
        raw = self.stt_whitelist_edit.text().strip()
        if not raw:
            return None
        return {n.strip() for n in raw.split(",") if n.strip()}

    def _on_whitelist_changed(self) -> None:
        names = self._parse_whitelist()
        if self.bot is not None:
            self.bot.set_voice_whitelist(names)
        if names is None:
            self._log("[stt] whitelist: (全监听)")
        else:
            self._log(f"[stt] whitelist: {sorted(names)}")

    # ---------- STT (utterance → transcribe → maybe trigger AI) ----------
    def _bot_utterance_cb(self, user: dict, pcm_bytes: bytes) -> None:
        # Called from pymumble's network thread — bounce to UI thread via signal.
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
                  f"{result.inference_ms:.0f}ms): {text!r}")
        if not text:
            return
        # Auto-respond: synthesize a fresh event and run the same decision pipeline
        # as the manual trigger. force_speak — per SPEC §5.3.2 player voice is a
        # priority event.
        if self.decision_worker is not None and self.decision_worker.isRunning():
            self._log("[stt] AI 还在说上一句,丢弃这一轮")
            return
        persona_id = self.persona_combo.currentData()
        if not persona_id:
            return
        try:
            persona = load_persona(persona_id)
        except Exception as e:
            self._log(f"[stt] persona load failed: {e}")
            return
        engine = self.tts_engine_combo.currentData() or "edge"
        qwen3_url = self.qwen3_url_edit.text().strip().rstrip("/")
        tts_lang = self.tts_lang_combo.currentData() or "Japanese"
        sub_lang = self.sub_lang_combo.currentData() or "Chinese"
        event = f'{name}: 「{text}」'
        self._log(f"=== {persona.display_name} ({persona_id}) — 对话 ===")
        self._log(f"事件: {event}")
        self.decision_worker = DecisionWorker(DecisionRequest(
            persona=persona,
            event=event,
            force_speak=True,  # player voice is always a priority trigger
            save_mp3=False,
            bot=self.bot if (self.bot is not None and self.mumble_speak_check.isChecked()) else None,
            engine=engine,
            qwen3_url=qwen3_url,
            tts_language=tts_lang,
            subtitle_language=sub_lang,
            synthesize_frame=False,   # voice chat — no game frame
            is_conversation=True,
        ))
        self.decision_worker.log.connect(self._log)
        self.decision_worker.finished_ok.connect(self._on_decision_done)
        self.decision_worker.failed.connect(self._on_decision_failed)
        self.decision_worker.finished.connect(self._cleanup_decision)
        self.decision_worker.start()

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
        size = self.stt_size_combo.currentText()
        self.stt_status.setText(f"已加载 · whisper-{size}")
        self.stt_status.setStyleSheet("color: #2a2;")
        self.stt_listen_check.setEnabled(True)
        self.stt_listen_check.setChecked(True)

    def _on_stt_failed(self, msg: str) -> None:
        self._log(f"[stt] load FAILED: {msg}")
        self.stt_status.setText(f"加载失败")
        self.stt_status.setStyleSheet("color: #c33;")
        self.stt_load_btn.setEnabled(True)

    def _cleanup_stt_load_worker(self) -> None:
        self.stt_load_worker = None
        if self.stt is None:
            self.stt_load_btn.setEnabled(True)

    def _on_mumble_connected(self, bot: object) -> None:
        assert isinstance(bot, MumbleBot)
        self.bot = bot
        ch = bot.current_channel_name or "(unknown)"
        self.mumble_status.setText(f"已连接 · 频道 {ch!r}")
        self.mumble_status.setStyleSheet("color: #2a2;")
        self.mumble_disconnect_btn.setEnabled(True)
        self.mumble_speak_check.setEnabled(True)
        # Re-check on every connect — otherwise a previous disconnect leaves the
        # checkbox unchecked, and AI silently "speaks" into the void.
        self.mumble_speak_check.setChecked(True)

    def _on_mumble_failed(self, msg: str) -> None:
        self._log(f"[mumble] connect FAILED: {msg}")
        self.mumble_status.setText("连接失败")
        self.mumble_status.setStyleSheet("color: #c33;")
        self.mumble_connect_btn.setEnabled(True)

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

    # ---------- Decision ----------
    def _on_trigger(self) -> None:
        if self.decision_worker is not None and self.decision_worker.isRunning():
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

        bot_for_speech = self.bot if (self.bot is not None and self.mumble_speak_check.isChecked()) else None

        engine = self.tts_engine_combo.currentData() or "edge"
        qwen3_url = self.qwen3_url_edit.text().strip().rstrip("/")
        tts_lang = self.tts_lang_combo.currentData() or "日语"
        sub_lang = self.sub_lang_combo.currentData() or "中文"

        self._log(f"\n=== {persona.display_name} ({persona_id}) ===")
        self._log(f"事件: {self.event_edit.text()}")
        self._log(f"TTS 引擎: {engine}{' @ ' + qwen3_url if engine == 'qwen3' else ''}  "
                  f"配音={tts_lang}  字幕={sub_lang}")
        if bot_for_speech is not None:
            self._log(f"(通过 Mumble 频道 {bot_for_speech.current_channel_name!r} 播放)")
        self.trigger_btn.setEnabled(False)
        self.statusBar().showMessage("运行中…")

        self.decision_worker = DecisionWorker(DecisionRequest(
            persona=persona,
            event=self.event_edit.text(),
            force_speak=self.force_check.isChecked(),
            save_mp3=self.save_mp3_check.isChecked(),
            bot=bot_for_speech,
            engine=engine,
            qwen3_url=qwen3_url,
            tts_language=tts_lang,
            subtitle_language=sub_lang,
        ))
        self.decision_worker.log.connect(self._log)
        self.decision_worker.finished_ok.connect(self._on_decision_done)
        self.decision_worker.failed.connect(self._on_decision_failed)
        self.decision_worker.finished.connect(self._cleanup_decision)
        self.decision_worker.start()

    def _on_decision_done(self, ai_text: str) -> None:
        if ai_text:
            self.last_line_lbl.setText(ai_text)
            self.statusBar().showMessage("完成")
        else:
            self.statusBar().showMessage("AI 沉默")

    def _on_decision_failed(self, msg: str) -> None:
        self._log(f"[ERR] {msg}")
        self.statusBar().showMessage("出错")

    def _cleanup_decision(self) -> None:
        self.trigger_btn.setEnabled(True)
        self.decision_worker = None

    # ---------- shutdown ----------
    def closeEvent(self, ev) -> None:
        if self.bot is not None:
            try:
                self.bot.stop()
            except Exception:
                pass
        super().closeEvent(ev)
