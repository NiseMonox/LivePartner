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

    def __init__(self, cfg: MumbleConfig):
        super().__init__()
        self.cfg = cfg

    def run(self) -> None:
        try:
            bot = MumbleBot(self.cfg)
            self.log.emit(f"[mumble] connecting to {self.cfg.host}:{self.cfg.port} …")
            bot.start(timeout=8.0)
            self.log.emit(f"[mumble] connected as {self.cfg.name!r}, channel "
                          f"{bot.current_channel_name!r}")
            self.connected.emit(bot)
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


class DecisionWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, req: DecisionRequest):
        super().__init__()
        self.req = req

    def run(self) -> None:
        try:
            img = _synth_you_died_frame()
            frame_b64 = _pil_to_b64(img)
            thumb = img.copy()
            thumb.thumbnail((256, 256))
            thumb_b64 = _pil_to_b64(thumb)
            self.log.emit(f"frame {img.size[0]}x{img.size[1]}, thumb {thumb.size[0]}x{thumb.size[1]}")

            if self.req.force_speak:
                self.log.emit("[gate] skipped (force_speak)")
                speak = True
                dt_gate = 0.0
            else:
                t0 = time.perf_counter()
                speak = gate(self.req.event, thumb_b64, self.req.persona)
                dt_gate = (time.perf_counter() - t0) * 1000
                self.log.emit(f"[gate] speak={speak}  ({dt_gate:.0f} ms)")

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
                target = "Mumble" if self.req.bot else "(dropped, no Mumble)"
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

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LivePartner (M1)")
        self.resize(960, 740)

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

        tts_row1.addWidget(QLabel("配音语种:"))
        self.tts_lang_combo = QComboBox()
        for label, code in [("日语", "日语"), ("中文", "中文"), ("英语", "英语"),
                            ("韩语", "韩语"), ("法语", "法语")]:
            self.tts_lang_combo.addItem(label, userData=code)
        tts_row1.addWidget(self.tts_lang_combo)

        tts_row1.addWidget(QLabel("字幕语种:"))
        self.sub_lang_combo = QComboBox()
        for label, code in [("中文", "中文"), ("日语", "日语"), ("英语", "英语")]:
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
        self.mumble_name = QLineEdit("LivePartner")
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
            name=self.mumble_name.text().strip() or "LivePartner",
            channel=self.mumble_channel.text().strip() or "LivePartner",
        )
        self.mumble_connect_btn.setEnabled(False)
        self.mumble_status.setText("连接中…")
        self.mumble_status.setStyleSheet("color: #c80;")
        self.connect_worker = MumbleConnectWorker(cfg)
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
