"""Minimal M1 main window per SPEC §13.

Provides:
  - persona dropdown
  - editable event description (manual trigger source)
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
    connected = Signal(object)  # MumbleBot
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
    do_tts: bool
    bot: MumbleBot | None  # if set, also stream PCM through it


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
            reply = generate(self.req.event, frame_b64, self.req.persona)
            dt_gen = (time.perf_counter() - t0) * 1000
            self.log.emit(f"[generate] {dt_gen:.0f} ms")
            self.log.emit(f"AI ({self.req.persona.display_name}) > {reply.text}")
            self.log.emit(f"total LLM: {dt_gate + dt_gen:.0f} ms")

            if (self.req.do_tts or self.req.bot is not None) and reply.text:
                t0 = time.perf_counter()
                tts = synthesize_for_persona(reply.text, self.req.persona)
                dt_tts = (time.perf_counter() - t0) * 1000
                self.log.emit(f"[tts] {dt_tts:.0f} ms  voice={tts.voice_id}  "
                              f"{len(tts.mp3)//1024} KB mp3")
                if self.req.do_tts:
                    Path("demo_tts.mp3").write_bytes(tts.mp3)

                if self.req.bot is not None:
                    t0 = time.perf_counter()
                    pcm = mp3_to_pcm48k(tts.mp3)
                    dur = pcm48k_duration_seconds(pcm)
                    self.log.emit(f"[pcm] {(time.perf_counter()-t0)*1000:.0f} ms  "
                                  f"{len(pcm)//1024} KB  {dur:.2f}s")
                    self.req.bot.send_pcm(pcm)
                    self.log.emit(f"[mumble] streaming {dur:.2f}s into channel "
                                  f"{self.req.bot.current_channel_name!r} …")

            self.finished_ok.emit(reply.text)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# ---------- main window ----------


class MainWindow(QMainWindow):
    DEFAULT_EVENT = (
        "玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。BGM 转为低沉死亡音乐。"
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LivePartner (M1)")
        self.resize(880, 640)

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
        self.tts_save_check = QCheckBox("保存 demo_tts.mp3")
        self.tts_save_check.setChecked(True)
        row1.addWidget(self.tts_save_check)
        root.addLayout(row1)

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
        self.bot: MumbleBot | None = None

    # ---------- logging ----------
    def _log(self, msg: str) -> None:
        self.log_view.append(msg)

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
        self.mumble_status.setText(f"连接失败")
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

        self._log(f"\n=== {persona.display_name} ({persona_id}) ===")
        self._log(f"事件: {self.event_edit.text()}")
        if bot_for_speech is not None:
            self._log(f"(将通过 Mumble 频道 {bot_for_speech.current_channel_name!r} 播放)")
        self.trigger_btn.setEnabled(False)
        self.statusBar().showMessage("运行中…")

        self.decision_worker = DecisionWorker(DecisionRequest(
            persona=persona,
            event=self.event_edit.text(),
            force_speak=self.force_check.isChecked(),
            do_tts=self.tts_save_check.isChecked(),
            bot=bot_for_speech,
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
