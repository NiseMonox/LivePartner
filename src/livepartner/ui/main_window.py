"""Minimal M1 main window per SPEC §13.

For M1 we have no capture card / Mumble plumbing yet, so the window provides:
  - persona dropdown
  - editable event description (manual trigger source)
  - "Trigger" button → runs gate→generate→TTS on a synthesized YOU DIED frame
  - live log of pipeline timings + AI output

Status bar shows current state. TTS mp3 is saved to demo_tts.mp3 for playback.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..decision import gate, generate
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


class DecisionWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(str)  # AI text
    failed = Signal(str)       # error message

    def __init__(self, persona: Persona, event: str, *, force_speak: bool, do_tts: bool):
        super().__init__()
        self.persona = persona
        self.event = event
        self.force_speak = force_speak
        self.do_tts = do_tts

    def run(self) -> None:
        try:
            img = _synth_you_died_frame()
            frame_b64 = _pil_to_b64(img)
            thumb = img.copy()
            thumb.thumbnail((256, 256))
            thumb_b64 = _pil_to_b64(thumb)
            self.log.emit(f"frame {img.size[0]}x{img.size[1]}, thumb {thumb.size[0]}x{thumb.size[1]}")

            if self.force_speak:
                self.log.emit("[gate] skipped (force_speak)")
                speak = True
                dt_gate = 0.0
            else:
                t0 = time.perf_counter()
                speak = gate(self.event, thumb_b64, self.persona)
                dt_gate = (time.perf_counter() - t0) * 1000
                self.log.emit(f"[gate] speak={speak}  ({dt_gate:.0f} ms)")

            if not speak:
                self.log.emit("AI 选择沉默。")
                self.finished_ok.emit("")
                return

            t0 = time.perf_counter()
            reply = generate(self.event, frame_b64, self.persona)
            dt_gen = (time.perf_counter() - t0) * 1000
            self.log.emit(f"[generate] {dt_gen:.0f} ms")
            self.log.emit(f"AI ({self.persona.display_name}) > {reply.text}")
            self.log.emit(f"total LLM: {dt_gate + dt_gen:.0f} ms")

            if self.do_tts and reply.text:
                t0 = time.perf_counter()
                tts = synthesize_for_persona(reply.text, self.persona)
                dt_tts = (time.perf_counter() - t0) * 1000
                out = Path("demo_tts.mp3")
                out.write_bytes(tts.mp3)
                self.log.emit(
                    f"[tts] {dt_tts:.0f} ms  voice={tts.voice_id}  {len(tts.mp3)//1024} KB → {out}"
                )

            self.finished_ok.emit(reply.text)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MainWindow(QMainWindow):
    DEFAULT_EVENT = (
        "玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。BGM 转为低沉死亡音乐。"
    )

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LivePartner (M1)")
        self.resize(820, 560)

        central = QWidget()
        root = QVBoxLayout(central)

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

        self.tts_check = QCheckBox("生成 TTS")
        self.tts_check.setChecked(True)
        row1.addWidget(self.tts_check)
        root.addLayout(row1)

        root.addWidget(QLabel("事件描述："))
        self.event_edit = QLineEdit(self.DEFAULT_EVENT)
        root.addWidget(self.event_edit)

        self.trigger_btn = QPushButton("触发一次 AI 反应（合成 YOU DIED 帧）")
        self.trigger_btn.clicked.connect(self._on_trigger)
        root.addWidget(self.trigger_btn)

        root.addWidget(QLabel("日志："))
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 10))
        root.addWidget(self.log_view, stretch=1)

        last_line_lbl = QLabel("AI 最近一句：")
        self.last_line_lbl = QLabel("(尚未生成)")
        self.last_line_lbl.setFont(QFont("Microsoft YaHei", 14, QFont.Weight.Bold))
        self.last_line_lbl.setWordWrap(True)
        self.last_line_lbl.setStyleSheet("color: #c33; padding: 8px;")
        root.addWidget(last_line_lbl)
        root.addWidget(self.last_line_lbl)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")
        self.worker: DecisionWorker | None = None

    def _log(self, msg: str) -> None:
        self.log_view.append(msg)

    def _on_trigger(self) -> None:
        if self.worker is not None and self.worker.isRunning():
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

        self._log(f"\n=== {persona.display_name} ({persona_id}) ===")
        self._log(f"事件: {self.event_edit.text()}")
        self.trigger_btn.setEnabled(False)
        self.statusBar().showMessage("运行中…")

        self.worker = DecisionWorker(
            persona,
            self.event_edit.text(),
            force_speak=self.force_check.isChecked(),
            do_tts=self.tts_check.isChecked(),
        )
        self.worker.log.connect(self._log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._cleanup)
        self.worker.start()

    def _on_done(self, ai_text: str) -> None:
        if ai_text:
            self.last_line_lbl.setText(ai_text)
            self.statusBar().showMessage("完成")
        else:
            self.statusBar().showMessage("AI 沉默")

    def _on_failed(self, msg: str) -> None:
        self._log(f"[ERR] {msg}")
        self.statusBar().showMessage("出错")

    def _cleanup(self) -> None:
        self.trigger_btn.setEnabled(True)
        self.worker = None
