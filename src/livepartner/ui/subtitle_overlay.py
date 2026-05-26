"""Always-on-top, frameless, semi-transparent subtitle overlay.

Designed to be readable as an OBS source (transparent background, big white
text with black outline, fades in/out per AI line). Bridged from MainWindow
via show_line(); MainWindow calls it from _on_decision_done so any spoken
line lands here automatically.

Default placement: bottom-center of the primary screen, ~80% width, 120px tall.
The user can drag it to a new spot — position persists for this session
(``self.move(...)``); we don't write to disk because the user usually wants
a fresh layout each session.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics, QGuiApplication, QMouseEvent, QPainter,
    QPainterPath, QPen,
)
from PySide6.QtWidgets import QWidget


class SubtitleOverlay(QWidget):
    """Frameless top-most window that draws stroked text in the middle.

    Click + drag anywhere to reposition. Text auto-clears after ``hold_ms``
    if no new line arrives — set hold_ms<=0 to keep it sticky.
    """

    def __init__(
        self,
        *,
        hold_ms: int = 6000,
        font_size: int = 32,
        opacity: float = 1.0,
    ) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool,  # don't show in taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        # Important on Windows: without ShowWithoutActivating the overlay
        # steals focus on every show().
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowOpacity(opacity)

        self._text = ""
        self._font = QFont("Microsoft YaHei", font_size, QFont.Weight.Bold)
        self._hold_ms = hold_ms

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_hide_timeout)

        # Default geometry: 80% width, 140px tall, bottom-center of primary
        # screen with 80px bottom margin so it's clear of taskbar.
        screen = QGuiApplication.primaryScreen().availableGeometry()
        w = int(screen.width() * 0.8)
        h = 140
        x = screen.x() + (screen.width() - w) // 2
        y = screen.y() + screen.height() - h - 80
        self.setGeometry(x, y, w, h)

        # Drag-to-move state.
        self._drag_origin: QPoint | None = None

    # ---------- public API ----------
    def show_line(self, text: str) -> None:
        """Show ``text`` and start the auto-hide timer (if configured)."""
        self._text = text or ""
        self.update()
        if not self.isVisible() and text:
            self.show()
        if text and self._hold_ms > 0:
            self._hide_timer.start(self._hold_ms)
        elif not text:
            self._hide_timer.stop()

    def clear(self) -> None:
        self._text = ""
        self.update()
        self.hide()
        self._hide_timer.stop()

    def set_font_size(self, pt: int) -> None:
        self._font = QFont("Microsoft YaHei", max(8, int(pt)), QFont.Weight.Bold)
        self.update()

    def set_opacity_pct(self, pct: int) -> None:
        self.setWindowOpacity(max(0.1, min(1.0, pct / 100.0)))

    def set_hold_ms(self, ms: int) -> None:
        self._hold_ms = max(0, int(ms))

    # ---------- internals ----------
    def _on_hide_timeout(self) -> None:
        # Just hide; don't clear text so re-showing remembers last line.
        self.hide()

    def paintEvent(self, _ev) -> None:
        if not self._text:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        p.setFont(self._font)

        # Stroked text: build a QPainterPath and stroke it with a thick black
        # pen, then fill with white. Much more readable on busy backgrounds
        # than plain text or shadow.
        path = QPainterPath()
        fm = QFontMetrics(self._font)
        # Word-wrap manually so we can center each line.
        margin = 24
        rect = self.rect().adjusted(margin, margin, -margin, -margin)
        text = self._text
        # Crude wrap: split by char boundaries to fit width. For typical
        # 1-2 line replies (≤30 chars) this is trivial.
        lines: list[str] = []
        cur = ""
        for ch in text:
            cur_test = cur + ch
            if fm.horizontalAdvance(cur_test) > rect.width() and cur:
                lines.append(cur)
                cur = ch
            else:
                cur = cur_test
        if cur:
            lines.append(cur)
        line_h = fm.height()
        total_h = line_h * len(lines)
        y = rect.top() + (rect.height() - total_h) // 2 + fm.ascent()
        for line in lines:
            x = rect.left() + (rect.width() - fm.horizontalAdvance(line)) // 2
            path.addText(x, y, self._font, line)
            y += line_h
        # Outline pen (black, thick) then fill (white).
        pen = QPen(QColor(0, 0, 0, 220))
        pen.setWidth(6)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(QColor(255, 255, 255))
        p.drawPath(path)

    # Drag to reposition.
    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = ev.globalPosition().toPoint() - self.frameGeometry().topLeft()
            ev.accept()

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag_origin is not None and ev.buttons() & Qt.MouseButton.LeftButton:
            self.move(ev.globalPosition().toPoint() - self._drag_origin)
            ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            ev.accept()
