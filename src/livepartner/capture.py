"""HDMI capture card → most-recent game frame, on demand.

Background thread polls cv2.VideoCapture at the card's native rate; the latest
frame is stashed in a thread-safe slot. Decision code calls latest_frame()
when it wants to send a frame to the LLM.

Audio capture (game BGM via sounddevice) is intentionally not wired yet — that
lands when we hook the signal layer (SPEC §5.2.2) in M2.
"""
from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


def list_video_devices() -> list[str]:
    """DirectShow video device names, in opencv's device-index order on Windows."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception as e:
        log.warning("pygrabber not available: %s", e)
        return []


@dataclass(frozen=True)
class FrameSnapshot:
    """A single grabbed frame.

    `frame` is BGR uint8 HxWx3 (opencv's default). Call to_png_b64() to get the
    base64-encoded PNG bytes the LLM expects.
    """
    frame: np.ndarray
    timestamp: float

    def to_png_b64(self) -> str:
        ok, buf = cv2.imencode(".png", self.frame)
        if not ok:
            raise RuntimeError("cv2.imencode PNG failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def thumbnail_png_b64(self, max_side: int = 256) -> str:
        small = self._scaled(max_side)
        ok, buf = cv2.imencode(".png", small)
        if not ok:
            raise RuntimeError("cv2.imencode PNG failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def to_vlm_b64(self, max_side: int = 1024, quality: int = 80) -> tuple[str, str]:
        """Compact JPEG suitable for vision LLM upload — ~50-150 KB typical at
        max_side=1024 / q=80, vs ~500 KB-1 MB for 1080p PNG. Returns (b64, mime).
        """
        small = self._scaled(max_side)
        ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode JPEG failed")
        return base64.b64encode(buf.tobytes()).decode("ascii"), "image/jpeg"

    def _scaled(self, max_side: int) -> np.ndarray:
        h, w = self.frame.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale >= 1.0:
            return self.frame
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(self.frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class CaptureSource:
    """Owns a cv2.VideoCapture + a background polling thread."""

    def __init__(
        self,
        device_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self.fps = fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[FrameSnapshot] = None
        self._error: Optional[str] = None
        self._negotiated_size: tuple[int, int] | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(f"could not open DirectShow device {self.device_index}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._negotiated_size = (actual_w, actual_h)
        self._cap = cap
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="livepartner-capture",
        )
        self._thread.start()
        log.info(
            "capture started: device=%d  negotiated=%dx%d @ %dfps target",
            self.device_index, actual_w, actual_h, self.fps,
        )

    def _run(self) -> None:
        cap = self._cap
        if cap is None:
            return
        consecutive_failures = 0
        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        self._error = "30 consecutive failed reads — source dropped?"
                        log.warning("capture: 30 consecutive failed reads")
                    time.sleep(0.05)
                    continue
                consecutive_failures = 0
                snap = FrameSnapshot(frame=frame, timestamp=time.monotonic())
                with self._lock:
                    self._latest = snap
        except Exception as e:
            with self._lock:
                self._error = f"{type(e).__name__}: {e}"
            log.exception("capture thread crashed")
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        self._cap = None
        with self._lock:
            self._latest = None
        log.info("capture stopped")

    def latest_frame(self, max_age_sec: float = 2.0) -> Optional[FrameSnapshot]:
        """Return the most-recent frame, or None if it's older than max_age_sec
        (or capture is stalled)."""
        with self._lock:
            snap = self._latest
        if snap is None:
            return None
        if (time.monotonic() - snap.timestamp) > max_age_sec:
            return None
        return snap

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def negotiated_size(self) -> tuple[int, int] | None:
        return self._negotiated_size

    @property
    def last_error(self) -> str | None:
        return self._error
