"""Owns the local Qwen3-TTS server subprocess lifecycle.

The TTS server lives in its own venv (``external/faster-qwen3-tts/.venv``) and
loads a ~600MB model at startup. We spawn it as a child process from the UI so
the user doesn't have to remember to start it manually, and so closing the UI
cleanly tears it down (no orphan GPU process).

stdout/stderr are redirected to .memory/tts_server.log so the user can tail it
if anything looks off; we don't try to stream it back to the Qt UI in real
time (would be noisy and adds threading complexity).
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TTS_VENV_PY = REPO_ROOT / "external" / "faster-qwen3-tts" / ".venv" / "Scripts" / "python.exe"
TTS_SCRIPT = REPO_ROOT / "services" / "qwen3_tts_server.py"
LOG_PATH = REPO_ROOT / ".memory" / "tts_server.log"


class TtsServerProcess:
    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self._log_fp = None  # kept open for the lifetime of the process

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.is_running else None

    @property
    def log_path(self) -> Path:
        return LOG_PATH

    def start(self) -> int:
        """Spawn the TTS server subprocess. Returns the child PID.

        Raises ``FileNotFoundError`` if the venv python or script isn't where
        we expect. Idempotent — re-calling while running returns the existing
        PID.
        """
        if self.is_running:
            return self.pid  # type: ignore[return-value]
        if not TTS_VENV_PY.exists():
            raise FileNotFoundError(
                f"TTS venv python not found: {TTS_VENV_PY} — "
                "did you run setup_windows.bat under external/faster-qwen3-tts/?"
            )
        if not TTS_SCRIPT.exists():
            raise FileNotFoundError(f"TTS server script not found: {TTS_SCRIPT}")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so consecutive starts share one file (easier to read
        # across sessions). Header line marks each restart.
        self._log_fp = LOG_PATH.open("ab", buffering=0)
        try:
            import datetime
            self._log_fp.write(
                f"\n\n===== TTS server start {datetime.datetime.now().isoformat()} =====\n"
                .encode("utf-8")
            )
        except Exception:
            pass

        # Win32-specific creation flags:
        # - CREATE_NEW_PROCESS_GROUP: detaches from our console so terminate()
        #   doesn't accidentally hit US (the UI) too
        # - CREATE_NO_WINDOW: don't pop up a console for the child
        creationflags = 0
        if sys.platform == "win32":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
                | subprocess.CREATE_NO_WINDOW       # type: ignore[attr-defined]
            )

        # -u: unbuffered stdout/stderr so the log file shows real-time progress
        # (without it, Python buffers print() chunks and the model-loading
        # output only appears all at once after ~60s, looking like a freeze).
        # PYTHONUNBUFFERED is a belt-and-braces for any C-level prints.
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            [str(TTS_VENV_PY), "-u", str(TTS_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdout=self._log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=creationflags,
            env=env,
        )
        log.info("TTS server spawned: pid=%d, log=%s", self.proc.pid, LOG_PATH)
        return self.proc.pid

    def stop(self, timeout: float = 6.0) -> None:
        """Terminate the child. Tries graceful first, then SIGKILL after timeout."""
        if not self.is_running:
            self._close_log()
            return
        proc = self.proc
        assert proc is not None
        try:
            proc.terminate()  # SIGTERM (Unix) / TerminateProcess (Win)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("TTS server didn't exit in %.1fs, killing", timeout)
            try:
                proc.kill()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        except Exception as e:
            log.warning("TTS server stop error: %s", e)
        finally:
            self._close_log()

    def _close_log(self) -> None:
        fp = self._log_fp
        self._log_fp = None
        if fp is not None:
            try:
                fp.close()
            except Exception:
                pass
