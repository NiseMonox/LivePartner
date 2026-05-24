"""faster-whisper wrapper for transcribing player utterances.

Designed to be cheap to call: model is loaded once, kept warm; subsequent
transcribes only pay the inference cost (~1 s for medium on RTX 3070 Laptop).

Windows note: ctranslate2 needs the CUDA runtime DLLs that ship in
nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels. We add their directories to
the DLL search path AND preload the key DLLs via ctypes before importing
faster_whisper — that combo is what actually works on this 3070 Laptop +
CUDA 13.2 driver setup.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path


# ---------- Windows CUDA DLL shim ----------
def _bootstrap_windows_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes
        import sysconfig
        site = Path(sysconfig.get_paths()["purelib"])
        nv = site / "nvidia"
        if not nv.is_dir():
            return
        for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
            p = nv / sub
            if p.is_dir():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")
        for rel in [
            "cublas/bin/cublas64_12.dll",
            "cublas/bin/cublasLt64_12.dll",
            "cudnn/bin/cudnn64_9.dll",
            "cudnn/bin/cudnn_ops64_9.dll",
            "cudnn/bin/cudnn_cnn64_9.dll",
        ]:
            f = nv / rel
            if f.is_file():
                try:
                    ctypes.WinDLL(str(f))
                except OSError:
                    pass
    except Exception:
        pass


_bootstrap_windows_cuda_dlls()

import numpy as np
from faster_whisper import WhisperModel


@dataclass(frozen=True)
class STTResult:
    text: str
    language: str
    language_probability: float
    duration: float
    inference_ms: float


def _pcm_to_float32(pcm_bytes: bytes, src_sr: int = 48000) -> np.ndarray:
    """48 kHz mono int16 PCM → 16 kHz mono float32 in [-1, 1] (whisper input format)."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if src_sr == 16000:
        return audio
    if src_sr == 48000:
        # 3:1 decimation. Crude (no antialiasing filter) but works at the
        # accuracy whisper cares about.
        return audio[::3]
    n_new = int(round(len(audio) * 16000 / src_sr))
    x_old = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


class STT:
    """Lazy-loaded faster-whisper. Construct cheaply, call transcribe_pcm() to use."""

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str | None = None,   # None = auto-detect each call
        beam_size: int = 1,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(self.model_size, device=self.device,
                                       compute_type=self.compute_type)
        return self._model

    def warm_load(self) -> None:
        _ = self.model  # touches the lazy property

    def transcribe_pcm(self, pcm_bytes: bytes, *, sample_rate: int = 48000) -> STTResult:
        audio16k = _pcm_to_float32(pcm_bytes, src_sr=sample_rate)
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            audio16k, language=self.language, beam_size=self.beam_size,
        )
        # Iterating segments is lazy — has to be materialized to time correctly.
        text = "".join(s.text for s in segments).strip()
        dt = (time.perf_counter() - t0) * 1000
        return STTResult(
            text=text,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            inference_ms=dt,
        )
