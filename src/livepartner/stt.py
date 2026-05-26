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
    # Non-empty when we deliberately dropped the transcription — the most
    # frequent cause is Whisper hallucinating standard YouTube tail phrases
    # ("Thanks for watching", "ご視聴ありがとうございました") on cough/breath/
    # mouse-click audio. Empty ``text`` AND ``dropped_reason`` set means the
    # caller should silently ignore this utterance.
    dropped_reason: str = ""
    raw_text: str = ""           # what the model actually emitted, pre-filter
    no_speech_prob: float = 0.0  # max segment no_speech_prob


# Phrases Whisper emits on silent/noise audio. Sourced from the openai/whisper
# issue tracker + observed in this project. Matched case-insensitive against
# stripped text. Keep this conservative — single-word valid replies (是, 嗯,
# yes, ok) should NOT live here.
# Entries are normalized (lowercase, outer punctuation already stripped).
_HALLUCINATION_PHRASES = frozenset({
    # Generic English noise tokens — typical when mic picks up cough/breath.
    "ahem", "ahem ahem",
    "pfft",
    "huh", "hmm",
    "mm-hmm", "mm hmm", "uh-huh", "uh huh",
    "you",
    # Whisper's YouTube training tail. We keep the full-sentence forms to
    # avoid false-positiving on plain "thanks" / "thank you" which a player
    # might really say.
    "thanks for watching", "thank you for watching",
    "subscribe to my channel",
    "see you next time",
    # Bare English closers Whisper hallucinates on Chinese-speaker breath /
    # cough / mouse-click audio. Players in this project address Eri in Chinese,
    # so a bare English "thank you" / "bye" is overwhelmingly likely a model
    # hallucination, not a real address. The short-English heuristic below
    # catches further variants we haven't enumerated.
    "thank you", "thank you.", "thanks", "thank", "thanks!",
    "bye", "bye.", "bye-bye", "bye bye", "goodbye",
    # Japanese training tail.
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "次回もお楽しみに",
    "それでは、また",
    # Korean training tail.
    "시청해주셔서 감사합니다",
    "구독과 좋아요 부탁드립니다",
    # Bracketed sound-effect labels — Whisper emits these as transcription.
    "music", "applause", "laughter", "silence",
})

# Heuristic to backstop _HALLUCINATION_PHRASES: a player who routinely speaks
# Chinese to Eri sending a 1-3 word English fragment with high language-detect
# confidence is overwhelmingly a Whisper YouTube-tail hallucination. We match
# the fragment against a permissive set of common English fillers, and drop it
# only if Whisper itself committed to "en" with high probability.
_EN_FILLER_FRAGMENTS = frozenset({
    "thank you", "thanks", "thank", "thx",
    "bye", "goodbye", "see ya", "see you",
    "hello", "hi", "hey",
    "yeah", "yep", "yes", "no", "nope",
    "okay", "ok", "alright", "right",
    "good", "great", "nice", "cool",
    "wow", "oh", "ah",
    "i see", "i know", "i don't know",
})


def _looks_like_en_filler(text: str) -> bool:
    """Lowercase + strip punctuation + check word count ≤ 3 + membership."""
    s = text.strip().lower().strip(_OUTER_PUNCT)
    if not s or len(s.split()) > 3:
        return False
    return s in _EN_FILLER_FRAGMENTS

# If max(no_speech_prob across segments) exceeds this, we treat the whole
# utterance as noise. faster-whisper's threshold is per-segment voting;
# we want a stricter trigger.
NO_SPEECH_PROB_DROP = 0.6


# Outer punctuation we want to strip both sides so e.g. "Ahem.", "Pfft!",
# "[Music]", "시청해주셔서 감사합니다!" all normalize to the bare phrase.
_OUTER_PUNCT = ".,!?！？。、，「」 \"'`()[]【】《》~～"


def _is_hallucination_phrase(text: str) -> bool:
    s = text.strip().lower().strip(_OUTER_PUNCT)
    return s in _HALLUCINATION_PHRASES


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
        # We collect segments first so we can also inspect no_speech_prob.
        seg_list = list(segments)
        raw_text = "".join(s.text for s in seg_list).strip()
        max_nsp = max((float(getattr(s, "no_speech_prob", 0.0) or 0.0)
                       for s in seg_list), default=0.0)
        dt = (time.perf_counter() - t0) * 1000

        # ---- noise filters ----
        text = raw_text
        dropped_reason = ""
        lang = getattr(info, "language", "") or ""
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        if not text:
            dropped_reason = "empty"
        elif max_nsp > NO_SPEECH_PROB_DROP:
            dropped_reason = f"no_speech_prob={max_nsp:.2f}"
            text = ""
        elif _is_hallucination_phrase(raw_text):
            dropped_reason = f"hallucination={raw_text!r}"
            text = ""
        elif lang == "en" and lang_prob > 0.6 and _looks_like_en_filler(raw_text):
            # User-profile heuristic: this app's primary speakers address Eri
            # in Chinese. A short English fragment with confident "en" detect
            # is almost always Whisper's YouTube-tail leakage on a breath /
            # cough / click, not an actual reply. lang_prob>0.6 catches the
            # cases where Whisper is sure it's English (real Chinese leak-
            # through into "en" usually scores lower).
            dropped_reason = f"en-filler={raw_text!r} (lang_prob={lang_prob:.2f})"
            text = ""

        return STTResult(
            text=text,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            inference_ms=dt,
            dropped_reason=dropped_reason,
            raw_text=raw_text,
            no_speech_prob=max_nsp,
        )
