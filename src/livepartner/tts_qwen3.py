"""HTTP client for the local Qwen3-TTS streaming server (services/qwen3_tts_server.py).

The server returns 48 kHz mono int16 PCM streamed as application/octet-stream.
Two call surfaces:
  - stream_pcm(text, persona) → iterator of bytes; pipe each chunk straight into Mumble
  - synthesize_pcm(text, persona) → full bytes (collected); useful for save-to-wav / testing
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Optional

import httpx

from .persona import Persona


DEFAULT_BASE_URL = "http://127.0.0.1:7001"
PCM_SAMPLE_RATE = 48_000  # what the server outputs
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * 2  # int16 mono


@dataclass(frozen=True)
class Qwen3TtsConfig:
    base_url: str = DEFAULT_BASE_URL
    chunk_size: int = 4          # codec steps per server-emitted audio chunk (~83 ms each)
    temperature: float = 0.9
    top_k: int = 50
    language: str = "Chinese"
    timeout: float = 60.0


def is_alive(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 1.5) -> bool:
    try:
        r = httpx.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200 and r.json().get("ok") is True
    except Exception:
        return False


def list_voices(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 2.0) -> list[str]:
    r = httpx.get(f"{base_url}/voices", timeout=timeout)
    r.raise_for_status()
    return list(r.json().get("voices") or [])


def voice_id_for(persona: Persona) -> str:
    """How a persona maps to a server-side voice.

    The .pt filename stem in personas/voices/ is keyed by persona.id by default.
    Edge TTS's voice_id field (e.g. "zh-CN-XiaoyiNeural") is engine-specific and not
    a valid Qwen3 voice id, so we ignore it here.
    """
    return persona.id


def stream_pcm(
    text: str,
    persona: Persona,
    *,
    cfg: Optional[Qwen3TtsConfig] = None,
) -> Iterator[bytes]:
    """POST /tts and yield 48kHz mono int16 PCM bytes as they arrive."""
    cfg = cfg or Qwen3TtsConfig()
    body = {
        "persona_id": voice_id_for(persona),
        "text": text,
        "language": cfg.language,
        "chunk_size": cfg.chunk_size,
        "temperature": cfg.temperature,
        "top_k": cfg.top_k,
    }
    with httpx.stream("POST", f"{cfg.base_url}/tts", json=body, timeout=cfg.timeout) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes(chunk_size=8192):
            if chunk:
                yield chunk


def synthesize_pcm(
    text: str,
    persona: Persona,
    *,
    cfg: Optional[Qwen3TtsConfig] = None,
) -> bytes:
    """Collect the whole stream into one bytes blob (useful for save-to-wav, tests)."""
    return b"".join(stream_pcm(text, persona, cfg=cfg))


def pcm_duration_seconds(pcm: bytes) -> float:
    return len(pcm) / PCM_BYTES_PER_SECOND
