"""Edge TTS wrapper — text → mp3 bytes via Microsoft Edge's free TTS.

For M1 we save mp3. PCM conversion + Mumble streaming will land with the bot integration.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import edge_tts

from .persona import Persona, VoiceConfig


@dataclass(frozen=True)
class TtsResult:
    mp3: bytes
    voice_id: str
    rate: str
    char_count: int


def _speed_to_rate(speed: float) -> str:
    pct = round((speed - 1.0) * 100)
    sign = "+" if pct >= 0 else "-"
    return f"{sign}{abs(pct)}%"


async def synthesize_mp3(
    text: str,
    *,
    voice_id: str = "zh-CN-XiaoyiNeural",
    rate: str = "+0%",
) -> bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice_id, rate=rate)
    chunks: list[bytes] = []
    async for ev in communicate.stream():
        if ev["type"] == "audio":
            chunks.append(ev["data"])
    return b"".join(chunks)


def synthesize_sync(
    text: str,
    *,
    voice: VoiceConfig | None = None,
) -> TtsResult:
    voice = voice or VoiceConfig()
    rate = _speed_to_rate(voice.speed)
    mp3 = asyncio.run(synthesize_mp3(text, voice_id=voice.voice_id, rate=rate))
    return TtsResult(mp3=mp3, voice_id=voice.voice_id, rate=rate, char_count=len(text))


def synthesize_for_persona(text: str, persona: Persona) -> TtsResult:
    return synthesize_sync(text, voice=persona.voice)
