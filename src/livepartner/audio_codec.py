"""Audio codec helpers — pull anything into 48 kHz mono int16 PCM for Mumble TX."""
from __future__ import annotations

import miniaudio


PCM_SAMPLE_RATE = 48_000
PCM_BYTES_PER_SECOND = PCM_SAMPLE_RATE * 2  # int16 mono


def mp3_to_pcm48k(mp3_bytes: bytes) -> bytes:
    """Decode arbitrary MP3 → 48 kHz mono int16 PCM bytes."""
    decoded = miniaudio.decode(
        mp3_bytes,
        output_format=miniaudio.SampleFormat.SIGNED16,
        nchannels=1,
        sample_rate=PCM_SAMPLE_RATE,
    )
    return bytes(decoded.samples)


def pcm48k_duration_seconds(pcm: bytes) -> float:
    return len(pcm) / PCM_BYTES_PER_SECOND
