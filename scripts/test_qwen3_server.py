"""POST /tts to the local Qwen3 TTS server, save streamed PCM to a wav for QA."""
from __future__ import annotations
import argparse, struct, sys, time
from pathlib import Path

import httpx


def write_wav(path: Path, pcm_bytes: bytes, sr: int = 48000) -> None:
    """Write a minimal mono int16 WAV from raw PCM bytes."""
    n_frames = len(pcm_bytes) // 2
    byte_rate = sr * 2
    block_align = 2
    header = b"RIFF"
    header += struct.pack("<I", 36 + len(pcm_bytes))
    header += b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, sr, byte_rate, block_align, 16)
    header += b"data"
    header += struct.pack("<I", len(pcm_bytes))
    path.write_bytes(header + pcm_bytes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7001/tts")
    ap.add_argument("--persona", default="snark")
    ap.add_argument("--text", default="第三次寄了？F 键都被你按包浆了吧。")
    ap.add_argument("--language", default="Chinese",
                    help="TTS language code (Chinese/Japanese/English/...) or Auto")
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("test_qwen3.wav"))
    args = ap.parse_args()

    body = {
        "persona_id": args.persona,
        "text": args.text,
        "language": args.language,
        "chunk_size": args.chunk_size,
    }
    print(f"POST {args.url}  body={body}")

    pcm_chunks: list[bytes] = []
    t0 = time.perf_counter()
    first_ms = None
    with httpx.stream("POST", args.url, json=body, timeout=60) as r:
        r.raise_for_status()
        sr = int(r.headers.get("X-Sample-Rate", "48000"))
        print(f"HTTP {r.status_code}  X-Sample-Rate={sr}")
        for chunk in r.iter_bytes(chunk_size=8192):
            if not chunk:
                continue
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
                print(f"  TTFB = {first_ms:.0f} ms  (first {len(chunk)} bytes)")
            pcm_chunks.append(chunk)
    total_ms = (time.perf_counter() - t0) * 1000

    pcm = b"".join(pcm_chunks)
    n_samples = len(pcm) // 2
    dur_s = n_samples / sr
    print(f"  total PCM: {len(pcm)//1024} KB  ({dur_s:.2f}s @ {sr}Hz)")
    print(f"  total time: {total_ms:.0f} ms  RTF≈{dur_s/(total_ms/1000):.2f}")

    write_wav(args.out, pcm, sr=sr)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
