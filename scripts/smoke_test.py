"""End-to-end LLM smoke test against the configured provider.

Stage 1: flash text call — gate-style yes/no classification.
Stage 2: pro vision call — describe a generated image.
"""
from __future__ import annotations

import base64
import struct
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livepartner.config import settings
from livepartner.llm import flash_text, pro_vision


def make_test_png() -> bytes:
    """64x64 red field with a black diagonal — pure stdlib, no PIL."""
    w = h = 64
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            if abs(x - y) < 3:
                raw += b"\x00\x00\x00"
            else:
                raw += b"\xff\x00\x00"
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def main() -> int:
    print(f"base_url = {settings.base_url}")
    print(f"key      = {settings.api_key[:8]}...{settings.api_key[-4:]}")
    print(f"flash    = {settings.model_flash}")
    print(f"pro      = {settings.model_pro}")
    print()

    print(f"[stage 1] {settings.model_flash} — text-only gate-style call")
    t0 = time.perf_counter()
    out = flash_text(
        system='你是一个二分类器。读取事件描述，只回答"是"或"否"，不要解释。',
        prompt="事件：玩家在 boss 战中第三次死亡。AI 陪玩是否应该说话？",
        max_tokens=20,
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"  -> {out!r}  ({dt:.0f} ms)")
    print()

    print(f"[stage 2] {settings.model_pro} — vision call with a 64x64 PNG")
    png = make_test_png()
    b64 = base64.b64encode(png).decode("ascii")
    t0 = time.perf_counter()
    out = pro_vision(
        system="你是一个看图说话助手。简短描述图片内容，不超过 30 字。",
        prompt="这张图里有什么？",
        image_b64=b64,
        max_tokens=80,
    )
    dt = (time.perf_counter() - t0) * 1000
    print(f"  -> {out!r}  ({dt:.0f} ms)")

    # Heuristic pass/fail: vision must mention red OR black to be considered working.
    if not any(s in (out or "") for s in ("红", "黑", "red", "black")):
        print("\n  [WARN] vision output didn't mention red/black — model may not have seen the image")
        return 1
    print("\n[OK] flash + pro vision both functional")
    return 0


if __name__ == "__main__":
    sys.exit(main())
