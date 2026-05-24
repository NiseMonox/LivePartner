"""Compare image-encoding strategies for VLM upload:

For each strategy, encode a test image (use test_capture.png or fall back to a
generated frame), send to Qwen Omni Plus via the same call decision.py uses,
and report file size + LLM latency + transcript.

Goal: find smallest encoding that the VLM still reads accurately.
"""
from __future__ import annotations
import base64, io, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from livepartner.llm import chat
from livepartner.config import settings


def pick_image() -> np.ndarray:
    for cand in ["test_capture.png", "test_clone.wav"]:  # fallback ignored if not image
        p = Path(cand)
        if p.exists() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            img = cv2.imread(str(p))
            if img is not None:
                return img
    # Synthesize a small test scene with text + colors
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (30, 60, 120)
    cv2.rectangle(img, (100, 100), (1180, 620), (200, 200, 50), 4)
    cv2.putText(img, "HEALTH 247/300", (200, 250), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)
    cv2.putText(img, "BOSS: SHADOW LORD", (200, 400), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (50, 50, 255), 3)
    return img


def scaled(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def call_vlm(image_bytes: bytes, mime: str, label: str) -> tuple[int, float, str]:
    b64 = base64.b64encode(image_bytes).decode()
    t0 = time.perf_counter()
    out = chat(
        settings.model_pro,
        prompt="简短描述这张游戏画面里看到了什么(HUD 数字、颜色、文字都尽量提)。控制在 30 字以内。",
        image_b64=b64,
        mime=mime,
        max_tokens=120,
    )
    dt = (time.perf_counter() - t0) * 1000
    return len(image_bytes), dt, out.strip()


img = pick_image()
print(f"source: {img.shape}", flush=True)

variants = [
    ("PNG 1024",   1024, ".png", []),
    ("JPEG q=80 1024", 1024, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 80]),
    ("JPEG q=70 768",   768, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 70]),
    ("JPEG q=60 768",   768, ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), 60]),
    ("WebP q=80 1024", 1024, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 80]),
    ("WebP q=75 768",   768, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 75]),
    ("WebP q=65 768",   768, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 65]),
    ("WebP q=70 512",   512, ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), 70]),
]
mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}

for label, side, ext, opts in variants:
    small = scaled(img, side)
    ok, buf = cv2.imencode(ext, small, opts)
    if not ok:
        print(f"  [{label:22s}] encode FAILED")
        continue
    image_bytes = buf.tobytes()
    try:
        size, dt, text = call_vlm(image_bytes, mime_map[ext], label)
        print(f"  [{label:22s}] {size//1024:>4d} KB  {dt:>5.0f} ms  → {text}")
    except Exception as e:
        print(f"  [{label:22s}] {len(image_bytes)//1024} KB  ERROR: {type(e).__name__}: {e}")
