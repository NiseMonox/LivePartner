"""Probe a list of candidate Qwen models for actual vision support.

Sends the same WebP test image + 'describe' prompt to each. Records: HTTP
status, latency, content, and whether it claims "can't see image"-style
phrases. Useful to verify before swapping LP_MODEL_PRO/FLASH.
"""
from __future__ import annotations
import base64, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import httpx
from livepartner.config import settings


def load_sample() -> tuple[str, str]:
    """Synthesize a small game-like frame and encode as WebP."""
    img = np.zeros((512, 768, 3), dtype=np.uint8)
    img[:, :] = (30, 60, 120)  # dark blue background
    cv2.rectangle(img, (60, 60), (708, 452), (200, 200, 50), 4)
    cv2.putText(img, "HEALTH 247/300", (120, 180), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 255, 255), 3)
    cv2.putText(img, "MANA 88/120", (120, 260), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (255, 200, 200), 3)
    cv2.putText(img, "BOSS: SHADOW LORD", (120, 360), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (50, 50, 255), 3)
    ok, buf = cv2.imencode(".webp", img, [int(cv2.IMWRITE_WEBP_QUALITY), 75])
    assert ok
    return base64.b64encode(buf.tobytes()).decode(), "image/webp"


B64, MIME = load_sample()
print(f"sample image: {len(B64)*3//4//1024} KB {MIME}", flush=True)


def hit(model: str) -> None:
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{MIME};base64,{B64}"}},
                {"type": "text", "text": "看到画面里有什么数字或文字？简要列出,不超过30字。"},
            ],
        }],
        "max_tokens": 80,
        "enable_thinking": False,  # turn off chain-of-thought reasoning if model supports it
    }
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{settings.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}",
                     "Content-Type": "application/json"},
            json=body, timeout=60,
        )
        dt = (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"  [{model:30s}] EXC  {type(e).__name__}: {e}")
        return
    if r.status_code != 200:
        print(f"  [{model:30s}] HTTP {r.status_code}  {dt:.0f}ms  {r.text[:140]}")
        return
    j = r.json()
    txt = (j["choices"][0]["message"].get("content") or "").strip()
    can_see = any(s in txt for s in ("HEALTH", "247", "300", "MANA", "SHADOW", "88", "120", "BOSS"))
    marker = "✓ READS IMAGE" if can_see else "✗ ignored / didn't read"
    print(f"  [{model:30s}] HTTP 200  {dt:>5.0f}ms  {marker}")
    print(f"     → {txt[:160]}")


import sys as _sys
candidates = _sys.argv[1:] or [
    "qwen3.5-omni-plus",       # current PRO
    "qwen3.5-omni-flash",      # current FLASH
    "qwen3.5-plus",            # user's question
    "qwen3.6-plus",            # mentioned in user's docs
    "qwen3.6-flash",           # user's docs
    "qwen3.6-max-preview",     # in earlier model list
    "qwen3.7-max",             # in earlier model list
    "qwen3-vl-plus",           # explicit vl
    "qwen3-vl-flash",          # explicit vl
]
# Two calls each — second call shows warm latency.
for m in candidates:
    print(f"\n=== {m} ===", flush=True)
    hit(m)
    hit(m)
