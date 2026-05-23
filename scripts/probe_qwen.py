"""Probe Qwen DashScope: list models, then test vision with a real-looking image."""
from __future__ import annotations
import base64, json, os, sys, time, zlib, struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.environ["QWEN_API_KEY"]
BASE = os.environ["QWEN_BASE_URL"].rstrip("/")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def make_test_png() -> bytes:
    """64x64 PNG: red field with black diagonal — easy to describe."""
    w = h = 64
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            if abs(x - y) < 3:
                raw += b"\x00\x00\x00"
            else:
                raw += b"\xff\x00\x00"
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def list_models():
    print(f"=== GET {BASE}/models")
    r = httpx.get(f"{BASE}/models", headers=HEADERS, timeout=30)
    print(f"HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:500]); return []
    j = r.json()
    ids = [m["id"] for m in j.get("data", [])]
    print(f"{len(ids)} models")
    # show first 30 + any vision-likely
    for mid in ids[:30]:
        print(f"  {mid}")
    if len(ids) > 30:
        print(f"  ... and {len(ids) - 30} more")
    vl = [m for m in ids if "vl" in m.lower() or "vision" in m.lower() or "omni" in m.lower()]
    if vl:
        print(f"vision-likely:")
        for m in vl[:20]:
            print(f"  {m}")
    flash = [m for m in ids if "flash" in m.lower()]
    pro = [m for m in ids if "pro" in m.lower() or "plus" in m.lower() or "max" in m.lower()]
    if flash:
        print(f"flash-named:")
        for m in flash[:10]:
            print(f"  {m}")
    if pro:
        print(f"pro/plus/max-named:")
        for m in pro[:10]:
            print(f"  {m}")
    return ids


def test_text(model, prompt):
    print(f"\n=== TEXT  model={model}")
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 50}
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS, json=body, timeout=60)
    dt = (time.perf_counter() - t0) * 1000
    print(f"HTTP {r.status_code}  {dt:.0f}ms")
    if r.status_code != 200:
        print(r.text[:400]); return False
    j = r.json()
    msg = j["choices"][0]["message"]
    print(f"  content: {(msg.get('content') or '')[:200]!r}")
    return True


def test_vision(model):
    print(f"\n=== VISION  model={model}")
    png = make_test_png()
    b64 = base64.b64encode(png).decode()
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": "这张图里有什么颜色和形状?用一句话回答。"},
            ],
        }],
        "max_tokens": 100,
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{BASE}/chat/completions", headers=HEADERS, json=body, timeout=60)
    dt = (time.perf_counter() - t0) * 1000
    print(f"HTTP {r.status_code}  {dt:.0f}ms")
    if r.status_code != 200:
        print(r.text[:400]); return False
    j = r.json()
    msg = j["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    print(f"  content: {content[:300]!r}")
    # Detect "I can't see images" failure modes
    bad = any(s in content for s in ["无法", "看不到", "无法查看", "unable", "cannot"])
    if bad:
        print("  [FAIL] model reports it can't see the image — vision NOT functional")
        return False
    if "红" in content or "黑" in content or "red" in content.lower() or "black" in content.lower():
        print("  [OK] vision recognized colors")
        return True
    print("  [?] response doesn't mention colors clearly — partial")
    return True  # call succeeded; partial credit


if __name__ == "__main__":
    ids = list_models()
    flash_guess = os.environ.get("LP_MODEL_FLASH", "qwen-flash-3.5")
    pro_guess = os.environ.get("LP_MODEL_PRO", "qwen-pro-3.5")
    print(f"\nTrying configured names: flash={flash_guess}, pro={pro_guess}")
    test_text(flash_guess, "回答\"你好\"")
    test_vision(pro_guess)
