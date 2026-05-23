"""End-to-end vision decision demo.

Reproduces SPEC §4.3 scenario without hardware:
1. Synthesize a dark-red "YOU DIED" game-over frame.
2. Run through flash gate (thumbnail) → pro vision generate (full frame).
3. Print AI's in-character response (snark persona by default).

Usage:
    python scripts/demo_decision.py
    python scripts/demo_decision.py --persona snark --image path/to/frame.png
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont

from livepartner.decision import decide, gate, generate
from livepartner.llm import encode_image
from livepartner.persona import load_persona
from livepartner.tts import synthesize_for_persona


def synth_you_died_frame(size: tuple[int, int] = (1280, 720)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), color=(40, 6, 6))  # dark red
    draw = ImageDraw.Draw(img)
    # Vignette-ish radial darkness
    for i in range(0, min(w, h) // 2, 6):
        a = max(0, 60 - i // 3)
        draw.rectangle([i, i, w - i, h - i], outline=(20 - a // 4, 0, 0))
    # Main text — try a real font, fall back to default bitmap font
    text = "YOU DIED"
    font = None
    for cand in [
        "C:/Windows/Fonts/seguibl.ttf",   # Segoe UI Black
        "C:/Windows/Fonts/arialbd.ttf",   # Arial Bold
        "C:/Windows/Fonts/Arial.ttf",
    ]:
        if Path(cand).exists():
            try:
                font = ImageFont.truetype(cand, size=120)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((w - tw) // 2, (h - th) // 2 - 20), text, fill=(180, 20, 20), font=font)
    # Small sub-line
    sub = "Press F to pay respects"
    try:
        sfont = ImageFont.truetype("C:/Windows/Fonts/Arial.ttf", size=28)
    except Exception:
        sfont = ImageFont.load_default()
    sbbox = draw.textbbox((0, 0), sub, font=sfont)
    sw = sbbox[2] - sbbox[0]
    draw.text(((w - sw) // 2, (h + th) // 2 + 40), sub, fill=(120, 90, 90), font=sfont)
    return img


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def make_thumbnail_b64(img: Image.Image, max_side: int = 256) -> str:
    thumb = img.copy()
    thumb.thumbnail((max_side, max_side))
    return pil_to_b64(thumb)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", default="snark")
    ap.add_argument("--image", type=Path, help="path to a game frame (PNG/JPG); default = synthesized YOU DIED")
    ap.add_argument("--event", default="玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。BGM 转为低沉死亡音乐。")
    ap.add_argument("--tts-out", type=Path, default=Path("demo_tts.mp3"),
                    help="where to write TTS mp3; pass empty string to skip TTS")
    ap.add_argument("--force-speak", action="store_true",
                    help="skip the LLM gate (use for high-priority events like death)")
    args = ap.parse_args()

    persona = load_persona(args.persona)
    print(f"persona: {persona.display_name} ({persona.id})")
    print(f"event  : {args.event}")
    print()

    if args.image:
        img = Image.open(args.image).convert("RGB")
        print(f"frame  : loaded {args.image} ({img.size[0]}x{img.size[1]})")
    else:
        img = synth_you_died_frame()
        print(f"frame  : synthesized YOU DIED ({img.size[0]}x{img.size[1]})")

    frame_b64 = pil_to_b64(img)
    thumb_b64 = make_thumbnail_b64(img)
    print(f"frame_b64 size: {len(frame_b64) // 1024} KB,  thumb_b64 size: {len(thumb_b64) // 1024} KB")
    print()

    # Stage 1: gate (skip if force_speak)
    dt_gate = 0.0
    if args.force_speak:
        print("[gate]      skipped (force_speak)")
        speak = True
    else:
        t0 = time.perf_counter()
        speak = gate(args.event, thumb_b64, persona)
        dt_gate = (time.perf_counter() - t0) * 1000
        print(f"[gate]      speak={speak}  ({dt_gate:.0f} ms)")

    if not speak:
        print("\nAI 选择沉默。")
        return 0

    # Stage 2
    t0 = time.perf_counter()
    reply = generate(args.event, frame_b64, persona)
    dt_gen = (time.perf_counter() - t0) * 1000
    print(f"[generate]  {dt_gen:.0f} ms")
    print()
    print(f"  AI ({persona.display_name}) > {reply.text}")
    print()
    print(f"total LLM latency: {dt_gate + dt_gen:.0f} ms")

    # Stage 3: TTS
    if str(args.tts_out):
        t0 = time.perf_counter()
        tts = synthesize_for_persona(reply.text, persona)
        dt_tts = (time.perf_counter() - t0) * 1000
        args.tts_out.write_bytes(tts.mp3)
        print()
        print(f"[tts]       {dt_tts:.0f} ms  voice={tts.voice_id}  rate={tts.rate}  chars={tts.char_count}")
        print(f"  saved {len(tts.mp3) // 1024} KB mp3 to {args.tts_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
