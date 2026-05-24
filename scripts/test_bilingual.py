"""Quick check that the bilingual JA/ZH split actually works against Qwen Omni Plus."""
import base64, io, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw, ImageFont
from livepartner.decision import generate
from livepartner.persona import load_persona


def synth() -> str:
    img = Image.new("RGB", (1280, 720), (40, 6, 6))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 120)
    except Exception:
        f = ImageFont.load_default()
    d.text((420, 300), "YOU DIED", fill=(180, 20, 20), font=f)
    buf = io.BytesIO(); img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


b64 = synth()
for pid in ["snark", "praise"]:
    p = load_persona(pid)
    r = generate(
        "玩家在 boss 战中第三次死亡。画面切换为 YOU DIED 红屏。",
        b64, p,
        bilingual=True, tts_language="日语", subtitle_language="中文",
    )
    print(f"\n=== {pid} ===")
    print(f"  subtitle (zh): {r.text!r}")
    print(f"  tts (ja)     : {r.tts_text!r}")
    if r.text == r.tts_text:
        print("  [WARN] zh == ja — parse fallback hit. Raw:")
        print(f"  {r.raw!r}")
