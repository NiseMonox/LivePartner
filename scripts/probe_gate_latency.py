"""Find the fastest way to make Qwen Omni Flash answer yes/no on a thumbnail."""
from __future__ import annotations
import base64, io, sys, time, zlib, struct
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import httpx
from livepartner.config import settings


def make_thumb_b64() -> str:
    w = h = 256
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes((40, 6, 6)) if (x + y) % 20 > 10 else bytes((180, 20, 20))
    def chunk(tag, data):
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    idat = zlib.compress(bytes(raw))
    png = sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode()


B64 = make_thumb_b64()
PROMPT_TEXT = "事件：玩家死亡。AI 是否该说话？只回答 yes 或 no。"

variants = [
    ("default",                       {}),
    ("enable_thinking=false",         {"enable_thinking": False}),
    ("extra_body.enable_thinking=F",  {"extra_body": {"enable_thinking": False}}),
    ("temperature=0",                 {"temperature": 0}),
    ("temperature=0 + max=5",         {"temperature": 0, "max_tokens_override": 5}),
]


def call(label, extra, model):
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{B64}"}},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }],
        "max_tokens": extra.pop("max_tokens_override", 20),
        **{k: v for k, v in extra.items() if not k.startswith("extra_")},
    }
    if "extra_body" in extra:
        body.update(extra["extra_body"])

    times = []
    last_content = ""
    last_status = 0
    for _ in range(2):
        t0 = time.perf_counter()
        r = httpx.post(
            f"{settings.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
            json=body, timeout=60,
        )
        dt = (time.perf_counter() - t0) * 1000
        times.append(dt)
        last_status = r.status_code
        if r.status_code == 200:
            last_content = (r.json()["choices"][0]["message"].get("content") or "").strip()[:40]
        else:
            last_content = r.text[:100]
    avg = sum(times) / len(times)
    print(f"  [{label:30s}]  HTTP {last_status}  avg {avg:>5.0f}ms  ({times[0]:.0f},{times[1]:.0f})  {last_content!r}")


for model in [settings.model_flash]:
    print(f"\n=== model = {model}")
    for name, extra in variants:
        call(name, dict(extra), model)

# Bonus: try lighter non-omni text-only models if you wanted text-only gate
print(f"\n=== text-only candidate models (gate without image, using event description only)")
text_only_body = {
    "messages": [{"role": "user", "content": PROMPT_TEXT}],
    "max_tokens": 5,
}
for m in ["qwen-flash", "qwen-turbo", "qwen-plus", "qwen3.6-flash"]:
    body = dict(text_only_body); body["model"] = m
    t0 = time.perf_counter()
    try:
        r = httpx.post(
            f"{settings.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"},
            json=body, timeout=30,
        )
        dt = (time.perf_counter() - t0) * 1000
        if r.status_code == 200:
            c = (r.json()["choices"][0]["message"].get("content") or "").strip()[:40]
            print(f"  [{m:25s}]  HTTP 200  {dt:>5.0f}ms  {c!r}")
        else:
            print(f"  [{m:25s}]  HTTP {r.status_code}  {r.text[:80]}")
    except Exception as e:
        print(f"  [{m:25s}]  ERR  {e!r}")
