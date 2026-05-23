from __future__ import annotations

import base64
from functools import cache
from pathlib import Path
from typing import Any

from openai import OpenAI

from .config import settings


@cache
def client() -> OpenAI:
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def chat(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    image_b64: str | None = None,
    mime: str = "image/png",
    max_tokens: int = 300,
    extra_body: dict[str, Any] | None = None,
) -> str:
    """One call surface for everything: text-only or vision, flash or pro.

    extra_body lets callers pass DashScope-specific params (e.g. enable_thinking=False).
    """
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})

    if image_b64 is None:
        user_content: Any = prompt
    else:
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]
    messages.append({"role": "user", "content": user_content})

    extra: dict[str, Any] = {}
    if extra_body:
        extra["extra_body"] = extra_body

    resp = client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        **extra,
    )
    return resp.choices[0].message.content or ""


def flash_text(prompt: str, **kw: Any) -> str:
    return chat(settings.model_flash, prompt, **kw)


def flash_vision(prompt: str, image_b64: str, **kw: Any) -> str:
    return chat(settings.model_flash, prompt, image_b64=image_b64, **kw)


def pro_text(prompt: str, **kw: Any) -> str:
    return chat(settings.model_pro, prompt, **kw)


def pro_vision(prompt: str, image_b64: str, **kw: Any) -> str:
    return chat(settings.model_pro, prompt, image_b64=image_b64, **kw)


def encode_image(path: str | Path) -> tuple[str, str]:
    """Read an image file → (base64, mime). Detects PNG/JPEG/WEBP/GIF by extension."""
    p = Path(path)
    ext = p.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }.get(ext, "application/octet-stream")
    return base64.b64encode(p.read_bytes()).decode("ascii"), mime
