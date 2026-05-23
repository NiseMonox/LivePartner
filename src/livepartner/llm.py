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


def _content_with_image(text: str, image_b64: str, mime: str) -> list[dict[str, Any]]:
    return [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        {"type": "text", "text": text},
    ]


def flash_text(prompt: str, *, system: str | None = None, max_tokens: int = 200) -> str:
    """Cheap, fast text call — for gate / summary / compression."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = client().chat.completions.create(
        model=settings.model_flash,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def pro_vision(
    prompt: str,
    image_b64: str,
    *,
    system: str | None = None,
    max_tokens: int = 300,
    mime: str = "image/png",
) -> str:
    """Vision call — game frame + text prompt → free-form response."""
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": _content_with_image(prompt, image_b64, mime)})
    resp = client().chat.completions.create(
        model=settings.model_pro,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def encode_image(path: str | Path) -> tuple[str, str]:
    """Read an image file → (base64, mime). Detects PNG/JPEG by extension."""
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
