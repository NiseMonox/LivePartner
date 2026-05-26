"""Visual self-awareness — periodic flash-vision describe of Eri's VTS window.

Every ~90 s we grab whatever VTube Studio is rendering (model + accessories +
background), send the image to flash vision asking for a short Chinese
description ("帽子、衣服、配饰、姿势"), and store the result. If the new
description differs from the last one, we mark a "你的形象刚刚变了" event
that gets injected into memory so Eri's next natural turn (ambient/STT)
notices and comments.

Why this is different from the scene describer:
- Scene describer looks at the **game** frame from the capture card, every 5 s.
- Self describer looks at the **VTS** window screenshot, every 90 s.
- Different targets, different cadences, different memory sections.

Why passive (vs force-triggering an AI turn on detection):
- Visual changes in VTS happen alongside game state changes; force-triggering
  could break game pacing. The next ambient tick or player utterance will
  naturally pick up the change from memory.
"""
from __future__ import annotations

import re
from io import BytesIO

from PIL import Image

from .llm import flash_vision


_SELF_DESCRIBE_SYSTEM = (
    "你是一个虚拟主播角色外观描述器。\n"
    "看一张虚拟主播角色的画面截图,用**一句中文**(不超过 40 字)描述这个角色当前外观特征。\n\n"
    "重点关注:\n"
    "- 发型/发色 (短发/长发/马尾/刘海长短/颜色)\n"
    "- 衣服款式 (制服/睡衣/外套/连衣裙/颜色)\n"
    "- 配饰 (耳朵/角/光环/眼镜/帽子/项链)\n"
    "- 尾巴 / 翅膀 / 特殊部件\n"
    "- 明显的姿势/状态\n\n"
    "忽略:\n"
    "- 背景颜色 / 桌面 / 周围的非角色元素 (我们只关心角色本身)\n"
    "- 微表情变化 / 嘴型 / 眨眼 (这些是动态的,跟外观无关)\n\n"
    "格式: 一句话,具体特征用顿号或逗号分隔。例:\n"
    "  '银白短发,蓝色 JK 制服,猫耳朵,白色蓬松尾巴,戴黑框眼镜'\n"
    "  '粉色双马尾,白色连衣裙,头顶光环,无配饰'\n\n"
    "**只输出那一句**,不要前缀、解释、markdown。"
)


def describe_self(image: Image.Image, *, max_tokens: int = 120) -> str:
    """One flash-vision call: image → short appearance description."""
    # Encode as JPEG ~70 quality to keep token cost down for periodic polls.
    buf = BytesIO()
    image_rgb = image.convert("RGB") if image.mode != "RGB" else image
    image_rgb.save(buf, format="JPEG", quality=70)
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    try:
        out = flash_vision(
            prompt="描述这个虚拟主播角色的外观",
            image_b64=b64,
            system=_SELF_DESCRIBE_SYSTEM,
            mime="image/jpeg",
            max_tokens=max_tokens,
            extra_body={"enable_thinking": False},
        )
    except Exception:
        return ""
    line = out.strip().strip("`'\"")
    if not line:
        return ""
    return line.splitlines()[0].strip()


# Same scene-similarity helpers, but tuned for self-description text — these
# tend to share many common Chinese chars (颜色/发型/尾巴) so the bigram
# threshold is set permissive to catch *real* outfit changes but not false-
# positive on synonyms / paraphrases.
_SELF_FILLERS = set("的了呢吗嘛啊哦呀嗯吧呐, 、,。!?，！？「」")


def _normalize_self(s: str) -> str:
    return "".join(c for c in s if c not in _SELF_FILLERS)


def self_description_changed(prev: str, new: str) -> bool:
    """True if the new description is substantively different from the previous.

    Threshold tuned for short Chinese (~30 chars) appearance lines: we look at
    char-set Jaccard — if even a few content words differ (e.g. 'red shirt'
    became 'blue shirt') the overlap drops past the cutoff. Below ~0.7 = real
    change. Returns False for first-time call (prev empty).
    """
    if not prev:
        return False  # baseline — not a change
    if not new:
        return False  # describer failed; keep previous
    a = _normalize_self(prev.strip())
    b = _normalize_self(new.strip())
    if not a or not b:
        return False
    if a == b:
        return False
    A, B = set(a), set(b)
    if not A or not B:
        return False
    jaccard = len(A & B) / len(A | B)
    # < 0.70 means roughly >30% of characters differ between descriptions.
    # In practice this triggers on outfit/hair/accessory changes but stays
    # silent on rephrasings (which usually have >0.85 overlap).
    return jaccard < 0.70
