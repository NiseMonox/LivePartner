"""Live2D Director — picks the avatar's expression / gesture per line.

This is a **separate flash LLM call** that runs in parallel with TTS once the
main decision LLM has produced its LINE + EMO. The director sees:

- The line Eri is about to say
- Her EMO prosody hint
- A small window of recent context (player / AI / event entries)

and returns a ``DirectorIntent`` — an expression name (from a palette the UI
filters down at VTS-model-load time) plus an optional hotkey trigger and
intensity. The intent goes to ``VTSController`` to drive the avatar.

Architecture rationale:
- Decoupling: the main persona prompt stays focused on dialog quality. Adding
  every model's hotkey IDs to the persona system prompt would blow it up.
- Parallel: director runs after ``text_ready`` (LLM text out, before TTS) and
  takes ~300-500 ms — under TTS TTFB so expression usually beats the first audio.
- Failure-safe: any HTTP / JSON / palette mismatch falls back to a Chinese
  keyword dictionary on EMO. The avatar never freezes on a missing director.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable

from .llm import flash_text
from .persona import Persona


@dataclass(frozen=True)
class DirectorIntent:
    expr: str           # must be ∈ palette passed in; "" if no palette/match
    intensity: float    # 0..1, used by intensity-aware drivers (idle decay timer reads it)
    trigger: str        # optional hotkey id (animation / movement); "" = none
    # Direct facial parameter offsets — each in [-1.0, 1.0]. Driven via
    # set_parameters_bulk after the expression sticker is set, so the avatar
    # gets both decorative overlay (sticker) AND actual face change at once.
    # Default 0 = neutral. Type-H3 maps:
    #   brow_l → BrowLeftY      (-1 = pressed down, +1 = raised)
    #   brow_r → BrowRightY     (same; mismatch L/R for "one eyebrow up")
    #   smile  → MouthSmile     (-1 = down-curved, +1 = smile up)
    #                            (also drives Eye Form + Mouth Form on this rig)
    brow_l: float = 0.0
    brow_r: float = 0.0
    smile: float = 0.0

    @classmethod
    def neutral(cls, palette: Iterable[str] = ()) -> "DirectorIntent":
        """A safe fallback intent — first palette entry if any (usually 'neutral'),
        or empty string. Always 0.5 intensity, no trigger, neutral face."""
        items = list(palette)
        return cls(
            expr=items[0] if items else "",
            intensity=0.5, trigger="",
            brow_l=0.0, brow_r=0.0, smile=0.0,
        )


# Keyword fallback table — matches the AI's LINE text directly (since EMO
# is no longer in the pipeline). First matching pattern wins; we apply the
# associated brow/smile offsets and pick the first expression candidate that
# exists in the model's palette.
# brow_l != brow_r produces lopsided "smug/quizzical" looks.
_LINE_KEYWORD_HINTS: list[tuple[str, list[str], float, float, float]] = [
    # pattern,                              expr_candidates,                              brow_l, brow_r, smile
    (r"哈哈|笑|开心|爽|好玩|好笑|有趣",      ["smile_big", "smile", "happy", "smug"],       0.6,    0.6,    0.7),
    (r"笨蛋|垃圾|拜托|服了|无语|算了|呵",    ["smug", "smug_smirk", "smirk", "pout"],       0.5,   -0.3,    0.3),
    (r"什么|啊\?|等等|怎么|真的假的|不会吧", ["wide_eyes", "surprised", "shock"],            0.8,    0.8,    0.0),
    (r"叹气|累了|没劲|无聊|懒得|可怜|失望",  ["facepalm", "averted_eyes", "sad", "neutral"],-0.4,   -0.4,   -0.3),
    (r"哼|才不|害羞|不是|别看|羞",           ["averted_eyes", "embarrassed", "shy"],        -0.3,   -0.3,    0.2),
    (r"生气|气|烦|讨厌|可恶|愤怒",           ["pout", "angry", "annoyed"],                  -0.6,   -0.6,   -0.4),
    (r"温柔|没事|加油|你能行|别担心",        ["smile", "smile_small", "neutral"],            0.2,    0.2,    0.4),
    (r"快|赶紧|动作快|看着我",               ["pout", "smug", "neutral"],                   -0.3,   -0.3,   -0.1),
]


_DIRECTOR_SYSTEM = (
    "你是 Live 2D VTuber 角色的表情导演。\n"
    "看一句 Eri 即将说的台词和最近的对话上下文,挑表情贴图 + 驱动面部参数。\n\n"
    "**输出严格 JSON 一行**,不要 markdown,不要解释:\n"
    '{{"expr": "<表情名>", "intensity": <0-1>, "trigger": "<可选>",'
    ' "brow_l": <-1..1>, "brow_r": <-1..1>, "smile": <-1..1>}}\n\n'
    "## 表情 (expr) — 装饰性贴图叠加\n"
    "必须精确从下面挑一个,不要发明新名:\n"
    "{expr_list}\n\n"
    "## 动作 trigger — 可选 hotkey\n"
    "{hotkey_list}\n"
    "留空字符串就是不触发额外动作。\n\n"
    "## 强度 intensity\n"
    "平淡 0.2-0.4 / 一般 0.5-0.7 / 强烈 0.8-1.0,默认 0.5\n\n"
    "## 面部参数 — 直接驱动脸部肌肉 (最重要,效果最明显!)\n"
    "都是 -1.0 ~ +1.0,默认 0.0 = 中性\n"
    "- **brow_l / brow_r**: 左右眉高度。-1 = 皱眉/压低(生气/担忧),+1 = 上扬(惊讶/开心)\n"
    "  - 不对称(brow_l 高 brow_r 低)= '挑单边眉' 嫌弃/疑惑/嘲讽\n"
    "  - 双眉同时高 = 惊讶 / 兴奋 / 开心\n"
    "  - 双眉同时低 = 害羞 / 严肃 / 难过\n"
    "- **smile**: 嘴角(同时影响眼形和眉形,因为 rig 是绑在 MouthSmile 上)\n"
    "  - -1 = 大下撇(嫌弃/沮丧),0 = 平嘴,+1 = 大笑\n\n"
    "## 几个常见组合示范\n"
    "- 开心:    brow_l=0.6 brow_r=0.6 smile=0.7\n"
    "- 嫌弃:    brow_l=0.5 brow_r=-0.3 smile=0.3  (单边挑眉的轻蔑)\n"
    "- 惊讶:    brow_l=0.8 brow_r=0.8 smile=0\n"
    "- 无奈:    brow_l=-0.4 brow_r=-0.4 smile=-0.3\n"
    "- 害羞:    brow_l=-0.3 brow_r=-0.3 smile=0.2\n"
    "- 生气:    brow_l=-0.6 brow_r=-0.6 smile=-0.4\n"
    "- 平淡:    全 0\n"
)


def decide_intent(
    line: str,
    context: list[str],
    expr_palette: list[str],
    hotkey_palette: list[str],
    persona: Persona,
) -> DirectorIntent:
    """Call flash, return the parsed DirectorIntent.

    Picks expression + face params from the AI line + recent context.
    EMO is no longer in the pipeline; the director infers tone from the
    line text alone. Falls back to ``fallback_from_line(line, expr_palette)``
    on any HTTP / JSON / palette error. Never raises.
    """
    if not expr_palette:
        return DirectorIntent(expr="", intensity=0.5, trigger="")

    expr_list = ", ".join(expr_palette)
    hotkey_list = ", ".join(hotkey_palette) if hotkey_palette else "(无)"
    system = _DIRECTOR_SYSTEM.format(expr_list=expr_list, hotkey_list=hotkey_list)
    ctx_text = "\n".join(context) if context else "(无)"
    user = f"台词: {line}\n最近:\n{ctx_text}"

    try:
        raw = flash_text(
            prompt=user,
            system=system,
            max_tokens=160,
            extra_body={"enable_thinking": False},
            temperature=0.6,
        )
    except Exception:
        return fallback_from_line(line, expr_palette)

    # Tolerant JSON extraction: model may wrap in markdown fences or stray prose.
    m = re.search(r"\{[^{}]*\}", raw, flags=re.DOTALL)
    if not m:
        return fallback_from_line(line, expr_palette)
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback_from_line(line, expr_palette)

    expr = str(obj.get("expr", "")).strip()
    if expr not in expr_palette:
        return fallback_from_line(line, expr_palette)
    try:
        intensity = float(obj.get("intensity", 0.5))
    except (TypeError, ValueError):
        intensity = 0.5
    intensity = max(0.0, min(1.0, intensity))
    trigger = str(obj.get("trigger", "")).strip()
    if trigger and trigger not in hotkey_palette:
        trigger = ""  # silently drop bogus trigger; expr still fine

    def _clamp_face(v) -> float:
        try:
            return max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0

    return DirectorIntent(
        expr=expr,
        intensity=intensity,
        trigger=trigger,
        brow_l=_clamp_face(obj.get("brow_l", 0.0)),
        brow_r=_clamp_face(obj.get("brow_r", 0.0)),
        smile=_clamp_face(obj.get("smile", 0.0)),
    )


def fallback_from_line(line: str, expr_palette: list[str]) -> DirectorIntent:
    """Keyword-driven expression pick from the AI line — used when the director
    LLM fails or its output is unusable.

    Walks ``_LINE_KEYWORD_HINTS`` top to bottom; the first pattern that matches
    the line text nominates a list of semantic names + face param offsets. We
    return the first expr that's actually in the palette. If nothing matches,
    returns ``DirectorIntent.neutral(palette)`` (first palette entry).
    """
    if not expr_palette:
        return DirectorIntent(expr="", intensity=0.5, trigger="")
    palette_set = set(expr_palette)
    if line:
        for pattern, candidates, brow_l, brow_r, smile in _LINE_KEYWORD_HINTS:
            if re.search(pattern, line):
                for c in candidates:
                    if c in palette_set:
                        return DirectorIntent(
                            expr=c, intensity=0.5, trigger="",
                            brow_l=brow_l, brow_r=brow_r, smile=smile,
                        )
                # Pattern matched but palette has no expr — at least apply
                # face params so the avatar reacts visually even without sticker.
                return DirectorIntent(
                    expr=expr_palette[0], intensity=0.5, trigger="",
                    brow_l=brow_l, brow_r=brow_r, smile=smile,
                )
    return DirectorIntent.neutral(expr_palette)
