"""Two-stage decision pipeline per SPEC §5.3.3.

flash gate (small thumbnail + event, GENERIC system) → bool
   ↓ if yes
pro vision generate (full frame + full persona + memory + event) → AI line

Per SPEC §5.3.2 we also allow `force_speak=True` for high-priority events
(death, achievement, player voice) — these skip the LLM gate entirely.

Bilingual output: when bilingual=True (default for the AI-voice-in-Japanese,
subtitle-in-Chinese setup), the LLM emits two labelled lines. We display ZH
and feed JA to TTS.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import flash_vision, pro_vision
from .persona import Persona


GATE_SYSTEM = (
    "你是一个二分类器。看一张游戏画面缩略图 + 一句事件描述，"
    '判断 AI 陪玩此刻是否应该开口说话。只回答 "yes" 或 "no"，不要解释。'
    "若画面平淡、事件平庸，应答 no；若画面有戏剧性变化或事件值得反应，应答 yes。"
)


def gate(
    event_summary: str,
    thumbnail_b64: str,
    persona: Persona,
    *,
    mime: str = "image/png",
) -> bool:
    out = flash_vision(
        prompt=f"事件: {event_summary}\n要说话吗?",
        image_b64=thumbnail_b64,
        system=GATE_SYSTEM,
        mime=mime,
        max_tokens=10,
        extra_body={"enable_thinking": False},
    )
    answer = out.strip().lower()
    return answer.startswith("yes") or answer.startswith("是")


@dataclass(frozen=True)
class AiReply:
    text: str         # what the UI displays (subtitle language — Chinese by default)
    tts_text: str     # what gets sent to TTS (voice language — Japanese by default)
    raw: str          # full model output for debugging


_BILINGUAL_FORMAT = (
    "\n# 输出格式（严格遵守）\n"
    "本次输出必须恰好两行，不要任何前缀、引号、markdown、解释:\n"
    "JA: <{tts_lang}台词，用于配音>\n"
    "ZH: <{sub_lang}字幕，用于显示>\n\n"
    "要求:\n"
    "- JA 行用自然的{tts_lang}口语表达你的人格反应\n"
    "- ZH 行是同一含义的{sub_lang}口语化字幕，符合该人格调性\n"
    "- 两行表达意思一致但不必逐字翻译\n"
    "- 不输出除这两行以外的任何字符"
)


_LINE_RE = re.compile(r"^(JA|ZH)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)


def _parse_bilingual(raw: str) -> tuple[str, str]:
    """Best-effort parse of the JA:/ZH: format. Returns (ja, zh).

    Robust to mid-output markdown fences, full-width colons, and missing labels —
    falls back to using the whole text for whichever side wasn't parsed.
    """
    ja, zh = "", ""
    cleaned = raw.strip().strip("`").replace("```", "")
    for line in cleaned.splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        tag = m.group(1).upper()
        if tag == "JA" and not ja:
            ja = m.group(2).strip()
        elif tag == "ZH" and not zh:
            zh = m.group(2).strip()
    return ja, zh


def generate(
    event_summary: str,
    frame_b64: str,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    max_tokens: int = 200,
    bilingual: bool = True,
    tts_language: str = "日语",
    subtitle_language: str = "中文",
) -> AiReply:
    """Stage 2: produce the actual line, with vision + full persona.

    When bilingual is True (default), the LLM emits one labelled line for the
    TTS language (Japanese by default) and one for the subtitle language
    (Chinese by default). When False, a single-language response is used for
    both fields.
    """
    system_parts = [persona.system_prompt]
    if memory:
        system_parts.append(f"\n# 记忆\n{memory}")

    if bilingual:
        system_parts.append(_BILINGUAL_FORMAT.format(
            tts_lang=tts_language, sub_lang=subtitle_language,
        ))
    else:
        # Single-language constraint, mirrors the older path.
        system_parts.append(
            f"\n# 输出约束\n严格不超过 {persona.silence_rules.max_words_per_response} 字。"
            "只输出 AI 要说的那一句话，不要解释、不要旁白、不要引号。"
        )

    out = pro_vision(
        prompt=f"当前事件: {event_summary}",
        image_b64=frame_b64,
        system="".join(system_parts),
        mime=mime,
        max_tokens=max_tokens,
    )
    raw = out.strip()

    if bilingual:
        ja, zh = _parse_bilingual(raw)
        if ja and zh:
            return AiReply(text=zh, tts_text=ja, raw=raw)
        # Partial / malformed — fall back gracefully.
        if zh and not ja:
            return AiReply(text=zh, tts_text=zh, raw=raw)
        if ja and not zh:
            return AiReply(text=ja, tts_text=ja, raw=raw)
        # No labels at all — treat the whole blob as both (will at least play).
        return AiReply(text=raw, tts_text=raw, raw=raw)

    return AiReply(text=raw, tts_text=raw, raw=raw)


def decide(
    event_summary: str,
    frame_b64: str,
    thumbnail_b64: str,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    force_speak: bool = False,
    bilingual: bool = True,
    tts_language: str = "日语",
    subtitle_language: str = "中文",
) -> AiReply | None:
    if not force_speak and not gate(event_summary, thumbnail_b64, persona, mime=mime):
        return None
    return generate(
        event_summary, frame_b64, persona,
        memory=memory, mime=mime,
        bilingual=bilingual,
        tts_language=tts_language,
        subtitle_language=subtitle_language,
    )
