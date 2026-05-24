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

from .llm import flash_vision, pro_text, pro_vision
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


_CONVERSATION_NOTE_NO_FRAME = (
    "\n# 当前模式：语音对话（无画面）\n"
    "玩家正在通过 Mumble 和你直接说话,你看不到任何游戏画面 —— 不要假装看到画面、"
    "不要描述画面、不要把这条当成游戏事件触发。\n"
    "玩家的话已经写在用户消息里,**必须**直接回应那句话(回答问题/接话/搭腔)。\n"
    "人格里关于\"只在画面有变化时说话\"、\"大多数时候保持沉默\"之类的规则**临时失效** —— "
    "玩家点名你必须答,但语气、用词、性格调性继续按人格走。"
)

_CONVERSATION_NOTE_WITH_FRAME = (
    "\n# 当前模式：语音对话 + 可看画面\n"
    "玩家正在通过 Mumble 和你直接说话,**同时**你能看到当前游戏画面(随用户消息附图)。\n"
    "玩家的话已经写在用户消息里,**必须**直接回应那句话。如果玩家问到画面/游戏内容(\"你看到什么\"、"
    "\"这是哪儿\"、\"我该往哪走\"…)就**真的看图回答**;如果只是闲聊,可以提一嘴画面也可以不提。\n"
    "人格里关于\"只在画面有变化时说话\"、\"大多数时候保持沉默\"之类的规则**临时失效** —— "
    "玩家点名你必须答,但语气、用词、性格调性继续按人格走。"
)


def generate(
    event_summary: str,
    frame_b64: str | None,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    max_tokens: int = 200,
    bilingual: bool = True,
    tts_language: str = "日语",
    subtitle_language: str = "中文",
    is_conversation: bool = False,
) -> AiReply:
    """Stage 2: produce the actual line.

    If frame_b64 is None, runs a text-only call (no vision). Use this for
    player-voice triggers — feeding a synthesized YOU DIED frame to an
    unrelated conversation confuses the model.

    is_conversation=True adds a system note telling the model the input is a
    direct verbal address (overrides persona's "stay silent" rules).
    """
    system_parts = [persona.system_prompt]
    if memory:
        system_parts.append(f"\n# 记忆\n{memory}")
    if is_conversation:
        system_parts.append(
            _CONVERSATION_NOTE_WITH_FRAME if frame_b64 is not None
            else _CONVERSATION_NOTE_NO_FRAME
        )

    if bilingual:
        system_parts.append(_BILINGUAL_FORMAT.format(
            tts_lang=tts_language, sub_lang=subtitle_language,
        ))
    else:
        system_parts.append(
            f"\n# 输出约束\n严格不超过 {persona.silence_rules.max_words_per_response} 字。"
            "只输出 AI 要说的那一句话，不要解释、不要旁白、不要引号。"
        )

    system = "".join(system_parts)
    prompt = event_summary if is_conversation else f"当前事件: {event_summary}"

    # Disable chain-of-thought reasoning for the generate step too — Qwen3.6
    # family has thinking ON by default which adds 5-10× latency for what is
    # essentially a "produce one in-character line" task.
    extra = {"enable_thinking": False}

    if frame_b64 is None:
        out = pro_text(prompt=prompt, system=system, max_tokens=max_tokens,
                       extra_body=extra)
    else:
        out = pro_vision(prompt=prompt, image_b64=frame_b64, system=system,
                         mime=mime, max_tokens=max_tokens, extra_body=extra)
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
    frame_b64: str | None,
    thumbnail_b64: str | None,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    force_speak: bool = False,
    bilingual: bool = True,
    tts_language: str = "日语",
    subtitle_language: str = "中文",
    is_conversation: bool = False,
) -> AiReply | None:
    if not force_speak:
        if thumbnail_b64 is None:
            return None  # no frame, no force → caller must opt-in via force_speak
        if not gate(event_summary, thumbnail_b64, persona, mime=mime):
            return None
    return generate(
        event_summary, frame_b64, persona,
        memory=memory, mime=mime,
        bilingual=bilingual,
        tts_language=tts_language,
        subtitle_language=subtitle_language,
        is_conversation=is_conversation,
    )
