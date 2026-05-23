"""Two-stage decision pipeline per SPEC §5.3.3.

flash gate (small thumbnail + event, GENERIC system) → bool
   ↓ if yes
pro vision generate (full frame + full persona + memory + event) → AI line

Per SPEC §5.3.2 we also allow `force_speak=True` for high-priority events
(death, achievement, player voice) — these skip the LLM gate entirely.
"""
from __future__ import annotations

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
    """Stage 1: should AI speak now? Uses flash + a small thumbnail with a generic system prompt
    (persona prompt would just bloat prefill — persona shines in generate).
    """
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
    text: str
    raw: str


def generate(
    event_summary: str,
    frame_b64: str,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    max_tokens: int = 120,
) -> AiReply:
    """Stage 2: produce the actual line, with vision + full persona."""
    system_parts = [persona.system_prompt]
    if memory:
        system_parts.append(f"\n# 记忆\n{memory}")
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
    return AiReply(text=out.strip(), raw=out)


def decide(
    event_summary: str,
    frame_b64: str,
    thumbnail_b64: str,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    force_speak: bool = False,
) -> AiReply | None:
    """Full decision: optional gate → generate. Returns None if AI stays silent.

    force_speak=True skips the LLM gate (use for high-priority events: death,
    achievement, player voice, etc. — these go straight to generate).
    """
    if not force_speak and not gate(event_summary, thumbnail_b64, persona, mime=mime):
        return None
    return generate(event_summary, frame_b64, persona, memory=memory, mime=mime)
