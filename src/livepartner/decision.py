"""Two-stage decision pipeline per SPEC §5.3.3.

flash gate (small thumbnail + event summary) → bool
   ↓ if yes
pro vision generate (full frame + persona + memory + event summary) → AI line
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import flash_vision, pro_vision
from .persona import Persona


GATE_INSTRUCTION = (
    "下面是当前游戏画面和事件描述。\n"
    "判断你这个 AI 陪玩此刻是否应该开口说话。\n"
    "如果该说话回答 yes，如果该沉默回答 no。\n"
    "不要解释，只输出一个词。"
)


def gate(
    event_summary: str,
    thumbnail_b64: str,
    persona: Persona,
    *,
    mime: str = "image/png",
) -> bool:
    """Stage 1: should AI speak now? Uses flash + a small thumbnail."""
    out = flash_vision(
        prompt=f"{GATE_INSTRUCTION}\n\n事件: {event_summary}",
        image_b64=thumbnail_b64,
        system=persona.system_prompt,
        mime=mime,
        max_tokens=20,
    )
    answer = out.strip().lower()
    return answer.startswith("yes") or answer.startswith("是")


@dataclass(frozen=True)
class AiReply:
    text: str
    raw: str  # full model output for debugging


def generate(
    event_summary: str,
    frame_b64: str,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    max_tokens: int = 120,
) -> AiReply:
    """Stage 2: produce the actual line, with vision."""
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
) -> AiReply | None:
    """Full decision: gate → (maybe) generate. Returns None if AI stays silent."""
    if not gate(event_summary, thumbnail_b64, persona, mime=mime):
        return None
    return generate(event_summary, frame_b64, persona, memory=memory, mime=mime)
