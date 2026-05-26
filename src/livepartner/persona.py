from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PERSONA_DIR = REPO_ROOT / "personas"


@dataclass(frozen=True)
class SilenceRules:
    cooldown_after_speak: int = 15
    silence_during_cutscene: bool = True
    silence_during_boss: bool = False
    silence_when_player_speaks: bool = True
    max_words_per_response: int = 40


@dataclass(frozen=True)
class VoiceConfig:
    tts_engine: str = "edge_tts"
    voice_id: str = "zh-CN-XiaoyiNeural"
    speed: float = 1.0
    default_emotion: str = "neutral"


@dataclass(frozen=True)
class Persona:
    id: str
    display_name: str
    description: str
    system_prompt: str
    voice: VoiceConfig
    silence_rules: SilenceRules
    trigger_bias: dict[str, float] = field(default_factory=dict)
    vts: dict[str, Any] = field(default_factory=dict)
    # How chatty the persona naturally is. Steers the gate's yes/no bias.
    # Values: "chatty" / "balanced" / "quiet". Default balanced.
    chat_style: str = "balanced"


def load_persona(persona_id: str, *, directory: Path | None = None) -> Persona:
    directory = directory or PERSONA_DIR
    path = directory / f"{persona_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"persona not found: {path}")
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Persona(
        id=data["id"],
        display_name=data.get("display_name", data["id"]),
        description=data.get("description", ""),
        system_prompt=data["system_prompt"].strip(),
        voice=VoiceConfig(**(data.get("voice") or {})),
        silence_rules=SilenceRules(**(data.get("silence_rules") or {})),
        trigger_bias=dict(data.get("trigger_bias") or {}),
        vts=dict(data.get("vts") or {}),
        chat_style=data.get("chat_style", "balanced"),
    )


def list_personas(directory: Path | None = None) -> list[str]:
    directory = directory or PERSONA_DIR
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))
