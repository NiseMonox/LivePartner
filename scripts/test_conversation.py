"""Verify the conversation-mode prompt: AI directly answers player questions
instead of inventing a game-event reaction."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livepartner.decision import generate
from livepartner.persona import load_persona


cases = [
    ("companion", "NiseMono: 「能做个自我介绍吗」"),
    ("companion", "NiseMono: 「早上好呀」"),
    ("snark",     "NiseMono: 「这boss也太难了吧」"),
    ("praise",    "NiseMono: 「你猜我多大」"),
    ("coach",     "NiseMono: 「下一步该怎么打」"),
]
for pid, event in cases:
    p = load_persona(pid)
    r = generate(event, None, p, is_conversation=True,
                 bilingual=True, tts_language="Japanese", subtitle_language="Chinese")
    print(f"\n=== {pid}  | {event}")
    print(f"  zh: {r.text!r}")
    print(f"  ja: {r.tts_text!r}")
