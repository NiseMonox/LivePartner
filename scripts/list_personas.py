"""List all available personas and verify they load cleanly."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livepartner.persona import list_personas, load_persona


def main() -> int:
    ids = list_personas()
    if not ids:
        print("(no personas found)")
        return 1
    print(f"{len(ids)} personas:")
    for pid in ids:
        try:
            p = load_persona(pid)
            print(f"  {p.id:12s}  {p.display_name:6s}  voice={p.voice.voice_id:30s} "
                  f"cooldown={p.silence_rules.cooldown_after_speak:>3d}s  "
                  f"max_words={p.silence_rules.max_words_per_response:>3d}")
            print(f"               {p.description}")
        except Exception as e:
            print(f"  {pid:12s}  FAILED  {e!r}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
