"""Entry point: python -m livepartner [--smoke]"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _ensure_cache_dirs_on_d() -> None:
    """Belt-and-braces: if HF_HOME / UV_CACHE_DIR weren't inherited from the
    user-level env (PowerShell sometimes misses propagation to existing shells),
    point them at the repo's D: caches so model downloads don't dribble to C:."""
    repo = Path(__file__).resolve().parents[2]
    defaults = {
        "HF_HOME": str(repo / ".hf-cache"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


_ensure_cache_dirs_on_d()


def main() -> int:
    ap = argparse.ArgumentParser(prog="livepartner")
    ap.add_argument("--smoke", action="store_true",
                    help="construct main window then exit (CI / agent verification)")
    args = ap.parse_args()

    from PySide6.QtWidgets import QApplication

    from .ui.main_window import MainWindow

    app = QApplication(sys.argv)
    w = MainWindow()
    if args.smoke:
        print(f"[smoke] MainWindow constructed: title={w.windowTitle()!r}, "
              f"personas={w.persona_combo.count()}")
        return 0
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
