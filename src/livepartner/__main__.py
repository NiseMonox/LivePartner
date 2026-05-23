"""Entry point: python -m livepartner [--smoke]"""
from __future__ import annotations

import argparse
import sys


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
