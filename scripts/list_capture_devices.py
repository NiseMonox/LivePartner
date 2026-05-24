"""Enumerate DirectShow video devices and audio devices on this machine.

Run any time you (re)plug a capture card and want to know its index/name.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def list_video_devices() -> list[str]:
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception as e:
        print(f"[video] pygrabber unavailable: {e}")
        return []
    fg = FilterGraph()
    names = fg.get_input_devices()
    return list(names)


def list_audio_devices() -> tuple[list[dict], int | None, int | None]:
    try:
        import sounddevice as sd
    except Exception as e:
        print(f"[audio] sounddevice unavailable: {e}")
        return [], None, None
    devs = sd.query_devices()
    default_in, default_out = sd.default.device
    return list(devs), default_in, default_out


def main() -> int:
    print("=== DirectShow video devices ===")
    vids = list_video_devices()
    if not vids:
        print("  (none)")
    for i, n in enumerate(vids):
        print(f"  [{i}] {n}")

    print()
    print("=== Audio devices ===")
    auds, di, do = list_audio_devices()
    for i, d in enumerate(auds):
        in_ch = d.get("max_input_channels", 0)
        out_ch = d.get("max_output_channels", 0)
        sr = d.get("default_samplerate", "?")
        host = d.get("hostapi", "?")
        marks = []
        if in_ch > 0:
            marks.append(f"IN×{in_ch}")
        if out_ch > 0:
            marks.append(f"OUT×{out_ch}")
        marks_s = "/".join(marks) or "—"
        kind = []
        if i == di:
            kind.append("DEFAULT-IN")
        if i == do:
            kind.append("DEFAULT-OUT")
        kind_s = "  ".join(kind)
        print(f"  [{i:>2}] {d.get('name','?'):40s} {marks_s:>14s}  sr={sr}  host={host}  {kind_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
