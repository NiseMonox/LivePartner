"""Probe a DirectShow video device — open it, grab one frame, save as PNG."""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=6, help="DirectShow device index")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", type=Path, default=Path("test_capture.png"))
    args = ap.parse_args()

    print(f"[capture] opening DSHOW device {args.index} …")
    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[capture] could not open"); return 2

    # Try to coax the resolution down (capture cards default to 1080p/4K which is slow)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[capture] negotiated {w}x{h} @ {fps:.0f}fps")

    # Drop the first 5 frames — capture cards often serve stale buffers initially
    for _ in range(5):
        cap.read()
    t0 = time.perf_counter()
    ok, frame = cap.read()
    dt = (time.perf_counter() - t0) * 1000
    if not ok or frame is None:
        print(f"[capture] read returned ok={ok} frame={frame!r}")
        cap.release(); return 3
    print(f"[capture] grabbed {frame.shape}  dtype={frame.dtype}  in {dt:.0f} ms")

    # Quick sanity: detect "no signal" → mostly-black frame
    mean = float(frame.mean())
    print(f"[capture] frame mean brightness = {mean:.1f}")
    if mean < 5:
        print("[capture] WARN: frame is nearly black — is the HDMI source on?")
    cv2.imwrite(str(args.out), frame)
    print(f"[capture] saved {args.out} ({args.out.stat().st_size//1024} KB)")
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
