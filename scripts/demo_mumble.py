"""Connect to local Mumble server, join channel, speak one snark line.

Prereq: mumble-server.exe running on 127.0.0.1:64738 (default port).
On Windows start the server via PowerShell:
    Start-Process -FilePath (Join-Path $env:ProgramFiles 'Mumble\server\mumble-server.exe') -WindowStyle Hidden
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livepartner.audio_codec import mp3_to_pcm48k, pcm48k_duration_seconds
from livepartner.mumble_bot import MumbleBot, MumbleConfig
from livepartner.persona import load_persona
from livepartner.tts import synthesize_for_persona


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=64738)
    ap.add_argument("--name", default="LivePartner")
    ap.add_argument("--channel", default="LivePartner")
    ap.add_argument("--password", default="")
    ap.add_argument("--persona", default="snark")
    ap.add_argument("--text", default="测试，损友连接 Mumble 完成。")
    args = ap.parse_args()

    persona = load_persona(args.persona)

    print(f"[1/4] synthesizing TTS for: {args.text!r}")
    t0 = time.perf_counter()
    tts = synthesize_for_persona(args.text, persona)
    print(f"      done in {(time.perf_counter()-t0)*1000:.0f} ms, {len(tts.mp3)//1024} KB mp3")

    print(f"[2/4] decoding mp3 → 48kHz mono int16 PCM")
    t0 = time.perf_counter()
    pcm = mp3_to_pcm48k(tts.mp3)
    dur = pcm48k_duration_seconds(pcm)
    print(f"      done in {(time.perf_counter()-t0)*1000:.0f} ms, "
          f"{len(pcm)//1024} KB PCM, {dur:.2f} s")

    print(f"[3/4] connecting Mumble bot to {args.host}:{args.port}, channel {args.channel!r}")
    cfg = MumbleConfig(host=args.host, port=args.port, name=args.name,
                       password=args.password, channel=args.channel)
    bot = MumbleBot(cfg)
    print(f"      bot.start() …", flush=True)
    bot.start(timeout=8.0)
    print(f"      current channel: {bot.current_channel_name!r}", flush=True)

    print(f"[4/4] streaming PCM to Mumble channel")
    t0 = time.perf_counter()
    bot.send_pcm(pcm)
    # Need to wait for pymumble to actually push the bytes; give it audio_duration + slack.
    bot.wait_until_silent(max_wait=dur + 3.0)
    # Pad small grace period
    time.sleep(0.5)
    print(f"      streamed in {(time.perf_counter()-t0)*1000:.0f} ms (audio was {dur:.2f}s long)")

    bot.stop()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
