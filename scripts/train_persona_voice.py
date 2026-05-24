"""Build a robust speaker embedding by slicing a long reference into ~10s windows,
extracting an x-vector per window, then averaging.

Run with the FASTER-QWEN3-TTS venv (it has torch + qwen_tts + soundfile):

    D:\\LivePartner\\external\\faster-qwen3-tts\\.venv\\Scripts\\python.exe ^
        D:\\LivePartner\\scripts\\train_persona_voice.py ^
        --ref D:\\LivePartner\\Voice_Source\\voice.wav ^
        --personas snark praise companion newbie coach ^
        --window-sec 10 ^
        --rms-threshold 0.01

Output:
  - slices saved under <ref>_slices/  (kept for QA)
  - <ref>_avg.pt with the averaged embedding
  - Copies <ref>_avg.pt to personas/voices/<persona>.pt for each persona

By default the server picks up new embeddings via POST /voices/reload (we hit
that endpoint at the end). Set --skip-reload to do it yourself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import torch


VOICES_DIR = Path(r"D:\LivePartner\personas\voices")
DEFAULT_MODEL = r"D:\LivePartner\external\faster-qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_RELOAD_URL = "http://127.0.0.1:7001/voices/reload"


def slice_audio(
    ref: Path,
    window_sec: float,
    rms_threshold: float,
    slice_dir: Path,
) -> list[Path]:
    print(f"[train] loading {ref}", flush=True)
    audio, sr = sf.read(ref, always_2d=True)
    print(f"[train]  channels={audio.shape[1]}  sr={sr}  frames={audio.shape[0]}  "
          f"dur={audio.shape[0]/sr:.1f}s", flush=True)

    # Stereo → mono, normalize peak to leave a little headroom.
    mono = audio.mean(axis=1).astype(np.float32)
    peak = float(np.abs(mono).max() or 1.0)
    if peak > 0:
        mono = mono * (0.99 / peak)

    win = int(round(window_sec * sr))
    if len(mono) < win:
        # File is shorter than one window — just use the whole thing as the only slice.
        print(f"[train] file shorter than {window_sec}s, using whole file", flush=True)
        return [ref]

    slice_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous slices to avoid stale data.
    for old in slice_dir.glob("slice_*.wav"):
        old.unlink()

    kept: list[Path] = []
    dropped_silence = 0
    for i, start in enumerate(range(0, len(mono) - win + 1, win)):
        sl = mono[start:start + win]
        rms = float(np.sqrt((sl ** 2).mean()))
        if rms < rms_threshold:
            dropped_silence += 1
            continue
        out_path = slice_dir / f"slice_{i:03d}.wav"
        sf.write(out_path, sl, sr)
        kept.append(out_path)
    print(f"[train] kept {len(kept)} slices, dropped {dropped_silence} silent ones "
          f"(rms < {rms_threshold})", flush=True)
    if not kept:
        raise RuntimeError("No non-silent slices found. Lower --rms-threshold?")
    return kept


def extract_embeddings(model_path: str, slices: list[Path]) -> torch.Tensor:
    from qwen_tts import Qwen3TTSModel

    print(f"[train] loading model {model_path}", flush=True)
    model = Qwen3TTSModel.from_pretrained(model_path, device_map="cuda:0", dtype=torch.bfloat16)

    embs: list[torch.Tensor] = []
    for i, p in enumerate(slices):
        items = model.create_voice_clone_prompt(ref_audio=str(p), ref_text="",
                                                x_vector_only_mode=True)
        # Cast to float32 for accurate averaging, store on CPU.
        emb = items[0].ref_spk_embedding.detach().cpu().float()
        embs.append(emb)
        print(f"[train]  [{i+1}/{len(slices)}] {p.name}  shape={tuple(emb.shape)}",
              flush=True)
    stacked = torch.stack(embs, dim=0)  # (N, 1024)
    avg = stacked.mean(dim=0)
    # Sanity: report std as a "voice consistency" signal.
    std = stacked.std(dim=0).mean().item()
    print(f"[train] averaged across {len(embs)} slices  mean_std={std:.4f} "
          "(lower = more consistent voice)", flush=True)
    return avg.to(torch.bfloat16)


def write_for_personas(emb: torch.Tensor, personas: list[str]) -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    for p in personas:
        out = VOICES_DIR / f"{p}.pt"
        torch.save(emb, out)
        print(f"[train] wrote {out}  size={out.stat().st_size} bytes", flush=True)


def try_reload(url: str) -> None:
    try:
        r = httpx.post(url, timeout=10.0)
        if r.status_code == 200:
            print(f"[train] server reloaded voices: {r.json()}", flush=True)
        else:
            print(f"[train] reload HTTP {r.status_code}: {r.text[:200]}", flush=True)
    except Exception as e:
        print(f"[train] server reload skipped ({e!r})  — restart it manually or "
              f"POST {url} when running.", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path, help="reference audio (any sr/channels)")
    ap.add_argument("--personas", nargs="+", required=True,
                    help="persona ids to write the averaged .pt for")
    ap.add_argument("--window-sec", type=float, default=10.0,
                    help="slice length in seconds (5-30 typical)")
    ap.add_argument("--rms-threshold", type=float, default=0.01,
                    help="drop slices whose RMS energy is below this (after peak normalize)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--slice-dir", type=Path, default=None,
                    help="where to write slice_*.wav (default: <ref>_slices/)")
    ap.add_argument("--reload-url", default=DEFAULT_RELOAD_URL)
    ap.add_argument("--skip-reload", action="store_true")
    args = ap.parse_args()

    slice_dir = args.slice_dir or args.ref.parent / (args.ref.stem + "_slices")

    slices = slice_audio(args.ref, args.window_sec, args.rms_threshold, slice_dir)
    avg = extract_embeddings(args.model_path, slices)
    write_for_personas(avg, args.personas)

    # Also keep a master copy next to the source for reference.
    master = args.ref.with_suffix("").with_name(args.ref.stem + "_avg.pt")
    torch.save(avg, master)
    print(f"[train] master copy: {master}", flush=True)

    if not args.skip_reload:
        try_reload(args.reload_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
