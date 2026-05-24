"""Extract a 4KB speaker embedding for one persona from a reference audio file.

The output is consumed by services/qwen3_tts_server.py at startup.

Run with the FASTER-QWEN3-TTS venv (it has torch + qwen_tts):

    D:\\LivePartner\\external\\faster-qwen3-tts\\.venv\\Scripts\\python.exe ^
        D:\\LivePartner\\scripts\\extract_persona_voice.py ^
        snark D:\\path\\to\\snark_ref.wav

Output: D:\\LivePartner\\personas\\voices\\snark.pt  (~4 KB)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

DEFAULT_MODEL = r"D:\LivePartner\external\faster-qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICES = Path(r"D:\LivePartner\personas\voices")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("persona_id", help="persona id, becomes <id>.pt in voices dir")
    ap.add_argument("ref_audio", help="reference audio path (wav recommended)")
    ap.add_argument("--model-path", default=DEFAULT_MODEL,
                    help="Qwen3-TTS base model dir")
    ap.add_argument("--voices-dir", default=str(DEFAULT_VOICES),
                    help="dir to write the .pt embedding into")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel

    voices_dir = Path(args.voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    out = voices_dir / f"{args.persona_id}.pt"

    print(f"[extract] loading model: {args.model_path}", flush=True)
    model = Qwen3TTSModel.from_pretrained(args.model_path, device_map=args.device, dtype=torch.bfloat16)
    print(f"[extract] reading ref audio: {args.ref_audio}", flush=True)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text="",
        x_vector_only_mode=True,
    )
    spk_emb = prompt_items[0].ref_spk_embedding.cpu()
    torch.save(spk_emb, out)
    print(f"[extract] saved {out}  shape={tuple(spk_emb.shape)}  dtype={spk_emb.dtype}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
