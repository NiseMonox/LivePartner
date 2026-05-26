"""Local Qwen3-TTS FastAPI server with precomputed per-persona speaker embeddings.

Run with the faster-qwen3-tts venv (NOT the main LivePartner venv):

    D:\\LivePartner\\external\\faster-qwen3-tts\\.venv\\Scripts\\python.exe ^
        D:\\LivePartner\\services\\qwen3_tts_server.py

Env vars (all optional):
    LP_TTS_MODEL_PATH   path to local Qwen3-TTS model dir
    LP_TTS_VOICES_DIR   dir of <persona_id>.pt speaker embeddings
    LP_TTS_PORT         listen port (default 7001)
    LP_TTS_DEVICE       cuda / cpu (default cuda)
    LP_TTS_DTYPE        bf16 / fp16 / fp32 (default bf16)

Endpoints:
    POST /tts        { persona_id, text, language="Chinese", chunk_size=4 }
                     → streamed application/octet-stream of 48kHz mono int16 PCM
    GET  /voices     → { "voices": [persona_id, ...] }
    GET  /health     → { ok, model_loaded, device, voices, sample_rate, ... }
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# faster_qwen3_tts is installed in this venv via `uv pip install -e .` from the external repo.
from faster_qwen3_tts import FasterQwen3TTS


# ---------- config ----------
DEFAULT_MODEL_PATH = r"D:\LivePartner\external\faster-qwen3-tts\models\Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_VOICES_DIR = r"D:\LivePartner\personas\voices"

MODEL_PATH = os.environ.get("LP_TTS_MODEL_PATH", DEFAULT_MODEL_PATH)
VOICES_DIR = Path(os.environ.get("LP_TTS_VOICES_DIR", DEFAULT_VOICES_DIR))
PORT = int(os.environ.get("LP_TTS_PORT", "7001"))
DEVICE = os.environ.get("LP_TTS_DEVICE", "cuda")
DTYPE_STR = os.environ.get("LP_TTS_DTYPE", "bf16")
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[DTYPE_STR]

# Mumble target — output what the bot needs.
OUTPUT_SAMPLE_RATE = 48_000


# ---------- app state ----------
class State:
    model: Optional[FasterQwen3TTS] = None
    voices: dict[str, torch.Tensor] = {}
    model_sr: int = 24000  # filled at startup


S = State()


def _resample_to_48k_int16(samples_f32: np.ndarray, src_sr: int) -> bytes:
    """Mono float32 [-1, 1] at src_sr → 48kHz mono int16 raw bytes."""
    if src_sr == 48000:
        out = samples_f32
    elif src_sr == 24000:
        # 2x nearest-neighbor upsample — crude but fast.
        out = np.repeat(samples_f32, 2)
    else:
        n_new = int(round(len(samples_f32) * 48000 / src_sr))
        t_old = np.linspace(0.0, 1.0, len(samples_f32), endpoint=False)
        t_new = np.linspace(0.0, 1.0, n_new, endpoint=False)
        out = np.interp(t_new, t_old, samples_f32).astype(np.float32)
    out = np.clip(out, -1.0, 1.0) * 32767.0
    return out.astype(np.int16).tobytes()


def _load_voices() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    S.voices.clear()
    for pt in sorted(VOICES_DIR.glob("*.pt")):
        try:
            emb = torch.load(pt, weights_only=True).to(S.model.device).to(DTYPE)
            S.voices[pt.stem] = emb
            print(f"[tts] voice loaded: {pt.stem}  shape={tuple(emb.shape)}", flush=True)
        except Exception as e:
            print(f"[tts] WARN: failed to load {pt}: {e}", flush=True)


def _prewarm_streaming() -> None:
    """Trigger the streaming code path once so its CUDA graphs are captured.

    Without this, the first real /tts call eats ~20 s of one-time graph capture
    (separate from the non-streaming capture done by FasterQwen3TTS internally).
    """
    if not S.voices:
        return
    sample_voice = next(iter(S.voices.values()))
    vcp = dict(
        ref_code=[None],
        ref_spk_embedding=[sample_voice],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )
    print("[tts] prewarming streaming path (one-shot, ~20s) …", flush=True)
    t0 = time.perf_counter()
    chunks = 0
    for _ in S.model.generate_voice_clone_streaming(
        text="预热",
        language="Chinese",
        voice_clone_prompt=vcp,
        chunk_size=4,
        max_new_tokens=24,  # generate just a tiny snippet
    ):
        chunks += 1
    print(f"[tts] prewarm done in {(time.perf_counter()-t0)*1000:.0f} ms ({chunks} chunks)",
          flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[tts] loading model: {MODEL_PATH}  dtype={DTYPE_STR} device={DEVICE}", flush=True)
    t0 = time.perf_counter()
    S.model = FasterQwen3TTS.from_pretrained(MODEL_PATH, device=DEVICE, dtype=DTYPE)
    print(f"[tts] model loaded in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
    try:
        S.model_sr = int(S.model.speech_tokenizer.sample_rate)
    except Exception:
        S.model_sr = 24000
    print(f"[tts] model sample rate: {S.model_sr} Hz", flush=True)

    print(f"[tts] scanning voices in: {VOICES_DIR}", flush=True)
    _load_voices()
    if not S.voices:
        print(f"[tts] WARNING: no voice embeddings found. POST /tts will 404 until you "
              f"drop *.pt files into {VOICES_DIR}", flush=True)
    else:
        _prewarm_streaming()
    yield
    # shutdown: nothing to clean — process exit handles GPU
    print("[tts] shutdown", flush=True)


app = FastAPI(lifespan=lifespan, title="LivePartner Qwen3-TTS Server")


class TtsRequest(BaseModel):
    persona_id: str = Field(..., description="Voice id; matches <persona_id>.pt in voices dir")
    text: str
    language: str = "Chinese"
    chunk_size: int = Field(4, ge=1, le=24, description="Codec steps per yielded chunk (~chunk_size/12 s)")
    temperature: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 2048
    # Natural-language style/prosody hint, prepended as a user instruction turn.
    # E.g. "请用兴奋的语气说" / "低沉无奈地说" / "轻笑着说". None = neutral.
    # Officially this is experimental in x-vector-only mode but in practice it
    # does steer prosody, just not as reliably as ICL mode.
    instruct: Optional[str] = None


# Languages the underlying Qwen3-TTS model accepts (config.talker_config.codec_language_id).
SUPPORTED_LANGUAGES = {
    "chinese", "english", "german", "italian", "portuguese", "spanish",
    "japanese", "korean", "french", "russian", "auto",
}


@app.post("/tts")
def tts(req: TtsRequest):
    if S.model is None:
        raise HTTPException(503, "model not loaded yet")
    if req.persona_id not in S.voices:
        raise HTTPException(
            404,
            f"voice {req.persona_id!r} not loaded. available: {sorted(S.voices)}",
        )
    # Normalize + validate language BEFORE we start a streaming response, so callers
    # get a clean 400 instead of an aborted chunked body.
    lang_normalized = req.language.strip().lower()
    if lang_normalized not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            400,
            f"language {req.language!r} not supported. accepted (case-insensitive): "
            f"{sorted(SUPPORTED_LANGUAGES)}",
        )
    req_language = lang_normalized
    spk_emb = S.voices[req.persona_id]
    vcp = dict(
        ref_code=[None],
        ref_spk_embedding=[spk_emb],
        x_vector_only_mode=[True],
        icl_mode=[False],
    )

    instruct = (req.instruct or "").strip() or None
    if instruct:
        print(f"[tts] instruct: {instruct!r}", flush=True)

    def gen():
        t0 = time.perf_counter()
        first = True
        total_chunks = 0
        total_samples_out = 0
        for audio_chunk, sr, timing in S.model.generate_voice_clone_streaming(
            text=req.text,
            language=req_language,
            voice_clone_prompt=vcp,
            chunk_size=req.chunk_size,
            temperature=req.temperature,
            top_k=req.top_k,
            max_new_tokens=req.max_new_tokens,
            instruct=instruct,
        ):
            if first:
                ttfb_ms = (time.perf_counter() - t0) * 1000
                print(f"[tts] {req.persona_id!r}  TTFB={ttfb_ms:.0f}ms  text_len={len(req.text)}",
                      flush=True)
                first = False
            pcm = _resample_to_48k_int16(audio_chunk.astype(np.float32, copy=False), sr)
            total_samples_out += len(pcm) // 2
            total_chunks += 1
            yield pcm
        dur = total_samples_out / OUTPUT_SAMPLE_RATE
        gen_s = time.perf_counter() - t0
        rtf = dur / gen_s if gen_s > 0 else 0
        print(f"[tts] done  chunks={total_chunks}  audio={dur:.2f}s  gen={gen_s:.2f}s  RTF={rtf:.2f}",
              flush=True)

    return StreamingResponse(
        gen(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(OUTPUT_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Format": "int16-le",
        },
    )


@app.get("/voices")
def voices():
    return {"voices": sorted(S.voices.keys())}


@app.post("/voices/reload")
def reload_voices():
    if S.model is None:
        raise HTTPException(503, "model not loaded yet")
    _load_voices()
    return {"voices": sorted(S.voices.keys())}


@app.get("/health")
def health():
    return JSONResponse({
        "ok": S.model is not None,
        "model_path": MODEL_PATH,
        "device": DEVICE,
        "dtype": DTYPE_STR,
        "model_sample_rate": S.model_sr,
        "output_sample_rate": OUTPUT_SAMPLE_RATE,
        "voices": sorted(S.voices.keys()),
        "voices_dir": str(VOICES_DIR),
    })


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
