"""Smoke-test faster-whisper on this machine.

Tries GPU first; falls back to CPU if cuDNN bindings aren't found. Picks any
test_*.wav lying around as the input."""
from __future__ import annotations
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# Help ctranslate2 find pip-installed CUDA runtime DLLs on Windows.
# Both add to PATH/DLL-search AND explicitly pre-load the key DLLs via ctypes —
# the explicit load is what actually fixes ctranslate2 on this setup.
if sys.platform == "win32":
    try:
        import ctypes
        import sysconfig
        # nvidia-* wheels are implicit namespace packages (no __init__.py), so look
        # them up via site-packages directly.
        _site = Path(sysconfig.get_paths()["purelib"])
        _root = _site / "nvidia"
        print(f"[whisper-preload] nvidia root: {_root}  exists={_root.is_dir()}", file=sys.stderr)
        for _sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
            _p = _root / _sub
            if _p.is_dir():
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(_p))
                os.environ["PATH"] = str(_p) + os.pathsep + os.environ.get("PATH", "")
        for _dll in [
            _root / "cublas" / "bin" / "cublas64_12.dll",
            _root / "cublas" / "bin" / "cublasLt64_12.dll",
            _root / "cudnn" / "bin" / "cudnn64_9.dll",
            _root / "cudnn" / "bin" / "cudnn_ops64_9.dll",
            _root / "cudnn" / "bin" / "cudnn_cnn64_9.dll",
        ]:
            if _dll.is_file():
                try:
                    ctypes.WinDLL(str(_dll))
                    print(f"[whisper-preload] loaded {_dll.name}", file=sys.stderr)
                except OSError as _e:
                    print(f"[whisper-preload] WARN couldn't preload {_dll.name}: {_e}",
                          file=sys.stderr)
            else:
                print(f"[whisper-preload] WARN missing {_dll}", file=sys.stderr)
    except Exception as _e:
        print(f"[whisper-preload] WARN nvidia DLL shim skipped: {_e}", file=sys.stderr)


from faster_whisper import WhisperModel


def pick_audio() -> Path:
    for cand in [
        "test_jp_ok.wav",
        "test_warm.wav",
        "test_qwen3_2.wav",
        "test_clone.wav",
        "demo_tts.mp3",
    ]:
        p = Path(cand)
        if p.exists():
            return p
    raise SystemExit("No test_*.wav found in cwd; run a TTS demo first")


def try_load(size: str, device: str, compute_type: str) -> WhisperModel | None:
    try:
        print(f"[whisper] loading {size}  device={device}  compute_type={compute_type}", flush=True)
        t0 = time.perf_counter()
        m = WhisperModel(size, device=device, compute_type=compute_type)
        print(f"[whisper] loaded in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)
        return m
    except Exception as e:
        print(f"[whisper] FAILED {device}/{compute_type}: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> int:
    audio = pick_audio()
    print(f"[whisper] audio: {audio.resolve()}  ({audio.stat().st_size//1024} KB)", flush=True)

    # Try the size we actually want (medium) on GPU; fall back to smaller / CPU.
    size = os.environ.get("LP_WHISPER_SIZE", "medium")
    model = (
        try_load(size, "cuda", "float16")
        or try_load(size, "cuda", "int8_float16")
        or try_load(size, "cpu", "int8")
    )
    if model is None:
        print("[whisper] could not load any backend"); return 2

    t0 = time.perf_counter()
    # language=None lets it auto-detect; force Chinese for first test
    segments, info = model.transcribe(str(audio), language=None, beam_size=1)
    text_parts = []
    for seg in segments:
        text_parts.append(seg.text)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[whisper] detected lang={info.language}  prob={info.language_probability:.2f}  "
          f"dur={info.duration:.2f}s  transcribe={dt:.0f}ms  "
          f"RTF={info.duration/(dt/1000):.2f}", flush=True)
    print(f"[whisper] transcript: {''.join(text_parts).strip()!r}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
