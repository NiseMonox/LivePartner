**English** | [简体中文](README.zh-CN.md)

# LivePartner

> A desktop AI gaming companion — "she" keeps you company while you play.
> She watches your game, listens to you, talks to you in a cloned voice, shows up as a
> Live2D avatar through VTube Studio, and **drives her own expressions/mouth movements/blinking/posture**.

LivePartner is a local Python + PySide6 desktop app that pieces together the following:

- **Vision**: reads game frames from an HDMI capture card → Qwen3.6 vision looks at them → decides whether to speak + what to say
- **Hearing**: the player's mic comes in over a Mumble voice channel → faster-whisper-large-v3-turbo transcribes it → she replies right away
- **Voice**: local Qwen3-TTS inference + voice cloning → streams PCM in the cloned character voice back into Mumble
- **Avatar**: VTube Studio + the pyvts WebSocket API → a Live2D model sits on your screen (desktop pet / streaming)
- **AI-directed expressions**: a separate Director flash call picks an expression sticker + eyebrow/mouth-corner parameter offsets for every line
- **Lip sync**: real-time RMS of the TTS PCM → drives VTS MouthOpen, paced to real-time audio playback rather than generation speed
- **Auto blink/head sway**: periodic parameter driving (EyeOpenLeft/Right, FaceAngleX/Y/Z) keeps the avatar "alive" at all times
- **Visual self-awareness**: screenshots the VTS window every 90s → flash vision describes her look → if the player changes her outfit/hairstyle, Eri comments on it unprompted
- **Live-stream danmaku**: can hook into a Bilibili live room; the AI replies to viewers' danmaku (bullet comments), rate-limited
- **Long-term memory**: player profile / automatic fact extraction / scene recall / repeat detection + reroll

> The current character, `Eri`, is a Japanese high-school girl (JK) and your childhood friend, tsundere at heart. She speaks Chinese, in a TTS voice with a slightly robotic edge.
> You can make your own persona based on `personas/eri.yaml`.

For the full product/architecture design, see [SPEC.md](SPEC.md); local dev notes are in [DEV.md](DEV.md).

---

## Architecture

```
               ┌──────────────────────────────────┐
Gaming PC ──── │  OBS NDI stream (optional)       │ ─── laptop OBS
               └──────────────────────────────────┘
                                                        │
HDMI ──── capture card ── laptop ── opencv frame grab ──┤
                                                        ▼
                  ┌─────────────────────────────────────────┐
                  │  LivePartner (PySide6 app)              │
                  │ ┌─────────────────────────────────────┐ │
                  │ │ decision (gate→generate)            │ │
                  │ │  ↓ LINE (Chinese)                   │ │
                  │ │  ├→ subtitle overlay (Qt+SSE)       │ │
                  │ │  ├→ Director (flash JSON)           │ │── pyvts ──→ VTube Studio
                  │ │  └→ Qwen3-TTS (local)               │ │            (Live2D avatar)
                  │ │      ↓ PCM stream                   │ │               │
                  │ │      ├→ Mumble Bot ─────────────────┼─┼─→ Mumble server ─ player's headset
                  │ │      └→ LipSync → MouthOpen         │ │
                  │ │                                     │ │
                  │ │ IdleMotion (20Hz):                  │ │
                  │ │  blink + head sway + breath         │ │
                  │ │                                     │ │
                  │ │ SelfDescriber (90s):                │ │
                  │ │  VTS screenshot → describe → memory │ │
                  │ └─────────────────────────────────────┘ │
                  │ Mumble bot ← player mic                 │
                  │   ↓                                     │
                  │ STT (faster-whisper)                    │
                  │   ↓ transcript                          │
                  │   queue → decision                      │
                  │                                         │
                  │ danmaku (Bilibili WSS) → rate limit     │
                  │   → decision                            │
                  └─────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- **Windows 10/11** (the Windows API + DirectShow capture + VTS are all Windows-first)
- **Python 3.11+**
- **uv** (recommended, much faster than pip) `winget install astral-sh.uv`
- **HDMI capture card** (the project prefers an AVerMedia Live Gamer Ultra by default; see `_PREFERRED_CAPTURE_DEVICE_HINTS`)
- **Mumble server** (Murmur) + a Mumble client (for the player)
- **VTube Studio** (free on Steam; optional — no VTS, no Live2D avatar)
- **Qwen3-TTS** local inference environment (the project manages it as the `external/faster-qwen3-tts/` sub-repo; see [DEV.md](DEV.md))
- A **DashScope API Key** (for Qwen3.6-flash LLM inference)

### Install dependencies

```bash
git clone https://github.com/NiseMonox/LivePartner.git
cd LivePartner

# Core dependencies
uv venv
uv pip install -e .

# Optional modules (as needed)
uv pip install -e ".[capture]"   # opencv + pygrabber for the capture card
uv pip install -e ".[vts]"        # pyvts (Live2D control)
uv pip install -e ".[danmaku]"    # bilibili-api-python (danmaku)
```

### Configuration

Copy `.env.example` to `.env` and fill in:

```
LP_DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
LP_MODEL_FLASH=qwen3.6-flash
LP_MODEL_PRO=qwen3.6-flash
LP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Extract the voice-cloning reference audio into `personas/voices/<persona_id>.pt`:

```bash
python external/faster-qwen3-tts/examples/extract_speaker.py \
  --ref-audio your_reference.wav \
  --ref-text "transcript of the reference audio" \
  --output personas/voices/eri.pt
```

In VTube Studio, go to Settings → API and turn on "Start API" (default port 8001), then load your Live2D model.

---

## Running

```bash
# Start the Mumble server (local dev)
mumble.bat

# Start the LivePartner UI
eri.bat
```

Or run it directly:

```bash
.venv\Scripts\python.exe -m livepartner
```

Once the UI is open:

1. **Capture Card tab** (`采集卡`) → select your HDMI capture card and start capture (`启动`)
2. **Voice tab** (`语音`) → start the TTS service (`启动 TTS 服务`; the first load takes 60-90s), then connect to Mumble (`连接 Mumble`)
3. **STT**: load large-v3-turbo (`加载模型`), turn on channel listening (`监听频道`), and put your Mumble username in the whitelist (`白名单`)
4. **Live2D tab** → click Connect (`连接`) to hook up VTS → click Allow in the VTS popup to authorize it (one-time)
5. **Run tab** (`运行`) → one-click start (`一键启动`; brings up everything above in order), or tick ambient mode (`环境模式`) to let Eri watch the screen and decide for herself when to speak

With that done, join with a Mumble client and start talking: Eri answers in her cloned voice, and the VTS model switches expressions + moves its mouth in sync.

---

## Key UI tabs

| Tab | Contents |
|---|---|
| **Run** (`运行`) | Trigger button, persona picker, subtitle overlay / OBS Browser Source / one-click start / ambient mode |
| **Capture Card** (`采集卡`) | DirectShow device selection, resolution, preview |
| **Voice/TTS/Mumble** (`语音/TTS/Mumble`) | TTS engine, Qwen3 server URL, Mumble server, STT model |
| **Memory** (`记忆`) | observed.md timeline, long-term memory editor (player/identity/jokes/progress/auto.md), session summary, review of auto-extracted facts |
| **Live2D** | VTS connection, model/expression/hotkey lists, manual expression testing, idle motion (blink/head sway/breathing) toggles, auto-applied appearance-preset hotkeys |
| **Streaming** (`直播`) | Bilibili room ID, toggle for AI replies to danmaku, cooldown + minimum-length filter, danmaku log |

---

## Key files

```
src/livepartner/
├── decision.py              # Main decision logic: gate → generate (LINE output in Chinese)
├── director.py              # Separate director flash call: picks expression + facial params
├── memory.py                # Long-term memory + repeat detection + self-appearance state
├── persona.py               # Character loading + dataclass
├── llm.py                   # OpenAI-compatible (DashScope) wrapper
├── mumble_bot.py            # pymumble 1.6 integration + streaming PCM output
├── stt.py                   # faster-whisper-large-v3-turbo transcription + hallucination filter
├── tts.py / tts_qwen3.py    # Edge TTS / Qwen3-TTS local client
├── capture.py               # DirectShow capture card + frame snapshot
├── vts_controller.py        # pyvts async-thread wrapper + Expression API
├── lip_sync.py              # PCM RMS → MouthOpen, scheduled at real-time audio speed
├── idle_motion.py           # 20Hz blink/head sway/breathing driver
├── appearance.py            # Visual self-awareness (VTS window screenshot → flash describe)
├── danmaku.py               # Bilibili danmaku WSS adapter
├── window_capture.py        # Windows ctypes window lookup + PIL screenshot
├── overlay_server.py        # HTTP + SSE pushing subtitles to OBS Browser Source
└── ui/
    ├── main_window.py       # The entire Qt UI (~2500 lines)
    └── subtitle_overlay.py  # Transparent always-on-top subtitle overlay

services/
└── qwen3_tts_server.py      # FastAPI wrapper around faster-qwen3-tts; the UI calls it over HTTP

personas/
├── eri.yaml                 # Default persona (tsundere JK childhood friend, Chinese)
└── voices/                  # .pt voice clone embeddings (gitignored)

.memory/                     # Player profile + auto-extracted facts + chat history (gitignored)
```

---

## Custom personas

Copy `personas/eri.yaml` and edit it into a new one, then use `extract_speaker.py` to extract the matching voice into `personas/voices/<new id>.pt`;
it'll show up automatically in the UI's Persona (`人格`) dropdown. See SPEC.md §4 "Persona design" for details.

---

## Documentation index

- **SPEC.md** — full product vision + architecture design (a design-phase doc; may differ from the current implementation)
- **DEV.md** — local dev notes: env vars / `.bat` scripts / common pitfalls / debugging tips
- **WINDOWS_SETUP_GUIDE** — in the `external/faster-qwen3-tts/` sub-repo; covers installing Qwen3-TTS on its own

## License

MIT (main repo) / each sub-repo under `external/` follows its own license.

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Alibaba's Qwen team
- [faster-qwen3-tts](https://github.com/Qwen-AI/faster-qwen3-tts) (vendored) — fast Qwen3-TTS inference
- [pyvts](https://github.com/Genteki/pyvts) — VTube Studio Python client
- [pymumble](https://github.com/azlux/pymumble) 1.6 — Python Mumble protocol
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper accelerated with CTranslate2
- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — danmaku WSS adapter
