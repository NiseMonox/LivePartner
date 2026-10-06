[English](README.md) | **简体中文**

# LivePartner

> 桌面 AI 游戏陪玩 — "她"陪着你玩。
> 看你的游戏画面、听你的声音、用克隆音色和你说话、通过 VTube Studio 显示
> Live2D 形象，并且**自己驱动表情/嘴型/眨眼/姿态**。

LivePartner 是一个 Python + PySide6 的本地桌面应用，把以下能力拼到一起：

- **视觉**：从 HDMI 采集卡读游戏画面 → Qwen3.6 vision 看 → 决定要不要说话 + 说什么
- **听觉**：Mumble 语音频道接玩家麦克风 → faster-whisper-large-v3-turbo 转写 → 立即回应
- **声音**：Qwen3-TTS 本地推理 + voice cloning → 用克隆的角色音色 PCM 流式回 Mumble
- **形象**：VTube Studio + pyvts WebSocket API → Live2D 模型挂在屏幕上 (桌宠 / 直播)
- **AI 主控表情**：独立的 Director flash 调用,给每句台词挑表情贴图 + 眉毛/嘴角参数偏移
- **嘴型同步**：TTS PCM 实时 RMS → 驱动 VTS MouthOpen,按音频实时速度而非生成速度
- **自动眨眼/头部摆动**：周期参数驱动 (EyeOpenLeft/Right、FaceAngleX/Y/Z),让形象始终"活的"
- **视觉自我感知**：每 90s 截一次 VTS 窗口 → flash vision 描述外观 → 玩家换衣/换发型 Eri 会主动评论
- **直播弹幕**：可接 B 站直播间, AI 限流回应观众弹幕
- **长期记忆**：玩家档案 / 自动事实抽取 / 画面回顾 / 反复读检测 + reroll

> 当前角色 `Eri` 是日系 JK 青梅竹马、傲娇为底色,用中文说话,带轻微机械感的 TTS 音色。
> 你可以基于 `personas/eri.yaml` 改一份新人格。

完整产品/架构设计请看 [SPEC.md](SPEC.md);本地开发笔记看 [DEV.md](DEV.md)。

---

## 架构

```
            ┌────────────────────────────┐
游戏 PC ──── │  OBS NDI 推流(可选)         │ ─── 笔记本 OBS
            └────────────────────────────┘
                                                 │
HDMI ──── 采集卡 ── 笔记本 ── opencv 抓帧 ────────┤
                                                 ▼
                          ┌──────────────────────────────┐
                          │  LivePartner (PySide6 应用)   │
                          │ ┌──────────────────────────┐ │
                          │ │ decision (gate→generate) │ │
                          │ │  ↓ LINE (中文)            │ │
                          │ │  ├→ 字幕 overlay (Qt+SSE) │ │
                          │ │  ├→ Director (flash JSON) │ │── pyvts ──→ VTube Studio
                          │ │  └→ Qwen3-TTS (本地)      │ │            (Live2D 形象)
                          │ │      ↓ PCM 流              │ │              │
                          │ │      ├→ Mumble Bot ───────┼─┼─→ Mumble 服务器 ─ 玩家耳机
                          │ │      └→ LipSync → MouthOpen│ │
                          │ │                            │ │
                          │ │ IdleMotion (20Hz):         │ │
                          │ │  blink + head sway + breath│ │
                          │ │                            │ │
                          │ │ SelfDescriber (90s):       │ │
                          │ │  VTS 窗口截图 → 描述 → memory│
                          │ └──────────────────────────┘ │
                          │ Mumble bot ← 玩家麦克风     │
                          │   ↓                         │
                          │ STT (faster-whisper)        │
                          │   ↓ 转写文本                │
                          │   排队 → decision           │
                          │                              │
                          │ 弹幕 (Bilibili WSS) → 限流  │
                          │   → decision                │
                          └──────────────────────────────┘
```

---

## 安装

### 前置

- **Windows 10/11** (Windows API + DirectShow capture + VTS 都是 Windows-first)
- **Python 3.11+**
- **uv** (推荐, 比 pip 快很多) `winget install astral-sh.uv`
- **HDMI 采集卡** (项目默认偏好 AVerMedia Live Gamer Ultra, 见 `_PREFERRED_CAPTURE_DEVICE_HINTS`)
- **Mumble 服务器** (Murmur) + Mumble 客户端 (玩家用)
- **VTube Studio** (Steam 免费版,可选 — 没有就没有 Live2D 形象)
- **Qwen3-TTS** 本地推理环境 (项目通过 `external/faster-qwen3-tts/` 子仓库管理, 见 [DEV.md](DEV.md))
- 一份 **DashScope API Key** (用于 Qwen3.6-flash LLM 推理)

### 装依赖

```bash
git clone https://github.com/NiseMonox/LivePartner.git
cd LivePartner

# 主依赖
uv venv
uv pip install -e .

# 可选模块 (按需)
uv pip install -e ".[capture]"   # opencv + pygrabber 采集卡
uv pip install -e ".[vts]"        # pyvts (Live2D 控制)
uv pip install -e ".[danmaku]"    # bilibili-api-python (弹幕)
```

### 配置

复制 `.env.example` 到 `.env`,填:

```
LP_DASHSCOPE_API_KEY=sk-xxxxxxxxxxxx
LP_MODEL_FLASH=qwen3.6-flash
LP_MODEL_PRO=qwen3.6-flash
LP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

把语音克隆参考音频抽成 `personas/voices/<persona_id>.pt`:

```bash
python external/faster-qwen3-tts/examples/extract_speaker.py \
  --ref-audio your_reference.wav \
  --ref-text "参考音频对应的文本" \
  --output personas/voices/eri.pt
```

VTube Studio 设置 → API → 打开 "Start API" (默认端口 8001);加载你的 Live2D 模型。

---

## 启动

```bash
# 启 Mumble 服务器 (本地开发)
mumble.bat

# 启 LivePartner UI
eri.bat
```

或者直接:

```bash
.venv\Scripts\python.exe -m livepartner
```

UI 打开后:

1. **采集卡 tab** → 选择 HDMI 采集卡, 启动 capture
2. **语音 tab** → 启动 TTS 服务 (首次加载 60-90s), 连 Mumble
3. **STT** 加载 large-v3-turbo, 监听频道, 白名单填你的 Mumble 用户名
4. **Live2D tab** → 点连接 VTS → VTS 弹窗里 Allow 授权 (一次性)
5. **运行 tab** → 一键启动 (按顺序拉起上面那些), 或勾环境模式让 Eri 自己看画面决定开口

完成后, 用 Mumble 客户端连进去说话, Eri 就会用克隆音色回应, VTS 模型同步切表情 + 嘴动。

---

## 关键 UI Tab

| Tab | 内容 |
|---|---|
| **运行** | 触发按钮、人格选择、字幕浮窗 / OBS Browser Source / 一键启动 / 环境模式 |
| **采集卡** | DirectShow 设备选择、分辨率、预览 |
| **语音/TTS/Mumble** | TTS 引擎、Qwen3 server URL、Mumble 服务器、STT 模型 |
| **记忆** | observed.md 时间线、长期记忆编辑 (player/identity/jokes/progress/auto.md)、会话总结、自动事实审核 |
| **Live2D** | VTS 连接、模型/表情/hotkey 列表、手动测试表情、idle 动作 (眨眼/头摆/呼吸) toggle、外观预设 hotkey 自动应用 |
| **直播** | Bilibili 房间号、AI 回应弹幕 toggle、冷却 + 最短长度过滤、弹幕日志 |

---

## 关键文件

```
src/livepartner/
├── decision.py              # 主决策: gate → generate (LINE 中文输出)
├── director.py              # 独立 director flash 调用,选表情 + 面部参数
├── memory.py                # 长期记忆 + 反复读检测 + 自我外观状态
├── persona.py               # 角色加载 + dataclass
├── llm.py                   # OpenAI-compatible (DashScope) 封装
├── mumble_bot.py            # pymumble 1.6 接入 + 流式 PCM 发送
├── stt.py                   # faster-whisper-large-v3-turbo 转写 + 幻觉过滤
├── tts.py / tts_qwen3.py    # Edge TTS / Qwen3-TTS 本地客户端
├── capture.py               # DirectShow 采集卡 + frame snapshot
├── vts_controller.py        # pyvts 异步线程封装 + Expression API
├── lip_sync.py              # PCM RMS → MouthOpen,按音频实时速度调度
├── idle_motion.py           # 20Hz 眨眼/头摆/呼吸驱动
├── appearance.py            # 视觉自我感知 (VTS 窗口截图 → flash describe)
├── danmaku.py               # Bilibili 弹幕 WSS adapter
├── window_capture.py        # Windows ctypes 找窗口 + PIL 截图
├── overlay_server.py        # HTTP + SSE 给 OBS Browser Source 推字幕
└── ui/
    ├── main_window.py       # 整个 Qt UI (~2500 行)
    └── subtitle_overlay.py  # 透明置顶字幕浮窗

services/
└── qwen3_tts_server.py      # FastAPI 包装 faster-qwen3-tts,UI 通过 HTTP 调

personas/
├── eri.yaml                 # 默认人格 (傲娇 JK 青梅竹马, 中文)
└── voices/                  # .pt voice clone embeddings (gitignored)

.memory/                     # 玩家档案 + 自动抽取的事实 + 对话历史 (gitignored)
```

---

## 自定义人格

复制 `personas/eri.yaml` 改一份新的, 然后用 `extract_speaker.py` 抽对应音色到 `personas/voices/<新 id>.pt`,
UI 里"人格"下拉会自动列出。详见 SPEC.md §4 "Persona 设计"。

---

## 文档索引

- **SPEC.md** — 完整产品愿景 + 架构设计 (设计阶段文档,可能跟当前实现有出入)
- **DEV.md** — 本地开发笔记: 环境变量 / `.bat` 脚本 / 常见踩坑 / 调试技巧
- **WINDOWS_SETUP_GUIDE** — 在 `external/faster-qwen3-tts/` 子仓库,Qwen3-TTS 单独安装

## License

MIT (主仓库) / `external/` 各子仓库按各自 license。

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 阿里 Qwen 团队
- [faster-qwen3-tts](https://github.com/Qwen-AI/faster-qwen3-tts) (vendored) — Qwen3-TTS 快速推理
- [pyvts](https://github.com/Genteki/pyvts) — VTube Studio Python 客户端
- [pymumble](https://github.com/azlux/pymumble) 1.6 — Python Mumble protocol
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper CTranslate2 加速
- [bilibili-api-python](https://github.com/Nemo2011/bilibili-api) — 弹幕 WSS adapter
