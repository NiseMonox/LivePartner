# LivePartner —— 产品与架构设计文档

> 一款桌面端 AI 游戏陪玩软件。AI 在你玩单机游戏时"坐在你身边"，
> 看你的屏幕、听游戏的音频和你的声音、偶尔说话、长期记忆，
> 可选直播搭子模式（接 OBS 出镜带 Live2D 形象 + 弹幕反应）。
>
> 当前版本：**v0.2**（2026-05-23）
> 状态：设计冻结，准备进入 M1
>
> **vs v0 关键变化**
> - 主软件**全部部署在笔记本**，游戏 PC 零安装（仅依赖已有 Mumble 客户端）
> - 视频和游戏音频通过 **HDMI 采集卡** USB 引入笔记本
> - AI 音频双向通过 **Mumble 自建语音频道**，替代虚拟声卡 + Porcupine 唤醒词 + 直采麦克风
> - 玩家按 Mumble PTT 等于"召唤 AI"，砍掉唤醒词系统

---

## 1. 产品愿景

### 1.1 是什么
一个常驻笔记本的 AI 陪玩。它不是助手、不是攻略、不是聊天机器人。它的角色是"和你一起经历这款游戏的朋友"。

### 1.2 解决什么
- 单机游戏的孤独感
- 开直播门槛高（要观众、要装备、要状态）
- 现有 AI 工具都是"问答式"，没有"陪伴 + 共同经历"

### 1.3 不是什么
- 不是攻略机器人（默认不剧透）
- 不是解说员（不会持续解说）
- 不是声控游戏助手（不操控游戏）

### 1.4 目标用户（v1）
开发者本人。后续可外推到单机重度玩家、个人主播。

---

## 2. 核心设计原则

| 原则 | 说明 |
|---|---|
| **陪伴感 ↔ 不打扰** | 所有设计在这对张力间权衡。宁可少说不要多说。 |
| **零剧透** | AI 假装第一次玩这款游戏，与玩家"一起首通"。 |
| **持续看，分层降本** | 本地预筛持续运行，云端 LLM 按需触发。 |
| **游戏 PC 零侵入** | 游戏机不装任何 LivePartner 软件、不改音频驱动、不改 Windows 设置。所有逻辑在笔记本。 |
| **复用已有基建** | TS/Mumble、OBS、VTube Studio 都是用户已经在用的工具，软件以"接入"姿态融入，不是再造一套。 |
| **模块插拔** | TTS / STT / LLM / 形象 / 弹幕源 / 语音传输 都做成可替换。 |

---

## 3. 功能范围（v1）

### 3.1 模式
单一模式："**陪玩 + 直播搭子**"合一。差异只在直播时启用弹幕反应模块和 OBS 推流。

### 3.2 输入
| 信号 | 路径 |
|---|---|
| 屏幕画面 | 游戏 PC HDMI → 采集卡 → USB → 笔记本 |
| 游戏音频 | 游戏 PC HDMI 音频 → 采集卡 → USB → 笔记本 |
| 玩家麦克风 | 玩家麦 → 游戏 PC Mumble 客户端 → Mumble 服务器 → Mumble bot → 笔记本主程序 |
| 弹幕 | 互联网 WebSocket（直播时启用） |

### 3.3 输出
| 通道 | 路径 |
|---|---|
| AI 语音给玩家 | 笔记本 TTS → Mumble bot → Mumble 服务器 → 游戏 PC Mumble 客户端 → 玩家耳机 |
| AI 语音给观众 | 笔记本 TTS → 本地 PCM → OBS 音频源 |
| Live2D 形象 | VTube Studio（笔记本）→ Spout2 → OBS |
| 字幕 | 笔记本 PySide6 透明浮窗 → OBS 窗口捕获 |

玩家自己不看字幕（v1 取舍）。

### 3.4 核心子系统
- 触发引擎（事件融合 + 沉默规则）
- 记忆系统（每游戏一档案）
- 人格系统（5 预设 + 用户自定义）
- 直播搭子模块（弹幕 + OBS 集成）
- Mumble bot（语音收发中继）

---

## 4. 系统架构

### 4.1 部署拓扑

```
┌─────────── 游戏 PC（零 LivePartner 安装）─────────────┐
│                                                       │
│  游戏 ─HDMI─→ 采集卡 ─HDMI 直通─→ 显示器（0 延迟）   │
│              │ │                                      │
│              │ └─USB─→ 笔记本（视频 + 游戏音）        │
│              │                                        │
│  耳麦 ─┬─→ Mumble Client（已有）─┬→ Mumble Server   │
│        │   （PTT 按住说话）       │  （在笔记本上）   │
│        └← 听 AI 声音 ─────────────┘                  │
│                                                       │
└───────────────────────────────────────────────────────┘
                            │
                            │ 局域网（采集卡 USB + Mumble UDP）
                            ▼
┌──────────────── 笔记本（4070, 32G, i7）────────────────┐
│                                                        │
│  Mumble Server（已有，新建 LivePartner 频道）          │
│       ↑   ↓                                            │
│  ┌────┴───┴───────────────────────────────────────┐   │
│  │  Mumble Bot Client（pymumble，新写）          │   │
│  │    RX: 玩家语音 PCM → 主程序                   │   │
│  │    TX: 主程序 PCM → 频道                       │   │
│  └────────────────────────────────────────────────┘   │
│                ↑                ↓                      │
│  ┌─────────────┴────────────────┴───────────────────┐ │
│  │           LivePartner 主程序（PySide6）          │ │
│  │  ┌──────────────────────────────────────────┐    │ │
│  │  │ 采集层：opencv/pyav（采集卡 DirectShow） │    │ │
│  │  │         sounddevice（采集卡音频）        │    │ │
│  │  ├──────────────────────────────────────────┤    │ │
│  │  │ 信号层：SSIM / CLIP / HUD / OCR /        │    │ │
│  │  │         RMS / 频谱 / BGM 检测            │    │ │
│  │  ├──────────────────────────────────────────┤    │ │
│  │  │ 决策层：事件融合 → 沉默规则 → DeepSeek   │    │ │
│  │  │       （V4 Flash gate + V4 Pro generate）│    │ │
│  │  ├──────────────────────────────────────────┤    │ │
│  │  │ 记忆系统：markdown 读写                  │    │ │
│  │  ├──────────────────────────────────────────┤    │ │
│  │  │ 输出：Mumble TX / VTS / 字幕浮窗          │    │ │
│  │  └──────────────────────────────────────────┘    │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  本地服务（同机进程）：                                │
│  • TTS Server (GPT-SoVITS, ~3GB VRAM 常驻)            │
│  • STT Server (faster-whisper, ~3GB VRAM 按需)        │
│  • VTube Studio (Windows GUI, ~1GB VRAM)              │
│  • OBS (直播时, ~1GB VRAM 含 NVENC)                   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 4.2 进程拓扑（笔记本内部）

```
livepartner.exe（主进程，PySide6 UI）
├── 采集进程（multiprocessing）
│   ├── 视频线程：DirectShow → numpy ndarray queue
│   └── 音频线程：sounddevice → int16 PCM queue
├── 信号进程
│   ├── 视频信号：SSIM / CLIP / HUD / OCR
│   └── 音频信号：RMS / 频谱 / BGM 状态机
├── 决策进程（asyncio）
│   ├── 事件总线（订阅信号 + Mumble RX + 弹幕）
│   ├── 沉默控制器
│   ├── LLM 调用器
│   └── 记忆读写
├── Mumble bot 进程（pymumble）
│   ├── RX 回调：解码用户语音 → 决策进程
│   └── TX 队列：接 TTS 输出 → 编码 Opus → 推到频道
├── 输出协调器（同主进程）
│   ├── TTS WS 客户端 → 收 PCM → Mumble bot TX + OBS PCM tap
│   ├── VTS WS 客户端 → 表情/口型控制
│   └── 字幕浮窗（PySide6 QWidget）
└── UI 线程：设置面板 + 状态托盘

外部进程（同机）：
- mumble-server (常驻 daemon)
- tts_server.py (GPT-SoVITS, FastAPI + WS)
- stt_server.py (faster-whisper, FastAPI)
- VTube Studio (Windows GUI)
- OBS (直播时手动启)
```

进程通信：`zmq` PUB/SUB（信号广播）+ `asyncio` 队列（同进程内）。

### 4.3 端到端事件流（典型 case：玩家死亡 + 自己吐槽）

```
t=0       游戏死亡画面（红屏 "YOU DIED"）+ BGM 切换为死亡音乐
t=30ms    采集卡 HDMI → USB → 笔记本（视频帧 + 音频 chunk）
t=80ms    SSIM 检测到大变化 + OCR 命中 "YOU DIED"
          音频信号检测到 BGM mood = "death"
t=100ms   事件总线收到 [hud:death, audio:bgm_death]，importance=0.95
t=120ms   沉默规则放行（损友人格 cooldown 已过，死亡是高优触发）
t=130ms   DeepSeek V4 Flash 预筛：should_speak=true（耗时 200ms）
t=350ms   DeepSeek V4 Pro 生成台词（含最近 3 帧 + 记忆）
t=1300ms  返回 {text: "哎呦，第几次了我都数不清了", emotion: smirk}
t=1320ms  TTS server 收到，GPT-SoVITS 流式合成（首块 300ms）
t=1620ms  首块 PCM 到达，写入 Mumble bot TX 队列
t=1640ms  Mumble bot Opus 编码 → 推到频道
t=1700ms  游戏 PC Mumble client 收到 → 解码 → 播放到玩家耳机
t=1750ms  VTube Studio 切换 "smirk" 表情；RMS → MouthOpenY
t=4500ms  TTS 播放完，记忆系统异步写入 observed.md
          —— 同时，t=2000ms 玩家按 PTT 说 "你才数不清" →
            Mumble 客户端发送 → bot RX 触发 → STT → 决策层下一轮
```

总体延迟 1.7s，主要在 LLM 调用（900ms）+ TTS 首块（300ms）+ Mumble 链路（80ms）。可接受。

---

## 5. 模块详细设计

### 5.1 采集层

| 模块 | 技术 | 频率 | 输出 |
|---|---|---|---|
| 屏幕 | `opencv-python` cv2.VideoCapture (DirectShow) 或 `pyav` | 30fps 可配 | numpy ndarray |
| 游戏音频 | `sounddevice` 读采集卡音频设备（USB Audio Class） | 50ms chunk | int16 PCM |
| 玩家麦 | **不直采**，由 Mumble bot RX 提供 | 事件驱动 | int16 PCM + 用户标签 |
| 弹幕 | `blivedm`（B 站）| 事件驱动 | DanmakuEvent |

**采集卡设备识别**：
- Windows 上采集卡注册为 DirectShow 视频设备和 USB Audio 设备
- 启动时枚举所有设备，按名称匹配（如 "Game Capture HD60 X"），用户在设置里选择
- 视频用 `pygrabber.dshow_graph.FilterGraph` 列设备，opencv 按设备 index 打开
- 音频用 `sounddevice.query_devices()` 列设备

### 5.2 本地信号层

#### 5.2.1 视频信号
- **SSIM/帧差分**：`skimage.metrics.structural_similarity` 比较前后帧，< 0.95 即"有变化"
- **CLIP 嵌入**（可选）：ViT-B/32（CPU 也能跑），512 维向量，用于"刚才看过的画面别重复"
- **HUD ROI 模板匹配**：每个游戏 profile 配置 ROI 区域 + 模板图，OpenCV `matchTemplate`
- **OCR**：`paddleocr`（中文好）或 `tesseract`

#### 5.2.2 音频信号（来自采集卡的游戏音）
- **RMS 包络**：音量突变 → 战斗 / 爆炸 / 重要时刻
- **频谱差分**：mel-spectrogram 对比，BGM 切换检测
- **BGM 状态机**：用低频特征聚类，分 calm / tense / boss / victory / death 等状态

#### 5.2.3 玩家语音触发（替代 v0 的唤醒词）

由 Mumble 自然驱动：
1. 玩家按 PTT，开始说话
2. Mumble bot RX 收到该用户的 audio chunks（pymumble 按用户分别回调）
3. Bot 累积音频
4. 用户停止 PTT（或检测 0.5s 静音）→ 把累积的 PCM 发给 STT
5. STT 返回文本，作为 `player_voice` 事件送入决策层

**好处**：
- 零唤醒词（PTT 是物理键，零误触发）
- 复用 Mumble 客户端已经调好的麦克风/降噪/增益
- 玩家正常和朋友聊天的肌肉记忆直接复用

### 5.3 决策层

#### 5.3.1 事件融合

```python
class FusedEvent:
    timestamp: float
    screen_change_score: float
    audio_change_score: float
    hud_events: list[str]              # ["death", "level_up", ...]
    ocr_text: str | None
    bgm_state: Literal["calm", "tense", "victory", "death", ...]
    player_voice: str | None           # Mumble + STT 来的文本
    danmaku_batch: list[DanmakuEvent]
    importance: float                  # 自动算
```

#### 5.3.2 沉默规则

```python
class SilenceController:
    def should_consult_llm(self, event: FusedEvent) -> bool:
        if event.player_voice: return True               # 玩家主动说话必应
        if event.hud_events & PRIORITY_EVENTS: return True
        if time.time() - self.last_speak < self.cooldown: return False
        if event.importance < self.threshold: return False
        if self.context.is_boss_focused and self.persona.silence_during_boss:
            return False
        if self.context.is_cutscene and self.persona.silence_during_cutscene:
            return False
        return True
```

#### 5.3.3 LLM 调用策略（两段）

```python
# 段 1: DeepSeek V4 Flash 二分类（~$0.001）
gate = flash.invoke(
    system=PERSONA_GATE_PROMPT,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": last_thumbnail_256},
        {"type": "text", "content": event_summary},
    ]}],
    max_tokens=50,
)

if gate["should_speak"]:
    # 段 2: DeepSeek V4 Pro 生成
    response = pro.invoke(
        system=full_persona_prompt + memory_files,   # 稳定前缀触发上下文缓存
        messages=[{"role": "user", "content": [
            {"type": "image", "source": frame_now},
            {"type": "image", "source": frame_n2},
            {"type": "image", "source": frame_n4},
            {"type": "text", "content": event_summary},
        ]}],
        max_tokens=200,
    )
    # 输出: {text, emotion, vts_expression, memory_update}
```

**上下文缓存**：DeepSeek API 对相同前缀的输入自动命中缓存（KV cache hit），system prompt（人格 + 记忆）保持稳定前缀以最大化命中率，输入成本显著降低。

#### 5.3.4 记忆读写
- **读**：每次 DeepSeek V4 Pro 调用前组装 `global/player.md` + 当前游戏的 `identity.md` + `observed.md`（最近 30 条）+ `jokes.md` + `progress.md`，总 token < 4K
- **写**：V4 Pro 输出的 `memory_update` 字段异步追加到 `observed.md`
- **会话末**：游戏退出或人格切换时，V4 Flash 把当日小结写 `sessions/<date>.md` 并更新 `highlights.md` / `progress.md`

### 5.4 输出层

#### 5.4.1 TTS 客户端
- 通过本机 WS 连 TTS server（`ws://127.0.0.1:7001`）
- 流式接收 48kHz mono int16 PCM chunks
- PCM 同时分发到：
  - Mumble bot TX 队列（推给玩家）
  - OBS 音频 tap（虚拟音频设备或直接 `obs-websocket` 注入；直播时启用）

#### 5.4.2 Mumble Bot 输出
- pymumble `sound_output.add_sound(pcm_bytes)`，内部处理 Opus 编码 + 包发送
- AI 音量调节由玩家在 Mumble 客户端右键 bot 用户做（无须 ducking 代码）

#### 5.4.3 VTube Studio 控制
- WS 连本机 `ws://127.0.0.1:8001`，首次启动请求权限令牌（持久化保存）
- 发送 `InjectParameterDataRequest`：
  - `MouthOpenY`：TTS 音频 RMS（每 50ms）
  - `MouthForm`：中文五元音粗映射
  - `FaceAngleX/Y/Z`：idle 微动画
  - 表情参数：按 V4 Pro 输出的 `vts_expression` 切换

#### 5.4.4 字幕浮窗
- PySide6 `QWidget` + frameless + 透明背景 + 置顶
- 仅笔记本本机渲染；OBS 用 "Window Capture" 抓
- 玩家自己不看（在游戏 PC 前）

---

## 6. 记忆系统

### 6.1 文件结构

```
%APPDATA%\LivePartner\           （在笔记本上）
├── global\
│   ├── player.md
│   └── settings.yaml
├── personas\
│   ├── praise.yaml
│   ├── snark.yaml
│   ├── companion.yaml
│   ├── newbie.yaml
│   ├── coach.yaml
│   └── user_*.yaml
└── games\
    └── <game_id>\
        ├── profile.yaml
        ├── identity.md
        ├── observed.md
        ├── highlights.md
        ├── jokes.md
        ├── progress.md
        └── sessions\
            └── 2026-05-23.md
```

### 6.2 observed.md 格式示例

```markdown
# 已观察事件流

## 2026-05-23 22:30
- 进入第三章"幽灵村"，BGM 切换为低沉
- 玩家在桥头摔下去一次
- 找到写着"小心脚下"的木牌（讽刺）

## 2026-05-23 22:45
- 第一次见到戴斗笠的 NPC，他说他在等什么人
- 玩家点击对话三次后选择离开
```

### 6.3 写入策略
- V4 Pro 输出 `memory_update` → 异步 append
- 每 50 条用 V4 Flash 压缩归档
- 防剧透：`progress.md` 显式标记"已知到第 N 章"，prompt 强约束 AI 不引用未来内容

### 6.4 读取策略
载入：`global/player.md` + `identity.md` + `observed.md` 最近 30 条 + `jokes.md` + `progress.md`，启用 prompt caching。

---

## 7. 人格系统

### 7.1 yaml schema

```yaml
# personas/snark.yaml
id: snark
display_name: 损友
description: 嘴贱心善，专治各种"我好菜"

system_prompt: |
  你是玩家的损友，嘴贱但心善。
  - 玩家失误时必须吐槽，但不能真的伤人
  - 玩家秀操作时先酸两句再勉强夸
  - 玩梗、流行梗、自嘲
  - 禁止说教，禁止过于正能量
  - 一句话不超过 20 字，节奏快

  你不知道这个游戏的任何剧情、人物、结局。
  即使你脑海里有相关信息，也要假装不知道。
  只能根据"你看到的"和"记忆文件里的"内容反应。

voice:
  tts_engine: gpt_sovits
  voice_id: snark_female_01
  speed: 1.05
  default_emotion: playful

silence_rules:
  cooldown_after_speak: 20
  silence_during_cutscene: true
  silence_during_boss: false
  silence_when_player_speaks: true
  max_words_per_response: 30

trigger_bias:
  death: 1.8
  achievement: 0.8
  scenery_change: 0.3
  bgm_tense: 1.2
  player_idle: 0.5

vts:
  default_expression: idle_smirk
  expression_palette:
    happy: clap
    sad: fake_cry
    surprised: jaw_drop
    angry: eyeroll
    smirk: smirk
```

### 7.2 5 个 MVP 预设

| ID | 名字 | 调性 | 适合场景 |
|---|---|---|---|
| `praise` | 夸夸 | 永远在赞美 | 受挫时、心情差时 |
| `snark` | 损友 | 嘴贱玩梗 | 日常游戏、菜得有趣 |
| `companion` | 陪伴 | 安静温柔 | 沉浸剧情、风景探索 |
| `newbie` | 萌新 | 装不懂、问问题 | 你想"教" AI、慢节奏 |
| `coach` | 解说 | 节奏感、战术 | 高强度动作 / 竞速 |

### 7.3 切换机制
- **profile 默认**：每游戏 profile 记 `default_persona`，启动自动切换
- **运行时切换**：
  - 热键（在笔记本上，因为玩家不在那）→ 改成"在 Mumble 频道发 `/persona snark` 文本指令"，bot 解析
  - 笔记本上的浮窗按钮（主播视角，自己看见）
  - 语音指令："换损友模式"（走 STT 路径）
- 切换时 cooldown 重置

### 7.4 用户自定义
- 浮窗"克隆 + 编辑"按钮
- 任一预设拷贝为 `user_*.yaml`，用户文本编辑器改 prompt / 音色 / 规则
- 热加载（文件 mtime 监听）

---

## 8. 沉默规则（四层叠加）

```
全局规则（settings.yaml）
  ↓
游戏 profile 规则（games/<id>/profile.yaml）
  ↓
人格规则（personas/*.yaml）
  ↓
运行时上下文（is_cutscene、is_boss_focused、time_since_last_speak）
```

冲突解决：**保守原则**，任意一层说"沉默"就沉默。例外：玩家 PTT 主动说话覆盖所有沉默。

---

## 9. 直播搭子模式

### 9.1 启用
笔记本上 LivePartner 浮窗的"直播模式"开关。启用后：
- 连接弹幕源
- 启用弹幕聚合
- 切换沉默规则到"广播向"配置（避免敏感词、节奏更主动）

### 9.2 弹幕处理

```python
class DanmakuAggregator:
    """每 30 秒聚合一次弹幕，决定是否回应"""
    def aggregate(self, batch: list[DanmakuEvent]) -> AggregatedDanmaku | None:
        direct = [d for d in batch if self._is_direct_address(d)]
        if direct: return self._compose_direct(direct)
        priority = [d for d in batch if d.is_super_chat]
        if priority: return self._compose_priority(priority)
        keywords = self._extract_keywords(batch)
        hot = [k for k, c in keywords.items() if c >= 3]
        if hot: return self._compose_topic(hot, batch)
        return None
```

### 9.3 OBS 集成（全部在笔记本）

```
OBS 场景（笔记本）
├── 视频源：采集卡（DirectShow，已经在用了，零额外成本）
├── Live2D：VTube Studio Spout2 输出
├── AI 字幕：Window Capture，抓 LivePartner 浮窗
├── AI 音频：直接读 TTS PCM tap（不绕 Mumble，零延迟、零损耗）
└── 弹幕监控（可选）：浏览器源指向弹幕展示页
```

**注意**：观众听到的 AI 音频走"本机 PCM tap"路径，不经过 Mumble，这样：
- 不受 Mumble 编解码损失
- 不受网络抖动影响
- AI 音量可以单独在 OBS 调，与玩家自己听到的音量独立

---

## 10. 笔记本一机部署

### 10.1 服务清单

| 服务 | 端口 | 协议 | VRAM | 状态 |
|---|---|---|---|---|
| Mumble Server | 64738 (UDP/TCP) | Mumble protocol | 0 | 常驻 |
| LivePartner 主程序 | — | — | <500MB | 启动 |
| LivePartner Mumble Bot（子进程） | — | — | 0 | 跟随主程序 |
| TTS Server (GPT-SoVITS) | 7001 | WS (流式 PCM) | ~3GB | 常驻 |
| STT Server (faster-whisper large-v3) | 7002 | HTTP | ~3GB | 按需加载 |
| 可选 VLM 预筛 (MiniCPM-V int4) | 7003 | HTTP | ~3GB | 按需，与 STT 互斥 |
| VTube Studio | 8001 (WS API) | VTS Plugin API | ~1GB | 常驻（GUI 应用） |
| OBS | — | — | ~1GB (NVENC) | 仅直播时启 |

**VRAM 计算（4070 Laptop = 8GB）**：
- 常驻基线：TTS 3GB + VTS 1GB = 4GB
- 加 STT 按需：4GB + 3GB = 7GB（紧但够）
- 直播时加 OBS：+1GB → 8GB（极限，需关 VLM 预筛）
- VLM 预筛不与 STT 同时加载；直播时直接禁用，预筛改走 DeepSeek V4 Flash 远程

**结论**：4070 8GB 跑得下，但 VLM 预筛建议不上本地，用 DeepSeek V4 Flash 远程做。

### 10.2 TTS 服务接口

```
WS ws://127.0.0.1:7001/tts
→ {"text": "哎呦，第几次了", "voice_id": "snark_female_01",
   "speed": 1.05, "emotion": "playful"}
← binary frame: 48kHz mono int16 PCM, ~200ms chunks
← {"type": "end"}
```

### 10.3 STT 服务接口

```
POST http://127.0.0.1:7002/transcribe
body: audio/wav, 48kHz mono int16
→ {"text": "...", "language": "zh", "duration": 2.3}
```

### 10.4 Mumble 服务配置

```ini
# /etc/mumble-server.ini 或 murmur.ini
serverpassword=...                    # 防止外人乱进
channels=
  - name: LivePartner
    description: AI 陪玩频道
    position: 0
welcometext=""
```

LivePartner 启动时自动确保 "LivePartner" 频道存在，bot 加入此频道。

### 10.5 LivePartner 启动顺序
```
1. mumble-server 自启动（systemd / Windows Service）
2. TTS server 自启动（PM2 / NSSM）
3. VTube Studio 手动启（Steam 应用）
4. LivePartner 主程序：
   a. 连接 Mumble server，bot 进入频道
   b. 启动采集卡设备（视频 + 音频）
   c. 启动 UI
5. 直播时手动启 OBS
```

---

## 11. 成本与性能预算

### 11.1 一次性硬件
| 项 | 估价 |
|---|---|
| HDMI 采集卡（Elgato HD60 X / 圆刚 GC553 同档） | ¥800-1800 |
| 3.5mm 音频线（若需要） | ¥10-30 |

### 11.2 LLM 月成本（每天 2 小时游戏）
| 项 | 单价 | 频率 | 月成本 |
|---|---|---|---|
| V4 Flash 预筛 | 待核（≤ ¥0.001/次） | 60 次/小时 | ≤ ¥3-5 |
| V4 Pro 生成（含图） | 待核（≤ ¥0.05/次） | 15-25 次/小时 | ≤ ¥45-80 |
| 会话末总结（V4 Flash） | 待核（≤ ¥0.01/次） | 2 次/天 | ≤ ¥0.6 |
| **合计** | | | **预估 < ¥50/月** |

DeepSeek 上下文缓存命中后输入成本再降 50%+；上表数字以 Anthropic 等价模型为上限参考，实际按 DeepSeek 当期价目重核。

### 11.3 本地资源
| 项 | 游戏 PC | 笔记本 |
|---|---|---|
| CPU | < 2%（仅 Mumble client） | 20-35% |
| GPU | 0 | 40-70%（TTS + VTS + OBS） |
| 内存 | < 100MB（Mumble client） | 6-10GB |
| 带宽 | 局域网采集卡 USB ~50Mbps + Mumble ~100Kbps | 同左 |

游戏 PC 几乎没额外负担，不影响帧率。

---

## 12. 技术栈与项目结构

### 12.1 语言与框架

| 项 | 选型 |
|---|---|
| 主语言 | Python 3.11+ |
| UI | PySide6（主窗 + 字幕浮窗） |
| 异步 / 进程通信 | asyncio + multiprocessing + zmq |
| 包管理 | uv + pyproject.toml |
| 打包 | PyInstaller（自用即可） |

### 12.2 关键依赖

```
# 采集
opencv-python              # 采集卡 DirectShow 视频
pygrabber                  # DirectShow 设备枚举
sounddevice                # 采集卡音频
pyav                       # （备选）ffmpeg 后端

# 语音传输
pymumble                   # Mumble bot 客户端

# 信号 / CV
numpy
scikit-image               # SSIM
paddleocr                  # OCR
open_clip_torch            # 可选：CLIP 嵌入

# 网络
openai                     # DeepSeek API（OpenAI 兼容客户端）
httpx                      # HTTP client
websockets                 # WS（TTS / VTS）

# UI
PySide6

# 弹幕（直播模式可选）
blivedm                    # B 站

# 服务端（独立 venv 或 docker）
gpt-sovits 或 bert-vits2   # TTS
faster-whisper             # STT
mumble-server              # 系统包安装，非 Python
```

**已砍掉**（vs v0）：`dxcam`、`pyaudiowpatch`、`pvporcupine`、VB-Audio Cable 驱动安装、Picovoice Console 注册。

### 12.3 项目结构

```
LivePartner/
├── SPEC.md
├── pyproject.toml
├── README.md
├── src/livepartner/
│   ├── __main__.py                   # 入口
│   ├── config.py                     # settings.yaml
│   ├── capture/
│   │   ├── card_video.py             # opencv + DirectShow
│   │   ├── card_audio.py             # sounddevice
│   │   └── device_picker.py          # 启动时枚举 + 选设备
│   ├── signals/
│   │   ├── frame_diff.py
│   │   ├── hud_detector.py
│   │   ├── audio_features.py
│   │   └── bgm_state.py
│   ├── voice/
│   │   ├── mumble_bot.py             # pymumble 封装
│   │   ├── tts_client.py
│   │   └── stt_client.py
│   ├── decision/
│   │   ├── event_bus.py
│   │   ├── fusion.py
│   │   ├── silence.py
│   │   ├── llm.py
│   │   └── memory.py
│   ├── persona/
│   │   ├── loader.py
│   │   └── prompt_builder.py
│   ├── output/
│   │   ├── vts_client.py
│   │   ├── subtitle_overlay.py
│   │   └── obs_audio_tap.py
│   ├── danmaku/
│   │   ├── bilibili.py
│   │   └── aggregator.py
│   └── ui/
│       ├── main_window.py            # 设置 + 状态
│       └── overlay.py
├── services/                         # 同机后台服务
│   ├── tts_server.py
│   ├── stt_server.py
│   └── mumble/                       # mumble-server 配置
├── personas/                         # 5 个预设
├── games/                            # 玩家数据，gitignore
└── tests/
```

---

## 13. MVP 路线图

### M1 主链路（2-3 周）
**目标**：开启游戏，AI 能"看到" + "说出"合理的话，玩家通过 Mumble 听到。

- [ ] 项目骨架（pyproject.toml + 配置加载 + 日志）
- [ ] 采集卡视频 + 音频抓取（opencv + sounddevice）
- [ ] Mumble bot：连接服务器、进入 LivePartner 频道、能 TX PCM
- [ ] SSIM 触发 → DeepSeek V4 Pro vision → 文本输出
- [ ] Edge TTS（先不部署 GPT-SoVITS）→ PCM → Mumble bot TX
- [ ] 最简 PySide6 主窗（开关 + 实时日志）

**验证**：玩 30 分钟游戏，主观打分每条 AI 发言的合理性。

### M2 智能化 + 记忆（2-3 周）
- [ ] V4 Flash 预筛
- [ ] 沉默规则系统（cooldown + cutscene 检测）
- [ ] HUD 模板匹配（先做 1 个游戏的死亡检测）
- [ ] 记忆系统 observed.md 自动写入
- [ ] 会话末 V4 Flash 总结
- [ ] 游戏 profile 切换

**验证**：连玩 3 天，看记忆是否在第二天被合理引用。

### M3 双向语音 + 形象（2-3 周）
- [ ] GPT-SoVITS TTS 服务（替换 Edge）
- [ ] faster-whisper STT 服务
- [ ] Mumble bot RX：玩家 PTT 说话 → STT → 决策层
- [ ] VTube Studio 集成（表情 + 口型）
- [ ] 字幕浮窗
- [ ] 5 个人格 yaml + 切换机制

**验证**：能像和朋友自然对话；Live2D 形象表情贴合 AI 情绪。

### M4 直播搭子（1-2 周）
- [ ] B 站弹幕接入 + 聚合
- [ ] OBS 场景模板（采集卡 + VTS + 字幕浮窗 + AI 音频 tap）
- [ ] AI 音频本地 PCM tap → OBS（不绕 Mumble）
- [ ] 直播模式开关 + 沉默规则切换

**验证**：开一场 1 小时直播自用，体验是否丝滑。

### M5+（持续）
- 抖音弹幕
- AI 心情 / 状态系统
- 多游戏 HUD 模板共享 / 众包
- Mumble 多人陪玩（朋友进频道一起玩，AI 区分用户）
- 移动端控制面板（手机调人格 / 音量 / 静音）

---

## 14. 风险与未决

### 14.1 已识别风险

| 风险 | 缓解方案 |
|---|---|
| DeepSeek V4 Pro vision 偶发延迟 > 3s | 5s timeout，超时跳过本轮 |
| GPT-SoVITS 进程崩溃 | systemd / NSSM `restart=always` |
| **Mumble bot 掉线** | pymumble 内置重连；UI 显示连接状态 |
| **采集卡驱动异常**（黑屏 / 无音） | 启动自检 + 设备重选 UI；记录最后健康设备 |
| **HDMI 音视频不同步**（部分采集卡） | 音频采集时记录时间戳，必要时手动 offset 校准 |
| **采集卡限制刷新率**（如 1080p120 → 1080p60） | SPEC 推荐型号验证过 120fps；用户选其他卡需自测 |
| PTT 按下太短被截断 | Mumble bot 端 buffer 至少 500ms 完整音频再发 STT |
| LLM 输出脏话 / 敏感词（直播） | 输出层敏感词过滤 + 人格 prompt 约束 |
| 记忆文件无限增长 | 每 50 条 V4 Flash 压缩 + 老条目按月归档 |
| Mumble 频道里同时有朋友 + AI bot，AI 该回谁 | bot 默认只响应"被点名"或"直接对话"；详见 §14.2 未决 |

### 14.2 未决（需要后续讨论）

- **AI 在频道里如何区分对话对象**：玩家自言自语 / 玩家和朋友说话 / 玩家和 AI 说话，bot 怎么判断要不要插嘴？候选：用关键词触发（"小伙伴"开头）/ 第二个 PTT 键专给 AI / 显式频道分离
- **AI 心情 / 状态系统**：AI 自己有"心情曲线"还是纯反应式？
- **跨游戏角色一致性**：换游戏后是"同一个 AI"还是被重置？
- **多语言**：v1 中文优先，英文什么时候做
- **TS / Mumble 共存**：用户原 TS 服务器是否保留？AI 仅在 Mumble，不在 TS

---

## 附录 A：术语
- **HUD**：Heads-Up Display
- **ROI**：Region of Interest
- **SSIM**：Structural Similarity Index
- **PTT**：Push-to-Talk
- **VTS**：VTube Studio
- **NVENC**：NVIDIA 硬件视频编码器
- **DirectShow**：Windows 媒体框架，采集卡用此接入

## 附录 B：参考实现 / 借鉴对象
- pymumble 官方示例（music bot）
- SinusBot（TS 音乐 bot，架构参考）
- VTube Studio + Bert-VITS2 的 VTuber 工具链
- Inworld AI（游戏 NPC 性格）

## 附录 C：v0 → v0.2 主要变化对照

| 模块 | v0 | v0.2 |
|---|---|---|
| 部署 | 游戏机装主软件 + 服务机装 TTS/STT | 笔记本一机部署，游戏 PC 仅 Mumble client |
| 屏幕采集 | 游戏 PC DXGI | 笔记本 opencv + 采集卡 |
| 游戏音频 | 游戏 PC WASAPI Loopback | 笔记本 sounddevice + 采集卡 USB 音频 |
| 玩家麦克风 | 游戏 PC sounddevice + Porcupine 唤醒词 | 玩家 PTT → Mumble client → bot RX |
| AI 音频输出 | 游戏 PC 虚拟声卡（VB-Cable）+ ducking | Mumble bot TX → 玩家 Mumble client |
| OBS 集成 | 游戏 PC OBS + 跨机数据传输 | 笔记本 OBS + 本机所有源 |
| 唤醒词依赖 | Porcupine（Picovoice 注册） | 砍掉（PTT 替代） |
| 虚拟声卡依赖 | VB-Audio Cable | 砍掉 |
| 多人陪玩 | 不支持 | 天然支持（朋友进 Mumble 频道） |
