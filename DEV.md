# DEV.md — 本地开发笔记

给我自己看的运行手册。环境变量、各个脚本、调试技巧、踩过的坑。

---

## 目录结构 (在 `D:\LivePartner` 下)

```
.env                   # API key + 模型 ID (gitignored)
.memory/               # 运行时数据 (gitignored)
   global/player.md       玩家档案 (跨游戏)
   games/default/         本游戏的记忆 (identity/jokes/progress/auto/observed/sessions)
   identity/Eri.crt|.key  持久化 Mumble 客户端 TLS 证书
   vts_token.txt          VTS 插件授权 token (一次性授权后这里持久化)
   ui.log                 UI 实时日志 (用来发给我看)
   tts_server.log         Qwen3-TTS 子进程日志
personas/
   eri.yaml               默认人格
   voices/eri.pt          eri 的声音克隆 (gitignored, 自己训)
external/                # 第三方子仓库 (gitignored)
   faster-qwen3-tts/      Qwen3-TTS 本地推理 (自带 venv)
.venv/                 # 主 venv (uv 管,gitignored)
```

---

## 环境变量 (.env)

```bash
LP_LLM_API_KEY=sk-xxx                                    # 必填,DashScope key
LP_LLM_BASE_URL=https://dashscope.aliyuncs.com/...       # 必填
LP_MODEL_FLASH=qwen3.6-flash                             # gate/director/fact/scene/self
LP_MODEL_PRO=qwen3.6-flash                               # 主 generate (我现在两个都用 flash)
```

阶梯计费下,单 LLM 调用都 < 256k token, 永远走第一档 (¥1.2/M 输入)。8 小时挂机 ~¥22。

---

## Windows `.bat` 启动脚本

| 脚本 | 用途 |
|---|---|
| `eri.bat` | 启 LivePartner UI (`.venv\Scripts\python.exe -m livepartner`) |
| `mumble.bat` | 启本地 Mumble 服务器 (Murmur),cwd 设为项目根,sqlite 和 log 落在项目里 |
| `cleanup.bat` | 杀掉所有 LivePartner 相关的孤儿进程 (`python.exe` 路径或 cmdline 含 `D:\LivePartner` 的,排除自身 PID),显示 nvidia-smi VRAM 情况 |
| `cleanup.bat -mumble` | 额外杀 mumble-server.exe |
| `cleanup.ps1` | cleanup.bat 调用的 PowerShell 实现,纯 ASCII 避免 PS 5.1 GBK 编码坑 |

**bat 文件必须是 CRLF 行尾 + UTF-8 无 BOM**。LF-only 会触发 "M not recognized" 错误。

---

## 启动顺序 (生产)

1. **mumble.bat** (服务器)
2. **eri.bat** (UI)
3. UI 里点 **一键启动** → 按顺序拉起:
   - capture (采集卡)
   - TTS server (Qwen3-TTS 子进程,等 60-90s 加载完)
   - STT (faster-whisper-large-v3-turbo,~3s)
   - Mumble 客户端连 (127.0.0.1:64738)
   - 全绿之后 ready
4. **VTS 那边**手动启动 + 加载模型 + 设置 → API → Start API
5. Live2D tab → 点连接 (首次授权 VTS popup → Allow,之后自动)
6. 玩家 Mumble 客户端连进同一个 Murmur,加入 Root 频道

---

## 关键 log tag

| Tag | 来源 |
|---|---|
| `[gate]` | ambient gate 判断 |
| `[generate]` | 主 LLM 调用耗时 |
| `[director]` | Live2D Director 选表情 |
| `[qwen3]` | TTS server 流式 PCM (TTFB / RTF) |
| `[mumble]` | TX 队列状态 |
| `[stt]` | STT 转写 + drop 原因 |
| `[scene]` | 场景描述 |
| `[fact]` | 自动事实抽取 |
| `[self]` | 视觉自我感知 |
| `[vts]` | VTS API 调用 |
| `[idle]` | IdleMotion (眨眼/头摆) |
| `[memory]` | 注入字符数 |
| `[reroll]` | 反复读检测触发 |
| `[弹幕]` | Bilibili 弹幕到达 |
| `[cancel]` | 关闭/取消路径 |

实时输出在 UI 底部 log 框,**同时写到 `.memory/ui.log`**,排错时把这个文件贴上来即可。

---

## TTS 子进程

`services/qwen3_tts_server.py` 是 FastAPI 包装,被 `_tts_server_proc.py` 当子进程拉起。

- 监听 `127.0.0.1:7001` (默认)
- `POST /tts` 接 `{persona_id, text, language, chunk_size}` 返回流式 PCM
- `GET /voices` 列出已加载的 `.pt` embedding
- 子进程 spawn 时用 `-u` + `PYTHONUNBUFFERED=1` 让 log 实时输出
- Windows 上 `CREATE_NEW_PROCESS_GROUP + CREATE_NO_WINDOW`,关 UI 时 SIGTERM 子树

如果 TTS 假死, `cleanup.bat` 直接杀,然后重启。

---

## VTS / Live2D 坑

- **VTS 必须可见**：我们用 PIL.ImageGrab 抓屏幕区域,VTS 被遮挡的话 SelfDescriber 拍到的是遮挡物
- **表情用 ExpressionActivationRequest, 不是 hotkey trigger**：toggle hotkey 会出现"再按一次自动关掉"的陷阱
- **连接成功后会自动清空所有 active 表情**：避免之前 session 残留叠加
- **Type-H3 模型 Body X/Y/Z 绑到了 FaceAngleX/Y/Z**：head sway 同时驱动了身体,不需要单独 BodyAngleX
- **Breath 是 VTS Auto-breathing 类型**：requestTrackingParameterList **不会**列出,我们的 force-add 路径已经取消(会让 bulk set 整体被 reject 导致眨眼/头摆都停)
- **`init_hotkey_name="Eri"`**：连接后自动触发名叫 "Eri" 的 hotkey,应用外观预设;Live2D tab UI 里可改名

---

## Mumble 坑

- Python 3.12 + pymumble 1.6 用了 `ssl.wrap_socket` (已 deprecated) — 用 `_mumble_cert.py` 注入 SSLContext shim
- 自签 RSA-2048 客户端证书存 `.memory/identity/Eri.crt|.key` — 持久化用户音量设置
- 本地 Murmur 用 `ssl.CERT_NONE` (开发用,不要生产)

---

## STT 坑

- **whisper 幻觉**：cough/breath 经常被识别成 "Thank you" / "Bye" — 黑名单见 `_HALLUCINATION_PHRASES` + 短英文兜底 heuristic (`_looks_like_en_filler`)
- **AI 还在说时玩家说话**：不打断,**排队**到 `pending_transcribed`,清理时 drain
- **default 模型** `large-v3-turbo` (~3s 加载 / ~300ms 推理 / 4GB VRAM)

---

## 调试技巧

- **看 .memory/ui.log**：UI 已经把所有 `[tag]` 行写进去了,直接 `tail -f` 或贴给 Claude 看
- **看 .memory/tts_server.log**：TTS 子进程的输出
- **手动 trigger LLM**：运行 tab "事件" 框填一句话 + 勾 "force speak" + 点触发 → 跳过 gate 直接走 generate
- **手动 trigger 表情**：Live2D tab 手动测试下拉 + 触发按钮
- **手动 trigger 字幕**：勾"启用字幕浮窗"后,toggle 会弹一句占位文字 2.5s,用来定位浮窗位置
- **Qt UI 闪退**：通常是显存爆,看 `nvidia-smi`,跑 `cleanup.bat` 清孤儿
- **VTS auto-blink 突然停**：检查 face tracking 设置,如果摄像头停掉 VTS 也会停眨眼,我们的 IdleMotion 会接管

---

## 重启 vs 不重启

- 改 Python 代码 → 必须重启 UI
- 改 `personas/eri.yaml` → 下一次 turn 自动重读,不用重启
- 改 `.memory/global/player.md` 或其它 md → 下一次 memory 渲染自动读到
- 改 `.env` → 必须重启 (settings 启动时加载)
- VTS 里手动换衣服/换发型 → 下次 SelfDescriber 90s 周期会检测到,**不用**重启

---

## Token 优化思路 (没做)

- prompt cache (DashScope 显式缓存,首次 ¥1.5 之后 ¥0.12,8h 挂机可省 80%)
- AFK 自动暂停 (30 分钟没玩家交互停 ambient/scene)
- scene describer fingerprint 阈值收紧 (现在每 5s 触发, 80% 被 SAME 跳过,但图 token 已经付了)
- gate 缩略图分辨率降 (256 → 192)

不急。8h 现在 ¥22 已经够便宜。

---

## 已知 pending

- E2E 验证 (task #31) — 一直没系统跑 checklist
- "外观预设 hotkey" 默认值 "Eri" 是硬编码,可以挪到 eri.yaml 里
- 弹幕过滤还很粗,只有冷却 + 最短长度,没区分订阅者 / SC / 普通弹幕
- Live2D 表情贴图层 + 面部参数层之间的 idle decay 时机可能错位,8s 都按同一个 timer
