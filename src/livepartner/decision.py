"""Two-stage decision pipeline per SPEC §5.3.3.

flash gate (small thumbnail + event, GENERIC system) → bool
   ↓ if yes
pro vision generate (full frame + full persona + memory + event) → AI line

Per SPEC §5.3.2 we also allow `force_speak=True` for high-priority events
(death, achievement, player voice) — these skip the LLM gate entirely.

Output format: model emits a single Chinese line + an EMO prosody hint.
The Chinese line is used as both subtitle AND TTS input (mono-language pipeline,
Chinese voice via cross-lingual cloning on the configured voice embedding).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import flash_text, flash_vision, pro_text, pro_vision
from .persona import Persona


GATE_SYSTEM_BASE = (
    "你是一个二分类器。看一张游戏画面缩略图 + 一句事件描述，"
    '判断 AI 陪玩此刻是否应该开口说话。只回答 "yes" 或 "no"，不要解释。'
)

_GATE_HINT_BY_STYLE = {
    "chatty": (
        "\n这个 AI 性格偏活泼,愿意主动聊天但也不是嘴停不下来。\n"
        "判断标准:画面里有**明显新内容**(场景切换/新角色/UI 数字明显变化/玩家有动作)答 yes,"
        "画面**几乎没变**(同一界面/同一菜单/光标小幅移动)答 no —— "
        "AI 已经评论过的同类画面再次出现也答 no,留点空间给玩家。\n"
        "目标:每 30-60 秒说 1 句,而不是每 tick 都说。"
    ),
    "balanced": (
        "\n判断标准:画面平淡、事件平庸答 no；"
        "画面有戏剧性变化或事件值得反应答 yes。"
    ),
    "quiet": (
        "\n这个 AI 性格偏沉默,默认不打扰玩家。\n"
        "判断标准:**默认答 no**。只在画面有强戏剧性变化"
        "(死亡、boss 出现、重大场景切换、玩家明显失误或秀操作)时答 yes。"
    ),
}


def gate(
    event_summary: str,
    thumbnail_b64: str,
    persona: Persona,
    *,
    mime: str = "image/png",
) -> bool:
    style = getattr(persona, "chat_style", "balanced")
    hint = _GATE_HINT_BY_STYLE.get(style, _GATE_HINT_BY_STYLE["balanced"])
    out = flash_vision(
        prompt=f"事件: {event_summary}\n要说话吗?",
        image_b64=thumbnail_b64,
        system=GATE_SYSTEM_BASE + hint,
        mime=mime,
        max_tokens=10,
        extra_body={"enable_thinking": False},
    )
    answer = out.strip().lower()
    return answer.startswith("yes") or answer.startswith("是")


@dataclass(frozen=True)
class AiReply:
    text: str          # the Chinese line — used as both subtitle and TTS input
    tts_text: str      # same as text in mono-Chinese pipeline; kept for API stability
    raw: str           # full model output for debugging
    tts_instruct: str = ""  # natural-language prosody hint for TTS, empty = neutral
    rerolled: int = 0  # how many extra LLM calls happened (0 = clean first try,
                       # 1 = first try detected as repeat, reroll produced output,
                       # 2 = reroll also detected as repeat → silence)


_OUTPUT_FORMAT = (
    "\n# 输出格式（最高优先级，严格遵守）\n"
    "**只输出 1 行中文台词**,不带任何前缀、标签、引号、markdown、解释。\n"
    "不要写 'LINE:'/'EMO:' 之类的前缀,不要换行,就一句话。\n\n"
    "# 语言规则\n"
    "- 台词**以中文为主**,无论玩家用什么语言对你说话\n"
    "- 自然、口语化的中文。可以带角色调性,但**不要用日式口头禅的中文转写**\n"
    "  (不要写「诶~」「哼」「呐」「哇——」「啊咧」「真是的」「笨蛋」之类的语气词起头)\n\n"
    "# 名字写法（重要 —— TTS 读音）\n"
    "- 提到**自己的名字**时,写成 **エリ** (片假名,TTS 会用日语发音)\n"
    "- 提到**玩家的名字**时,写成 **似曾** (中文,TTS 按中文 \"sì céng\" 读)\n"
    "- 不要写 Eri / 艾莉 / 伊利 / NiseMono / ニセモノ\n"
    "- 不需要硬塞名字,中文里 \"我\"/\"你\" 自然就用 \"我\"/\"你\"\n\n"
    "# 语气词限制（最高优先级 —— 严禁违反）\n"
    "- **严禁**用语气词起头或结尾(「诶」「哎」「啊」「哦」「呀」「嗯」「哼」"
    "「哇」「呐」「呃」「咦」「唉」「啧」「呵」「嗤」「唔」「哈」「嘿」「嗨」等)\n"
    "- 即使想表达不爽/嫌弃/嘲讽,也**不能**用 `啧,` / `呵,` / `哼,` 这种起头\n"
    "  - ❌ 「啧,JK 制服...」 ❌ 「呵,你这水平...」 ❌ 「哼,才不告诉你」\n"
    "  - ✓ 「JK 制服...」 ✓ 「你这水平...」 ✓ 「才不告诉你」\n"
    "- 这些字单独 TTS 出来音色像噪音,人格的不爽情绪靠**实词措辞**传递,不靠语气词\n"
)


# Fixed TTS prosody instruction. Eri's voice is supposed to feel slightly
# mechanical — a stable monotone with a hint of synthetic edge. Sent as the
# `instruct` field on every Qwen3-TTS call; the LLM no longer outputs EMO.
DEFAULT_TTS_INSTRUCT = "用略带机械感的语气说"


_LINE_RE = re.compile(r"^LINE\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)


# Defensive post-process: strip leading interjection chars even if the LLM
# slipped one in despite the prompt forbidding it. 哈/嘿/嗨 are excluded from
# auto-strip because they're commonly part of intentional laughter
# (「哈哈」/「嘿嘿」). 呵 is INCLUDED — single 呵 followed by content is almost
# always dismissive ("呵, 你这水平"); 呵呵 alone gets kept by the null-guard.
_BAD_STARTERS = "诶哎啊哦呀嗯哼哇呐呃咦唉啧嗤唔呵"
_LEADING_INTERJECTION_RE = re.compile(
    rf"^[{re.escape(_BAD_STARTERS)}]+[~\-—,，、!！?？.。 \t]*"
)


def _strip_leading_interjection(line: str) -> str:
    """Strip a leading interjection + its trailing punctuation if present.

    Conservative: returns the original line if stripping would leave it empty
    (don't delete a one-word legit reply like just "啊?").
    """
    m = _LEADING_INTERJECTION_RE.match(line)
    if not m:
        return line
    stripped = line[m.end():].lstrip()
    if not stripped:
        return line  # whole line was interjection — keep it rather than emit nothing
    return stripped


def _parse_response(raw: str) -> str:
    """Extract the AI's single Chinese line from a model response.

    The model is instructed to output just one line of dialog (no LINE: prefix
    needed any more). But we still tolerate legacy `LINE: ...` prefixed output
    + markdown fences for robustness — strip them and return the text.
    """
    cleaned = raw.strip().strip("`").replace("```", "").strip()
    if not cleaned:
        return ""
    extracted = ""
    # Tolerate legacy LINE: prefix if a model still emits it.
    for ln in cleaned.splitlines():
        m = _LINE_RE.match(ln.strip())
        if m:
            extracted = m.group(1).strip()
            break
    if not extracted:
        # No prefix — take the first non-empty line (model might emit multiple
        # by accident; we only want the first).
        for ln in cleaned.splitlines():
            s = ln.strip()
            if s:
                extracted = s
                break
    if not extracted:
        extracted = cleaned
    # Defensive: strip leading interjection ("啧,", "呵,", "哎呀" etc) even
    # though the system prompt forbids them — LLMs occasionally slip one in.
    return _strip_leading_interjection(extracted)


# Dialog-line repeat detection. Tuned more permissive than scene dedup
# (scene_similar uses 0.30/0.60 on bigram/charset) — dialog lines are shorter
# and even ~20% bigram overlap after normalization is a strong "you just said
# this" signal (you typically need 3-4 shared meaningful Chinese bigrams).
# Skip charset overlap entirely: long Chinese sentences naturally share many
# chars (的/了/呢/啊 + common nouns) so charset triggers too many false positives.
# False positives only cost one extra LLM call, missed repeats are visible to
# the user — bias toward more aggressive catching.
_DIALOG_BIGRAM_THRESHOLD = 0.20

# Stripped before comparing — these characters don't carry the meaning of the
# line. "桌面好乱" vs "桌面真乱呢" should compare the same after stripping.
_DIALOG_FILLERS = set("的了呢吗嘛啊哦呀嗯吧呐哼呐，。！？、…—~·\"'「」 ")


def _normalize_for_compare(s: str) -> str:
    return "".join(c for c in s if c not in _DIALOG_FILLERS)


def _dialog_similar(a: str, b: str) -> bool:
    """Decide if two dialog lines are 'effectively the same'.

    Used to detect when the LLM is about to repeat a recent AI line so we can
    re-roll. Higher precision than recall: a missed repeat just means we let it
    through; a false positive forces a re-roll which has cost (extra LLM call,
    possibly worse output). Threshold tuned conservatively.

    Hits on:
    - exact match (after strip)
    - one is contained in the other (after stripping fillers/punctuation)
    - char-level bigram Jaccard > 0.25, computed on the NORMALIZED strings so
      that "桌面，全是图标呢" vs "桌面全是图标吗" don't get penalized by their
      different fillers/punctuation.
    """
    a = a.strip()
    b = b.strip()
    if not a or not b:
        return False
    if a == b:
        return True
    an = _normalize_for_compare(a)
    bn = _normalize_for_compare(b)
    if not an or not bn:
        return False
    if an in bn or bn in an:
        return True
    if len(an) < 2 or len(bn) < 2:
        return False
    A = {an[i:i + 2] for i in range(len(an) - 1)}
    B = {bn[i:i + 2] for i in range(len(bn) - 1)}
    if not A or not B:
        return False
    union_size = len(A | B)
    if union_size == 0:
        return False
    return len(A & B) / union_size > _DIALOG_BIGRAM_THRESHOLD


def _find_dialog_match(line: str, recent: list[str]) -> str | None:
    """If line is too close to any of `recent`, return the matched past line.
    Otherwise None. Walks newest → oldest so the most recent match wins."""
    for past in reversed(recent):
        if _dialog_similar(line, past):
            return past
    return None


_CONVERSATION_NOTE_NO_FRAME = (
    "\n# 当前模式：语音对话（无画面）\n"
    "玩家正在通过 Mumble 和你直接说话,你看不到任何游戏画面 —— 不要假装看到画面、"
    "不要描述画面、不要把这条当成游戏事件触发。\n"
    "玩家的话已经写在用户消息里,**必须**直接回应那句话(回答问题/接话/搭腔)。\n"
    "人格里关于\"只在画面有变化时说话\"、\"大多数时候保持沉默\"之类的规则**临时失效** —— "
    "玩家点名你必须答,但语气、用词、性格调性继续按人格走。"
)

_CONVERSATION_NOTE_WITH_FRAME = (
    "\n# 当前模式：语音对话 + 可看画面\n"
    "玩家正在通过 Mumble 和你直接说话,**同时**你能看到当前游戏画面(随用户消息附图)。\n"
    "玩家的话已经写在用户消息里,**必须**直接回应那句话。如果玩家问到画面/游戏内容(\"你看到什么\"、"
    "\"这是哪儿\"、\"我该往哪走\"…)就**真的看图回答**;如果只是闲聊,可以提一嘴画面也可以不提。\n"
    "人格里关于\"只在画面有变化时说话\"、\"大多数时候保持沉默\"之类的规则**临时失效** —— "
    "玩家点名你必须答,但语气、用词、性格调性继续按人格走。"
)


def _call_pro(
    *, prompt: str, system: str, frame_b64: str | None,
    mime: str, max_tokens: int, extra: dict,
) -> str:
    """Single helper for the pro_text / pro_vision LLM call. Same anti-rep knobs
    in both paths. Returns raw stripped output."""
    if frame_b64 is None:
        out = pro_text(
            prompt=prompt, system=system, max_tokens=max_tokens,
            extra_body=extra,
            temperature=0.95, frequency_penalty=0.6, presence_penalty=0.5,
        )
    else:
        out = pro_vision(
            prompt=prompt, image_b64=frame_b64, system=system,
            mime=mime, max_tokens=max_tokens, extra_body=extra,
            temperature=0.95, frequency_penalty=0.6, presence_penalty=0.5,
        )
    return out.strip()


def generate(
    event_summary: str,
    frame_b64: str | None,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    max_tokens: int = 200,
    is_conversation: bool = False,
    recent_ai_lines: list[str] | None = None,
) -> AiReply:
    """Stage 2: produce the actual line (Chinese, single line + EMO).

    If frame_b64 is None, runs a text-only call (no vision). Use this for
    player-voice triggers — feeding a synthesized YOU DIED frame to an
    unrelated conversation confuses the model.

    is_conversation=True adds a system note telling the model the input is a
    direct verbal address (overrides persona's "stay silent" rules).

    recent_ai_lines: last N AI lines (texts) to compare the output against.
    If the first LLM call produces something too close to any of them we make
    one re-roll attempt with an explicit "this is a repeat, pivot" instruction.
    If the re-roll also lands on a near-repeat we return empty text (silence is
    better than the third try wasting tokens AND still repeating). Pass None or
    [] to disable the check (used in tests / when there's nothing to compare).
    """
    system_parts = [persona.system_prompt]
    if memory:
        system_parts.append(f"\n# 记忆\n{memory}")
    if is_conversation:
        system_parts.append(
            _CONVERSATION_NOTE_WITH_FRAME if frame_b64 is not None
            else _CONVERSATION_NOTE_NO_FRAME
        )
    system_parts.append(_OUTPUT_FORMAT)

    system = "".join(system_parts)
    prompt = event_summary if is_conversation else f"当前事件: {event_summary}"

    # Disable chain-of-thought reasoning for the generate step — Qwen3.6 family
    # has thinking ON by default which adds 5-10× latency for what is
    # essentially a "produce one in-character line" task.
    extra = {"enable_thinking": False}

    # Anti-repetition knobs (in the LLM sampler): temperature 0.95 +
    # frequency_penalty 0.6 + presence_penalty 0.5. Plus the structural
    # anti-repeat memory section. The repeat-detect-and-reroll below is the
    # third layer on top of those — catches cases the model still slips through.
    raw = _call_pro(
        prompt=prompt, system=system, frame_b64=frame_b64,
        mime=mime, max_tokens=max_tokens, extra=extra,
    )
    line = _parse_response(raw)

    # Output-side anti-repeat: compare against last few AI lines, re-roll once
    # if too similar. Cheap (one extra LLM call) and only fires on actual repeats.
    if recent_ai_lines and line:
        match = _find_dialog_match(line, recent_ai_lines)
        if match:
            anti_repeat = (
                "\n\n# 紧急换话题指令（最高优先级）\n"
                f"你刚才打算说: 「{line}」\n"
                f"但这跟之前说过的 「{match}」 意思太接近 —— 这是复读。\n"
                "**这一轮必须换一个完全不同的角度**：关心玩家(吃饭/休息/几点了)、"
                "吐槽别的、引用长期记忆、或者直接沉默(输出空也行)。\n"
                "绝对不要再讨论刚才那个话题。"
            )
            raw2 = _call_pro(
                prompt=prompt, system=system + anti_repeat, frame_b64=frame_b64,
                mime=mime, max_tokens=max_tokens, extra=extra,
            )
            line2 = _parse_response(raw2)
            if line2 and not _find_dialog_match(line2, recent_ai_lines):
                return AiReply(
                    text=line2, tts_text=line2, raw=raw2,
                    tts_instruct=DEFAULT_TTS_INSTRUCT, rerolled=1,
                )
            # Reroll still hit a repeat (or came back empty) — silence beats
            # forcing a third try with the same context.
            return AiReply(
                text="", tts_text="", raw=raw2,
                tts_instruct="", rerolled=2,
            )

    # In the mono-Chinese pipeline subtitle text == TTS text always.
    return AiReply(
        text=line, tts_text=line, raw=raw,
        tts_instruct=DEFAULT_TTS_INSTRUCT,
    )


_FACT_EXTRACT_SYSTEM = (
    "你是一个长期记忆抽取器。给你一段游戏中的事件 + AI 的回复，"
    "判断这里有没有值得**长期**记住的事实。\n\n"
    "'值得长期记住'指（只算这些）:\n"
    "- 玩家自报的名字 / 外号\n"
    "- 玩家对游戏的稳定偏好或吐槽（e.g. '玩家讨厌 QTE'）\n"
    "- 反复的失败模式（e.g. '玩家在 boss A 死了 5 次'）\n"
    "- 重要剧情进度（e.g. '通关了第三章'、'选了 A 路线'）\n"
    "- 内部梗 / 反复出现的笑话\n\n"
    "**不要**抽取一次性的事件（例如普通对话、寒暄、单次死亡）。\n\n"
    "输出规则（严格）:\n"
    "- 有值得记的: 一行 markdown bullet 起手 `- `，简短中性，不超过 30 字\n"
    "- 没有: 只输出大写单词 `NONE`\n"
    "- 不要解释、不要其他任何字符"
)


_SCENE_DESCRIBE_SYSTEM = (
    "你是一个游戏画面描述器。看一张画面，用**一句中文(不超过 50 字)**描述。\n\n"
    "格式: <场景类型大类>, <1-2 个具体内容>\n\n"
    "**具体内容**要点(尽量从画面里读出原文):\n"
    "- 可见的视频/物品/任务标题(读封面文字)\n"
    "- 角色/敌人/NPC 的名字\n"
    "- UI 关键数字(血量、金币、伤害值)\n"
    "- 玩家正在做的具体动作(挥剑/对话/翻菜单)\n\n"
    "**例子**:\n"
    "- 差: \"战斗场景, 有敌人\"\n"
    "- 好: \"boss 战, 敌人「黑兽」剩 30% 血, 玩家血 60%\"\n"
    "- 差: \"B站网页界面\"\n"
    "- 好: \"B站游戏页, 顶部封面「黑神话悟空全流程」「艾尔登法环」\"\n"
    "- 差: \"菜单界面\"\n"
    "- 好: \"角色装备菜单, 当前选中「火焰长剑」, 玩家等级 42\"\n\n"
    "**只输出那一句**,不要前缀、解释、引号、markdown。"
)


_SCENE_DESCRIBE_DIFF_SYSTEM = (
    "你是画面变化检测器。我告诉你上次的描述和现在的画面。\n\n"
    "**SAME 判定从严** —— 只有当画面**完全静止**(玩家彻底没动,光标没动,UI 没变化)"
    "才输出 `SAME`。其他情况一律描述新画面。\n\n"
    "判定细则:\n"
    "- 视频列表/菜单**滚动了**(哪怕只滚一小段) → 描述当前可见的内容\n"
    "- 光标/选中项移动了 → 描述新选中是什么\n"
    "- UI 数字/血量/进度有变化 → 描述新数字\n"
    "- 玩家动作变了(从静止到走/挥剑/对话) → 描述新动作\n"
    "- 出现新角色/敌人/弹窗 → 描述\n"
    "- 切场景/换游戏 → 描述新场景\n"
    "- **完全没变**(暂停界面/坐着发呆/纯黑过场) → `SAME`\n\n"
    "如果你读不清具体文字,**别犹豫**,根据布局变化/颜色变化/可见元素位置变化判断,"
    "宁可多描述、不要轻易 SAME(让 SAME 是少数派)。\n\n"
    "输出格式: 跟正常描述一样, <场景类型大类>, <1-2 个具体内容>, 不超过 50 字。\n\n"
    "**只输出 `SAME` 或一句新描述**,不要前缀、解释、markdown。"
)


def describe_scene(
    frame_b64: str,
    *,
    mime: str = "image/webp",
    max_tokens: int = 80,
    last_description: str | None = None,
) -> str:
    """One-line objective description of a frame, for time-stamped scene memory.

    When ``last_description`` is provided, the model gets a diff-style prompt:
    it can return ``SAME`` (we map to empty string) to signal "no meaningful
    change", letting the caller skip the scene-memory append entirely.

    Uses flash (cheap, fast). Empty string on no-change or failure.
    """
    try:
        if last_description:
            out = flash_vision(
                prompt=f"上一次的描述: {last_description}\n这次画面有实质变化吗?",
                image_b64=frame_b64,
                system=_SCENE_DESCRIBE_DIFF_SYSTEM,
                mime=mime,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False},
            )
        else:
            out = flash_vision(
                prompt="描述这张画面",
                image_b64=frame_b64,
                system=_SCENE_DESCRIBE_SYSTEM,
                mime=mime,
                max_tokens=max_tokens,
                extra_body={"enable_thinking": False},
            )
    except Exception:
        return ""
    line = out.strip().strip("`'\"")
    line = line.splitlines()[0].strip() if line else ""
    # SAME sentinel from the diff prompt = no meaningful change; tell caller
    # to skip the append by returning empty.
    if line.upper().strip(" .。") in ("SAME", "NO", "NONE", "无变化"):
        return ""
    return line


_SESSION_SUMMARY_SYSTEM = (
    "你是一个游戏会话总结器。给你今天一整天的事件流(observed.md 内容,"
    "包含 player/ai/event/scene 多种类型条目),请输出一份 markdown 格式的会话小结。\n\n"
    "格式要求(严格):\n"
    "```\n"
    "# 会话小结\n"
    "\n"
    "## 玩了什么\n"
    "- (3-5 条短 bullet,描述玩家游玩的主要内容和进度)\n"
    "\n"
    "## 印象深的瞬间\n"
    "- (2-4 条短 bullet,值得长期记的具体事件,带具体细节)\n"
    "\n"
    "## 玩家状态/情绪\n"
    "- (1-3 条短 bullet,描述玩家今天的情绪/状态/口头禅)\n"
    "\n"
    "## 你和玩家的互动\n"
    "- (2-3 条短 bullet,Eri 和玩家的对话亮点或新梗)\n"
    "```\n\n"
    "- 中文,口语化,简洁\n"
    "- 不要剧透未发生的事(只总结这段日志里出现的)\n"
    "- 不输出除上述格式之外的任何字符"
)


def summarize_session(events_text: str) -> str:
    """Flash-summarize a day's observed.md content into a session recap.

    Returns empty string on failure. Caller (memory.summarize_today_async)
    writes the result to sessions/<date>.md.
    """
    if not events_text.strip():
        return ""
    try:
        out = flash_text(
            prompt=f"今天的事件流:\n\n{events_text[:6000]}",
            system=_SESSION_SUMMARY_SYSTEM,
            max_tokens=600,
            extra_body={"enable_thinking": False},
        )
    except Exception:
        return ""
    return out.strip().strip("`")


def extract_fact(event_summary: str, ai_text: str) -> str | None:
    """Ask flash whether this turn yielded a durable fact. Returns the bullet
    line (starting with ``- ``) or ``None``.
    """
    out = flash_text(
        prompt=f"事件: {event_summary}\nAI 回复: {ai_text}",
        system=_FACT_EXTRACT_SYSTEM,
        max_tokens=60,
        extra_body={"enable_thinking": False},
    )
    line = out.strip()
    if not line or line.upper().startswith("NONE"):
        return None
    # Some models prefix with "```" or stray quotes — clean up best-effort.
    line = line.strip("`'\"")
    if not line.startswith("-"):
        return None
    return line


def decide(
    event_summary: str,
    frame_b64: str | None,
    thumbnail_b64: str | None,
    persona: Persona,
    *,
    memory: str = "",
    mime: str = "image/png",
    force_speak: bool = False,
    is_conversation: bool = False,
    recent_ai_lines: list[str] | None = None,
) -> AiReply | None:
    if not force_speak:
        if thumbnail_b64 is None:
            return None  # no frame, no force → caller must opt-in via force_speak
        if not gate(event_summary, thumbnail_b64, persona, mime=mime):
            return None
    return generate(
        event_summary, frame_b64, persona,
        memory=memory, mime=mime,
        is_conversation=is_conversation,
        recent_ai_lines=recent_ai_lines,
    )
