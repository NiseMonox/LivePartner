"""Persistent memory system per SPEC §6.

Layout (rooted at <repo>/.memory/):
    global/player.md         — cross-game player profile (name, taste, etc.)
    games/<game_id>/
        identity.md          — who Eri is in this specific game
        observed.md          — append-only event stream (markdown)
        jokes.md             — recurring callbacks / inside jokes
        progress.md          — known story progress (anti-spoiler boundary)

Read flow: render_for_prompt() returns a single markdown blob to inject into
the LLM system prompt's `# 记忆` section.

Write flow: add_event / add_player_voice / add_ai_response append a
timestamped line to observed.md and update an in-memory cache (this session
only — disk is the durable copy).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_MEMORY_ROOT = Path(__file__).resolve().parent.parent.parent / ".memory"
# Gap above which we start a new "## YYYY-MM-DD HH:MM" session block.
NEW_BLOCK_GAP_SEC = 30 * 60
# Cap on in-session entries kept in memory.
RECENT_CACHE_LIMIT = 100
# How many recent observed entries to feed into the prompt by default.
DEFAULT_PROMPT_RECENT = 20
# Scene-description memory: how long to keep entries by default. Tuned around
# 90s so dialog can naturally reference "刚才" / "5分钟前" without bloating
# the system prompt.
DEFAULT_SCENE_RETENTION_SEC = 90.0
# Hard cap on scene entries kept in memory at once, in case the user cranks
# the describe interval very low.
SCENE_CACHE_LIMIT = 40


@dataclass(frozen=True)
class MemoryEntry:
    ts: float                  # "last seen" time — used for relative-time labels
    kind: str                  # event / player / ai / fact / scene
    text: str
    # For dedup'd scene entries: when did this same content FIRST appear?
    # None for non-deduped entries. Render uses (ts - first_ts) to show
    # "持续 X 分钟" hints so the model knows the scene is static.
    first_ts: float | None = None
    # For scene entries: AI lines spoken while this scene was active. Each
    # element is (ts, text). Carries forward across same-scene dedup so the
    # render can show the model "在这个画面下你已经说过 N 次" — the strongest
    # anti-repetition signal we have. Empty tuple for non-scene entries.
    ai_lines: tuple[tuple[float, str], ...] = ()


_KIND_LABEL = {
    "event": "事件",
    "player": "玩家说",
    "ai": "你刚说",
    "fact": "记住",
}

# Local dedup is a safety net — primary dedup is the model-side SAME signal
# returned by decision.describe_scene() when given the last description.
# These thresholds are tuned for "verbatim or near-verbatim paraphrase":
#  - bigram_jaccard > 0.30  OR
#  - charset_overlap > 0.60 (Chinese paraphrases reuse content chars even
#    when bigrams differ — e.g. "B站网页" vs "浏览器网页"  share chars but
#    not bigrams).
# Semantic-but-rephrased duplicates that slip past both still get caught by
# the model returning SAME on the next tick.
SCENE_BIGRAM_THRESHOLD = 0.30
SCENE_CHARSET_THRESHOLD = 0.60


def _scene_similar(a: str, b: str) -> bool:
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return a == b
    if a == b:
        return True
    # Bigram Jaccard
    A = {a[i:i + 2] for i in range(len(a) - 1)}
    B = {b[i:i + 2] for i in range(len(b) - 1)}
    if A and B:
        bigram_j = len(A & B) / len(A | B)
        if bigram_j > SCENE_BIGRAM_THRESHOLD:
            return True
    # Character-set overlap (helps for Chinese paraphrases)
    CA, CB = set(a), set(b)
    if CA and CB:
        charset_j = len(CA & CB) / len(CA | CB)
        if charset_j > SCENE_CHARSET_THRESHOLD:
            return True
    return False


class MemoryStore:
    """One store per game. Thread-safe (called from UI + worker threads)."""

    def __init__(self, game_id: str = "default", root: Path | None = None):
        self.game_id = game_id
        self.root = (root or DEFAULT_MEMORY_ROOT).resolve()
        self.game_dir = self.root / "games" / game_id
        self.global_dir = self.root / "global"
        self.game_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._recent: list[MemoryEntry] = []
        self._last_block_ts: float | None = None
        # Separate buffer for periodic scene descriptions. Each entry is a
        # MemoryEntry with kind="scene". Kept separate from _recent because
        # scenes are time-windowed (~90s) while _recent is count-windowed,
        # and renders into a distinct prompt section.
        self._scenes: list[MemoryEntry] = []
        # Visual self-awareness — flash-vision describes Eri's VTS window
        # every ~90 s. We hold the current description so the prompt can
        # show "this is what you look like right now" plus the previous one
        # IF it differs meaningfully (signal a "you just changed appearance"
        # event that the next AI turn comments on, then consumes).
        self._self_description: str = ""
        self._self_changed_from: str = ""  # the prev description if a change is pending

    # ---------- paths ----------

    @property
    def observed_path(self) -> Path:
        return self.game_dir / "observed.md"

    @property
    def player_path(self) -> Path:
        return self.global_dir / "player.md"

    @property
    def identity_path(self) -> Path:
        return self.game_dir / "identity.md"

    @property
    def jokes_path(self) -> Path:
        return self.game_dir / "jokes.md"

    @property
    def progress_path(self) -> Path:
        return self.game_dir / "progress.md"

    @property
    def sessions_dir(self) -> Path:
        return self.game_dir / "sessions"

    def session_path(self, date_str: str | None = None) -> Path:
        """Path to a session file. Default: today (YYYY-MM-DD)."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        return self.sessions_dir / f"{date_str}.md"

    def write_session_summary(self, summary: str, date_str: str | None = None) -> Path:
        """Write a session-end summary to sessions/<date>.md.

        Overwrites if a file already exists — meant to be re-runnable as the
        day progresses (later call captures more events).
        """
        p = self.session_path(date_str)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(summary, encoding="utf-8")
        return p

    def read_observed_today(self) -> str:
        """Return the observed.md content from today's `## YYYY-MM-DD HH:MM`
        block onwards. Used as input to the LLM summarizer.

        If no block header for today exists, returns the full file
        (sometimes the day started in the middle of the previous block).
        """
        if not self.observed_path.exists():
            return ""
        try:
            text = self.observed_path.read_text(encoding="utf-8")
        except OSError:
            return ""
        today_marker = "## " + datetime.now().strftime("%Y-%m-%d")
        idx = text.find(today_marker)
        if idx < 0:
            return text
        return text[idx:]

    @property
    def auto_path(self) -> Path:
        """Auto-extracted facts (LLM-curated, user-editable).

        Separate from the manually-curated jokes/progress files so the user can
        review and promote/delete auto-extracted lines without worrying about
        clobbering their own notes.
        """
        return self.game_dir / "auto.md"

    # ---------- write ----------

    def add_event(self, text: str) -> None:
        self._append("event", text)

    def add_player_voice(self, text: str) -> None:
        self._append("player", text)

    def add_ai_response(self, text: str) -> None:
        self._append("ai", text)

    def add_fact(self, text: str) -> None:
        self._append("fact", text)

    def add_scene_description(
        self,
        text: str,
        *,
        retention_sec: float = DEFAULT_SCENE_RETENTION_SEC,
    ) -> bool:
        """Append a one-line scene description tagged with the current time.

        Returns True if a new entry was added, False if it was merged into
        the most recent entry as a near-duplicate (>55% bigram overlap).
        Merging means the timestamp gets bumped to "now" — useful because
        the prompt renderer compresses long runs into "持续 X 分钟" hints.
        """
        text = text.strip()
        if not text:
            return False
        now = time.time()
        with self._lock:
            # Dedup against the most recent scene. If similar, just slide the
            # "last seen" timestamp forward — same scene, no new info.
            if self._scenes:
                last = self._scenes[-1]
                if _scene_similar(last.text, text):
                    # Keep the older description (it might be more informative
                    # than the latest paraphrase); refresh ts AND remember when
                    # this scene first appeared so the render can show "持续 X".
                    # CRUCIAL: carry forward ai_lines so accumulated AI history
                    # under this scene survives dedup — that's the whole point
                    # of attaching them to the scene.
                    self._scenes[-1] = MemoryEntry(
                        ts=now, kind="scene", text=last.text,
                        first_ts=(last.first_ts if last.first_ts is not None else last.ts),
                        ai_lines=last.ai_lines,
                    )
                    cutoff = now - retention_sec
                    self._scenes = [e for e in self._scenes if e.ts >= cutoff]
                    return False
            self._scenes.append(MemoryEntry(ts=now, kind="scene", text=text))
            cutoff = now - retention_sec
            self._scenes = [e for e in self._scenes if e.ts >= cutoff]
            if len(self._scenes) > SCENE_CACHE_LIMIT:
                self._scenes = self._scenes[-SCENE_CACHE_LIMIT:]
            return True

    def attach_ai_to_current_scene(
        self,
        text: str,
        *,
        max_lines: int = 12,
    ) -> bool:
        """Append an AI line to the most recent scene entry's ai_lines.

        Called right after add_ai_response in the decision-done handler so the
        next prompt can show the model "在这个画面下你已经说过 N 次了" — by far
        the strongest anti-repetition signal we have.

        ``max_lines`` caps the per-scene tail so a stuck scene with 30+ AI
        comments doesn't bloat the prompt. We keep the oldest 2 (so the model
        sees how the repetition started) and the most recent (max_lines-2).

        Returns True if attached, False if there's no scene to attach to yet
        (scenes get described on a separate timer, so the first AI line of a
        session may happen before any scene exists).
        """
        text = text.strip()
        if not text:
            return False
        now = time.time()
        with self._lock:
            if not self._scenes:
                return False
            last = self._scenes[-1]
            combined = last.ai_lines + ((now, text),)
            if len(combined) > max_lines:
                # Keep 2 oldest + (max_lines - 2) newest so the model sees the
                # full "you've been at this for a while" arc without bloat.
                head = combined[:2]
                tail = combined[-(max_lines - 2):]
                combined = head + tail
            self._scenes[-1] = MemoryEntry(
                ts=last.ts, kind=last.kind, text=last.text,
                first_ts=last.first_ts, ai_lines=combined,
            )
            return True

    def scene_entries(self, *, retention_sec: float | None = None) -> list[MemoryEntry]:
        """Active scene entries (oldest first). ``retention_sec`` overrides
        the default cutoff for read-only queries."""
        now = time.time()
        with self._lock:
            scenes = list(self._scenes)
        if retention_sec is not None:
            cutoff = now - retention_sec
            scenes = [e for e in scenes if e.ts >= cutoff]
        return scenes

    def clear_scenes(self) -> None:
        with self._lock:
            self._scenes.clear()

    def dedup_auto_facts(self) -> int:
        """One-shot pass: drop near-duplicate bullets from auto.md (in-place).
        Returns how many lines were dropped. Header + non-bullet lines kept.
        """
        p = self.auto_path
        if not p.exists():
            return 0
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            return 0
        out_lines: list[str] = []
        kept_bullets: list[str] = []
        dropped = 0
        for raw in text.splitlines():
            s = raw.strip()
            if not s.startswith("-"):
                out_lines.append(raw)
                continue
            bare = s.lstrip("- ").strip()
            is_dup = any(_scene_similar(bare, k) for k in kept_bullets)
            if is_dup:
                dropped += 1
                continue
            kept_bullets.append(bare)
            out_lines.append(raw)
        if dropped > 0:
            try:
                p.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
            except OSError:
                pass
        return dropped

    def append_auto_fact(self, line: str) -> None:
        """Append an LLM-extracted bullet line to auto.md (persistent).

        Skips if the new fact is a near-duplicate of an existing bullet
        (Jaccard + charset overlap, same as scene-memory dedup) — this is the
        common case because the extractor re-extracts "player's name" every
        time the player says their name, with different wording each turn.

        Also reflects accepted facts as a ``[fact]`` entry in the in-session
        timeline so the user sees them immediately in the memory tab.
        """
        line = line.strip()
        if not line:
            return
        if not line.startswith("-"):
            line = "- " + line
        bare_new = line.lstrip("- ").strip()

        # Dedup against the existing file. Skip the write entirely if a
        # similar bullet is already in there.
        p = self.auto_path
        if p.exists():
            try:
                existing = p.read_text(encoding="utf-8")
            except OSError:
                existing = ""
            for existing_line in existing.splitlines():
                s = existing_line.strip()
                if not s.startswith("-"):
                    continue
                bare_old = s.lstrip("- ").strip()
                if _scene_similar(bare_old, bare_new):
                    return  # already remembered

        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            is_new = (not p.exists()) or p.stat().st_size == 0
            with p.open("a", encoding="utf-8") as f:
                if is_new:
                    f.write("# 自动抽取的事实（LLM 自动写入，你可以手动编辑）\n\n")
                f.write(line + "\n")
        except OSError:
            pass
        self._append("fact", bare_new)

    def _append(self, kind: str, text: str) -> None:
        text = text.strip()
        if not text:
            return
        now = time.time()
        with self._lock:
            self._recent.append(MemoryEntry(ts=now, kind=kind, text=text))
            if len(self._recent) > RECENT_CACHE_LIMIT:
                self._recent = self._recent[-RECENT_CACHE_LIMIT:]
            need_block = (
                self._last_block_ts is None
                or (now - self._last_block_ts) > NEW_BLOCK_GAP_SEC
            )
            self._last_block_ts = now
            dt = datetime.fromtimestamp(now)
            line = f"- [{dt.strftime('%H:%M:%S')}] [{kind}] {text}\n"
            try:
                p = self.observed_path
                p.parent.mkdir(parents=True, exist_ok=True)
                is_new = (not p.exists()) or p.stat().st_size == 0
                with p.open("a", encoding="utf-8") as f:
                    if is_new:
                        f.write("# 已观察事件流\n")
                    if need_block:
                        f.write(f"\n## {dt.strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(line)
            except OSError:
                pass

    # ---------- read ----------

    def _read_md(self, p: Path) -> str:
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def render_for_prompt(
        self,
        max_recent: int = DEFAULT_PROMPT_RECENT,
        scene_retention_sec: float = DEFAULT_SCENE_RETENTION_SEC,
    ) -> str:
        """Return the body for the LLM's `# 记忆` system-prompt section.

        Empty string if nothing useful is loaded — caller can short-circuit
        the join.
        """
        parts: list[str] = []
        player = self._read_md(self.player_path)
        if player:
            parts.append(f"## 关于玩家\n{player}")
        ident = self._read_md(self.identity_path)
        if ident:
            parts.append(f"## 这个游戏里的你 (设定)\n{ident}")

        # Visual self-awareness — current appearance + change-since-last marker
        # (set when the periodic VTS-window snapshot's description shifted).
        with self._lock:
            current_self = self._self_description
            changed_from = self._self_changed_from
        if current_self:
            parts.append(f"## 你现在的样子\n{current_self}")
        if changed_from and changed_from != current_self:
            parts.append(
                "## 你的形象刚刚变了 (优先级最高 — 这一轮务必主动评论一句)\n"
                f"之前: {changed_from}\n"
                f"现在: {current_self}"
            )

        # Scene memory: periodic frame descriptions with relative-time labels
        # ("刚才" / "30s前" / "1分钟前") so dialog can naturally reference
        # "刚才那个怪" without the model getting confused about timing.
        # Persistent scenes (dedup'd over time) also get a "(持续 X 分钟)"
        # tag so the model sees "this hasn't changed" rather than re-commenting.
        scenes = self.scene_entries(retention_sec=scene_retention_sec)
        if scenes:
            now = time.time()
            lines = []
            for e in scenes:
                ago = max(0, int(now - e.ts))
                if ago < 10:
                    when = "刚才"
                elif ago < 60:
                    when = f"{ago}秒前"
                else:
                    when = f"{ago // 60}分{ago % 60}秒前" if ago % 60 else f"{ago // 60}分钟前"
                persist = ""
                if e.first_ts is not None:
                    dur = int(e.ts - e.first_ts)
                    if dur >= 20:
                        if dur < 60:
                            persist = f" (持续 {dur} 秒)"
                        else:
                            persist = f" (持续 {dur // 60} 分钟)"
                lines.append(f"- [{when}{persist}] {e.text}")
                # AI lines spoken under this scene — the strongest anti-repeat
                # signal. The model sees its own past comments grouped with the
                # exact scene they were about, so the repetition is unmistakable.
                if e.ai_lines:
                    count = len(e.ai_lines)
                    for ai_ts, ai_text in e.ai_lines:
                        ai_ago = max(0, int(now - ai_ts))
                        if ai_ago < 60:
                            ai_when = f"{ai_ago}秒前"
                        else:
                            ai_when = (
                                f"{ai_ago // 60}分{ai_ago % 60}秒前"
                                if ai_ago % 60 else f"{ai_ago // 60}分钟前"
                            )
                        lines.append(f"  ↳ [{ai_when}] 你已说过: 「{ai_text}」")
                    if count >= 3:
                        lines.append(
                            f"  ⚠ 同一画面下你已经说了 {count} 次 —— "
                            "不要再原地复读这个场景,要么换话题/关心玩家,"
                            "要么直接 SAME(沉默)。"
                        )
            parts.append(
                f"## 画面回顾 (最近 {int(scene_retention_sec)} 秒, 越往下越新)\n"
                + "\n".join(lines)
            )

        with self._lock:
            recent = list(self._recent[-max_recent:])
        if recent:
            now = time.time()
            lines = []
            for e in recent:
                ago = max(0, int(now - e.ts))
                if ago < 60:
                    when = f"{ago}s前"
                elif ago < 3600:
                    when = f"{ago // 60}分前"
                else:
                    when = datetime.fromtimestamp(e.ts).strftime("%H:%M")
                label = _KIND_LABEL.get(e.kind, e.kind)
                lines.append(f"- [{when}] {label}: {e.text}")
            parts.append("## 最近发生 (越往下越新)\n" + "\n".join(lines))
        jokes = self._read_md(self.jokes_path)
        if jokes:
            parts.append(f"## 梗 / 内部笑话\n{jokes}")
        progress = self._read_md(self.progress_path)
        if progress:
            parts.append(
                f"## 剧情进度 (严格只引用此处提到的内容,不剧透未来)\n{progress}"
            )
        auto = self._read_md(self.auto_path)
        if auto:
            parts.append(f"## 累积事实 (自动抽取)\n{auto}")
        return "\n\n".join(parts)

    def recent_entries(self, n: int = 50) -> list[MemoryEntry]:
        """In-session entries (oldest first) for UI display."""
        with self._lock:
            return list(self._recent[-n:])

    def update_self_appearance(self, description: str) -> bool:
        """Update the cached self-description. If it differs substantively
        from the previous one (per ``self_description_changed``), also stash
        the previous string in ``_self_changed_from`` so the next memory
        render injects a 'you just changed' marker.

        Returns True if the new description was treated as a change.
        """
        from .appearance import self_description_changed
        description = (description or "").strip()
        if not description:
            return False
        with self._lock:
            prev = self._self_description
            changed = self_description_changed(prev, description)
            if changed:
                self._self_changed_from = prev
            self._self_description = description
        return changed

    def consume_self_appearance_change(self) -> None:
        """Clear the 'you just changed' marker after Eri has had a chance to
        comment on it. Current description stays."""
        with self._lock:
            self._self_changed_from = ""

    def recent_ai_lines(self, n: int = 5) -> list[str]:
        """Last N AI-spoken lines (oldest → newest), texts only.

        Used by the dialog-repeat check in decision.generate() — passed in as
        the "what you just said" baseline against which new LINE output gets
        similarity-tested. Empty list if AI hasn't spoken yet this session.
        """
        with self._lock:
            ai_only = [e.text for e in self._recent if e.kind == "ai"]
        return ai_only[-n:]

    def recent_context_lines(self, n: int = 3) -> list[str]:
        """Last N session entries (player / AI / event mix, oldest → newest),
        each prefixed by its kind label. Used by the Live2D Director AI so the
        avatar's expression decision sees the small "what just happened" slice
        without the model needing to parse the full memory dump.

        Format: ``"玩家说: ..." / "你刚说: ..." / "事件: ..." / "记住: ..."``
        """
        with self._lock:
            entries = list(self._recent[-n:])
        out: list[str] = []
        for e in entries:
            label = _KIND_LABEL.get(e.kind, e.kind)
            out.append(f"{label}: {e.text}")
        return out

    def clear_session_cache(self) -> None:
        """Drop in-memory recent entries without touching observed.md."""
        with self._lock:
            self._recent.clear()
            self._last_block_ts = None

    # ---------- game switching ----------

    def list_games(self) -> list[str]:
        d = self.root / "games"
        if not d.exists():
            return []
        return sorted([p.name for p in d.iterdir() if p.is_dir()])
