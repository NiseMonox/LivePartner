"""Idle motion — keeps the Live2D avatar feeling alive between turns.

Many VTS-rigged models (including Type-H3) expose standard parameters like
``EyeOpenLeft`` / ``ParamEyeLOpen``, ``ParamAngleX/Y/Z``, ``ParamBodyAngleX``,
``ParamBreath``. When face tracking is off these stay at default → frozen pose.

IdleMotion drives them programmatically at 20 Hz, combining three loops:

- **AutoBlink** — random 2-5 s interval; per blink: close 80 ms, hold 50 ms,
  open 120 ms. Drives ``EyeOpenLeft`` + ``EyeOpenRight`` symmetrically.
- **IdleSway** — slow decoupled sinewaves on head + body angles. Small
  amplitudes (~3-5°) with 5-11 s periods so the model "breathes" subtly without
  looking robotic.
- **Breath** — 4 s period 0↔1 on ``ParamBreath`` for chest rise/fall.

All three combine into a single 20 Hz tick that bulk-pushes via
``VTSController.set_parameters_bulk()``. Mouth (driven by LipSyncDriver at
~12 Hz on ``MouthOpen``) is a separate channel — no overlap.

The Director's expressions are decorative overlays (ExpStar / ExpSweat etc on
Type-H3); they don't touch the base body/head/eye parameters, so idle motion
and expressions compose freely.
"""
from __future__ import annotations

import math
import random
import time
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QTimer

if TYPE_CHECKING:
    from .vts_controller import VTSController


# Parameter-name candidates by purpose — first match in
# ``vts.status.parameters`` wins. Different rigs label them differently:
# VTS-mapped input names (EyeOpenLeft) vs raw Cubism (ParamEyeLOpen).
#
# NOTE on body movement: many models (incl. Type-H3) bind Body X/Y/Z to the
# SAME inputs as Face X/Y/Z (FaceAngleX/Y/Z). So driving head sway also moves
# the body — we don't list a separate body_x candidate. If a rig truly has an
# independent body angle input, we can add it back; for now head sway covers it.
_PARAM_CANDIDATES: dict[str, list[str]] = {
    "eye_left":  ["EyeOpenLeft", "ParamEyeLOpen", "ParamEyeOpenL"],
    "eye_right": ["EyeOpenRight", "ParamEyeROpen", "ParamEyeOpenR"],
    "head_x":    ["FaceAngleX", "ParamAngleX", "HeadAngleX"],
    "head_y":    ["FaceAngleY", "ParamAngleY", "HeadAngleY"],
    "head_z":    ["FaceAngleZ", "ParamAngleZ", "HeadAngleZ"],
    "breath":    ["Breath", "ParamBreath"],
}


class IdleMotion:
    """Drives idle parameters at 20 Hz. Lifecycle::

        idle = IdleMotion(vts, logger=print)
        idle.start()
        # ... avatar now blinks, sways, breathes ...
        idle.set_auto_blink(False)   # toggle features at runtime
        idle.stop()

    Three feature toggles (``auto_blink`` / ``idle_sway`` / ``breath``) can be
    flipped at any time; the next tick honors the new state.
    """
    TICK_MS = 50  # 20 Hz

    def __init__(
        self,
        vts: "VTSController",
        *,
        logger: Callable[[str], None] | None = None,
    ):
        self.vts = vts
        self._log = logger or (lambda _m: None)
        # Public toggles — UI checkboxes write directly here.
        self.auto_blink = True
        self.idle_sway = True
        self.breath = True

        # Auto-blink state.
        self._next_blink_at: float = 0.0
        self._blink_started_at: float = 0.0
        self._is_blinking: bool = False

        # Sway/breath time base (re-initialized on start so phase is reset).
        self._t0: float = 0.0

        # Resolved param names filled on start() against vts.status.parameters.
        # Empty dict = nothing to drive on this model.
        self._params: dict[str, str] = {}

        # Decorrelate the per-axis sinewaves so the model doesn't move on a
        # single beat. Periods and amplitudes chosen for a calm idle pose.
        self._sway_periods = {"head_x": 5.7, "head_y": 7.3, "head_z": 11.0}
        self._sway_amps = {"head_x": 5.0, "head_y": 4.0, "head_z": 2.5}
        self._sway_phase = {"head_x": 0.0, "head_y": 1.7, "head_z": 3.3}

        # Timer — created lazily on start() so this class is instantiable
        # outside the Qt thread (start() must be called from UI thread).
        self._timer: QTimer | None = None

    # ---------- public toggles ----------

    def set_auto_blink(self, enabled: bool) -> None:
        self.auto_blink = bool(enabled)

    def set_idle_sway(self, enabled: bool) -> None:
        self.idle_sway = bool(enabled)

    def set_breath(self, enabled: bool) -> None:
        self.breath = bool(enabled)

    # ---------- lifecycle ----------

    def start(self) -> None:
        if not self.vts.is_connected:
            self._log("[idle] VTS not connected — IdleMotion start ignored")
            return
        # Resolve param names against the currently loaded model. We pull the
        # parameter list snapshot from vts.status — it's populated by
        # _async_refresh_capabilities right after auth.
        available = set(self.vts.status.parameters)
        self._params = {}
        for purpose, candidates in _PARAM_CANDIDATES.items():
            for cand in candidates:
                if cand in available:
                    self._params[purpose] = cand
                    break
        # No force-adding: VTS's requestSetMultiParameterValue is fail-all-or-
        # none — if even one parameter name is unknown, the entire bulk request
        # is rejected and other parameters (eye/head sway) also stop updating.
        # If "Breath" isn't in the trackable list (e.g. Type-H3's Breath is an
        # "Auto-breathing" type VTS auto-supplies), we just skip the breath
        # driver entirely. User keeps blink + head sway.
        resolved = ", ".join(f"{k}={v}" for k, v in self._params.items())
        missing = [k for k in _PARAM_CANDIDATES if k not in self._params]
        if resolved:
            tail = f"  ·  missing: {missing}" if missing else ""
            self._log(f"[idle] params: {resolved}{tail}")
            if "breath" not in self._params:
                self._log(
                    "[idle] breath skipped — 'Breath' / 'ParamBreath' not in "
                    "trackable params (model uses VTS auto-breathing instead, "
                    "or face tracking would need to be on)"
                )
        else:
            self._log("[idle] no usable params on this model — IdleMotion noop")
            return

        self._t0 = time.monotonic()
        self._is_blinking = False
        self._schedule_next_blink(self._t0)
        if self._timer is None:
            self._timer = QTimer()
            self._timer.setInterval(self.TICK_MS)
            self._timer.timeout.connect(self._tick)
        self._timer.start()
        self._log("[idle] started")

    def stop(self) -> None:
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
            self._log("[idle] stopped")

    # ---------- internals ----------

    def _schedule_next_blink(self, now: float) -> None:
        # Humans blink ~10-15 times/min ≈ every 4-6 s; 2-5 s adds a touch of
        # restlessness which suits Eri's "always watching" energy.
        self._next_blink_at = now + random.uniform(2.0, 5.0)

    def _blink_value(self, phase_t: float) -> float:
        """Return blink eye-open value for a given phase time in seconds.

        Phases:  close 80 ms (1→0) | hold 50 ms (0) | open 120 ms (0→1)
        """
        if phase_t < 0.08:
            return 1.0 - (phase_t / 0.08)
        if phase_t < 0.13:
            return 0.0
        if phase_t < 0.25:
            return (phase_t - 0.13) / 0.12
        return 1.0

    def _tick(self) -> None:
        if not self.vts.is_connected or not self._params:
            return
        now = time.monotonic()
        t = now - self._t0
        updates: list[tuple[str, float]] = []

        # --- Auto-blink (eyes) ---
        # NOTE: we push an eye-open value EVERY tick (even between blinks),
        # not just during a blink. Reason: if we stop pushing, VTS hands the
        # parameter back to its default / tracker source, which can sit at
        # ~0.9 instead of 1.0 — and the eye "jumps" closed a hair on transition.
        # Continuous push at 1.0 keeps full control and the transition smooth.
        if self.auto_blink and "eye_left" in self._params:
            # State machine: trigger a new blink if it's time.
            if not self._is_blinking and now >= self._next_blink_at:
                self._is_blinking = True
                self._blink_started_at = now
            if self._is_blinking:
                phase_t = now - self._blink_started_at
                if phase_t >= 0.25:
                    # Blink finished — schedule next, hold open.
                    self._is_blinking = False
                    self._schedule_next_blink(now)
                    v = 1.0
                else:
                    v = self._blink_value(phase_t)
            else:
                # Idle between blinks — keep eyes open under our control.
                v = 1.0
            updates.append((self._params["eye_left"], v))
            if "eye_right" in self._params:
                updates.append((self._params["eye_right"], v))

        # --- Idle sway (head — body follows via shared FaceAngle inputs) ---
        if self.idle_sway:
            for purpose in ("head_x", "head_y", "head_z"):
                if purpose in self._params:
                    period = self._sway_periods[purpose]
                    amp = self._sway_amps[purpose]
                    phase = self._sway_phase[purpose]
                    v = amp * math.sin(2.0 * math.pi * (t + phase) / period)
                    updates.append((self._params[purpose], v))

        # Bulk-push eye + head params first — these are always known-good, so
        # the multi-param request can't be rejected en bloc by an unknown name.
        if updates:
            self.vts.set_parameters_bulk(updates)

        # --- Breath (separate channel) ---
        # Driven via single set_parameter so a VTS reject (e.g. parameter
        # write-locked because tracking auto-supplies it) doesn't tank the
        # head/eye updates above.
        if self.breath and "breath" in self._params:
            v = 0.5 + 0.5 * math.sin(2.0 * math.pi * t / 4.0)
            self.vts.set_parameter(self._params["breath"], v)
