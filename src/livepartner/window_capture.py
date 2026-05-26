"""Find + capture an arbitrary application window on Windows.

Used by the self-awareness path: every ~90 s we grab whatever VTube Studio is
showing right now, feed it to a flash vision call ("describe this avatar's
appearance"), and store the result in memory so Eri can react to outfit /
accessory / scene changes the user makes inside VTS.

Windows-only — we use ctypes against user32.dll so there's no extra dep
beyond Pillow (already on requirements). On non-Windows the helpers just
return None and the caller no-ops the self-describer.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from typing import Callable

from PIL import Image, ImageGrab


def _is_windows() -> bool:
    return sys.platform.startswith("win")


# ctypes bindings (Windows only — guard against import-time failure on
# non-Windows so the rest of the module imports cleanly there too).
if _is_windows():
    _user32 = ctypes.windll.user32

    _EnumWindowsProc = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )
    _user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.IsIconic.argtypes = [wintypes.HWND]  # minimized?
    _user32.IsIconic.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [
        wintypes.HWND, wintypes.LPWSTR, ctypes.c_int
    ]
    _user32.GetWindowTextW.restype = ctypes.c_int
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.GetClientRect.restype = wintypes.BOOL
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.ClientToScreen.restype = wintypes.BOOL
else:  # pragma: no cover
    _user32 = None
    _EnumWindowsProc = None


def _get_window_title(hwnd: int) -> str:
    if not _is_windows():
        return ""
    n = _user32.GetWindowTextLengthW(hwnd)
    if n <= 0:
        return ""
    buf = ctypes.create_unicode_buffer(n + 1)
    _user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def find_window(title_substr: str) -> int | None:
    """Return the HWND of the first visible, non-minimized top-level window
    whose title contains ``title_substr`` (case-insensitive). None on no match.
    """
    if not _is_windows():
        return None
    needle = title_substr.lower()
    found: list[int] = []

    @_EnumWindowsProc
    def _cb(hwnd, _lparam):
        if not _user32.IsWindowVisible(hwnd):
            return True  # keep enumerating
        if _user32.IsIconic(hwnd):  # minimized — can't capture
            return True
        title = _get_window_title(hwnd)
        if title and needle in title.lower():
            found.append(hwnd)
            return False  # stop enumerating
        return True

    _user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def get_client_screen_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Return (left, top, right, bottom) in SCREEN coords of the client area
    (drawable surface, excluding titlebar/border). None if hwnd is invalid."""
    if not _is_windows() or not hwnd:
        return None
    rect = wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    # GetClientRect returns 0-based (origin at window's top-left client). Map to
    # screen coordinates by adding the window position.
    pt = wintypes.POINT(0, 0)
    if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    left = pt.x
    top = pt.y
    right = left + (rect.right - rect.left)
    bottom = top + (rect.bottom - rect.top)
    if right <= left or bottom <= top:
        return None  # zero-size window (probably minimized)
    return left, top, right, bottom


def capture_window_by_title(
    title_substr: str,
    *,
    logger: Callable[[str], None] | None = None,
) -> Image.Image | None:
    """One-shot: find a top-level window matching ``title_substr`` and grab
    its client area as a PIL image. None on no match / non-Windows."""
    log = logger or (lambda _m: None)
    if not _is_windows():
        log("[capture] non-Windows platform — window capture unavailable")
        return None
    hwnd = find_window(title_substr)
    if hwnd is None:
        return None
    bbox = get_client_screen_rect(hwnd)
    if bbox is None:
        return None
    # PIL.ImageGrab.grab(bbox) captures from the actual screen, so anything
    # occluding the VTS window will be in the image. User should keep VTS
    # visible during periodic self-awareness checks.
    try:
        img = ImageGrab.grab(bbox=bbox, all_screens=True)
    except Exception as e:
        log(f"[capture] ImageGrab failed: {type(e).__name__}: {e}")
        return None
    return img
