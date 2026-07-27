"""
AETHER v3.0 — Native Windows Smart Window Manager
Provides Win32 window snapping, tiling, minimizing, restoring, and app switching.
"""
import os
import sys
import time
import ctypes
from ctypes import wintypes

# Win32 API Functions & Constants
user32 = ctypes.windll.user32

SWP_NOZORDER = 0x0004
SWP_SHOWWINDOW = 0x0040
SW_MINIMIZE = 6
SW_MAXIMIZE = 3
SW_RESTORE = 9
WM_CLOSE = 0x0010

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long)
    ]


def get_screen_bounds():
    """Returns primary monitor screen width and height."""
    sw = user32.GetSystemMetrics(0)
    sh = user32.GetSystemMetrics(1)
    return sw, sh


def get_foreground_hwnd():
    """Returns HWND handle of the current active foreground window."""
    return user32.GetForegroundWindow()


def snap_window(direction: str = "left") -> str:
    """
    Snaps the current foreground window to desktop screen regions.
    Options: 'left', 'right', 'top', 'bottom', 'maximize', 'minimize', 'restore'
    """
    hwnd = get_foreground_hwnd()
    if not hwnd:
        return "No active foreground window found to snap."

    direction = direction.lower().strip()
    sw, sh = get_screen_bounds()

    # Subtract taskbar height (~40px)
    work_h = sh - 40

    try:
        if direction in ("left", "left_half", "snap_left"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetWindowPos(hwnd, 0, 0, 0, sw // 2, work_h, SWP_NOZORDER | SWP_SHOWWINDOW)
            return "Snapped active window to the left half of the screen."

        elif direction in ("right", "right_half", "snap_right"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetWindowPos(hwnd, 0, sw // 2, 0, sw // 2, work_h, SWP_NOZORDER | SWP_SHOWWINDOW)
            return "Snapped active window to the right half of the screen."

        elif direction in ("top", "top_half", "snap_top"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetWindowPos(hwnd, 0, 0, 0, sw, work_h // 2, SWP_NOZORDER | SWP_SHOWWINDOW)
            return "Snapped active window to the top half of the screen."

        elif direction in ("bottom", "bottom_half", "snap_bottom"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            user32.SetWindowPos(hwnd, 0, 0, work_h // 2, sw, work_h // 2, SWP_NOZORDER | SWP_SHOWWINDOW)
            return "Snapped active window to the bottom half of the screen."

        elif direction in ("maximize", "max", "full_screen"):
            user32.ShowWindow(hwnd, SW_MAXIMIZE)
            return "Maximized active window."

        elif direction in ("minimize", "min"):
            user32.ShowWindow(hwnd, SW_MINIMIZE)
            return "Minimized active window."

        elif direction in ("restore", "normal"):
            user32.ShowWindow(hwnd, SW_RESTORE)
            return "Restored active window size."

        return f"Unknown snap direction: '{direction}'. Options: left, right, top, bottom, maximize, minimize."

    except Exception as e:
        return f"[WindowManager Error]: {str(e)}"


def close_foreground_window() -> str:
    """Closes the current active foreground window cleanly via Win32 WM_CLOSE signal."""
    hwnd = get_foreground_hwnd()
    if not hwnd:
        return "No active window found."
    user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    return "Closed active window."


def tile_windows() -> str:
    """Auto-tiles all visible desktop windows side-by-side across the screen."""
    sw, sh = get_screen_bounds()
    work_h = sh - 40

    visible_hwnds = []

    def enum_handler(hwnd, lParam):
        if user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buff = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buff, length + 1)
                title = buff.value.strip()
                # Exclude Windows system utility background windows
                if title and title not in ("Program Manager", "Settings", "Windows Input Experience"):
                    visible_hwnds.append(hwnd)
        return True

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
    user32.EnumWindows(EnumWindowsProc(enum_handler), 0)

    # Limit tiling to top 4 main windows
    targets = visible_hwnds[:4]
    count = len(targets)
    if count == 0:
        return "No open desktop windows to tile."

    col_w = sw // count
    for idx, hwnd in enumerate(targets):
        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetWindowPos(hwnd, 0, idx * col_w, 0, col_w, work_h, SWP_NOZORDER | SWP_SHOWWINDOW)

    return f"Successfully tiled {count} active window(s) across the desktop."
