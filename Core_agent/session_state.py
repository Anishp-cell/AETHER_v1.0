"""
AETHER Session State & Lifecycle Manager
Provides thread-safe and cross-process awareness of whether AETHER is currently running.
Used as a privacy gate for features like URL tracking and background telemetry.
"""

import os
import sys
import time
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
SESSION_FLAG_FILE = WORKSPACE_ROOT / ".aether" / "session_active.flag"

# In-memory session state flag for the current process
_aether_active: bool = False
_last_heartbeat: float = 0.0


def set_active(state: bool = True):
    """
    Activates or deactivates the AETHER active session state.
    Updates both the in-memory flag and the persistent cross-process flag file.
    """
    global _aether_active, _last_heartbeat
    _aether_active = bool(state)
    _last_heartbeat = time.time()

    SESSION_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if _aether_active:
        try:
            with open(SESSION_FLAG_FILE, "w", encoding="utf-8") as f:
                f.write(f"ACTIVE:{time.time()}\n")
        except Exception:
            pass
    else:
        try:
            if SESSION_FLAG_FILE.exists():
                SESSION_FLAG_FILE.unlink()
        except Exception:
            pass


def is_active() -> bool:
    """
    Returns True if AETHER is actively running in the current process
    or in another connected AETHER process (via session flag file heartbeat).
    """
    global _aether_active
    if _aether_active:
        return True

    # Check cross-process flag file
    if SESSION_FLAG_FILE.exists():
        try:
            mtime = SESSION_FLAG_FILE.stat().st_mtime
            # Valid if touched within the last 30 minutes
            if (time.time() - mtime) < 1800:
                return True
        except Exception:
            pass

    return False


def record_heartbeat():
    """Refreshes the active session heartbeat timestamp."""
    if is_active():
        set_active(True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd in ("on", "true", "start", "active"):
            set_active(True)
            print("[AETHER SessionState] Session activated.")
        elif cmd in ("off", "false", "stop", "inactive"):
            set_active(False)
            print("[AETHER SessionState] Session deactivated.")
        else:
            print(f"[AETHER SessionState] Active: {is_active()}")
    else:
        print(f"[AETHER SessionState] Active: {is_active()}")
