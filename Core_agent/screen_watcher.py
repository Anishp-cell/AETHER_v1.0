"""
AETHER v3.0 — Ambient Screen Watcher
Monitors screen regions in a background daemon thread for custom user conditions (e.g. "download complete", "build finished") and speaks an alert upon detection.
"""
import os
import sys
import time
import threading
import subprocess
import requests
import base64
from io import BytesIO

try:
    import pyautogui
except ImportError:
    pass

ACTIVE_WATCHER = None


class ScreenWatcher:
    """
    Background polling screen watcher for ambient desktop automation.
    """
    def __init__(self):
        self.is_watching = False
        self.condition = ""
        self.poll_interval = 5.0
        self._thread = None

    def start_watch(self, condition: str, poll_interval: float = 5.0) -> str:
        """
        Arms the background screen watcher to look for `condition`.
        Example: start_watch("download finishes")
        """
        if self.is_watching:
            self.stop_watch()

        self.condition = condition
        self.poll_interval = float(poll_interval)
        self.is_watching = True

        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

        return f"Ambient screen watcher armed. Watching for: '{condition}' every {poll_interval}s."

    def stop_watch(self) -> str:
        """Stops active screen watcher."""
        if not self.is_watching:
            return "No screen watcher is currently running."

        self.is_watching = False
        return "Screen watcher disarmed and stopped."

    def _watch_loop(self):
        print(f"\n👁️ [ScreenWatcher] Daemon loop started. Watching for '{self.condition}'...")
        ollama_url = "http://127.0.0.1:11434/api/generate"

        while self.is_watching:
            time.sleep(self.poll_interval)

            if not self.is_watching:
                break

            try:
                # Capture screenshot
                image = pyautogui.screenshot()
                buffered = BytesIO()
                image.save(buffered, format="JPEG", quality=75)
                img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

                prompt = f"""You are analyzing a desktop screenshot to check if a specific condition has occurred.
Condition to check: '{self.condition}'.
Respond with ONLY 'YES' if the condition is clearly true/met on screen.
Respond with ONLY 'NO' if the condition is not yet met.
Do not explain."""

                payload = {
                    "model": "llava-phi3",
                    "prompt": prompt,
                    "images": [img_b64],
                    "stream": False
                }
                res = requests.post(ollama_url, json=payload, timeout=10.0).json()
                answer = res.get("response", "").strip().upper()

                if "YES" in answer:
                    print(f"\n🎯 [ScreenWatcher] Condition MET: '{self.condition}'!")
                    self.is_watching = False

                    # Trigger native TTS alert via PowerShell
                    alert_msg = f"Sir, your screen watcher alert has triggered. Condition met: {self.condition}"
                    ps_cmd = f"Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{alert_msg}')"
                    subprocess.Popen(["powershell", "-Command", ps_cmd], creationflags=subprocess.CREATE_NO_WINDOW)
                    break

            except Exception as e:
                print(f"[ScreenWatcher Warning] Poll error: {e}")


# Singleton Instance
SCREEN_WATCHER = ScreenWatcher()

def start_screen_watch(condition: str, poll_seconds: float = 5.0) -> str:
    """Arms a background screen watcher that continuously monitors the desktop for a condition (e.g. 'download finished', 'build successful') and alerts you when complete."""
    return SCREEN_WATCHER.start_watch(condition, poll_seconds)

def stop_screen_watch() -> str:
    """Stops the active background screen watcher."""
    return SCREEN_WATCHER.stop_watch()
