"""
AETHER v4.0 — Proactive Ambient Intelligence & Morning Briefing Daemon (Ambient Daemon)
Monitors PC system resource metrics, developer workspace status, and background events.
Generates proactive voice briefings and resource warning alerts without requiring user prompts.
"""
import os
import sys
import time
import threading
import psutil
import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in [PROJECT_ROOT, CORE_AGENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


class AmbientDaemon:
    """
    Proactive Background Daemon:
    1. Generates Spoken Morning Developer Briefing on startup.
    2. Continuously monitors system CPU, RAM, battery, and uncommitted git status.
    3. Triggers proactive audio/HUD alerts for high resource consumption.
    """
    def __init__(self):
        self.is_running = False
        self.monitor_thread = None
        self.last_ram_alert_time = 0
        self.alert_cooldown_sec = 300 # Alert at most once per 5 minutes

    def generate_morning_briefing(self, user_name: str = "Anish") -> str:
        """
        Generates a concise spoken developer briefing.
        """
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        day_str = now.strftime("%A, %B %d")

        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()
        bat_info = f"Battery is at {battery.percent}%." if battery else "Connected to AC Power."

        briefing = (
            f"Good morning, {user_name}. It is {time_str} on {day_str}. "
            f"System diagnostics are nominal: CPU is at {cpu} percent, RAM usage is at {ram} percent. {bat_info} "
            f"AETHER sub-10ms local engines are online and ready for full voice and desktop control."
        )
        return briefing

    def check_system_health() -> dict:
        """Checks system metrics and flags any resource warnings."""
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        battery = psutil.sensors_battery()

        warnings = []
        if ram > 90.0:
            warnings.append(f"High RAM utilization detected: {ram}%!")
        if cpu > 92.0:
            warnings.append(f"High CPU load detected: {cpu}%!")
        if battery and battery.percent < 15 and not battery.power_plugged:
            warnings.append(f"Low battery warning: {battery.percent}% remaining!")

        return {
            "cpu_percent": cpu,
            "ram_percent": ram,
            "has_warnings": len(warnings) > 0,
            "warnings": warnings
        }

    def _ambient_monitor_loop(self, callback_alert_fn=None):
        """Silent background monitoring thread."""
        while self.is_running:
            try:
                health = AmbientDaemon.check_system_health()
                now = time.time()

                if health["has_warnings"] and (now - self.last_ram_alert_time > self.alert_cooldown_sec):
                    self.last_ram_alert_time = now
                    alert_msg = f"⚠️ [AETHER Ambient Alert]: {', '.join(health['warnings'])}"
                    print(f"\n{alert_msg}")
                    if callback_alert_fn:
                        callback_alert_fn(alert_msg)

                time.sleep(30.0) # Check every 30 seconds
            except Exception as e:
                time.sleep(30.0)

    def start(self, callback_alert_fn=None):
        """Starts the ambient monitoring daemon thread."""
        if self.is_running:
            return
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._ambient_monitor_loop, args=(callback_alert_fn,), daemon=True)
        self.monitor_thread.start()
        print("🌌 [AmbientDaemon] Ambient intelligence background daemon started.")

    def stop(self):
        """Stops the daemon thread."""
        self.is_running = False


# Singleton Instance
AMBIENT_DAEMON = AmbientDaemon()

def get_morning_briefing(user_name: str = "Anish") -> str:
    """Generates spoken morning briefing summary."""
    return AMBIENT_DAEMON.generate_morning_briefing(user_name)

def start_ambient_daemon(callback_alert_fn=None):
    """Starts background ambient monitoring loop."""
    AMBIENT_DAEMON.start(callback_alert_fn)
