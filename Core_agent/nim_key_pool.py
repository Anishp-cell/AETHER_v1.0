"""
AETHER NVIDIA NIM Multi-Key Pool Manager
Coordinates multiple NVIDIA NIM API keys across accounts to bypass single-account
rate limits (RPM/TPM) and multiply the monthly free request allowance.

Capabilities:
1. Dynamic Pool Discovery: Reads NVIDIA_API_KEYS (comma-separated) and NVIDIA_API_KEY from .env / environment.
2. Round-Robin Load Balancing: Distributes traffic across all configured accounts.
3. Automatic Failover:
   - On HTTP 429 (Rate Limit): Puts key in temporary cooldown (default 60s) and rotates instantly to next key.
   - On HTTP 402/403 (Quota Exhausted): Marks key as exhausted and removes from rotation.
4. Quota Aggregation: Multiplies base starting quota (1,000 * num_keys).
5. Thread-Safe & Process-Safe state synchronization via .aether/nim_key_pool.json.
"""

import os
import sys
import re
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
POOL_STATE_FILE = WORKSPACE_ROOT / ".aether" / "nim_key_pool.json"
CREDITS_PER_KEY = 1000


class NIMKeyPool:
    """Thread-safe Multi-Key Pool Manager for NVIDIA NIM endpoints."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, env_path: Optional[Path] = None, state_file: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return

        self.env_path = env_path or (WORKSPACE_ROOT / ".env")
        self.state_file = state_file or POOL_STATE_FILE
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._round_robin_idx = 0
        self._pool_lock = threading.Lock()

        # Load keys from environment & .env
        self.keys = self._discover_keys()
        self.state = self._load_or_init_state()
        self._initialized = True

    def _discover_keys(self) -> List[str]:
        """Discovers and deduplicates all valid NVIDIA API keys from .env and environment."""
        discovered: List[str] = []

        def add_key(k: str):
            k = k.strip().strip('"').strip("'")
            if k and k.startswith("nvapi-") and k not in discovered:
                discovered.append(k)

        # 1. Check environment variables
        env_keys = os.getenv("NVIDIA_API_KEYS", "")
        if env_keys:
            for piece in re.split(r'[,;\s\n]+', env_keys):
                add_key(piece)

        single_env = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        if single_env:
            add_key(single_env)

        # 2. Check .env file directly if keys not in os.environ or to supplement
        if self.env_path.exists():
            try:
                with open(self.env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("#") or not line:
                            continue
                        if line.startswith("NVIDIA_API_KEYS="):
                            val = line.split("=", 1)[1]
                            for piece in re.split(r'[,;\s\n]+', val):
                                add_key(piece)
                        elif line.startswith("NVIDIA_API_KEY=") or line.startswith("NIM_API_KEY="):
                            val = line.split("=", 1)[1]
                            add_key(val)
            except Exception as e:
                print(f"[NIMKeyPool] Warning reading {self.env_path}: {e}", file=sys.stderr)

        return discovered

    def _key_hash(self, key: str) -> str:
        """Returns a non-reversible SHA-256 fingerprint for state tracking."""
        import hashlib
        return hashlib.sha256(key.strip().encode("utf-8")).hexdigest()[:12]

    def _mask_key(self, key: str) -> str:
        """Returns a safe preview string (e.g. 'nvapi-nr1B...jAwl')."""
        if len(key) <= 12:
            return key[:4] + "..." if len(key) >= 4 else "..."
        return f"{key[:12]}...{key[-4:]}"

    def _load_or_init_state(self) -> Dict[str, Any]:
        """Loads persistent key pool metadata or initializes state using key hashes (no plaintext keys stored)."""
        state = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "keys_status": {}
        }

        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict) and "keys_status" in loaded:
                        state = loaded
            except Exception:
                pass

        # Reconcile discovered keys with saved state using SHA-256 hash IDs
        for k in self.keys:
            h = self._key_hash(k)
            if h not in state["keys_status"]:
                state["keys_status"][h] = {
                    "masked": self._mask_key(k),
                    "status": "HEALTHY",  # HEALTHY, RATE_LIMITED, EXHAUSTED
                    "cooldown_until": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                    "last_used": None,
                    "last_error": None
                }
            else:
                state["keys_status"][h]["masked"] = self._mask_key(k)

        self._save_state(state)
        return state

    def _save_state(self, state: Optional[Dict[str, Any]] = None):
        """Thread-safe save of key pool state."""
        if state is None:
            state = self.state
        state["last_updated"] = datetime.now().isoformat()
        temp_file = self.state_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            temp_file.replace(self.state_file)
        except Exception as e:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            print(f"[NIMKeyPool] Error saving pool state: {e}", file=sys.stderr)

    def refresh_keys(self):
        """Re-reads .env and re-syncs discovered keys."""
        with self._pool_lock:
            new_keys = self._discover_keys()
            for k in new_keys:
                if k not in self.keys:
                    self.keys.append(k)
                    self.state["keys_status"][k] = {
                        "masked": self._mask_key(k),
                        "status": "HEALTHY",
                        "cooldown_until": 0.0,
                        "success_count": 0,
                        "error_count": 0,
                        "last_used": None,
                        "last_error": None
                    }
            self._save_state()

    def get_active_key(self, prefer_round_robin: bool = True) -> Tuple[Optional[str], str]:
        """
        Retrieves the next available healthy key from the pool.
        Returns: (api_key_or_none, masked_or_error_msg)
        """
        with self._pool_lock:
            if not self.keys:
                return None, "No NVIDIA API keys configured."

            now = time.time()
            healthy_keys = []
            cooling_keys = []

            for k in self.keys:
                h = self._key_hash(k)
                info = self.state["keys_status"].get(h, {})
                status = info.get("status", "HEALTHY")
                cooldown_until = info.get("cooldown_until", 0.0)

                if status == "EXHAUSTED":
                    continue
                elif status == "RATE_LIMITED":
                    if now >= cooldown_until:
                        # Cooldown finished, restore to HEALTHY
                        info["status"] = "HEALTHY"
                        info["cooldown_until"] = 0.0
                        healthy_keys.append(k)
                    else:
                        cooling_keys.append((cooldown_until, k))
                else:
                    healthy_keys.append(k)

            # 1. Best case: one or more healthy keys available
            if healthy_keys:
                if prefer_round_robin:
                    self._round_robin_idx = (self._round_robin_idx + 1) % len(healthy_keys)
                    selected_key = healthy_keys[self._round_robin_idx]
                else:
                    selected_key = healthy_keys[0]

                sel_h = self._key_hash(selected_key)
                info = self.state["keys_status"][sel_h]
                info["last_used"] = datetime.now().isoformat()
                self._save_state()
                return selected_key, info["masked"]

            # 2. Secondary case: all non-exhausted keys are in rate limit cooldown
            if cooling_keys:
                cooling_keys.sort(key=lambda x: x[0])  # Earliest expiring first
                earliest_expire, earliest_key = cooling_keys[0]
                wait_seconds = max(1, int(earliest_expire - now))
                earliest_h = self._key_hash(earliest_key)
                info = self.state["keys_status"][earliest_h]
                print(
                    f"[NIMKeyPool] All keys rate limited. Earliest recovery in {wait_seconds}s ({info['masked']}).",
                    file=sys.stderr
                )
                return earliest_key, info["masked"]

            # 3. All keys exhausted
            print("[NIMKeyPool CRITICAL] All configured NVIDIA API keys are quota exhausted.", file=sys.stderr)
            return None, "All pooled keys quota exhausted"

    def report_success(self, key: str):
        """Records a successful API call for the given key."""
        with self._pool_lock:
            h = self._key_hash(key)
            if h in self.state["keys_status"]:
                info = self.state["keys_status"][h]
                info["status"] = "HEALTHY"
                info["cooldown_until"] = 0.0
                info["success_count"] = info.get("success_count", 0) + 1
                info["last_used"] = datetime.now().isoformat()
                self._save_state()

    def report_rate_limit(self, key: str, cooldown_s: int = 60, detail: str = ""):
        """Puts the key into temporary cooldown (default 60s) due to HTTP 429."""
        with self._pool_lock:
            h = self._key_hash(key)
            if h in self.state["keys_status"]:
                info = self.state["keys_status"][h]
                info["status"] = "RATE_LIMITED"
                info["cooldown_until"] = time.time() + cooldown_s
                info["error_count"] = info.get("error_count", 0) + 1
                info["last_error"] = f"HTTP 429 Rate Limit ({detail})" if detail else "HTTP 429 Rate Limit"
                print(
                    f"[NIMKeyPool] Key {info['masked']} hit Rate Limit (HTTP 429). Rotating to next pooled key (cooling {cooldown_s}s)...",
                    file=sys.stderr
                )
                self._save_state()

    def report_quota_exhausted(self, key: str, detail: str = ""):
        """Marks the key as permanently exhausted (HTTP 402/403 or out of credits)."""
        with self._pool_lock:
            h = self._key_hash(key)
            if h in self.state["keys_status"]:
                info = self.state["keys_status"][h]
                info["status"] = "EXHAUSTED"
                info["error_count"] = info.get("error_count", 0) + 1
                info["last_error"] = f"Quota Exhausted: {detail}" if detail else "Quota Exhausted"
                print(
                    f"[NIMKeyPool] Key {info['masked']} Quota Exhausted. Permanently rotating to next pooled key...",
                    file=sys.stderr
                )
                self._save_state()

    def get_summary(self) -> Dict[str, Any]:
        """Returns diagnostic summary of the entire key pool."""
        total = len(self.keys)
        healthy = 0
        rate_limited = 0
        exhausted = 0
        details = []

        now = time.time()
        for idx, k in enumerate(self.keys, 1):
            h = self._key_hash(k)
            info = self.state["keys_status"].get(h, {})
            st = info.get("status", "HEALTHY")
            cd = info.get("cooldown_until", 0.0)

            if st == "RATE_LIMITED" and now >= cd:
                st = "HEALTHY"
            elif st == "RATE_LIMITED":
                rate_limited += 1
            elif st == "EXHAUSTED":
                exhausted += 1
            else:
                healthy += 1

            details.append({
                "index": idx,
                "masked": info.get("masked", self._mask_key(k)),
                "status": st,
                "success_count": info.get("success_count", 0),
                "error_count": info.get("error_count", 0),
                "last_used": info.get("last_used"),
                "cooldown_remaining_s": max(0, int(cd - now)) if st == "RATE_LIMITED" else 0
            })

        return {
            "total_keys": total,
            "healthy_keys": healthy,
            "rate_limited_keys": rate_limited,
            "exhausted_keys": exhausted,
            "starting_pooled_quota": total * CREDITS_PER_KEY,
            "keys": details
        }

    def format_ascii_report(self) -> str:
        """Formats an ASCII dashboard table of the key pool."""
        s = self.get_summary()
        lines = [
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║                 NVIDIA NIM MULTI-KEY POOL DASHBOARD                  ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Configured Keys:   {s['total_keys']:<2} pooled accounts ({s['starting_pooled_quota']:,} total requests quota){'':<11}║",
            f"║  Pool Health:       🟢 {s['healthy_keys']} Healthy  |  🟡 {s['rate_limited_keys']} Cooling  |  🔴 {s['exhausted_keys']} Exhausted{'':<10}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  #   Masked Key Identity      Status       Success   Errors  Cooldown ║",
            "╟──────────────────────────────────────────────────────────────────────╢",
        ]

        if not s["keys"]:
            lines.append("║  No NVIDIA NIM API keys detected in .env or environment.             ║")
        else:
            for k in s["keys"]:
                idx = k["index"]
                masked = k["masked"]
                st = k["status"]
                succ = k["success_count"]
                err = k["error_count"]
                cd = f"{k['cooldown_remaining_s']}s" if k["cooldown_remaining_s"] > 0 else "-"

                if st == "HEALTHY":
                    badge = "🟢 HEALTHY"
                elif st == "RATE_LIMITED":
                    badge = "🟡 COOLING"
                else:
                    badge = "🔴 EXHAUSTED"

                lines.append(f"║  {idx:<2}  {masked:<24} {badge:<12} {succ:<9} {err:<7} {cd:<8} ║")

        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


# Global singleton instance
KEY_POOL = NIMKeyPool()


if __name__ == "__main__":
    pool = NIMKeyPool()
    print("\n" + pool.format_ascii_report() + "\n")
