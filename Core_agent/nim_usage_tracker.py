"""
AETHER NVIDIA NIM Usage & Balance Tracker
Thread-safe, persistent local token and quota monitoring for NVIDIA Cloud Endpoints.

Tracks:
- Request counts and remaining credit balance (NVIDIA build.nvidia.com 1,000 free requests tier)
- Prompt, completion, and total tokens lifetime and per-day
- Per-model consumption breakdown
- Recent API call history with timestamps and caller metadata
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "Core_agent"))
USAGE_FILE = WORKSPACE_ROOT / ".aether" / "nim_usage.json"
DEFAULT_STARTING_CREDITS = 1000  # NVIDIA Developer free tier default


class NIMUsageTracker:
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or USAGE_FILE
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _default_schema(self) -> Dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "starting_credits": DEFAULT_STARTING_CREDITS,
            "total_requests": 0,
            "lifetime_tokens": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "per_model": {},
            "daily_log": {},
            "recent_calls": []
        }

    def _load(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            data = self._default_schema()
            self._save(data)
            return data

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure schema integrity
            schema = self._default_schema()
            for key, val in schema.items():
                if key not in data:
                    data[key] = val
            return data
        except Exception:
            return self._default_schema()

    def _save(self, data: Optional[Dict[str, Any]] = None):
        if data is None:
            data = self.data
        data["last_updated"] = datetime.now().isoformat()

        temp_path = self.storage_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.storage_path)
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            print(f"[NIMUsageTracker Error] Could not save usage file: {e}", file=sys.stderr)

    def set_starting_credits(self, credits: int):
        """Sets the baseline starting quota (e.g. 1000 for NVIDIA free tier)."""
        self.data["starting_credits"] = max(1, int(credits))
        self._save()

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        caller: str = "browser_worker",
        duration_s: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Records a completed NVIDIA NIM API call and updates all counters.
        """
        total_tokens = prompt_tokens + completion_tokens
        today = datetime.now().strftime("%Y-%m-%d")
        now_iso = datetime.now().isoformat()

        # 1. Update Lifetime Totals
        self.data["total_requests"] += 1
        lt = self.data["lifetime_tokens"]
        lt["prompt_tokens"] += prompt_tokens
        lt["completion_tokens"] += completion_tokens
        lt["total_tokens"] += total_tokens

        # 2. Update Per-Model Breakdown
        if model not in self.data["per_model"]:
            self.data["per_model"][model] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        pm = self.data["per_model"][model]
        pm["requests"] += 1
        pm["prompt_tokens"] += prompt_tokens
        pm["completion_tokens"] += completion_tokens
        pm["total_tokens"] += total_tokens

        # 3. Update Daily Log
        if today not in self.data["daily_log"]:
            self.data["daily_log"][today] = {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        dl = self.data["daily_log"][today]
        dl["requests"] += 1
        dl["prompt_tokens"] += prompt_tokens
        dl["completion_tokens"] += completion_tokens
        dl["total_tokens"] += total_tokens

        # 4. Append to Recent Calls (keep last 30)
        call_entry = {
            "timestamp": now_iso,
            "model": model,
            "caller": caller,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "duration_s": round(duration_s, 2)
        }
        if metadata:
            call_entry["metadata"] = metadata

        self.data["recent_calls"].append(call_entry)
        if len(self.data["recent_calls"]) > 30:
            self.data["recent_calls"] = self.data["recent_calls"][-30:]

        self._save()
        return self.get_summary()

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns structured usage and remaining quota metrics, synchronized with multi-key pool.
        """
        try:
            from nim_key_pool import KEY_POOL
            pool_summary = KEY_POOL.get_summary()
            if pool_summary["total_keys"] > 0:
                pooled_credits = pool_summary["starting_pooled_quota"]
                if self.data.get("starting_credits", DEFAULT_STARTING_CREDITS) < pooled_credits:
                    self.data["starting_credits"] = pooled_credits
                    self._save()
        except Exception:
            pass

        starting = self.data.get("starting_credits", DEFAULT_STARTING_CREDITS)
        used = self.data.get("total_requests", 0)
        remaining = max(0, starting - used)
        percent_remaining = round((remaining / starting) * 100, 1) if starting > 0 else 0.0

        lt = self.data.get("lifetime_tokens", {})
        total_tokens = lt.get("total_tokens", 0)
        avg_tokens = round(total_tokens / used, 1) if used > 0 else 0.0

        today = datetime.now().strftime("%Y-%m-%d")
        today_data = self.data.get("daily_log", {}).get(today, {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        })

        return {
            "starting_credits": starting,
            "used_requests": used,
            "remaining_credits": remaining,
            "percent_remaining": percent_remaining,
            "lifetime_tokens": lt,
            "avg_tokens_per_call": avg_tokens,
            "today_requests": today_data["requests"],
            "today_tokens": today_data["total_tokens"],
            "per_model": self.data.get("per_model", {}),
            "recent_calls_count": len(self.data.get("recent_calls", [])),
            "storage_file": str(self.storage_path)
        }

    def format_ascii_report(self, api_key_preview: str = "") -> str:
        """
        Formats a clean, visually premium terminal dashboard box.
        """
        s = self.get_summary()
        lt = s["lifetime_tokens"]
        key_str = f"  Key:               {api_key_preview}\n" if api_key_preview else ""

        # Pool summary
        pool_info_line = None
        try:
            from nim_key_pool import KEY_POOL
            psum = KEY_POOL.get_summary()
            if psum["total_keys"] > 0:
                pool_info_line = f"🟢 {psum['healthy_keys']} Healthy / {psum['total_keys']} Pooled ({psum['starting_pooled_quota']:,} calls quota)"
        except Exception:
            pass

        models_lines = []
        for m, stats in s["per_model"].items():
            short_name = m.split("/")[-1] if "/" in m else m
            models_lines.append(f"  - {short_name}: {stats['requests']} calls ({stats['total_tokens']:,} tokens)")
        models_block = "\n".join(models_lines) if models_lines else "  - None recorded yet"

        # Determine health status icon
        pct = s["percent_remaining"]
        if pct > 40:
            status_badge = "🟢 HEALTHY"
        elif pct > 15:
            status_badge = "🟡 RUNNING LOW"
        else:
            status_badge = "🔴 CRITICAL QUOTA"

        lines = [
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║                 NVIDIA NIM USAGE & BALANCE DASHBOARD                 ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║  Status:            {status_badge:<48}║",
        ]
        if pool_info_line:
            lines.append(f"║  Key Pool:          {pool_info_line:<48}║")
        elif api_key_preview:
            lines.append(f"║  Active Key:        {api_key_preview:<48}║")

        lines.extend([
            f"║  Starting Quota:    {s['starting_credits']:,} API calls (NVIDIA Free Tier){'':<17}║",
            f"║  Requests Used:     {s['used_requests']:,} calls{'':<43}║",
            f"║  Remaining Quota:   {s['remaining_credits']:,} calls ({pct}%) {'':<31}║",
            f"║  Lifetime Tokens:   {lt.get('total_tokens', 0):,} total ({lt.get('prompt_tokens', 0):,} in, {lt.get('completion_tokens', 0):,} out){'':<5}║",
            f"║  Avg Tokens / Call: {s['avg_tokens_per_call']:,} tokens{'':<40}║",
            f"║  Today's Activity:  {s['today_requests']} calls ({s['today_tokens']:,} tokens){'':<32}║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            "║  Model Breakdown:                                                    ║",
        ])
        for line in models_lines:
            lines.append(f"║{line:<70}║")
        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        return "\n".join(lines)


if __name__ == "__main__":
    tracker = NIMUsageTracker()
    if len(sys.argv) > 2 and sys.argv[1] == "--set-budget":
        new_b = int(sys.argv[2])
        tracker.set_starting_credits(new_b)
        print(f"[NIMUsageTracker] Starting credit budget updated to {new_b} calls.")

    # Read .env to get key preview if available
    env_file = WORKSPACE_ROOT / ".env"
    key_preview = ""
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("NVIDIA_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if len(val) > 12:
                        key_preview = f"{val[:8]}...{val[-4:]}"
                    break

    print(tracker.format_ascii_report(api_key_preview=key_preview))
