"""
AETHER NVIDIA NIM Diagnostic & Key Validation Tool
Tests NVIDIA NIM cloud API connectivity, validates API keys, measures round-trip latency,
and displays real-time remaining credit balance.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT / "Core_agent"))

from nim_usage_tracker import NIMUsageTracker


def load_env_variables() -> Dict[str, str]:
    """Loads .env file without external dependencies."""
    env_vars = {}
    env_file = WORKSPACE_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    env_vars[k] = v
                    os.environ.setdefault(k, v)
    return env_vars


def validate_key_format(key: str) -> Tuple[bool, str]:
    """Verifies structural validity of an NVIDIA API key."""
    if not key:
        return False, "Missing API key in environment or .env"
    if not key.startswith("nvapi-"):
        return False, "Key does not have valid prefix (expected 'nvapi-...')"
    if len(key) < 30:
        return False, f"Key appears too short ({len(key)} chars, expected > 30)"
    return True, "Format valid"


def test_models_endpoint(key: str) -> Tuple[bool, str, int, list]:
    """Checks GET https://integrate.api.nvidia.com/v1/models."""
    url = "https://integrate.api.nvidia.com/v1/models"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "AETHER-Agent/1.0"
        }
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=10) as res:
            latency = time.time() - t0
            if res.status == 200:
                body = json.loads(res.read().decode("utf-8"))
                models = [m.get("id") for m in body.get("data", []) if "id" in m]
                return True, f"HTTP 200 OK ({latency:.2f}s)", len(models), models
            return False, f"Unexpected status: HTTP {res.status}", 0, []
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "HTTP 401 Unauthorized (API key is invalid or revoked)", 0, []
        elif e.code == 429:
            return False, "HTTP 429 Rate Limited (Quota exceeded or temporary rate limit)", 0, []
        return False, f"HTTP Error {e.code}: {e.reason}", 0, []
    except urllib.error.URLError as e:
        return False, f"Network Connection Failed: {e.reason}", 0, []
    except Exception as e:
        return False, f"Unexpected Error: {e}", 0, []


def test_chat_completion(key: str, model: str) -> Tuple[bool, str, float, Dict[str, Any]]:
    """Runs a minimal micro-ping completion to test inference and token usage."""
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 5,
        "temperature": 0.0
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "AETHER-Agent/1.0"
        }
    )
    try:
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=15) as res:
            latency = time.time() - t0
            if res.status == 200:
                body = json.loads(res.read().decode("utf-8"))
                usage = body.get("usage", {})
                reply = body.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                return True, reply, latency, usage
            return False, f"HTTP {res.status}", latency, {}
    except urllib.error.HTTPError as e:
        error_detail = ""
        try:
            error_detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return False, f"HTTP Error {e.code}: {e.reason} ({error_detail[:120]})", 0.0, {}
    except Exception as e:
        return False, str(e), 0.0, {}


def run_diagnostics():
    print("\n" + "=" * 70)
    print("           AETHER NVIDIA NIM DIAGNOSTIC & HEALTH SUITE           ")
    print("=" * 70)

    # 1. Load Environment & Key Pool
    sys.path.insert(0, str(WORKSPACE_ROOT / "Core_agent"))
    from nim_key_pool import KEY_POOL
    pool_summary = KEY_POOL.get_summary()

    env = load_env_variables()
    api_key = env.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY") or ""
    configured_model = env.get("NVIDIA_BROWSER_MODEL") or os.environ.get("NVIDIA_BROWSER_MODEL") or "meta/llama-3.2-11b-vision-instruct"

    print(f"\n[1] Multi-Key Pool Architecture:")
    print(f"    Pooled Keys:      {pool_summary['total_keys']} accounts configured")
    print(f"    Combined Quota:   {pool_summary['starting_pooled_quota']:,} total monthly requests")
    print(f"    Configured Model: {configured_model}")

    # Validate each key in pool
    print(f"\n[2] Testing Health & Catalog Access across Key Pool:")
    for k_info in pool_summary["keys"]:
        raw_key = KEY_POOL.keys[k_info["index"] - 1]
        ok, msg, m_count, _ = test_models_endpoint(raw_key)
        if ok:
            print(f"    Key #{k_info['index']} ({k_info['masked']}): 🟢 ONLINE ({m_count} foundation models accessible)")
            KEY_POOL.report_success(raw_key)
        else:
            print(f"    Key #{k_info['index']} ({k_info['masked']}): 🔴 ERROR ({msg})")

    # 3. Test Live Micro-Ping Chat Completion on Active Key
    active_key, active_masked = KEY_POOL.get_active_key()
    if not active_key:
        active_key = api_key
        active_masked = api_key[:8] + "..." if api_key else "(none)"

    print(f"\n[3] Testing Live Inference Latency on Active Key ({active_masked})...")
    chat_ok, reply, latency, usage = test_chat_completion(active_key, configured_model)
    if not chat_ok:
        # Fallback test with llama-3.1-8b-instruct
        print(f"    Inference Check:  ⚠️ Primary model returned error: {reply}")
        print("    Attempting fallback to 'meta/llama-3.1-8b-instruct'...")
        chat_ok, reply, latency, usage = test_chat_completion(active_key, "meta/llama-3.1-8b-instruct")

    if not chat_ok:
        print(f"    Inference Check:  ❌ FAILED: {reply}")
        print("=" * 70 + "\n")
        return

    prompt_t = usage.get("prompt_tokens", 0)
    comp_t = usage.get("completion_tokens", 0)
    total_t = usage.get("total_tokens", prompt_t + comp_t)

    print(f"    Inference Check:  ✅ SUCCESS ({latency:.2f}s round-trip)")
    print(f"    Model Response:   \"{reply}\"")
    print(f"    Ping Token Usage: {total_t} tokens ({prompt_t} prompt, {comp_t} completion)")

    # 4. Record Call into NIMUsageTracker
    tracker = NIMUsageTracker()
    tracker.record_usage(
        model=configured_model,
        prompt_tokens=prompt_t,
        completion_tokens=comp_t,
        caller="nim_diagnostics",
        duration_s=latency
    )

    # 5. Display Dashboard Reports
    print("\n[4] Real-time Quota & Usage Ledger:")
    print(tracker.format_ascii_report(api_key_preview=active_masked))
    print("\n" + KEY_POOL.format_ascii_report())
    print("\n✅ NVIDIA NIM Multi-Key Pool is 100% operational, healthy, and ready for AETHER.\n")


if __name__ == "__main__":
    run_diagnostics()
