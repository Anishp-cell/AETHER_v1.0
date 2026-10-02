"""
AETHER Autonomous Browser Control Agent
Powered by Browser-Use + Playwright + NVIDIA NIM / Cloud Inference

Provides autonomous web browser control for AETHER:
- Navigating complex multi-page workflows (flight booking, e-commerce, forms)
- Extracting dynamic JavaScript rendered data
- Handling interactive elements (clicks, dropdowns, inputs)
- Supporting both headed (visible desktop UI) and headless modes
- Seamless cross-environment bridging (.venv / .venv_browser)
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Root workspace path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

# Attempt loading .env if present
env_file = WORKSPACE_ROOT / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    except ImportError:
        pass


def _find_browser_python() -> Optional[str]:
    """
    Locates a Python 3.11+ environment capable of running browser-use.
    Checks:
    1. Current sys.executable (if Python >= 3.11)
    2. .venv_browser in workspace root
    3. System py launcher (py -3.12 / py -3.11)
    """
    if sys.version_info >= (3, 11):
        try:
            import browser_use
            return sys.executable
        except ImportError:
            pass

    # Check .venv_browser in workspace
    venv_browser_python = WORKSPACE_ROOT / ".venv_browser" / "Scripts" / "python.exe"
    if venv_browser_python.exists():
        return str(venv_browser_python)

    # Check Linux/macOS path just in case
    venv_browser_posix = WORKSPACE_ROOT / ".venv_browser" / "bin" / "python"
    if venv_browser_posix.exists():
        return str(venv_browser_posix)

    return None


def _run_via_worker_process(task: str, headless: bool = False, max_steps: int = 25) -> Dict[str, Any]:
    """
    Runs the browser task in an external worker process using .venv_browser (Python 3.12).
    This cleanly isolates Playwright and Browser-Use from the host Python 3.10 environment.
    """
    worker_script = Path(__file__).resolve().parent / "browser_worker.py"
    target_python = _find_browser_python()

    if not target_python:
        return {
            "status": "not_installed",
            "error": (
                "[AetherBrowser] Python 3.11+ browser environment not found.\n"
                "To set it up, run in your terminal:\n"
                "  py -3.12 -m venv .venv_browser\n"
                "  .venv_browser\\Scripts\\pip install browser-use playwright langchain-openai python-dotenv\n"
                "  .venv_browser\\Scripts\\playwright install chromium"
            )
        }

    payload = json.dumps({
        "task": task,
        "headless": headless,
        "max_steps": max_steps
    })

    cmd = [target_python, str(worker_script), "--json-input"]
    if headless:
        cmd.append("--headless")
    else:
        cmd.append("--headed")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(WORKSPACE_ROOT),
            encoding="utf-8",
            errors="replace"
        )
        stdout, stderr = proc.communicate(input=payload, timeout=180)

        # Print worker stderr logs in real-time for AETHER visibility
        if stderr and "[AetherBrowserWorker]" in stderr:
            for line in stderr.splitlines():
                if "[AetherBrowserWorker]" in line:
                    print(f"  {line}")

        # Find the last valid JSON output line
        last_json = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    last_json = json.loads(line)
                except Exception:
                    pass

        if last_json:
            return last_json

        if proc.returncode != 0:
            return {
                "status": "error",
                "error": f"Worker process failed with code {proc.returncode}: {stderr[:300]}"
            }

        return {
            "status": "error",
            "error": f"Invalid output from browser worker: {stdout[:200]}"
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        return {
            "status": "error",
            "error": "Browser task timed out after 180 seconds."
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to execute browser worker process: {e}"
        }


def _run_in_process_direct(task: str, headless: bool = False, max_steps: int = 25) -> Dict[str, Any]:
    """Runs directly in-process if browser_use is importable in current Python."""
    try:
        from browser_worker import execute_task
        return asyncio.run(execute_task(task=task, headless=headless, max_steps=max_steps))
    except Exception as e:
        return {"status": "error", "error": str(e)}


def browser_task(task: str, headless: bool = False, max_steps: int = 25) -> str:
    """
    Main AETHER Tool Function for Autonomous Browser Tasks.
    
    Args:
        task: Natural language description of what to do in the browser
              (e.g., 'Go to amazon.in, search for RTX 5070 laptops, and extract top 3 models and prices')
        headless: If True, browser runs in the background. If False, opens a visible browser window.
        max_steps: Maximum interaction steps before stopping (default 25)

    Returns:
        Spoken/Text summary of the browser action result for AETHER.
    """
    print(f"\n🌐 [AetherBrowser] Initiating Autonomous Browser Task:")
    print(f"   Task: \"{task}\"")
    print(f"   Mode: {'Headless (Background)' if headless else 'Headed (Visible Window)'}")

    # Check if direct in-process execution is possible
    can_direct = False
    if sys.version_info >= (3, 11):
        try:
            import browser_use
            can_direct = True
        except ImportError:
            can_direct = False

    if can_direct:
        result_dict = _run_in_process_direct(task, headless=headless, max_steps=max_steps)
    else:
        result_dict = _run_via_worker_process(task, headless=headless, max_steps=max_steps)

    status = result_dict.get("status", "error")
    if status == "success":
        final_text = result_dict.get("final_result", "Browser task completed.")
        urls = result_dict.get("urls", [])
        if urls:
            try:
                from site_history_manager import SiteHistoryManager
                SiteHistoryManager().record_urls(urls)
            except Exception:
                pass
        url_context = f"\n(Visited: {', '.join(urls[:3])})" if urls else ""
        tok = result_dict.get("token_usage", {})
        if tok and tok.get("calls", 0) > 0:
            print(f"📊 [AetherBrowser] NIM Tokens: {tok.get('total_tokens', 0):,} ({tok.get('prompt_tokens', 0):,} prompt, {tok.get('completion_tokens', 0):,} completion across {tok.get('calls')} steps)")
        print(f"✅ [AetherBrowser] Task completed successfully.{url_context}")
        return f"[AetherBrowser Action Result]:\n{final_text}{url_context}\n\n(Summarize this outcome concisely for the user)."

    elif status == "not_installed":
        msg = result_dict.get("error", "browser-use environment not configured.")
        print(f"⚠️ [AetherBrowser] {msg}")
        return f"[AetherBrowser Notice]: The browser automation engine requires setup. Please install the browser dependencies in .venv_browser:\n1. .venv_browser\\Scripts\\pip install browser-use playwright langchain-openai python-dotenv\n2. .venv_browser\\Scripts\\playwright install chromium"

    else:
        err = result_dict.get("error", "Unknown browser error.")
        print(f"❌ [AetherBrowser Error] {err}")
        return f"[AetherBrowser Error]: Could not complete browser task. Details: {err}"


def browser_open_and_interact(url: str, goal: str, headless: bool = False) -> str:
    """
    Navigates to a specific URL and completes an objective on the page.
    """
    full_task = f"Navigate to '{url}'. Once loaded, {goal}"
    return browser_task(task=full_task, headless=headless)


async def async_browser_task(task: str, headless: bool = False, max_steps: int = 25) -> str:
    """Async variant for direct event-loop dispatch."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, browser_task, task, headless, max_steps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AETHER Browser Agent & Diagnostics CLI")
    parser.add_argument("task", nargs="*", default=["Go to duckduckgo.com and search for 'NVIDIA Cosmos'"], help="Task description")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--usage", action="store_true", help="Display NVIDIA NIM token usage and remaining quota dashboard")
    parser.add_argument("--diag", action="store_true", help="Run comprehensive NVIDIA NIM key and latency diagnostics")
    parser.add_argument("--sites", action="store_true", help="Display most-visited base domain history recorded during AETHER sessions")
    args = parser.parse_args()

    if args.usage:
        from nim_usage_tracker import NIMUsageTracker
        tracker = NIMUsageTracker()
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
        sys.exit(0)

    if args.diag:
        from nim_diagnostics import run_diagnostics
        run_diagnostics()
        sys.exit(0)

    if args.sites:
        from site_history_manager import SiteHistoryManager
        print(SiteHistoryManager().format_ascii_report())
        sys.exit(0)

    test_task = " ".join(args.task)
    output = browser_task(test_task, headless=args.headless)
    print("\nResult:")
    print(output)
