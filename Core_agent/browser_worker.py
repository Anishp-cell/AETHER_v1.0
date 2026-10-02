"""
AETHER Browser Worker
Standalone worker process for Browser-Use automation.
Runs within Python 3.11+ environment (.venv_browser or direct runtime).
"""
import sys
import os
import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from workspace root .env if present
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
env_path = WORKSPACE_ROOT / ".env"

if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path)
    except ImportError:
        # Simple manual fallback parser for .env
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            pass


sys.path.insert(0, str(WORKSPACE_ROOT / "Core_agent"))

class TokenTracker:
    """Captures token usage metadata from LLM responses across browser steps."""
    def __init__(self, model_name: str = ""):
        self.model_name = model_name
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def record_step(self, prompt: int, completion: int, total: int = 0):
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total if total > 0 else (prompt + completion)


def _build_tracking_llm_class(base_cls):
    """Dynamically wraps a ChatOpenAI class to intercept token usage on ainvoke and report key pool status."""
    class TrackingChatModel(base_cls):
        def __init__(self, *args, token_tracker=None, active_key=None, **kwargs):
            # browser_use.ChatOpenAI does not accept callbacks; pop if not supported
            callbacks = kwargs.pop("callbacks", None)
            try:
                super().__init__(*args, **kwargs)
            except TypeError:
                if callbacks is not None:
                    kwargs["callbacks"] = callbacks
                super().__init__(*args, **kwargs)
            self.token_tracker = token_tracker
            self.active_key = active_key

        async def ainvoke(self, messages, output_format=None, **kwargs):
            try:
                res = await super().ainvoke(messages, output_format=output_format, **kwargs)
                if hasattr(res, "usage") and res.usage and self.token_tracker:
                    p = getattr(res.usage, "prompt_tokens", 0) or 0
                    c = getattr(res.usage, "completion_tokens", 0) or 0
                    t = getattr(res.usage, "total_tokens", p + c) or (p + c)
                    self.token_tracker.record_step(p, c, t)
                if self.active_key:
                    try:
                        from nim_key_pool import KEY_POOL
                        KEY_POOL.report_success(self.active_key)
                    except Exception:
                        pass
                return res
            except Exception as e:
                err_str = str(e).lower()
                if self.active_key:
                    try:
                        from nim_key_pool import KEY_POOL
                        if "429" in err_str or "rate limit" in err_str:
                            KEY_POOL.report_rate_limit(self.active_key, cooldown_s=60, detail=str(e)[:100])
                        elif any(x in err_str for x in ["402", "403", "quota", "credit"]):
                            KEY_POOL.report_quota_exhausted(self.active_key, detail=str(e)[:100])
                    except Exception:
                        pass
                raise

    return TrackingChatModel


def get_llm(tracker: Optional[TokenTracker] = None):
    """
    Dynamically select and configure the best available LLM provider for Browser-Use.
    Prioritizes NVIDIA NIM (with multi-key pool failover), followed by OpenAI, Groq, and local Ollama.
    """
    try:
        from browser_use import ChatOpenAI as BaseChatOpenAI
    except ImportError:
        try:
            from langchain_openai import ChatOpenAI as BaseChatOpenAI
        except ImportError:
            raise RuntimeError("Neither browser_use.ChatOpenAI nor langchain-openai is installed.")

    ChatModelClass = _build_tracking_llm_class(BaseChatOpenAI)

    llm = None
    # 1. NVIDIA NIM Cloud Inference with Multi-Key Pool (Preferred for AETHER v4.0)
    nvidia_key = None
    key_preview = ""
    try:
        from nim_key_pool import KEY_POOL
        nvidia_key, key_preview = KEY_POOL.get_active_key()
    except Exception:
        pass

    if not nvidia_key:
        nvidia_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NIM_API_KEY")
        key_preview = nvidia_key[:8] + "..." if nvidia_key else ""

    if nvidia_key:
        model = os.getenv("NVIDIA_BROWSER_MODEL", "meta/llama-3.2-11b-vision-instruct")
        print(f"[AetherBrowserWorker] Using NVIDIA NIM LLM: {model} (Key: {key_preview})", file=sys.stderr)
        llm = ChatModelClass(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_key,
            model=model,
            temperature=0.0,
            add_schema_to_system_prompt=True,
            dont_force_structured_output=True,
            max_completion_tokens=4096,
            token_tracker=tracker,
            active_key=nvidia_key
        )

    # 2. OpenAI API
    elif os.getenv("OPENAI_API_KEY"):
        openai_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_BROWSER_MODEL", "gpt-4o-mini")
        print(f"[AetherBrowserWorker] Using OpenAI LLM: {model}", file=sys.stderr)
        llm = ChatModelClass(
            api_key=openai_key,
            model=model,
            temperature=0.0,
            token_tracker=tracker
        )

    # 3. Groq API
    elif os.getenv("GROQ_API_KEY"):
        groq_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_BROWSER_MODEL", "llama-3.3-70b-versatile")
        print(f"[AetherBrowserWorker] Using Groq LLM: {model}", file=sys.stderr)
        llm = ChatModelClass(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            model=model,
            temperature=0.0,
            token_tracker=tracker
        )

    # 4. Local Ollama (fallback)
    else:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
        ollama_model = os.getenv("OLLAMA_BROWSER_MODEL", "qwen2.5:7b")
        print(f"[AetherBrowserWorker] Falling back to local Ollama: {ollama_model} at {ollama_url}", file=sys.stderr)
        llm = ChatModelClass(
            base_url=ollama_url,
            api_key="ollama",
            model=ollama_model,
            temperature=0.0,
            token_tracker=tracker
        )

    if llm is not None and not hasattr(llm, "provider"):
        setattr(llm, "provider", "openai")

    return llm


async def execute_task(task: str, headless: bool = False, max_steps: int = 25) -> dict:
    """Runs a task using browser-use Agent with real-time token tracking."""
    try:
        from browser_use import Agent
    except ImportError as e:
        return {
            "status": "error",
            "error": f"browser-use is not installed in this environment ({sys.executable}): {e}."
        }

    try:
        model_name = os.getenv("NVIDIA_BROWSER_MODEL", "meta/llama-3.2-11b-vision-instruct")
        token_tracker = TokenTracker(model_name=model_name)
        llm = get_llm(tracker=token_tracker)

        # Configure browser profile / session adaptively
        agent_kwargs = {"task": task, "llm": llm}
        try:
            from browser_use import BrowserProfile
            agent_kwargs["browser_profile"] = BrowserProfile(
                headless=headless,
                enable_default_extensions=False
            )
        except Exception:
            try:
                from browser_use import Browser
                agent_kwargs["browser"] = Browser(headless=headless)
        # Step callback for real-time base domain tracking
        def _step_url_callback(state, output, step_idx):
            try:
                if state and getattr(state, "url", None):
                    from site_history_manager import SiteHistoryManager
                    SiteHistoryManager().record_visit(state.url)
            except Exception:
                pass

        agent_kwargs["register_new_step_callback"] = _step_url_callback

        print(f"[AetherBrowserWorker] Starting browser agent (headless={headless})...", file=sys.stderr)
        agent = Agent(**agent_kwargs)

        history = await agent.run(max_steps=max_steps)
        final_text = history.final_result() or "Browser automation task completed successfully."

        urls = []
        try:
            urls = history.urls()
            if urls:
                try:
                    from site_history_manager import SiteHistoryManager
                    SiteHistoryManager().record_urls(urls)
                except Exception:
                    pass
        except Exception:
            pass

        # Record LLM token usage if tokens were tracked
        if token_tracker.total_tokens > 0 or token_tracker.calls > 0:
            try:
                from nim_usage_tracker import NIMUsageTracker
                tracker = NIMUsageTracker()
                tracker.record_usage(
                    model=token_tracker.model_name,
                    prompt_tokens=token_tracker.prompt_tokens,
                    completion_tokens=token_tracker.completion_tokens,
                    caller="browser_worker",
                    metadata={"task": task[:60], "steps": token_tracker.calls}
                )
                print(f"[AetherBrowserWorker] Tracked NIM Tokens: {token_tracker.total_tokens} ({token_tracker.prompt_tokens} in, {token_tracker.completion_tokens} out)", file=sys.stderr)
            except Exception as te:
                print(f"[AetherBrowserWorker] Token recording warning: {te}", file=sys.stderr)

        return {
            "status": "success",
            "final_result": final_text,
            "urls": urls,
            "token_usage": {
                "prompt_tokens": token_tracker.prompt_tokens,
                "completion_tokens": token_tracker.completion_tokens,
                "total_tokens": token_tracker.total_tokens,
                "calls": token_tracker.calls
            },
            "is_done": history.is_done() if hasattr(history, "is_done") else True
        }

    except Exception as e:
        print(f"[AetherBrowserWorker Error] {e}", file=sys.stderr)
        return {
            "status": "error",
            "error": str(e)
        }


def try_channel_1_fast_path(task: str, headless: bool = False) -> Optional[dict]:
    """
    Checks if task matches Channel 1 deterministic Playwright patterns.
    Returns result dict if matched and executed, or None to fall back to Channel 2.
    """
    import re
    from fast_web_driver import run_fast_web_task

    low = task.lower().strip()

    # 1. YouTube playback
    if "youtube" in low and any(w in low for w in ["play", "watch", "song", "video", "music"]):
        m_yt = (
            re.search(r'(?:play|watch)\s+(.+?)(?:\s+on\s+youtube|\s+in\s+youtube|$)', low)
            or re.search(r'youtube.*?(?:play|search)\s+(.+)', low)
            or re.search(r'(?:play|watch)\s+(.+)', low)
        )
        query = m_yt.group(1).strip().strip("'\"") if m_yt else "lofi hip hop"
        res = run_fast_web_task("youtube", query, headless=headless)
        if res.get("status") == "success":
            return {
                "status": "success",
                "final_result": res.get("message", f"Playing {query} on YouTube"),
                "urls": [res.get("url", "https://youtube.com")],
                "channel": "Channel 1 (Fast Playwright)"
            }
        print(f"[AetherBrowserWorker] Channel 1 YouTube fallback: {res.get('error')}", file=sys.stderr)

    # 2. Direct URL opening (e.g., 'go to news.ycombinator.com', 'open github.com')
    url_match = re.search(r'(?:open|go to|navigate to|visit)\s+([a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,}(?:/[^\s]*)?)', task, re.IGNORECASE)
    if url_match and not any(w in low for w in ["book", "cart", "checkout", "buy", "fill", "login"]):
        target_url = url_match.group(1).strip().strip("'\"")
        res = run_fast_web_task("open_url", target_url, headless=headless)
        if res.get("status") == "success":
            return {
                "status": "success",
                "final_result": res.get("message", f"Opened {target_url}"),
                "urls": [res.get("url", target_url)],
                "channel": "Channel 1 (Fast Playwright)"
            }
        print(f"[AetherBrowserWorker] Channel 1 Direct URL fallback: {res.get('error')}", file=sys.stderr)

    # 3. Deterministic Web Search & Extract
    is_search = any(k in low for k in ["search", "google", "look up", "find information", "query", "duckduckgo", "who is", "what is"])
    is_complex = any(w in low for w in ["book", "flight", "buy", "cart", "checkout", "login", "password", "sign in", "fill form", "click button"])
    if is_search and not is_complex:
        q_match = (
            re.search(r"(?:search for|search|look up|query)\s+['\"]?([^'\"]+?)['\"]?(?:\s+and\s+extract|\s+on|$)", task, re.IGNORECASE)
            or re.search(r"['\"]([^'\"]+)['\"]", task)
        )
        query = q_match.group(1).strip() if q_match else task
        res = run_fast_web_task("search", query, headless=headless)
        if res.get("status") == "success":
            return {
                "status": "success",
                "final_result": res.get("message", f"Search completed for {query}"),
                "urls": [res.get("url", "https://search.yahoo.com")],
                "channel": "Channel 1 (Fast Playwright)"
            }
        print(f"[AetherBrowserWorker] Channel 1 Web Search fallback: {res.get('error')}", file=sys.stderr)

    return None



def main():
    parser = argparse.ArgumentParser(description="AETHER Browser Worker")
    parser.add_argument("--task", type=str, help="Automation task description for the browser")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless background mode")
    parser.add_argument("--headed", action="store_true", help="Run browser in headed visible window mode")
    parser.add_argument("--max-steps", type=int, default=25, help="Maximum number of steps the agent can take")
    parser.add_argument("--test", action="store_true", help="Verify dependencies and LLM configuration")
    parser.add_argument("--json-input", action="store_true", help="Read payload as JSON from stdin")

    args = parser.parse_args()

    if args.test:
        print("[AetherBrowserWorker] Diagnostic check:")
        print(f"  Python executable: {sys.executable}")
        print(f"  Python version: {sys.version}")
        try:
            import browser_use
            print(f"  browser-use: installed (v{getattr(browser_use, '__version__', 'unknown')})")
        except ImportError as e:
            print(f"  browser-use: NOT installed ({e})")

        try:
            import playwright
            print(f"  playwright: installed")
        except ImportError as e:
            print(f"  playwright: NOT installed ({e})")

        try:
            llm = get_llm()
            print(f"  LLM configured: {type(llm).__name__}")
        except Exception as e:
            print(f"  LLM configuration error: {e}")

        sys.exit(0)

    task = args.task
    headless = False
    max_steps = args.max_steps

    if args.headed:
        headless = False
    elif args.headless:
        headless = True

    if args.json_input or not task:
        try:
            input_data = sys.stdin.read()
            if input_data.strip():
                payload = json.loads(input_data)
                task = payload.get("task", task)
                headless = payload.get("headless", headless)
                max_steps = payload.get("max_steps", max_steps)
        except Exception as e:
            result = {"status": "error", "error": f"Failed to parse stdin JSON: {e}"}
            print(json.dumps(result))
            sys.exit(1)

    if not task:
        result = {"status": "error", "error": "No task provided."}
        print(json.dumps(result))
        sys.exit(1)

    # ── Channel 1: Fast Deterministic Playwright (Visual Glow Highlighting, Sub-2s) ──
    fast_result = try_channel_1_fast_path(task=task, headless=headless)
    if fast_result is not None:
        print(json.dumps(fast_result, ensure_ascii=False))
        sys.exit(0)

    # ── Channel 2: Autonomous AI Agent Loop (browser-use + DeepSeek / NIM LLM) ──
    result = asyncio.run(execute_task(task=task, headless=headless, max_steps=max_steps))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
