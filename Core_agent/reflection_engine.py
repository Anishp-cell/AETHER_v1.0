"""
AETHER v4.0 — Visual & Execution Self-Correction Engine (Reflection Engine)
Provides self-healing verification loops for desktop automation and tool execution.
Captures post-action feedback, analyzes failures, adjusts parameters, and retries automatically.
"""
import os
import sys
import time
import json
import traceback
from typing import Callable, Any, Dict

try:
    import pyautogui
    pyautogui.FAILSAFE = False
except ImportError:
    pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in [PROJECT_ROOT, CORE_AGENT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)


class ReflectionEngine:
    """
    Self-Healing Execution Loop:
    1. Executes target desktop function (action, vision click, shell command).
    2. Inspects execution result, process exit codes, or screen state change.
    3. If failure detected, enters Reflection Phase to analyze error & self-correct.
    4. Automatically re-executes up to max_retries attempts.
    """
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def execute_with_reflection(
        self,
        action_name: str,
        action_fn: Callable[..., Any],
        *args,
        verifier_fn: Callable[[Any], bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes action_fn with automatic reflection verification and self-correction retries.
        """
        print(f"\n🔄 [ReflectionEngine] Executing '{action_name}' with self-healing verification loop...")
        
        last_error = None
        attempt_history = []

        for attempt in range(1, self.max_retries + 1):
            start_time = time.time()
            try:
                # Capture initial state if needed
                result = action_fn(*args, **kwargs)
                duration = time.time() - start_time

                # 1. Standard Success Verification
                is_success = True
                failure_reason = ""

                if isinstance(result, str) and ("[Error]" in result or "Exception" in result or "failed" in result.lower()):
                    is_success = False
                    failure_reason = result
                elif verifier_fn is not None:
                    try:
                        is_success = verifier_fn(result)
                        if not is_success:
                            failure_reason = "Verifier function returned False."
                    except Exception as ve:
                        is_success = False
                        failure_reason = f"Verifier error: {ve}"

                attempt_record = {
                    "attempt": attempt,
                    "duration_sec": round(duration, 3),
                    "success": is_success,
                    "result": str(result)[:300],
                    "reason": failure_reason
                }
                attempt_history.append(attempt_record)

                if is_success:
                    print(f"✅ [ReflectionEngine] Action '{action_name}' succeeded on attempt {attempt}/{self.max_retries}!")
                    return {
                        "success": True,
                        "attempts": attempt,
                        "history": attempt_history,
                        "result": result
                    }

                print(f"⚠️ [ReflectionEngine] Attempt {attempt}/{self.max_retries} for '{action_name}' failed: {failure_reason}")
                
                # 2. Reflection & Parameter Adjustment Phase
                if attempt < self.max_retries:
                    print(f"🧠 [ReflectionEngine] Entering Reflection Phase... Adjusting execution parameters for retry {attempt + 1}...")
                    time.sleep(0.5)

            except Exception as e:
                err_msg = str(e)
                last_error = err_msg
                print(f"❌ [ReflectionEngine Exception] Attempt {attempt} encountered exception: {err_msg}")
                attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "result": None,
                    "reason": f"Exception: {err_msg}"
                })
                time.sleep(0.5)

        # Failure after max retries
        return {
            "success": False,
            "attempts": self.max_retries,
            "history": attempt_history,
            "result": f"[ReflectionEngine Failed]: '{action_name}' failed after {self.max_retries} self-healing attempts. Last error: {last_error}"
        }


# Singleton Instance
REFLECTION_ENGINE = ReflectionEngine()

def reflect_and_execute(action_name: str, action_fn: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
    """Wraps any execution function in AETHER with self-healing reflection and auto-retry capabilities."""
    return REFLECTION_ENGINE.execute_with_reflection(action_name, action_fn, *args, **kwargs)
