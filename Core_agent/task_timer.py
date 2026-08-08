"""
AETHER v4.0 — Dynamic Task Timer & Adaptive Verification Engine
Implements Antigravity-style background task timing with dynamic duration estimation,
adaptive verification checks, and calculated timer extensions.
"""
import os
import sys
import time
import threading
from typing import Callable, Any, Dict, Optional

class DynamicTaskTimer:
    """
    Antigravity-Style Background Task Timer:
    1. Estimates realistic expected duration T0 based on task metadata.
    2. Runs background execution with a non-blocking adaptive timer.
    3. Verifies task completion at T0; if still in progress, calculates dynamic remaining time deltaT.
    4. Provides vocal and HUD progress notifications.
    """
    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    @staticmethod
    def estimate_duration(task_type: str, payload_size: int = 1, item_count: int = 1) -> float:
        """
        Calculates expected execution duration T0 mathematically based on task physics.
        No random time guessing.
        """
        task_lower = task_type.lower()
        if "math" in task_lower or "calc" in task_lower or "fibonacci" in task_lower:
            return 1.2
        elif "web" in task_lower or "scrape" in task_lower or "search" in task_lower:
            # 1.5s network base + 1.2s per URL/query
            return max(2.5, 1.5 + (1.2 * item_count))
        elif "stock" in task_lower or "crypto" in task_lower or "api" in task_lower:
            return max(2.0, 1.0 + (0.8 * item_count))
        elif "forge" in task_lower or "train" in task_lower or "compile" in task_lower:
            # Code synthesis + subprocess test sandbox takes ~8-15s
            return max(8.0, 6.0 + (0.5 * payload_size))
        elif "screen" in task_lower or "vision" in task_lower or "llava" in task_lower:
            return 3.5
        return 3.0

    def run_with_adaptive_timer(
        self,
        task_name: str,
        task_fn: Callable[..., Any],
        *args,
        task_type: str = "general",
        item_count: int = 1,
        on_heads_up: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[Any, float], None]] = None,
        **kwargs
    ) -> Any:
        """
        Executes task_fn with an adaptive timer, dynamically calculating progress.
        """
        estimated_t0 = self.estimate_duration(task_type, item_count=item_count)
        start_time = time.time()

        task_id = f"task_{int(start_time * 1000)}"
        with self.lock:
            self.active_tasks[task_id] = {
                "name": task_name,
                "type": task_type,
                "status": "running",
                "estimated_t0": estimated_t0,
                "start_time": start_time,
                "extensions": 0,
            }

        print(f"⏱️ [TaskTimer] Started '{task_name}' (Estimated Duration: {estimated_t0:.1f}s)...")

        result_box = [None]
        error_box = [None]
        done_event = threading.Event()

        def worker():
            try:
                result_box[0] = task_fn(*args, **kwargs)
            except Exception as e:
                error_box[0] = e
            finally:
                done_event.set()

        t_thread = threading.Thread(target=worker, daemon=True)
        t_thread.start()

        # Adaptive Timer Loop
        current_timeout = estimated_t0
        while not done_event.wait(timeout=current_timeout):
            elapsed = time.time() - start_time
            # Task is still running past estimated T0 — calculate dynamic remaining time
            extension_time = max(2.0, elapsed * 0.4)
            with self.lock:
                if task_id in self.active_tasks:
                    self.active_tasks[task_id]["extensions"] += 1
                    ext_count = self.active_tasks[task_id]["extensions"]
                else:
                    ext_count = 1

            msg = f"Task '{task_name}' is processing complex data, adding {extension_time:.1f}s timer..."
            print(f"⏳ [TaskTimer] {msg} (Elapsed: {elapsed:.1f}s)")
            if on_heads_up and ext_count <= 2:
                try:
                    on_heads_up(f"Task is still processing, continuing in background...")
                except Exception:
                    pass

            current_timeout = extension_time
            if elapsed > 60.0:  # Max guard timeout
                break

        elapsed_total = time.time() - start_time
        with self.lock:
            self.active_tasks.pop(task_id, None)

        if error_box[0]:
            print(f"❌ [TaskTimer] '{task_name}' failed after {elapsed_total:.2f}s: {error_box[0]}")
            raise error_box[0]

        print(f"✅ [TaskTimer] '{task_name}' completed in {elapsed_total:.2f}s (Est: {estimated_t0:.1f}s)")
        if on_complete:
            try:
                on_complete(result_box[0], elapsed_total)
            except Exception:
                pass

        return result_box[0]


# Singleton Instance
TASK_TIMER = DynamicTaskTimer()
