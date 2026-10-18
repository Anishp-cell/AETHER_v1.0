"""
AETHER High-Precision Memory Probe
Measures process RSS (RAM), VRAM (via PyNVML/torch.cuda/nvidia-smi), and Python heap (tracemalloc).
"""
import os
import sys
import time
import subprocess
import tracemalloc
import psutil

# VRAM measurement detection
_PYNVML_OK = False
_GPU_HANDLE = None

try:
    import pynvml
    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _PYNVML_OK = True
except Exception:
    _PYNVML_OK = False

# Fallback: torch.cuda memory allocated or nvidia-smi
_TORCH_OK = False
try:
    import torch
    if torch.cuda.is_available():
        _TORCH_OK = True
except Exception:
    _TORCH_OK = False

_proc = psutil.Process(os.getpid())


def get_vram_mb() -> float:
    """Returns currently allocated or total used VRAM in MB."""
    if _PYNVML_OK and _GPU_HANDLE is not None:
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
            return float(info.used / (1024 ** 2))
        except Exception:
            pass

    if _TORCH_OK:
        try:
            return float(torch.cuda.memory_allocated() / (1024 ** 2))
        except Exception:
            pass

    # Subprocess fallback (slow, cached if needed)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            encoding="utf-8",
            timeout=1
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def snap(label: str, verbose: bool = True) -> dict:
    """
    Takes a snapshot of current memory consumption.
    Returns: {
        'label': str,
        'ram_mb': float,
        'vram_mb': float,
        'heap_mb': float,
        'ts': float
    }
    """
    ram_mb = _proc.memory_info().rss / (1024 ** 2)
    vram_mb = get_vram_mb()

    heap_mb = 0.0
    if tracemalloc.is_tracing():
        cur, _ = tracemalloc.get_traced_memory()
        heap_mb = cur / (1024 ** 2)

    if verbose:
        print(f"[MEM] {label:<48} | RAM: {ram_mb:7.2f} MB | VRAM: {vram_mb:7.2f} MB | Heap: {heap_mb:6.2f} MB")

    return {
        "label": label,
        "ram_mb": round(ram_mb, 2),
        "vram_mb": round(vram_mb, 2),
        "heap_mb": round(heap_mb, 2),
        "ts": time.time()
    }


class StageProbe:
    """Context manager to measure delta memory of an operation."""
    def __init__(self, stage_num: int, label: str):
        self.stage_num = stage_num
        self.label = label
        self.before = None
        self.after = None
        self.result = None
        self.start_t = 0.0

    def __enter__(self):
        self.start_t = time.perf_counter()
        self.before = snap(f"PRE : {self.label}", verbose=False)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_t) * 1000.0
        self.after = snap(f"POST: {self.label}", verbose=False)
        
        delta_ram = round(self.after["ram_mb"] - self.before["ram_mb"], 2)
        delta_vram = round(self.after["vram_mb"] - self.before["vram_mb"], 2)
        delta_heap = round(self.after["heap_mb"] - self.before["heap_mb"], 2)

        self.result = {
            "stage": self.stage_num,
            "label": self.label,
            "ram_before_mb": self.before["ram_mb"],
            "ram_after_mb": self.after["ram_mb"],
            "delta_ram_mb": delta_ram,
            "vram_before_mb": self.before["vram_mb"],
            "vram_after_mb": self.after["vram_mb"],
            "delta_vram_mb": delta_vram,
            "delta_heap_mb": delta_heap,
            "elapsed_ms": round(elapsed_ms, 2)
        }

        sign = "+" if delta_ram >= 0 else ""
        print(f"[{self.stage_num:02d}] {self.label:<44} -> RAM: {self.after['ram_mb']:7.2f} MB ({sign}{delta_ram:6.2f} MB) | VRAM: {self.after['vram_mb']:6.1f} MB | {elapsed_ms:6.1f}ms")
