"""
AETHER Full-Stack Memory Ablation Study Runner
Measures RAM (RSS), VRAM (RTX 5050), and Heap for every sub-component incrementally.
Generates:
  - ablation_results.json
  - ablation_report.md
"""
import os
import sys
import gc
import json
import time
import tracemalloc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Core_agent"))
sys.path.insert(0, str(PROJECT_ROOT / "speech"))
sys.path.insert(0, str(PROJECT_ROOT / "vision"))
sys.path.insert(0, str(PROJECT_ROOT / "memory"))
sys.path.insert(0, str(PROJECT_ROOT / "ARN"))

from mem_probe import snap, StageProbe

RESULTS = []

def record_stage(probe: StageProbe, category: str, notes: str = ""):
    data = probe.result
    data["category"] = category
    data["notes"] = notes
    RESULTS.append(data)


def run_ablation():
    print("=" * 80)
    print("  AETHER v4.0 DEEP MEMORY ABLATION STUDY")
    print("  Target: Process RSS, VRAM (RTX 5050), and Heap allocation per component")
    print("=" * 80)
    
    tracemalloc.start()
    
    # ── Stage 00: Baseline ───────────────────────────────────────────
    with StageProbe(0, "Baseline (Python runtime & probe)") as p:
        time.sleep(0.05)
    record_stage(p, "Baseline", "Empty interpreter baseline")

    # ── Stage 01: FastAPI + Uvicorn ───────────────────────────────────
    with StageProbe(1, "FastAPI + Uvicorn + Starlette") as p:
        import fastapi
        import uvicorn
        from fastapi.middleware.cors import CORSMiddleware
        app = fastapi.FastAPI(title="AETHER Ablation")
        app.add_middleware(CORSMiddleware, allow_origins=["*"])
    record_stage(p, "Infrastructure", "Web server framework")

    # ── Stage 02: sounddevice + numpy ────────────────────────────────
    with StageProbe(2, "sounddevice + numpy (Audio I/O)") as p:
        import numpy as np
        import sounddevice as sd
    record_stage(p, "Infrastructure", "C-extensions and audio stream bindings")

    # ── Stage 03: faster_whisper import ──────────────────────────────
    with StageProbe(3, "faster_whisper (import only)") as p:
        import faster_whisper
        from faster_whisper import WhisperModel
    record_stage(p, "Speech (STT)", "CTranslate2 runtime and Whisper loader")

    # ── Stage 04: WhisperModel weights init ──────────────────────────
    with StageProbe(4, "WhisperModel('base.en', int8 CPU) init") as p:
        try:
            whisper_inst = WhisperModel("base.en", device="cpu", compute_type="int8")
        except Exception as e:
            whisper_inst = None
            print(f"      [Warning] Whisper init skipped: {e}")
    record_stage(p, "Speech (STT)", "Loaded 74M param STT int8 weights into CPU RAM")

    # ── Stage 05: wake_word ──────────────────────────────────────────
    with StageProbe(5, "wake_word detector (import & init)") as p:
        try:
            import wake_word
            ww_detector = wake_word.WakeWordDetector()
        except Exception as e:
            ww_detector = None
            print(f"      [Warning] Wake word detector skipped: {e}")
    record_stage(p, "Speech (Wake)", "Energy threshold gate and keywords")

    # ── Stage 06: TTS_kokoro import ──────────────────────────────────
    with StageProbe(6, "TTS_kokoro (import only)") as p:
        try:
            import TTS_kokoro
            from TTS_kokoro import KokoroTTS
        except Exception as e:
            KokoroTTS = None
            print(f"      [Warning] KokoroTTS import skipped: {e}")
    record_stage(p, "Speech (TTS)", "Kokoro pipeline bindings")

    # ── Stage 07: KokoroTTS init ─────────────────────────────────────
    with StageProbe(7, "KokoroTTS (82M weights init)") as p:
        kokoro_inst = None
        if KokoroTTS:
            try:
                kokoro_inst = KokoroTTS()
            except Exception as e:
                print(f"      [Warning] Kokoro init error: {e}")
    record_stage(p, "Speech (TTS)", "Kokoro-82M speech synthesis weights")

    # ── Stage 08: AudioDSPChain ──────────────────────────────────────
    with StageProbe(8, "audio_dsp (AudioDSPChain)") as p:
        try:
            import audio_dsp
            dsp = audio_dsp.AudioDSPChain()
        except Exception as e:
            dsp = None
    record_stage(p, "Speech (DSP)", "Broadcast audio filter chain (NumPy ops)")

    # ── Stage 09: dynamic_turn_taking ────────────────────────────────
    with StageProbe(9, "dynamic_turn_taking (Sparrow-2 engine)") as p:
        try:
            import dynamic_turn_taking
            conv_ctx = dynamic_turn_taking.ConversationContext()
        except Exception as e:
            conv_ctx = None
    record_stage(p, "Speech (Logic)", "Turn taking state machine and silence thresholds")

    # ── Stage 10: streaming_stt ──────────────────────────────────────
    with StageProbe(10, "streaming_stt (StreamingSTT engine)") as p:
        try:
            import streaming_stt
            sstt = streaming_stt.STREAMING_STT
        except Exception as e:
            sstt = None
    record_stage(p, "Speech (STT)", "Streaming speech buffer manager")

    # ── Stage 11: interruption_recovery ──────────────────────────────
    with StageProbe(11, "interruption_recovery engine") as p:
        try:
            import interruption_recovery
            rec = interruption_recovery.INTERRUPTION_RECOVERY
        except Exception as e:
            rec = None
    record_stage(p, "Speech (Logic)", "State rollback and interruption detection")

    # ── Stage 12: ARN ONNX session ───────────────────────────────────
    with StageProbe(12, "ARN Router (Int8 ONNX session)") as p:
        import onnxruntime as ort
        from config import ARNconfig
        from tokenizer_utils import TOKENIZER
        onnx_file = PROJECT_ROOT / "ARN" / "checkpoints" / "arn_model_int8.onnx"
        if onnx_file.exists():
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            arn_ort = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
        else:
            arn_ort = None
    record_stage(p, "Routing (ARN)", "4.5MB Int8 ONNX graph in ONNX Runtime")

    # ── Stage 13: ARN PyTorch shell ──────────────────────────────────
    with StageProbe(13, "ARN PyTorch shell (CRF Viterbi)") as p:
        import torch
        from model import AetherRoutingNetwork
        arn_cfg = ARNconfig()
        arn_pt_shell = AetherRoutingNetwork(arn_cfg, load_pretrained=False)
        arn_pt_shell.eval()
    record_stage(p, "Routing (ARN)", "PyTorch shell for CRF sequence decoding")

    # ── Stage 14: hybrid_router ──────────────────────────────────────
    with StageProbe(14, "hybrid_router (dispatcher)") as p:
        import hybrid_router
    record_stage(p, "Routing", "Dispatcher logic between fast-path and full reasoning")

    # ── Stage 15: policy_router ──────────────────────────────────────
    with StageProbe(15, "policy_router (PolicyRouter)") as p:
        import policy_router
        pol_router = policy_router.PolicyRouter()
    record_stage(p, "Routing", "Heuristic policy evaluation engine")

    # ── Stage 16: micro_expert ───────────────────────────────────────
    with StageProbe(16, "micro_expert (cross-encoder classifier)") as p:
        try:
            import micro_expert
            m_expert = micro_expert.MicroExpert()
        except Exception as e:
            m_expert = None
            print(f"      [Warning] MicroExpert error: {e}")
    record_stage(p, "Auxiliary Transformer", "cross-encoder/nli-MiniLM2-L6-H768 zero-shot intent classifier")

    # ── Stage 17: FrozenLLMEngine init ───────────────────────────────
    with StageProbe(17, "FrozenLLMEngine (HTTP client & history)") as p:
        import LLM_Engine
        llm = LLM_Engine.FrozenLLMEngine()
    record_stage(p, "LLM Engine", "Client coordinator and sliding window conversation history")

    # ── Stage 18: ExperienceMemory ───────────────────────────────────
    with StageProbe(18, "ExperienceMemory (Tier 2 profile)") as p:
        try:
            import experience_memory
            exp_mem = experience_memory.ExperienceMemory()
        except Exception as e:
            exp_mem = None
    record_stage(p, "Memory", "User profile and persistent preferences (JSON dict)")

    # ── Stage 19: FAISS index load ───────────────────────────────────
    with StageProbe(19, "FAISS index load (faiss_index.bin)") as p:
        import faiss
        faiss_file = PROJECT_ROOT / "memory" / "faiss_index.bin"
        if faiss_file.exists():
            faiss_idx = faiss.read_index(str(faiss_file))
        else:
            faiss_idx = faiss.IndexFlatL2(384)
    record_stage(p, "Memory (RAG)", "FAISS vector index in RAM")

    # ── Stage 20: documents.npy ──────────────────────────────────────
    with StageProbe(20, "documents.npy (RAG document records)") as p:
        docs_file = PROJECT_ROOT / "memory" / "documents.npy"
        if docs_file.exists():
            rag_docs = np.load(str(docs_file), allow_pickle=True).tolist()
        else:
            rag_docs = []
    record_stage(p, "Memory (RAG)", "Document metadata array")

    # ── Stage 21: SentenceTransformer ────────────────────────────────
    with StageProbe(21, "SentenceTransformer('all-MiniLM-L6-v2')") as p:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    record_stage(p, "Auxiliary Transformer", "384-dimensional embedding model on CPU")

    # ── Stage 22: tools_executor ─────────────────────────────────────
    with StageProbe(22, "tools_executor (AVAILABLE_TOOLS)") as p:
        import tools_executor
        n_tools = len(tools_executor.AVAILABLE_TOOLS)
    record_stage(p, "Tools", f"Registry of {n_tools} native tools and function pointers")

    # ── Stage 23: skill_registry + skill_factory ────────────────────
    with StageProbe(23, "skill_registry + skill_factory") as p:
        import skill_registry
        import skill_factory
        s_reg = skill_registry.LocalSkillRegistry()
    record_stage(p, "Skills", "Dynamic skill discovery and code generation templates")

    # ── Stage 24: aether_coder + auto_skill_forge ────────────────────
    with StageProbe(24, "aether_coder + auto_skill_forge") as p:
        import aether_coder
        import auto_skill_forge
    record_stage(p, "Skills", "Self-refining coding tools and auto skill forge")

    # ── Stage 25: OpenCV (camera_agent + screen_capture) ─────────────
    with StageProbe(25, "OpenCV (camera_agent + screen_capture)") as p:
        try:
            import cv2
            import camera_agent
            import screen_capture
            cam = camera_agent.VisionPerception()
            sc = screen_capture.ScreenCapture()
        except Exception as e:
            print(f"      [Warning] Vision load error: {e}")
    record_stage(p, "Vision", "OpenCV cv2 bindings and screen capture frame grabber")

    # ── Stage 26: desktop_agent ──────────────────────────────────────
    with StageProbe(26, "desktop_agent (pyautogui + win32)") as p:
        try:
            import desktop_agent
        except Exception as e:
            pass
    record_stage(p, "Desktop Automation", "OS mouse, keyboard, and window control")

    # ── Stage 27: screen_annotator + screen_canvas_copilot ───────────
    with StageProbe(27, "screen_annotator + canvas_copilot (PIL)") as p:
        try:
            import screen_annotator
            import screen_canvas_copilot
        except Exception as e:
            pass
    record_stage(p, "Desktop Automation", "PIL image manipulation and canvas overlays")

    # ── Stage 28: ambient_daemon ─────────────────────────────────────
    with StageProbe(28, "ambient_daemon (thread structures)") as p:
        import ambient_daemon
    record_stage(p, "Background Services", "Ambient state watcher")

    # ── Stage 29: desktop_overlay HUD ────────────────────────────────
    with StageProbe(29, "desktop_overlay (HUD Tkinter structures)") as p:
        try:
            import desktop_overlay
        except Exception as e:
            pass
    record_stage(p, "UI / Overlay", "Floating Windows HUD overlay structures")

    # ── Stage 30: NIMKeyPool ─────────────────────────────────────────
    with StageProbe(30, "NIMKeyPool (Multi-Key Pool Manager)") as p:
        from nim_key_pool import KEY_POOL
        n_keys = len(KEY_POOL.keys)
    record_stage(p, "Credentials & API", f"Multi-account pool manager ({n_keys} active keys, rotation lock)")

    # ── Stage 31: browser_worker ─────────────────────────────────────
    with StageProbe(31, "browser_worker (config & worker interface)") as p:
        import browser_worker
    record_stage(p, "Browser Subprocess", "Async browser worker dispatcher")

    # ── Stage 32: FULL STACK IDLE ────────────────────────────────────
    gc.collect()
    time.sleep(0.2)
    with StageProbe(32, "FULL STACK IDLE (All components loaded)") as p:
        time.sleep(0.1)
    record_stage(p, "System State", "Complete AETHER system at rest, ready for input")

    # ── Stage 33: ACTIVE INFERENCE — ARN Fast-Path ───────────────────
    from arn_router import ARN_ROUTER
    with StageProbe(33, "ACTIVE INFERENCE: ARN Fast-Path Tool Query") as p:
        res = ARN_ROUTER.route_query("what is the weather in Tokyo right now")
    record_stage(p, "Active Inference", f"Sub-5ms local routing execution (Result: {res.get('tools', [])})")

    # ── Stage 34: ACTIVE INFERENCE — NIM Cloud Pool ──────────────────
    with StageProbe(34, "ACTIVE INFERENCE: NVIDIA NIM Cloud Pool Query") as p:
        key, status = KEY_POOL.get_active_key()
        if key:
            try:
                import requests
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": "meta/llama-3.3-70b-instruct",
                    "messages": [{"role": "user", "content": "Respond with 5 words."}],
                    "max_tokens": 15,
                    "temperature": 0.2
                }
                t0 = time.perf_counter()
                r = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=8)
                req_time = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    ans = r.json()["choices"][0]["message"]["content"].strip()
                    print(f"      [NIM Response ({req_time:.0f}ms)]: {ans}")
                else:
                    print(f"      [NIM Status]: {r.status_code}")
            except Exception as e:
                print(f"      [NIM Error]: {e}")
    record_stage(p, "Active Inference", "Real-world cloud query across multi-key pool (measures local memory delta)")

    # ── Stage 35: ACTIVE INFERENCE — Knowledge RAG Search ────────────
    with StageProbe(35, "ACTIVE INFERENCE: RAG Vector Search Query") as p:
        q_emb = emb_model.encode(["search query about user profile"], convert_to_numpy=True)
        D, I = faiss_idx.search(q_emb, k=1)
    record_stage(p, "Active Inference", "FAISS L2 distance search and embedding execution")

    # Post-inference idle
    gc.collect()
    with StageProbe(36, "POST-INFERENCE IDLE (After Garbage Collection)") as p:
        time.sleep(0.1)
    record_stage(p, "System State", "System footprint after queries and garbage collection")

    # Save results
    out_json = PROJECT_ROOT / "ablation_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\n[Done] Raw ablation results saved to: {out_json}")

    # Generate Markdown Report
    generate_report(PROJECT_ROOT / "ablation_report.md")


def generate_report(report_path: Path):
    lines = []
    lines.append("# AETHER v4.0 Full-Stack Memory Ablation Report")
    lines.append(f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} on NVIDIA GeForce RTX 5050")
    lines.append("\n## 1. Incremental Component Breakdown\n")
    lines.append("| Stage | Category | Component | RAM (MB) | Delta RAM | VRAM (MB) | Heap (MB) | Elapsed | Notes |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    
    for r in RESULTS:
        stg = r["stage"]
        cat = r["category"]
        lbl = r["label"]
        ram = f"{r['ram_after_mb']:.1f}"
        d_ram = f"{'+' if r['delta_ram_mb'] >= 0 else ''}{r['delta_ram_mb']:.1f}"
        vram = f"{r['vram_after_mb']:.1f}"
        heap = f"{r.get('delta_heap_mb', 0):.1f}"
        el = f"{r['elapsed_ms']:.0f}ms"
        notes = r.get("notes", "")
        lines.append(f"| {stg:02d} | {cat} | **{lbl}** | {ram} | {d_ram} | {vram} | {heap} | {el} | {notes} |")

    # Top consumers
    import_deltas = [r for r in RESULTS if r["stage"] <= 31]
    top_ram = sorted(import_deltas, key=lambda x: x["delta_ram_mb"], reverse=True)[:7]

    lines.append("\n## 2. Top RAM Consumers (Initialization & Weights)\n")
    lines.append("| Rank | Component | Delta RAM (MB) | Category | Description |")
    lines.append("|---|---|---|---|---|")
    for idx, item in enumerate(top_ram, 1):
        lines.append(f"| #{idx} | **{item['label']}** | **+{item['delta_ram_mb']:.1f} MB** | {item['category']} | {item['notes']} |")

    # Category aggregation
    cat_totals = {}
    for r in import_deltas:
        c = r["category"]
        cat_totals[c] = cat_totals.get(c, 0.0) + max(0.0, r["delta_ram_mb"])

    lines.append("\n## 3. Cumulative Category Footprint\n")
    lines.append("| Category | Total Contributed RAM (MB) | % of Contributed RAM |")
    lines.append("|---|---|---|")
    tot = sum(cat_totals.values()) or 1.0
    for c, val in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
        pct = (val / tot) * 100.0
        lines.append(f"| **{c}** | {val:.1f} MB | {pct:.1f}% |")

    # Inference comparison
    inf_stages = [r for r in RESULTS if r["stage"] in (32, 33, 34, 35, 36)]
    lines.append("\n## 4. Idle vs. Active Inference Memory\n")
    lines.append("| State | RAM (MB) | VRAM (MB) | Delta RAM | Elapsed |")
    lines.append("|---|---|---|---|---|")
    for r in inf_stages:
        lines.append(f"| {r['label']} | {r['ram_after_mb']:.1f} MB | {r['vram_after_mb']:.1f} MB | {'+' if r['delta_ram_mb']>=0 else ''}{r['delta_ram_mb']:.1f} MB | {r['elapsed_ms']:.0f}ms |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Done] Formatted report written to: {report_path}")


if __name__ == "__main__":
    run_ablation()
