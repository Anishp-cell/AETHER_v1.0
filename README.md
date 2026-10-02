# AETHER v4.0: System-Level Multimodal Agentic Runtime & Dual-Channel Browser MoE

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-INT8_Quantized-005CED.svg)](https://onnxruntime.ai/)
[![NVIDIA CUDA](https://img.shields.io/badge/CUDA-12.x_%7C_RTX_5050-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![NVIDIA NIM](https://img.shields.io/badge/NVIDIA_NIM-Llama--3.2--Vision-76B900.svg)](https://build.nvidia.com/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14_(R3F)-black.svg)](https://nextjs.org/)
[![Playwright](https://img.shields.io/badge/Playwright-Dual--Channel-45BA4B.svg)](https://playwright.dev/)
[![Kokoro TTS](https://img.shields.io/badge/Kokoro-82M_Voicebox-FF6F00.svg)](https://github.com/hexgrad/kokoro)

**AETHER** (*Adaptive, Evolving, Tactical, Heuristic-Engine Response*) is a hardware-aware, multimodal artificial intelligence runtime built for hybrid edge execution and cloud acceleration. AETHER fuses an ultra-fast **sub-3ms local neural routing network (ARN)**, a **triple-model local Mixture of Experts (MoE)**, a **dual-channel autonomous browser agent**, and a **cloud-backed multimodal vision loop powered by NVIDIA NIM**.

Instead of routing all user intent through slow, expensive LLM prompts, AETHER resolves desktop and browser tasks in milliseconds on local silicon, scaling dynamically to multi-step visual agents only when needed.

---

## 🌟 What's New in v4.0

- ⚡ **Aether Routing Network (ARN) v2**: Upgraded to an 18-tool classification & BIO sequence tagging transformer (4.4M params), quantized to INT8 ONNX for **<3ms CPU inference**.
- 🌐 **Dual-Channel Autonomous Browser Engine**:
  - **Channel 1 (Fast Playwright Automation)**: Direct DOM manipulation with visual glowing bounding box highlights for rapid navigation, clicks, and scraping.
  - **Channel 2 (Vision-Guided Multi-Step Agent)**: Powered by `browser-use` and NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct`), capable of complex multi-turn web research, dynamic SPAs, and visual reasoning.
- 🛡️ **Subprocess-Decoupled Browser Worker**: Chromium instances and Playwright event loops run in an isolated operating system worker process (`browser_worker.py`), guaranteeing zero lag or frame drops on the primary FastAPI WebSocket thread.
- 🔑 **NVIDIA NIM Quota & Multi-Key API Pool**:
  - Live token ledger and request quota monitoring (`.aether/nim_usage.json`) tracking Developer Tier free allowances (1,000 requests/month per key).
  - High-water mark alerts at 80% (800 requests) and 95% (950 requests).
  - Multi-key API pool architecture supporting round-robin rotation and automatic failover across multiple developer accounts to massively scale request volume.
- 🔒 **Privacy-Safe Base Domain Preference Tracker**:
  - Session-gated tracking of user web habits (`site_history_manager.py`).
  - Strict privacy sanitization: strips all paths, session IDs, and query parameters, persisting only base domain frequencies (`.aether/site_history.json`).
  - Automatically injects `<USER_SITE_PREFERENCES>` into conversational prompts to resolve ambiguous commands (e.g., "play music" → `spotify.com`, "check pull requests" → `github.com`).
- 🎙️ **Conversational Audio DSP & Full-Duplex Dynamic Turn-Taking**:
  - Dual-stream WebRTC VAD with spectral noise suppression.
  - Dynamic silence thresholding to distinguish mid-sentence breath pauses from terminal utterances.
  - Sub-50ms barge-in interruption with speaker syllable-level context truncation to prevent memory hallucinations.

---

## 🏛️ Core Architectural Pillars

```
                                      ┌─────────────────────────────────────────────────────────────┐
                                      │                      USER UTTERANCE                         │
                                      │             (Microphone / WebSocket Stream)                 │
                                      └──────────────────────────────┬──────────────────────────────┘
                                                                     │
                                                      ┌──────────────▼──────────────┐
                                                      │  Conversational Audio DSP   │
                                                      │   (WebRTC VAD + Denoiser)   │
                                                      └──────────────┬──────────────┘
                                                                     │
                                                      ┌──────────────▼──────────────┐
                                                      │  Acoustic STT (Whisper GPU) │
                                                      └──────────────┬──────────────┘
                                                                     │
                                    ┌────────────────────────────────┴────────────────────────────────┐
                                    │                                                                 │
                     [Fast Path: Tool Routing]                                        [Conversational Fallback]
                                    │                                                                 │
                     ┌──────────────▼──────────────┐                                   ┌──────────────▼──────────────┐
                     │    Aether Routing Network   │                                   │      Base LLM Engine        │
                     │    (ARN 4.4M INT8 ONNX)     │                                   │     (Qwen2.5-Coder 1.5B)    │
                     │   <3ms CPU Neural Router    │                                   │    + User Site Prefs RAG    │
                     └──────┬───────────────┬──────┘                                   └──────────────┬──────────────┘
                            │               │                                                         │
             [18-Class Intent]             [25-Slot BIO Tagging]                                      │
             (Which tool to run)           (Extract args / URLs / sites)                              │
                            │               │                                                         │
                            └───────┬───────┘                                                         │
                                    │                                                                 │
                                    ▼                                                                 │
                  ┌──────────────────────────────────┐                                                │
                  │   Tiered Security Authorization  │                                                │
                  │   (Auto-Safe vs GUI Confirmation)│                                                │
                  └─────────────────┬────────────────┘                                                │
                                    │                                                                 │
               ┌────────────────────┴────────────────────┐                                            │
               │                                         │                                            │
     [Desktop / OS Actions]                     [Browser Subsystem]                                   │
     - WhatsApp Desktop                         (Worker Process)                                      │
     - Window Snapping / Tile                            │                                            │
     - Media / Volume COM              ┌─────────────────┴─────────────────┐                          │
     - Custom Script Sandbox           │                                   │                          │
                                 [Channel 1: Fast]                  [Channel 2: Vision]               │
                                 Playwright Engine                  NVIDIA NIM + Llama Vision         │
                                 - Visual Glow Overlay              - Multi-turn Autonomous Agent     │
                                 - Instant DOM Scraping             - Complex SPA Workflows           │
                                       │                                   │                          │
                                       └─────────────────┬─────────────────┘                          │
                                                         │                                            │
                                                         ▼                                            │
                                            ┌─────────────────────────┐                               │
                                            │ Privacy Base Domain Log │                               │
                                            │ (.aether/site_history)  │                               │
                                            └────────────┬────────────┘                               │
                                                         │                                            │
                                                         ▼                                            ▼
                                            ┌─────────────────────────────────────────────────────────┐
                                            │                 VERBAL RESPONSE SYNTHESIS               │
                                            │           (Kokoro-82M TTS / Dynamic Turn-Taking)        │
                                            └─────────────────────────────────────────────────────────┘
```

---

### 1. Aether Routing Network (ARN) — 4.4M Int8 Edge Neural Router

The **Aether Routing Network (ARN)** replaces slow, recursive LLM tool-calling loops with an ultra-lightweight, dual-head transformer based on `google/bert_uncased_L-2_H-128_A-2`.

- **Dual-Head Architecture**:
  1. **Intent Classification Head**: Evaluates the input utterance across **18 discrete tool classes** with calibrated softmax confidence.
  2. **BIO Sequence Tagging Head**: Concurrently predicts token-level slot arguments across 25 tag states (`B-site`, `I-site`, `B-browser_task`, `B-url`, `B-query`, `B-app`, `B-contact`, etc.).
- **Sub-3ms Latency**: Quantized to an INT8 ONNX graph (`checkpoints/arn_expanded_18tool_int8.onnx`), ARN executes on CPU in 1.8ms–2.9ms, completely decoupling tool detection from heavy LLM forward passes.
- **Fine-Tuning on RTX 5050**: Trained on an NVIDIA GeForce RTX 5050 Laptop GPU over 5,000 synthetic balanced pairs (`aether_orchestrator_dataset.jsonl`), converging to a final loss of `0.0003` with 100% routing accuracy on regression benchmarks.
- **Checkpoint Expansion Engine (`expand_checkpoint.py`)**: Seamlessly expands classification and sequence heads from 16 to 18 tools, copying pretrained tensor weights and zero-initializing new tool slots without catastrophic forgetting.

---

### 2. Dual-Channel Autonomous Browser Engine

Web automation in AETHER is handled via two specialized execution channels:

```
User Query: "Go to github.com and star this repository"
                       │
                       ▼
          [AETHER Browser Subsystem]
                       │
        ┌──────────────┴──────────────┐
        │                             │
   [Channel 1]                   [Channel 2]
Playwright Direct             Vision-Guided NIM
(Single-Step Actions)         (Multi-Step Tasks)
        │                             │
  - Visual Glow Bounding Box    - Llama-3.2-11B-Vision
  - Fast DOM Click/Fill         - browser-use Autonomous Loop
  - Content Scraping            - SPA & Multi-Page Navigation
        │                             │
        └──────────────┬──────────────┘
                       ▼
              [Target Web Browser]
```

- **Channel 1 — Fast Playwright Automation (`browser_open_and_interact`)**:
  - Built for deterministic tasks: navigating to URLs, clicking buttons, filling forms, and reading content.
  - **Visual Script Injector**: Injects a pulsating visual bounding box overlay around the active DOM element, giving the user immediate visual confirmation of the agent's actions on screen.
- **Channel 2 — Vision-Guided Multi-Step Autonomous Agent (`browser_task`)**:
  - Powered by the `browser-use` framework and NVIDIA NIM cloud endpoints (`meta/llama-3.2-11b-vision-instruct`).
  - Analyzes rendered screenshots and DOM trees to execute complex, multi-turn goals (e.g., flight comparisons, dashboard filtering, multi-page form fills).
- **Subprocess Worker Isolation (`Core_agent/browser_worker.py`)**:
  - Browser processes run in an isolated operating system worker subprocess.
  - Communication flows via multiprocessing pipes and JSON-RPC, ensuring heavy Chromium rendering or network lag never blocks FastAPI's WebSocket event loop.

---

### 3. NVIDIA NIM Cloud Intelligence & Multi-Key Credit Pool

AETHER integrates with **NVIDIA NIM (Inference Microservices)** cloud endpoints for heavy multimodal vision:

- **Local Token & Quota Ledger (`Core_agent/nim_usage_tracker.py`)**:
  - Persists all API activity to `.aether/nim_usage.json`.
  - Tracks prompt tokens, completion tokens, total lifetime tokens, and daily request counts.
  - Enforces quota thresholds against the NVIDIA Developer Free Tier (1,000 requests/month per key).
  - High-water mark safety alerts trigger at 80% (800 requests) and 95% (950 requests).
- **Multi-Key API Pool & Round-Robin Failover**:
  - Overcomes single-account rate limits by pooling multiple API keys in `.env` (`NVIDIA_API_KEYS=nvapi-key1,nvapi-key2,...`).
  - Implements automatic failover on HTTP 429 (Rate Limit) or HTTP 402/403 (Quota Exhaustion), dynamically rotating to the next healthy key in the pool.
- **CLI Balance Dashboard**:
  ```bash
  python -m Core_agent.nim_usage_tracker
  ```
  Displays an ASCII dashboard showing quota health, active key preview, lifetime tokens, and per-model consumption.

---

### 4. Privacy-Safe Base Domain Preference Tracker

AETHER learns your browsing preferences without compromising privacy:

- **Strict URL Sanitization (`Core_agent/site_history_manager.py`)**:
  - Strips all GET query strings, session IDs, URL paths, and fragments.
  - Extracts and logs **only the clean base domain name** (e.g., `github.com`, `youtube.com`, `stackoverflow.com`).
  - Automatically filters browser internals (`about:blank`, `chrome://`, `localhost`).
- **Session-Gated Recording**:
  - Tracking is strictly gated by `session_state.is_active()`. Zero recording occurs when AETHER is idle or shut down.
- **Dynamic System Prompt Context Injection**:
  - Visited domain frequencies are injected into the LLM context under `<USER_SITE_PREFERENCES>`.
  - Enables the conversational base model to naturally resolve vague human commands (e.g., "open my music" → `spotify.com`, "check my code" → `github.com`).
- **CLI History Inspection**:
  ```bash
  python -m Core_agent.site_history_manager
  ```

---

### 5. Conversational Audio DSP & Full-Duplex Speech Engine

- **WebRTC 16kHz Audio Pipeline**: Dual-stream VAD with spectral noise gating for continuous, hands-free voice interaction.
- **Dynamic Turn-Taking State Machine**: Probabilistic silence thresholding dynamically adjusts pause tolerance based on linguistic completeness (distinguishes mid-sentence pauses from finished thoughts).
- **Sub-50ms Barge-In Cancellation**: When the user speaks while AETHER is talking, the audio output immediately cuts out, and contextual memory is truncated to only the syllables that actually exited the speaker.
- **Acoustic Synthesis**: Local high-fidelity **Kokoro-82M TTS** with multiple voice profiles (`en-GB-RyanNeural`, `af_heart`) backed by an Edge-TTS fallback.

---

### 6. Hybrid Triple-Model Local MoE Engine

AETHER operates on a triple-model local Mixture of Experts pipeline hosted on Ollama:

| Model | Role | Parameters | Execution Target | Purpose |
|-------|------|-----------|------------------|---------|
| `ARN-Int8` | **Neural Router** | 4.4M | Local CPU (ONNX) | Sub-3ms zero-shot tool classification and argument slot filling. |
| `Aether-Orchestrator` | **Tool Fallback** | 1.5B (Q4_K_M) | Local GPU (Ollama) | SFT Qwen2.5-Coder for complex multi-tool JSON arrays. |
| `qwen2.5-coder:1.5b` | **Conversational Brain** | 1.5B | Local GPU (Ollama) | General dialogue, message drafting, and verbal synthesis. |
| `deepseek-r1:1.5b` | **Reasoning Engine** | 1.5B | Local GPU (Ollama) | Chain-of-thought logic, math, and code generation via `<think>` tags. |
| `llama-3.2-11b-vision` | **Vision & Browser Brain** | 11B | Cloud (NVIDIA NIM) | Visual DOM understanding and multi-step autonomous browsing. |

---

## 🔧 Complete Tool Arsenal (18 Native Tools)

| # | Tool Name | Category | Engine / Subsystem | Description |
|---|-----------|----------|-------------------|-------------|
| 1 | `browser_task` | Autonomous Browser | NVIDIA NIM + `browser-use` | Multi-step vision-guided autonomous web research, shopping, dynamic forms. |
| 2 | `browser_open_and_interact` | Fast Browser | Playwright + Visual Glow | High-speed page navigation, clicks, typing, and scraping with visual glow overlays. |
| 3 | `get_nim_usage` | Cloud Quota | NIM Usage Tracker | Checks real-time NVIDIA NIM token usage, request balance, and quota health. |
| 4 | `get_most_visited_sites` | Personalization | Site History Manager | Returns user's most frequently visited privacy-safe base domains. |
| 5 | `search_web` | Web Search | Native Chrome / Edge | Opens visible Google search in the default system browser. |
| 6 | `open_url` | Web Navigation | Desktop Browser | Navigates directly to a specified URL or domain. |
| 7 | `search_and_read_web` | Headless Scraper | DuckDuckGo HTML API | Silent web search returning structured text snippets directly to LLM context. |
| 8 | `read_specific_url` | Headless Scraper | BeautifulSoup4 Parser | Fetches, parses, and extracts markdown/text content from any web page. |
| 9 | `run_computer_command` | OS Automation | PyAutoGUI + Win32 | Executes shell commands, keyboard hotkeys, mouse clicks, and application launches. |
| 10 | `open_app_and_type` | OS Automation | Win32 + PyAutoGUI | Launches an application and pastes text (includes Spotify URI handling). |
| 11 | `analyze_screen_with_llava` | Desktop Vision | Llava-Phi3 (Local) | Captures desktop screenshot and performs multimodal visual analysis. |
| 12 | `send_whatsapp_message` | Messaging | WhatsApp Desktop | Full WhatsApp automation with Content Engine intercept to clean message text. |
| 13 | `media_control` | Media Automation | PyCaw + Windows COM | Adjusts system volume, mutes audio, and toggles media playback. |
| 14 | `get_system_diagnostics` | System Telemetry | psutil | Returns live CPU load, RAM utilization, and battery status. |
| 15 | `set_timer` | Utilities | Python Threading | Spawns a background timer with audio alarm notification upon completion. |
| 16 | `get_current_time` | Utilities | Native System Time | Returns formatted local system time. |
| 17 | `handle_smart_home` | IoT Automation | Smart Device Stub | Controls local IoT lights, plugs, and smart home appliances. |
| 18 | `write_and_run_script` | Code Execution | AetherCoder Sandbox | Generates and executes Python code in a safe workspace with a 30s timeout. |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Core Runtime** | Python 3.10+, FastAPI, WebSockets, Uvicorn, Multiprocessing |
| **Neural Router** | ONNX Runtime INT8, HuggingFace Transformers, PyTorch 2.x |
| **Edge Hardware Acceleration** | NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 12.x), TensorRT / ONNX |
| **Cloud Acceleration** | NVIDIA NIM API (`build.nvidia.com`), OpenAI-compatible REST |
| **Browser Subsystem** | Playwright, `browser-use`, Chromium Headless/Headful |
| **Local LLM Engine** | Ollama (localhost:11434), Qwen2.5-Coder 1.5B, DeepSeek-R1 1.5B |
| **Speech & Audio DSP** | Faster-Whisper (CUDA float16), Kokoro-82M TTS, WebRTC VAD |
| **Frontend Dashboard** | Next.js 14, React Three Fiber, Three.js, TailwindCSS |
| **OS Automation** | PyAutoGUI, Pyperclip, PyCaw (COM), ctypes (Win32 API) |
| **Local Memory** | FAISS Vector Store, all-MiniLM-L6-v2 Embeddings, JSON Ledgers |

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- Windows 10/11 (64-bit)
- Python 3.10+
- Node.js 18+ & npm
- [Ollama](https://ollama.ai/) installed and running
- NVIDIA GPU with CUDA 12.x (Recommended) or Multi-core CPU

### 2. Clone & Environment Setup

```bash
git clone https://github.com/Anishp-cell/AETHER_v1.0.git
cd AETHER_v1.0
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure Environment Variables (`.env`)

Create a `.env` file in the root directory:

```env
# NVIDIA NIM Cloud Endpoints (Single Key or Multi-Key Pool)
NVIDIA_API_KEY=nvapi-your-primary-key-here
NVIDIA_API_KEYS=nvapi-key1,nvapi-key2,nvapi-key3

# Browser Automation Settings
HEADLESS_BROWSER=false
NIM_VISION_MODEL=meta/llama-3.2-11b-vision-instruct

# Local LLM Host
OLLAMA_HOST=http://localhost:11434
```

> [!TIP]
> **Multi-Key Pooling**: Adding multiple comma-separated keys to `NVIDIA_API_KEYS` enables automatic failover and load rotation, multiplying your monthly free tier quota across multiple accounts.

### 4. Pull Local Ollama Models

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull deepseek-r1:1.5b
ollama pull llava-phi3
```

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 6. Launch AETHER

**Terminal 1 — AI Backend Engine**:
```bash
.venv\Scripts\activate
cd backend
python server.py
```

**Terminal 2 — 3D Holographic Frontend**:
```bash
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser. AETHER will boot into `Idle` state, greet you with a localized morning briefing, and initialize the audio-reactive 3D orb.

---

## 📊 CLI Diagnostics & Operational Suite

AETHER provides specialized CLI diagnostic utilities to inspect routing, quotas, and history:

### 1. NVIDIA NIM Quota & Token Dashboard
```bash
python -m Core_agent.nim_usage_tracker
```
Displays active API key status, remaining calls, lifetime token counts, and per-model consumption breakdown.

### 2. Privacy-Safe Site History Inspection
```bash
python -m Core_agent.site_history_manager
```
Displays an ASCII table of your most frequently visited base domains and tracking session status.

### 3. Zero-Shot ARN Lexical Evaluation
```bash
python -m ARN.zero_shot_lexical_eval
```
Benchmarks the 4.4M-parameter ARN ONNX router against the test evaluation dataset and prints latency statistics.

### 4. Retrain or Export ARN Checkpoints
```bash
# Fine-tune ARN on GPU
python -m ARN.train --epochs 10 --lr 1e-4 --device cuda

# Quantize and export to INT8 ONNX
python -m ARN.export_onnx
```

---

## 📂 Project Structure

```
AETHER_V1.0/
├── ARN/                               # Aether Routing Network (Edge Neural Router)
│   ├── arn_dual_head.py               # Dual-head BERT architecture (Intent + BIO Slots)
│   ├── config.py                      # Model hyperparameters & 18-tool configuration
│   ├── tokenizer_utils.py             # BIO slot tag mappings & subword alignment
│   ├── expand_checkpoint.py           # Checkpoint weight expansion utility (16 → 18 tools)
│   ├── export_onnx.py                 # PyTorch → ONNX INT8 dynamic quantization pipeline
│   ├── train.py                       # CUDA fine-tuning script
│   ├── zero_shot_lexical_eval.py      # Benchmark evaluation suite
│   └── checkpoints/                   # Checkpoints & ONNX quantized runtime graphs
├── backend/
│   └── server.py                      # FastAPI WebSocket server & full-duplex event loop
├── Core_agent/
│   ├── LLM_Engine.py                  # MoE cognitive engine & context injection
│   ├── browser_agent.py               # Dual-channel browser automation controller
│   ├── browser_worker.py              # Isolated subprocess browser worker (Playwright + NIM)
│   ├── nim_usage_tracker.py           # Thread-safe NVIDIA NIM token & quota ledger
│   ├── site_history_manager.py        # Privacy-safe base domain tracking & prompt context
│   ├── session_state.py               # Global active-session gating manager
│   ├── desktop_agent.py               # PyAutoGUI OS automation & window management
│   ├── web_agent.py                   # DuckDuckGo scraper & HTML URL reader
│   ├── aether_coder.py                # Python code execution sandbox
│   ├── tools_executor.py              # Tool registry (18 tools) & execution dispatcher
│   ├── micro_expert.py                # Zero-shot emotion & intent classifier
│   └── policy_router.py               # Behavioral system prompt adapter rules
├── speech/
│   ├── audio_dsp.py                   # WebRTC VAD & dual-stream spectral audio filter
│   ├── streaming_stt.py               # Streaming Faster-Whisper GPU transcriber
│   ├── turn_taking.py                 # Dynamic probabilistic turn-taking state machine
│   ├── TTS_kokoro.py                  # Kokoro-82M local high-fidelity TTS engine
│   └── TTS_edge.py                    # Edge-TTS cloud synthesis fallback
├── vision/
│   ├── camera_agent.py                # MediaPipe Face Mesh & emotion estimation
│   └── screen_capture.py              # High-speed desktop screen capture
├── memory/
│   ├── knowledge_rag.py               # FAISS vector store & document retrieval
│   └── experience_memory.py           # User profile & long-term facts ledger
├── frontend/                          # Next.js 14 + React Three Fiber 3D Dashboard
│   ├── app/                           # App router pages & WebSocket client hooks
│   └── components/                    # 3D Holographic Orb & Auth Confirmation Modals
└── .aether/                           # Local runtime ledgers (NIM quota, site history)
```

---

## 🔒 Privacy, Sandboxing & Security Core

- **100% Edge Processing by Default**: Audio frames, voice transcription, intent classification, and OS actions run locally on your hardware.
- **Privacy URL Sanitization**: Zero tracking of query parameters, private search terms, or authentication tokens.
- **Decoupled Process Sandboxing**: The browser engine executes in an isolated worker process, preventing memory leaks or browser crashes from impacting the core voice assistant.
- **Tiered Authorization Gate**:
  - **Auto-Authorized (Safe)**: Search, diagnostics, time, media volume, site history queries.
  - **Confirmation Required (High-Risk)**: WhatsApp messaging, terminal script execution, OS command dispatch, screen capture analysis.

---

## 📄 License

*All architectural implementations regarding local neural routing (ARN), dual-channel browser workers, dynamic audio DSP turn-taking, and quota-pooling ledgers are open for research and development.*
