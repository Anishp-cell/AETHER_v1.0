# AETHER v3.0: System-Level MoE Runtime

**AETHER** *(Adaptive, Evolving, Tactical, Heuristic-Engine Response)* is a multi-modal, hardware-aware artificial intelligence runtime designed exclusively for 100% offline edge execution. Instead of relying on monolithic cloud APIs, AETHER dynamically routes sub-tasks across a local Mixture of Experts (MoE) pipeline, dynamically balancing CPU and GPU workloads for ultra-low latency conversational AI with native desktop automation.

## 🌟 Core Architecture

### 1. Hybrid Triple-Model LLM Engine

AETHER operates on a **Triple-Model Mixture of Experts** pipeline, each model serving a distinct cognitive role via Ollama (localhost:11434):

| Model | Role | Parameters | Purpose |
|-------|------|-----------|---------|
| `Aether-Orchestrator` | **Tool Router** | 1.5B (Q4_K_M) | Custom SFT fine-tuned Qwen2.5-Coder for zero-shot JSON tool-call generation. Outputs raw `[{"name": "...", "arguments": {...}}]` arrays with zero conversational filler. |
| `qwen2.5-coder:1.5b` | **Conversational Brain** | 1.5B | General conversation, content generation, WhatsApp message drafting, and spoken response synthesis. |
| `deepseek-r1:1.5b` | **Reasoning Engine** | 1.5B | Complex multi-step logic, mathematics, physics, and deep reasoning via chain-of-thought `<think>` tags. |

**Model Selection is fully automatic.** The `LLM_Engine.py` checks the current user utterance for action-trigger keywords. If detected, control is delegated to the fine-tuned Orchestrator in a **zero-shot, history-stripped** prompt (System + User only). If no keywords are found, the conversational Qwen base model handles the response with full chat history context.

### 2. Custom Fine-Tuned Orchestrator Pipeline

The `Aether-Orchestrator` model was trained entirely in-house:

1. **Dataset Generation** (`training/dataset_generator.py`): 500 high-quality instruction→JSON-tool-call pairs were synthetically generated using OpenRouter API (Llama-3.1-8B-Instruct), covering all 15 tools with diverse phrasing.
2. **Fine-Tuning** (`training/kaggle_finetune.py`): SFT training on Kaggle GPUs using Unsloth + LoRA (r=16, alpha=32) on `Qwen2.5-Coder-1.5B-Instruct` for 3 epochs.
3. **Quantization & Export**: Merged LoRA weights → GGUF Q4_K_M quantization → `qwen2.5-coder-1.5b-instruct.Q4_K_M.gguf`.
4. **Deployment** (`training/Modelfile_strict`): Loaded into Ollama via a custom Modelfile with a strict system prompt enforcing raw JSON output, temperature 0.1, and 4096 context window.

### 3. Single-Pass Sequential Tool Execution

Unlike recursive ReAct loops, AETHER uses a **Single-Pass Sequential Architecture**:

```
User Speech → STT → Micro-Expert (Intent+Emotion) → Policy Router
                                                         ↓
                                        ┌─ Orchestrator (Tool Keywords Detected)
                                        │    → Zero-shot JSON tool array
                                        │    → Sequential tool execution
                                        │    → WhatsApp Content Engine intercept
                                        │    → Base Model verbal summary
                                        │
                                        └─ Base Model (No Keywords)
                                             → Conversational response
                                                         ↓
                                                   TTS → Speaker
```

**Key Design Decision:** The fine-tuned Orchestrator predicts the *entire* multi-tool chain in a single inference pass (e.g., `[search_web, send_whatsapp_message]`). Tools execute sequentially, with intermediate results appended to the context for downstream tools.

### 4. WhatsApp Content Engine Intercept

When `send_whatsapp_message` is triggered in a multi-tool chain, the engine detects if the Orchestrator hallucinated JSON syntax or placeholder text into the message body. If so, it:
1. Scrubs the Orchestrator's JSON history from context.
2. Hands the tool execution logs to `qwen2.5-coder:1.5b` with a strict "professional secretary" system prompt.
3. Generates clean, human-readable message text.
4. Injects this text back into the tool arguments before physical execution.

### 5. Hardware-Aware Ultra-Fast Intent Classification

AETHER runs a ~20M parameter **Micro-Expert** (`cross-encoder/nli-MiniLM2-L6-H768`) on CPU for zero-shot classification of:
- **Intent**: Desktop Command, Web Search, Knowledge Question, Creative Request, Screen Analysis, Casual Conversation, Debugging
- **Emotion**: Neutral, Frustrated, Excited, Urgent

Results are cached (LRU, 64 entries) and fed to the **Policy Router**, which applies strict IF/THEN rules (not learned gating) to compose behavioral adapters into the system prompt.

A **Keyword Hard Override** layer sits above the Micro-Expert, forcibly reclassifying intent to `desktop command` if tool-specific keywords (e.g., "whatsapp", "spotify", "volume", "google") are present, ensuring deterministic tool routing regardless of classifier uncertainty.

### 6. Live Context Truncation

If you interrupt AETHER mid-sentence using Voice Activity Detection (VAD) barge-in, AETHER halts its TTS engine, tracks exactly which syllables exited the speaker, and **truncates its own contextual memory log** so it cannot hallucinate what it didn't get to say.

### 7. Security Core (Tiered Authorization)

Tool execution passes through a tiered authorization gate:
- **Auto-Authorized (Safe):** `search_web`, `open_url`, `media_control`, `get_system_diagnostics`, `set_timer`, `get_current_time`, `search_and_read_web`, `read_specific_url`
- **Requires GUI Confirmation:** `send_whatsapp_message`, `run_computer_command`, `open_app_and_type`, `write_and_run_script`, `analyze_screen_with_llava`

The frontend displays a real-time Auth Modal for dangerous actions, blocking the AI thread until the user clicks Allow/Deny.

### 8. Audio-Reactive 3D Holographic UI
The frontend is a completely decoupled Next.js + React Three Fiber application connected via WebSockets. The 3D Hologram Orb physically maps to the backend's cognitive AI states (`Booting` → `Idle` → `Listening` → `Thinking` → `Speaking` → `Muted`).

### 9. Dynamic Skill Factory (Hot-Loading Tools)
AETHER can write and deploy its own tools on the fly. When a user asks for functionality that doesn't exist, AETHER writes a Python skill using `qwen2.5-coder:1.5b`, saves it to the `skills/` directory, and hot-loads it into the active execution context without requiring a system restart. A deterministic auto-correct layer intercepts hallucinated skill-creation triggers to ensure it only creates new tools when explicitly asked.

### 10. Daily-Driver Robustness & Optimizations
To make AETHER usable as a 24/7 background assistant, several optimizations were implemented:
- **Low-VRAM Audio Stack:** Whisper STT defaults to `base.en` on CPU (`int8`), freeing up ~800MB of GPU VRAM for the core reasoning models.
- **Lazy Vision Processing:** The camera loop drops from 5fps to a low-power 2s poll when the GUI dashboard is not actively open.
- **Execution Safeguards:** All dynamically loaded tools and web scraping functions are wrapped in a strict 15-second `ThreadPoolExecutor` timeout to prevent bad API endpoints from hanging the main voice pipeline.
- **Startup Briefing:** Automatically fetches weather, filters Hacker News for top AI/ML headlines, and reads any stored profile reminders on boot.

---

## 🔧 Complete Tool Arsenal (15 Native Tools)

| Tool | Category | Description |
|------|----------|-------------|
| `search_web` | Browser | Opens visible Google search in Chrome |
| `open_url` | Browser | Navigates to named sites (LinkedIn, YouTube, etc.) |
| `search_and_read_web` | Headless Scraper | Silent DuckDuckGo scrape → returns text snippets to LLM |
| `read_specific_url` | Headless Scraper | Fetches and parses a specific URL's content |
| `run_computer_command` | OS Automation | Opens apps, presses keys, runs shell commands, mouse clicks/scrolls, keyboard hotkeys |
| `open_app_and_type` | OS Automation | Opens an app + pastes text (with special Spotify URI handling) |
| `analyze_screen_with_llava` | Vision | Screenshots desktop → Llava-Phi3 multimodal analysis |
| `send_whatsapp_message` | Messaging | Full WhatsApp Desktop automation (search contact → paste → send) |
| `media_control` | Media | Play/Pause, Next/Prev track, Mute, Set absolute volume (PyCaw + COM) |
| `get_system_diagnostics` | System | CPU%, RAM%, Battery via psutil |
| `set_timer` | Utility | Background thread timer with native Windows TTS alarm |
| `get_current_time` | Utility | Returns formatted current time |
| `handle_smart_home` | IoT | Smart device control stub (lights, plugs) |
| `write_and_run_script` | Code Execution | DeepSeek-R1 writes Python → saves to workspace → executes with 30s timeout |
| `route_to_deepseek` | Routing | Redirects complex reasoning to DeepSeek-R1 chain-of-thought |

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend Core** | Python 3.10+, FastAPI, WebSockets, Uvicorn |
| **Frontend Dashboard** | Next.js 14, React Three Fiber, TailwindCSS |
| **LLM Inference** | Ollama (localhost:11434), GGUF Quantized Models |
| **Custom SFT Training** | Unsloth, LoRA (PEFT), Kaggle GPU, OpenRouter (Dataset Gen) |
| **Acoustics (STT)** | Faster-Whisper (CUDA, float16) |
| **Acoustics (TTS)** | Edge-TTS (Cloud, en-GB-RyanNeural) / Kokoro-82M (Local CPU) |
| **Acoustics (VAD)** | Energy-threshold VAD with barge-in interruption |
| **Vision** | MediaPipe Face Mesh, PyAutoGUI Screenshots, Llava-Phi3 |
| **Memory (RAG)** | FAISS Vector DB + all-MiniLM-L6-v2 Embeddings |
| **Memory (Personal)** | JSON flat-file user profile + LLM-based entity extraction |
| **OS Automation** | PyAutoGUI, Pyperclip, PyCaw (COM), subprocess, ctypes (Win32) |
| **Web Scraping** | BeautifulSoup4, DuckDuckGo HTML API |

## 🚀 Quick Start (Local Deployment)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/yourusername/aether-v3.git
cd aether-v3
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Setup Ollama Models

```bash
ollama pull qwen2.5-coder:1.5b
ollama pull deepseek-r1:1.5b
ollama pull llava-phi3
cd training
ollama create Aether-Orchestrator -f Modelfile_strict
```

### 3. Setup Frontend

```bash
cd frontend
npm install
```

### 4. Boot the Runtime

Open two terminal instances.
Terminal 1 (AI Backend Engine):

```bash
cd backend
python server.py
```

Terminal 2 (GUI Dashboard):

```bash
cd frontend
npm run dev
```

Navigate to `http://localhost:3000` to interact with the AETHER runtime. AETHER will immediately greet you with a personalized startup briefing containing local weather and trending AI news.

---

## 📂 Project Structure

```
AETHER_V1.0/
├── backend/
│   └── server.py              # FastAPI WebSocket server, VAD loop, AI orchestration
├── Core_agent/
│   ├── LLM_Engine.py          # Triple-Model MoE Engine with Single-Pass tool execution
│   ├── desktop_agent.py       # 10+ OS automation tools (WhatsApp, Spotify, Volume, etc.)
│   ├── web_agent.py           # Headless web scraping (DuckDuckGo + URL reader)
│   ├── aether_coder.py        # Autonomous code generation & execution sub-agent
│   ├── tools_executor.py      # Tool registry (15 tools) + Ollama function definitions
│   ├── micro_expert.py        # ~20M param zero-shot intent/emotion classifier
│   └── policy_router.py       # Rule-based behavioral adapter composition
├── speech/
│   ├── STT.py                 # Faster-Whisper GPU transcription
│   ├── TTS_edge.py            # Edge-TTS cloud synthesis
│   ├── TTS_kokoro.py          # Kokoro-82M local synthesis
│   ├── TTS_piper.py           # Piper TTS fallback
│   └── wake_word.py           # "Hey Jarvis" wake word detection
├── vision/
│   ├── camera_agent.py        # MediaPipe face mesh + emotion tracking
│   └── screen_capture.py      # Desktop screenshot capture for LLM vision
├── memory/
│   ├── knowledge_rag.py       # FAISS vector DB + embedding search
│   └── experience_memory.py   # User profile extraction & persistent storage
├── training/
│   ├── dataset_generator.py   # Synthetic dataset generation via OpenRouter
│   ├── kaggle_finetune.py     # Unsloth + LoRA SFT training script
│   ├── Modelfile_strict       # Ollama Modelfile for Aether-Orchestrator
│   ├── test_orchestrator.py   # Validation script for JSON output quality
│   └── aether_orchestrator_dataset.jsonl  # 500-pair training dataset
└── frontend/                  # Next.js 14 + React Three Fiber dashboard
```

---

## 🔒 Privacy & Sandboxing

AETHER is completely isolated. Audio frames, vision feeds, and code execution occur entirely on local silicon. The `AetherCoder` module statically analyzes intent commands via Regex constraints before firing any OS-level logic. All tool executions requiring physical OS interaction pass through the Security Core authorization gate.

## 📄 License

_All architectural claims regarding local MoE routing, policy-gated continual learning, and acoustic-driven context-truncation are currently proprietary and pending patent disclosure._
