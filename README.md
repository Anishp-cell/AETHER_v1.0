# AETHER v3.0: System-Level MoE Runtime

**AETHER** is a multi-modal, hardware-aware artificial intelligence runtime designed exclusively for 100% offline edge execution. Instead of relying on monolithic cloud APIs, AETHER dynamically routes sub-tasks across a local Mixture of Experts (MoE) pipeline, dynamically balancing CPU and GPU workloads for ultra-low latency conversational AI.

## 🌟 Core Architecture

### 1. Hardware-Aware Dynamic Routing

AETHER runs an ultra-fast local NLP "Micro-Expert" to classify user intent and emotion in milliseconds. Based on the classification, the **Policy Router** forwards the semantic payload to specialized models:

- **Fast Path:** `qwen2.5:1.5b` (For casual conversation & generic Q/A)
- **Deep Memory Path:** FAISS Vector Database + Local RAG embedding models (`all-MiniLM-L6-v2`)
- **Reasoning Path:** `deepseek-r1:1.5b` (For complex multi-step logical deduction)
- **System Path:** `AetherCoder` (For deterministic, sandboxed execution of generated Python/OS commands)

### 2. Live Context Truncation

Traditional smart speakers suffer from conversational hallucinations when interrupted. AETHER features a mathematically unique **Asynchronous Grapheme-Tracking Pipeline**.
If you interrupt AETHER mid-sentence using the Voice Activity Detection (VAD) barge-in, AETHER halts its TTS engine, tracks exactly which syllables exited the speaker, and **truncates its own contextual memory log** so it mathematically cannot hallucinate what it didn't get to say.

### 3. Audio-Reactive 3D Holographic UI

The frontend is a completely decoupled Next.js + React Three Fiber application connected via WebSockets. The 3D Hologram Orb physically maps to the backend's cognitive AI states (`Booting` -> `Idle` -> `Listening` -> `Thinking` -> `Speaking` -> `Muted`).

---

## 🛠 Tech Stack

- **Backend Core**: Python 3.10+, FastAPI, WebSockets
- **Frontend Dashboard**: Next.js 14, React Three Fiber, TailwindCSS
- **Machine Learning**: Ollama, Transformers, FAISS
- **Acoustics (VAD/STT/TTS)**: Faster-Whisper, Kokoro-82M, WebRTC VAD
- **Vision**: Camera Mesh Parsing + Llava-Phi3

## 🚀 Quick Start (Local Deployment)

### 1. Clone & Setup Backend

```bash
git clone https://github.com/yourusername/aether-v3.git
cd aether-v3
python -m venv .venv
source .venv/Scripts/activate  # Windows
pip install -r requirements.txt
```

### 2. Setup Frontend

```bash
cd frontend
npm install
```

### 3. Boot the Runtime

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

Navigate to `http://localhost:3000` to interact with the AETHER runtime.

---

## 🔒 Privacy & Sandboxing

AETHER is completely isolated. Audio frames, vision feeds, and code execution occur entirely on local silicon. The `AetherCoder` module statically analyzes intent commands via Regex constraints before firing any OS-level logic.

## 📄 License

_All architectural claims regarding local MoE routing and acoustic-driven context-truncation are currently proprietary and pending patent disclosure._
