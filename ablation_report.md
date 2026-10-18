# AETHER v4.0 Full-Stack Memory Ablation Report
> Generated: 2026-10-02 10:46:34 on NVIDIA GeForce RTX 5050

## 1. Incremental Component Breakdown

| Stage | Category | Component | RAM (MB) | Delta RAM | VRAM (MB) | Heap (MB) | Elapsed | Notes |
|---|---|---|---|---|---|---|---|---|
| 00 | Baseline | **Baseline (Python runtime & probe)** | 528.7 | +4.6 | 1589.0 | 0.0 | 52ms | Empty interpreter baseline |
| 01 | Infrastructure | **FastAPI + Uvicorn + Starlette** | 554.4 | +25.7 | 1586.2 | 9.6 | 802ms | Web server framework |
| 02 | Infrastructure | **sounddevice + numpy (Audio I/O)** | 560.4 | +6.0 | 1586.2 | 0.3 | 49ms | C-extensions and audio stream bindings |
| 03 | Speech (STT) | **faster_whisper (import only)** | 624.9 | +64.4 | 1583.2 | 23.7 | 3162ms | CTranslate2 runtime and Whisper loader |
| 04 | Speech (STT) | **WhisperModel('base.en', int8 CPU) init** | 728.6 | +103.8 | 1583.2 | 0.5 | 757ms | Loaded 74M param STT int8 weights into CPU RAM |
| 05 | Speech (Wake) | **wake_word detector (import & init)** | 916.7 | +188.1 | 1583.2 | 53.5 | 2485ms | Energy threshold gate and keywords |
| 06 | Speech (TTS) | **TTS_kokoro (import only)** | 1171.3 | +254.6 | 1577.8 | 118.2 | 14566ms | Kokoro pipeline bindings |
| 07 | Speech (TTS) | **KokoroTTS (82M weights init)** | 2251.9 | +1080.5 | 1577.8 | 96.0 | 3956ms | Kokoro-82M speech synthesis weights |
| 08 | Speech (DSP) | **audio_dsp (AudioDSPChain)** | 2252.0 | +0.1 | 1577.8 | 0.0 | 2ms | Broadcast audio filter chain (NumPy ops) |
| 09 | Speech (Logic) | **dynamic_turn_taking (Sparrow-2 engine)** | 2252.2 | +0.2 | 1577.8 | 0.1 | 4ms | Turn taking state machine and silence thresholds |
| 10 | Speech (STT) | **streaming_stt (StreamingSTT engine)** | 2252.2 | +0.0 | 1577.8 | 0.1 | 5ms | Streaming speech buffer manager |
| 11 | Speech (Logic) | **interruption_recovery engine** | 2252.3 | +0.1 | 1577.8 | 0.0 | 2ms | State rollback and interruption detection |
| 12 | Routing (ARN) | **ARN Router (Int8 ONNX session)** | 2286.7 | +34.4 | 1577.8 | 3.9 | 1256ms | 4.5MB Int8 ONNX graph in ONNX Runtime |
| 13 | Routing (ARN) | **ARN PyTorch shell (CRF Viterbi)** | 2287.0 | +0.3 | 1577.8 | 0.1 | 19ms | PyTorch shell for CRF sequence decoding |
| 14 | Routing | **hybrid_router (dispatcher)** | 2299.9 | +12.9 | 1577.8 | 0.0 | 38ms | Dispatcher logic between fast-path and full reasoning |
| 15 | Routing | **policy_router (PolicyRouter)** | 2300.1 | +0.2 | 1577.8 | 0.0 | 15ms | Heuristic policy evaluation engine |
| 16 | Auxiliary Transformer | **micro_expert (cross-encoder classifier)** | 2371.4 | +71.3 | 1533.3 | 19.0 | 1658ms | cross-encoder/nli-MiniLM2-L6-H768 zero-shot intent classifier |
| 17 | LLM Engine | **FrozenLLMEngine (HTTP client & history)** | 2471.2 | +99.8 | 1533.3 | 40.2 | 7661ms | Client coordinator and sliding window conversation history |
| 18 | Memory | **ExperienceMemory (Tier 2 profile)** | 2471.2 | +0.0 | 1533.3 | 0.0 | 0ms | User profile and persistent preferences (JSON dict) |
| 19 | Memory (RAG) | **FAISS index load (faiss_index.bin)** | 2471.2 | +0.0 | 1533.3 | 0.0 | 0ms | FAISS vector index in RAM |
| 20 | Memory (RAG) | **documents.npy (RAG document records)** | 2471.2 | +0.0 | 1533.3 | 0.0 | 0ms | Document metadata array |
| 21 | Auxiliary Transformer | **SentenceTransformer('all-MiniLM-L6-v2')** | 2488.5 | +17.2 | 1556.1 | 4.1 | 5323ms | 384-dimensional embedding model on CPU |
| 22 | Tools | **tools_executor (AVAILABLE_TOOLS)** | 2488.5 | +0.0 | 1556.1 | 0.0 | 0ms | Registry of 53 native tools and function pointers |
| 23 | Skills | **skill_registry + skill_factory** | 2488.5 | +0.0 | 1556.1 | 0.0 | 0ms | Dynamic skill discovery and code generation templates |
| 24 | Skills | **aether_coder + auto_skill_forge** | 2488.5 | +0.0 | 1556.1 | 0.0 | 0ms | Self-refining coding tools and auto skill forge |
| 25 | Vision | **OpenCV (camera_agent + screen_capture)** | 2535.1 | +46.6 | 1556.1 | 18.0 | 1592ms | OpenCV cv2 bindings and screen capture frame grabber |
| 26 | Desktop Automation | **desktop_agent (pyautogui + win32)** | 2535.2 | +0.0 | 1556.1 | 0.0 | 0ms | OS mouse, keyboard, and window control |
| 27 | Desktop Automation | **screen_annotator + canvas_copilot (PIL)** | 2535.2 | +0.0 | 1556.1 | 0.0 | 0ms | PIL image manipulation and canvas overlays |
| 28 | Background Services | **ambient_daemon (thread structures)** | 2535.2 | +0.0 | 1556.1 | 0.0 | 0ms | Ambient state watcher |
| 29 | UI / Overlay | **desktop_overlay (HUD Tkinter structures)** | 2536.8 | +1.6 | 1556.1 | 0.0 | 2ms | Floating Windows HUD overlay structures |
| 30 | Credentials & API | **NIMKeyPool (Multi-Key Pool Manager)** | 2541.3 | +4.3 | 1556.1 | 0.1 | 9ms | Multi-account pool manager (4 active keys, rotation lock) |
| 31 | Browser Subprocess | **browser_worker (config & worker interface)** | 2541.5 | +0.2 | 1556.1 | 0.0 | 3ms | Async browser worker dispatcher |
| 32 | System State | **FULL STACK IDLE (All components loaded)** | 2561.8 | +0.0 | 1556.1 | 0.0 | 108ms | Complete AETHER system at rest, ready for input |
| 33 | Active Inference | **ACTIVE INFERENCE: ARN Fast-Path Tool Query** | 2580.7 | +18.8 | 1556.1 | 0.0 | 54ms | Sub-5ms local routing execution (Result: []) |
| 34 | Active Inference | **ACTIVE INFERENCE: NVIDIA NIM Cloud Pool Query** | 2581.3 | +0.6 | 1556.1 | 0.2 | 166ms | Real-world cloud query across multi-key pool (measures local memory delta) |
| 35 | Active Inference | **ACTIVE INFERENCE: RAG Vector Search Query** | 2625.5 | +44.2 | 1556.1 | 0.1 | 167ms | FAISS L2 distance search and embedding execution |
| 36 | System State | **POST-INFERENCE IDLE (After Garbage Collection)** | 2625.5 | +0.0 | 1556.1 | 0.0 | 103ms | System footprint after queries and garbage collection |

## 2. Top RAM Consumers (Initialization & Weights)

| Rank | Component | Delta RAM (MB) | Category | Description |
|---|---|---|---|---|
| #1 | **KokoroTTS (82M weights init)** | **+1080.5 MB** | Speech (TTS) | Kokoro-82M speech synthesis weights |
| #2 | **TTS_kokoro (import only)** | **+254.6 MB** | Speech (TTS) | Kokoro pipeline bindings |
| #3 | **wake_word detector (import & init)** | **+188.1 MB** | Speech (Wake) | Energy threshold gate and keywords |
| #4 | **WhisperModel('base.en', int8 CPU) init** | **+103.8 MB** | Speech (STT) | Loaded 74M param STT int8 weights into CPU RAM |
| #5 | **FrozenLLMEngine (HTTP client & history)** | **+99.8 MB** | LLM Engine | Client coordinator and sliding window conversation history |
| #6 | **micro_expert (cross-encoder classifier)** | **+71.3 MB** | Auxiliary Transformer | cross-encoder/nli-MiniLM2-L6-H768 zero-shot intent classifier |
| #7 | **faster_whisper (import only)** | **+64.4 MB** | Speech (STT) | CTranslate2 runtime and Whisper loader |

## 3. Cumulative Category Footprint

| Category | Total Contributed RAM (MB) | % of Contributed RAM |
|---|---|---|
| **Speech (TTS)** | 1335.2 MB | 66.2% |
| **Speech (Wake)** | 188.1 MB | 9.3% |
| **Speech (STT)** | 168.2 MB | 8.3% |
| **LLM Engine** | 99.8 MB | 4.9% |
| **Auxiliary Transformer** | 88.5 MB | 4.4% |
| **Vision** | 46.6 MB | 2.3% |
| **Routing (ARN)** | 34.8 MB | 1.7% |
| **Infrastructure** | 31.8 MB | 1.6% |
| **Routing** | 13.1 MB | 0.6% |
| **Baseline** | 4.6 MB | 0.2% |
| **Credentials & API** | 4.3 MB | 0.2% |
| **UI / Overlay** | 1.6 MB | 0.1% |
| **Speech (Logic)** | 0.3 MB | 0.0% |
| **Browser Subprocess** | 0.2 MB | 0.0% |
| **Speech (DSP)** | 0.1 MB | 0.0% |
| **Memory** | 0.0 MB | 0.0% |
| **Memory (RAG)** | 0.0 MB | 0.0% |
| **Tools** | 0.0 MB | 0.0% |
| **Skills** | 0.0 MB | 0.0% |
| **Desktop Automation** | 0.0 MB | 0.0% |
| **Background Services** | 0.0 MB | 0.0% |

## 4. Idle vs. Active Inference Memory

| State | RAM (MB) | VRAM (MB) | Delta RAM | Elapsed |
|---|---|---|---|---|
| FULL STACK IDLE (All components loaded) | 2561.8 MB | 1556.1 MB | +0.0 MB | 108ms |
| ACTIVE INFERENCE: ARN Fast-Path Tool Query | 2580.7 MB | 1556.1 MB | +18.8 MB | 54ms |
| ACTIVE INFERENCE: NVIDIA NIM Cloud Pool Query | 2581.3 MB | 1556.1 MB | +0.6 MB | 166ms |
| ACTIVE INFERENCE: RAG Vector Search Query | 2625.5 MB | 1556.1 MB | +44.2 MB | 167ms |
| POST-INFERENCE IDLE (After Garbage Collection) | 2625.5 MB | 1556.1 MB | +0.0 MB | 103ms |