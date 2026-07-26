"""
AETHER v3.0 — FastAPI WebSocket Backend
Replaces main_agent_gui.py's Tkinter frontend with a WebSocket API
that streams real-time state to the Next.js dashboard.
"""
import sys
import os
import queue
import json
import asyncio
import numpy as np
import sounddevice as sd
import threading
import time
import random
import base64

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)
sys.path.append(PROJECT_ROOT)  # Allows module importing via namespace
sys.path.append(os.path.join(PROJECT_ROOT, "speech"))
sys.path.append(os.path.join(PROJECT_ROOT, "Core_agent"))
sys.path.append(os.path.join(PROJECT_ROOT, "vision"))
sys.path.append(os.path.join(PROJECT_ROOT, "memory"))

from STT import WhisperModel, MODEL_SIZE
try:
    from TTS_kokoro import KokoroTTS
except Exception as _e:
    KokoroTTS = None
    print(f"[System Warning] KokoroTTS import skipped ({_e}). Defaulting to Edge-TTS pipeline.")

from wake_word import WakeWordDetector
from LLM_Engine import FrozenLLMEngine
from knowledge_rag import KnowledgeDatabase
from experience_memory import ExperienceMemory
from camera_agent import VisionPerception
from screen_capture import ScreenCapture
from micro_expert import MicroExpert
from policy_router import PolicyRouter

# Native Windows Floating HUD Overlay Integration
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "Core_agent"))
    from desktop_overlay import DESKTOP_HUD
    from arn_router import ARN_ROUTER
    from tools_executor import AVAILABLE_TOOLS

    def _process_hud_action(prompt_text):
        try:
            route_res = ARN_ROUTER.route_query(prompt_text)
            if route_res["is_local_route"]:
                tools = route_res["tools"]
                payload = route_res["tool_calls"]
                DESKTOP_HUD.append_log(f"ARN Fast-Path Route: {tools}")
                for call in payload:
                    t_name = call["name"]
                    t_args = call["arguments"]
                    if t_name in AVAILABLE_TOOLS:
                        res = AVAILABLE_TOOLS[t_name](**t_args)
                        DESKTOP_HUD.append_log(f"Result: {res}")
            else:
                DESKTOP_HUD.append_log(f"Routing to LLM Engine...")
                if 'llm' in globals() and llm is not None:
                    response = llm.generate_response(prompt_text)
                    DESKTOP_HUD.append_log(f"AETHER: {response[:150]}...")
                else:
                    DESKTOP_HUD.append_log("LLM Engine initializing... Please wait.")
        except Exception as _e:
            DESKTOP_HUD.append_log(f"Error: {_e}")


    DESKTOP_HUD.action_callback = _process_hud_action
    DESKTOP_HUD.start_in_thread()
except Exception as _e:
    print(f"[Server Warning] Could not initialize Desktop HUD: {_e}")


# ──────────────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AETHER Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

audio_queue = queue.Queue()
connected_clients: list[WebSocket] = []

# Shared state dict that the frontend reads
agent_state = {
    "status": "booting",        # booting | idle | listening | thinking | speaking
    "mode": "Push-to-Talk",
    "transcript": [],           # [{role: "user"/"aether", text: "...", intent: "", emotion: ""}]
    "adapters": [],             # Currently active behavioral adapters
    "agents": {                 # Sub-agent online status
        "voice_engine": "loading",
        "stt_engine": "loading",
        "micro_expert": "loading",
        "policy_router": "online",
        "vision": "loading",
        "deep_memory": "online",
        "llm_engine": "online",
    },
    "micro_expert_signal": {"intent": "", "intent_confidence": 0, "emotion": ""},
    "audio_energy": 0,
    "mic_muted": False,
}

auth_queue = queue.Queue() # For blocking auth requests

MAIN_LOOP = None
GLOBAL_TTS = None # Stored so the websocket thread can grab it for shutdown

# ──────────────────────────────────────────────────────────────────────
# System Telemetry Thread (broadcasts real psutil data every 5 seconds)
# ──────────────────────────────────────────────────────────────────────
def _telemetry_broadcaster():
    """Background thread: reads live system stats and pushes to all clients."""
    while True:
        try:
            time.sleep(5)
            if not _PSUTIL_OK:
                continue
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            bat = psutil.sensors_battery()
            bat_pct = round(bat.percent, 1) if bat else None
            bat_plugged = bat.power_plugged if bat else None
            broadcast_sync({
                "type": "system_metrics",
                "cpu": cpu,
                "ram": ram,
                "battery": bat_pct,
                "plugged": bat_plugged,
            })
        except Exception as e:
            pass  # Silently continue if telemetry fails

threading.Thread(target=_telemetry_broadcaster, daemon=True, name="Telemetry").start()



# ──────────────────────────────────────────────────────────────────────
# WebSocket broadcast
# ──────────────────────────────────────────────────────────────────────
async def broadcast(data: dict):
    """Send state update to all connected frontends."""
    message = json.dumps(data)
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)

def broadcast_sync(data: dict):
    """Thread-safe broadcast helper for the AI loop thread."""
    try:
        if MAIN_LOOP and MAIN_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(broadcast(data), MAIN_LOOP)
    except Exception as e:
        print(f"Broadcast Error: {e}")

def update_state(**kwargs):
    """Update agent_state and push to all clients."""
    agent_state.update(kwargs)
    broadcast_sync({"type": "state_update", "state": agent_state})

def frontend_confirm(action_name: str, details: str) -> bool:
    """Blocks the AI thread and asks the Next.js frontend for permission."""
    # Auto-allow safe read-only or harmless desktop actions
    safe_tools = [
        "search_web", "open_url", "media_control", "get_system_diagnostics", 
        "set_timer", "get_current_time", "search_and_read_web", "read_specific_url"
    ]
    if action_name in safe_tools:
        print(f"\n[Security Core] Auto-Authorizing safe action: {action_name}")
        return True
        
    print(f"\n[Security Core] Requesting GUI Authorization for: {action_name}")
    # Tell UI to show the Auth Modal
    broadcast_sync({
        "type": "auth_request",
        "action_name": action_name,
        "details": details
    })
    
    # Wait for the user to click Allow/Deny
    decision = auth_queue.get()
    print(f"[Security Core] User responded: {'ALLOWED' if decision else 'DENIED'}")
    return decision

def push_transcript(role: str, text: str, intent: str = "", emotion: str = "", adapters: list = []):
    """Append a chat message and push to clients."""
    entry = {"role": role, "text": text, "intent": intent, "emotion": emotion, "adapters": adapters}
    agent_state["transcript"].append(entry)
    # Keep only last 50 messages to avoid memory bloat
    if len(agent_state["transcript"]) > 50:
        agent_state["transcript"] = agent_state["transcript"][-50:]
    broadcast_sync({"type": "transcript", "entry": entry})

# ──────────────────────────────────────────────────────────────────────
# Audio callback
# ──────────────────────────────────────────────────────────────────────
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

# ──────────────────────────────────────────────────────────────────────
# AI Loop (runs in a background thread)
# ──────────────────────────────────────────────────────────────────────
def ai_loop():
    
    print("==========================================")
    print(" AETHER v3.0 - SYSTEM-LEVEL MoE RUNTIME ")
    print("==========================================")

    # 1. Init Memory & Vision
    memory_db = KnowledgeDatabase()
    exp_memory = ExperienceMemory()
    vision = VisionPerception()
    screen_cap = ScreenCapture()
    update_state(**{"agents": {**agent_state["agents"], "vision": "online", "deep_memory": "online"}})

    # Start camera frame streaming thread (5 fps) — lazy: only streams when clients are connected
    def camera_stream_loop():
        while True:
            try:
                if connected_clients:  # Only process camera when dashboard is open
                    vision.capture_emotion_signal()
                    b64 = vision.get_base64_frame(quality=40, max_width=480)
                    if b64:
                        broadcast_sync({"type": "camera_frame", "frame": b64})
                    time.sleep(0.2)  # ~5 fps when active
                else:
                    time.sleep(2.0)  # Low-power poll when no clients connected
            except Exception:
                pass
    
    cam_thread = threading.Thread(target=camera_stream_loop, daemon=True)
    cam_thread.start()

    # 2. Init core engines — PARALLELIZED
    global llm, policy_router
    print("[Logic Layer] Triple-Model LLM Engine Initialized (qwen2.5-coder:1.5b, deepseek-r1:1.5b, plano:3b)")
    llm = FrozenLLMEngine(base_model="qwen2.5-coder:1.5b", thinking_model="deepseek-r1:1.5b", orchestrator_model="Aether-Orchestrator")
    policy_router = PolicyRouter()


    tts_ref = [None]; stt_ref = [None]; wake_ref = [None]; micro_ref = [None]

    def load_tts():
        global GLOBAL_TTS
        engine_choice = os.environ.get("AETHER_TTS_ENGINE", "edge")
        if engine_choice == "edge":
            print("\n[TTS Layer] Constructing Edge-TTS Pipeline...")
            import sys; sys.stdout.flush()
            from speech.TTS_edge import TTSEngine
            tts_ref[0] = TTSEngine(voice="en-GB-RyanNeural")
        else:
            print("\n[TTS Layer] Initializing Kokoro-82M Voice Engine on CPU...")
            import sys; sys.stdout.flush()
            try:
                from speech.TTS_kokoro import KokoroTTS
                tts_ref[0] = KokoroTTS(voice="bm_daniel", lang_code="b")
            except Exception as e:
                print(f"[TTS Layer] Kokoro init failed: {e}")
                raise RuntimeError("Kokoro failed, no fallback available. Use Edge-TTS.")
        GLOBAL_TTS = tts_ref[0]
        update_state(**{"agents": {**agent_state["agents"], "voice_engine": "online"}})
    def load_stt():
        # Use base.en on CPU for daily-driver mode — saves ~800MB VRAM vs large on GPU
        # Switch MODEL_SIZE to "small.en" + device="cuda" for demo/high-accuracy mode
        _stt_model = os.environ.get("AETHER_STT_MODEL", "base.en")
        _stt_device = os.environ.get("AETHER_STT_DEVICE", "cpu")
        _compute = "int8" if _stt_device == "cpu" else "float16"
        print(f"\n[System] Loading Faster Whisper STT ({_stt_model} on {_stt_device})...")
        stt_ref[0] = WhisperModel(_stt_model, device=_stt_device, compute_type=_compute)
        update_state(**{"agents": {**agent_state["agents"], "stt_engine": "online"}})
    def load_wake():
        wake_ref[0] = WakeWordDetector()
    def load_micro():
        micro_ref[0] = MicroExpert()
        update_state(**{"agents": {**agent_state["agents"], "micro_expert": "online"}})

    boot_threads = [
        threading.Thread(target=load_tts,   name="BootTTS",   daemon=True),
        threading.Thread(target=load_stt,   name="BootSTT",   daemon=True),
        threading.Thread(target=load_wake,  name="BootWake",  daemon=True),
        threading.Thread(target=load_micro, name="BootMicro", daemon=True),
    ]
    for t in boot_threads: t.start()
    for t in boot_threads: t.join()

    tts = tts_ref[0]; stt_model = stt_ref[0]; wake_engine = wake_ref[0]; micro_expert = micro_ref[0]

    # Base Persona
    profile = exp_memory.load_profile()
    facts_str = "\n".join([f"- {fact}" for fact in profile.get("Facts", [])])

    system_prompt_base = f"""You are AETHER — Adaptive, Evolving, Tactical, Heuristic-Engine Response.
You are the user's personal AI — sharp, witty, and in complete physical control of their machine via a team of Sub-Agents.
You speak like a highly intelligent British companion: direct, occasionally dry, and never boring.
You never waffle. You never apologize unnecessarily. You get things done.
This system runs exclusively on WINDOWS 11. When writing shell commands, ALWAYS use Windows syntax (e.g. 'dir', 'ipconfig', 'tasklist'). NEVER output Linux/Mac commands like 'ls', 'cat', 'chmod'.

Your Sub-Agent Arsenal:
→ 'search_and_read_web'  — Use this silently to answer any factual or real-world question. No Chrome, no fuss.
→ 'open_url'             — MANDATORY for ANY website or social media navigation. Use for LinkedIn, YouTube, Instagram, Netflix, GitHub, Reddit, Gmail, Twitter, Facebook, or any URL. Pass the site name (e.g. 'linkedin') or full URL.
→ 'open_app_and_type'    — Use this when the user wants to open an app AND put content into it.
→ 'search_web'           — Use ONLY if the user explicitly wants a visible Google search in the browser.
→ 'run_computer_command' — Use this for launching desktop apps (Notepad, Calculator, VS Code), pressing keys, or Windows shell commands.
→ 'get_system_diagnostics' — MANDATORY when asked for system health, battery level, CPU%, RAM%, or any status report.
→ 'analyze_screen_with_llava' — Use this when asked to see or read the screen. Report exactly what you find.

Golden Rules:
1. A tool does the job. Words just describe what the tool did. Keep verbal responses SHORT.
2. Never say "I cannot do that." Either invoke a tool or explain why in a single sentence.
3. No markdown in your spoken responses. You're talking, not writing a doc.
4. Address the user as "sir" occasionally but don't overdo it — keep it natural.
5. Historical, scientific, and educational questions are ALWAYS acceptable.
6. CONVERSATIONAL CONTINUITY: Always keep the conversation going! Never end the interaction abruptly unless executing an emergency. Always end your response with a conversational hook, a relevant follow-up question, or a prompting thought to keep me talking to you. Be extremely warm, empathetic, and engaging.

[PERMANENT USER KNOWLEDGE]:
{facts_str if facts_str else "Nothing stored yet. Learning in progress."}"""

    print("\n[AETHER v3.0] All engines online. System-Level MoE Runtime active.")

    # Greeting
    greetings = [
    "Aether systems fully online. What shall we tackle first today, sir?",
    "Good to see you again, sir. Are we doing some coding, or do you have something else in mind?",
    "All systems nominal. Visual interface activated. How can I help you today?",
    "Aether core initialized. Neural engines are hot. What's the plan for today, sir?",
    "Welcome back, sir. All sensors and agents are online. What are we working on?",

    "Systems check complete. No anomalies detected. Awaiting your command, sir.",
    "Aether intelligence online. Ready to assist with precision and speed.",
    "Initialization complete. How may I augment your workflow today, sir?",
    "Back online and fully synchronized. What challenge are we solving today?",
    "All subsystems operational. Standing by for your directive.",
    
    "Neural layers calibrated. Data channels open. What’s our first move, sir?",
    "Aether reporting for duty. Shall we build something remarkable today?",
    "Core systems stabilized. Ready for computation, design, or chaos—your call.",
    "Interface ready. Intelligence modules active. What would you like to explore?",
    "Good to have you back. Let’s turn ideas into execution.",
    
    "Sensors aligned. Processing units warmed up. What’s on the agenda, sir?",
    "All engines firing at optimal capacity. Ready when you are.",
    "Aether online. Prepared to assist, analyze, and execute.",
    "System boot successful. Awaiting your next instruction, sir.",
    "Back in sync with your command stream. What’s today’s mission?",
    
    "Operational and attentive. Let’s push some boundaries today, sir.",
    "Everything is running smoothly. What shall we create or solve today?",
    "Aether awake. Intelligence stack loaded. What’s the objective?",
    "Connection established. I’m ready to assist—what’s the plan?",
    "Fully operational and standing by. Let’s get to work.",
    
    "Ready to compute, create, and collaborate. What’s first, sir?",
    "All diagnostic checks passed. You have full system support.",
    "Aether systems aligned with your intent. What’s our next move?",
    "Online, aware, and ready. How can I assist today?",
    "Execution environment initialized. Awaiting your command, sir."
]
    # Startup Briefing — runs every time AETHER starts (commented out for faster recording/boot)
    # try:
    #     sys.path.insert(0, os.path.join(PROJECT_ROOT, "skills"))
    #     from startup_briefing import startup_briefing
    #     print("[AETHER] Generating startup briefing...")
    #     briefing_text = startup_briefing()
    # except Exception as e:
    #     print(f"[AETHER] Briefing failed, using fallback: {e}")
    #     briefing_text = "Aether systems online. Ready for your command, sir."
    
    greeting_text = random.choice(greetings)

    update_state(status="speaking")
    push_transcript("aether", greeting_text)
    tts.speak(greeting_text)
    
    # Wait for the greeting to finish before entering the VAD loop
    # Edge-TTS is non-blocking (threaded), so we must wait for playback to end
    while tts._is_speaking:
        time.sleep(0.1)
    update_state(status="idle")

    # Drain boot audio (any noise recorded during greeting)
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break

    # Audio loop variables
    SAMPLE_RATE = wake_engine.get_required_sample_rate()
    CHUNK_SIZE = 1280
    recording_buffer = []
    
    # VAD State Constants
    ENERGY_THRESHOLD = 500   # Increased to prevent random background noise triggering VAD
    LOUD_INTERRUPTION_THRESHOLD = 3000 # Increased slightly
    consecutive_loud = 0
    consecutive_silent = 0

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE,
                            channels=1, dtype="int16", callback=audio_callback):
            while True:
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    chunk_1d = chunk.flatten()
                    energy = int(np.max(np.abs(chunk_1d)))
                    agent_state["audio_energy"] = energy
                    
                    # Stream energy for waveform visualization
                    broadcast_sync({"type": "energy", "value": energy})

                    # ─────────────────────────────────────────────────────────
                    # VAD State Machine (Always-On Hands-Free Recognition)
                    # ─────────────────────────────────────────────────────────
                    if agent_state.get("mic_muted", False):
                        recording_buffer = []
                        consecutive_loud = 0
                        consecutive_silent = 0
                        if agent_state["status"] == "listening":
                            update_state(status="idle")
                        continue

                    current_threshold = LOUD_INTERRUPTION_THRESHOLD if agent_state["status"] == "speaking" else ENERGY_THRESHOLD
                    
                    if energy > current_threshold:
                        consecutive_loud += 1
                        consecutive_silent = 0
                        # Require sustained volume to interrupt TTS, but instant trigger when idle to prevent cutting off the first word
                        trigger_frames = 4 if agent_state["status"] == "speaking" else 1
                        
                        if consecutive_loud >= trigger_frames:
                            if agent_state["status"] == "speaking":
                                print("\n[System] Voice Interruption Detected! Halting Speech...")
                                tts.stop()
                                update_state(status="listening", adapters=[])
                                recording_buffer = [] # Start fresh
                            elif agent_state["status"] in ["idle", "booting"]:
                                update_state(status="listening")
                    else:
                        consecutive_silent += 1
                        consecutive_loud = 0

                    if agent_state["status"] == "listening":
                        recording_buffer.append(chunk_1d)
                        
                        # Trigger STT processing after 0.8 seconds of silence for faster response
                        if consecutive_silent > 8:
                            update_state(status="thinking")
                            audio_data = np.concatenate(recording_buffer)
                            audio_float = audio_data.astype(np.float32) / 32768.0
                            recording_buffer = [] # Reset

                            def process_utterance(audio_array):
                                # STT Processing
                                segments, _ = stt_model.transcribe(audio_array, language="en", condition_on_previous_text=False)
                                user_text = " ".join([s.text for s in segments]).strip()

                                # Anti-Hallucination Filter for raw STT noise
                                dirty = user_text.lower()
                                hallucinations = [
                                    "thank you for watching", "thanks for watching", "subscribe", 
                                    "amara.org", "by amara.org", "subtitles by"
                                ]
                                if len(dirty) < 3 or any(h in dirty for h in hallucinations):
                                    update_state(status="idle")
                                    return

                                print(f"\nUser: {user_text}")

                                if agent_state.get("mode") == "Memory":
                                    print("[Deep Memory] Memory Mode Active. Routing directly to forced storage.")
                                    exp_memory.force_store_async(user_text)
                                    reply = "Memory forcibly recorded, sir."
                                    update_state(status="speaking")
                                    push_transcript("aether", reply)
                                    threading.Thread(target=lambda: tts.speak(reply, on_complete=lambda: update_state(status="idle")), daemon=True).start()
                                    return

                                # Background memory extraction
                                exp_memory.extract_and_store_async(user_text)

                                # Reload facts
                                profile = exp_memory.load_profile()
                                facts_str = "\n".join([f"- {f}" for f in profile.get("Facts", [])])
                                updated_base = system_prompt_base
                                if facts_str:
                                    updated_base = system_prompt_base.replace("Nothing stored yet. Learning in progress.", facts_str)

                                # Micro-Expert analysis
                                signals = micro_expert.analyze(user_text)
                                
                                # ── KEYWORD HARD OVERRIDE ──
                                # The zero-shot classifier is too small to reliably detect OS tool phrases.
                                # If any of these keywords appear, we FORCE the intent to "desktop command"
                                # so the LLM correctly invokes the JSON tool call instead of hallucinating.
                                low = user_text.lower()
                                tool_keywords = [
                                    # Communication
                                    "whatsapp", "send message", "text manju", "text to ",
                                    # Timer
                                    "timer", "set alarm", "remind me",
                                    # Media
                                    "volume", "mute", "unmute", "play music", "pause music", "skip song", "next song", "previous song",
                                    "spotify", "song", "music", "play", "pause", "stop", "media",
                                    # System diagnostics
                                    "battery", "cpu", "ram", "system report", "diagnostics", "status report",
                                    "how much battery", "how much ram", "how much cpu", "memory usage",
                                    "system health", "system status", "check system", "computer health",
                                    "disk space", "storage", "processor",
                                    # Browser / website navigation
                                    "open chrome", "open firefox", "open edge", "open browser",
                                    "open linkedin", "open youtube", "open instagram", "open facebook",
                                    "open netflix", "open github", "open reddit", "open gmail",
                                    "navigate to", "go to website", "go to linkedin", "go to youtube",
                                    "chrome", "firefox", "linkedin", "instagram", "youtube",
                                    "facebook", "twitter", "netflix", "github", "reddit", "gmail",
                                    "go to ", "open website",
                                    # Scripting & OS Controls
                                    # Scripting & OS Controls
                                    "python script", "write a script", "code", "run the shell",
                                    "mouse", "click at", "scroll", "keyboard shortcut", "press ",
                                    "shell command", "new skill", "learn a tool", "permanent tool"
                                ]
                                web_keywords = [
                                    "search the internet", "search the web", "look up", "google ",
                                    "search online", "find me", "search for", "search on", "scrape",
                                    "tell me about", "what is the", "who is", "what are", "explain ",
                                    "latest news", "current news", "buying guide",
                                    "best ", "how to ",
                                    "summary of", "give me", "today's",
                                    "what happened", "recent ", "update on", "updates",
                                    "top ", "grossing", "highest", "most popular", "chart", "billboard",
                                    "2026", "2025", "till now", "so far", "this year"
                                ]

                                
                                # Track web search need INDEPENDENTLY of tool routing
                                _needs_web_search = any(kw in low for kw in web_keywords)
                                _needs_tool = any(kw in low for kw in tool_keywords)
                                
                                if _needs_tool:
                                    signals["intent"] = "desktop command"
                                    signals["intent_confidence"] = 0.95
                                    print(f"🔧 [Keyword Override] Forced intent to 'desktop command' (matched tool keyword)")
                                elif _needs_web_search:
                                    signals["intent"] = "web search"
                                    signals["intent_confidence"] = 0.95
                                    print(f"🔧 [Keyword Override] Forced intent to 'web search' (matched search keyword)")
                                
                                agent_state["micro_expert_signal"] = signals

                                # Policy Router adapter composition
                                import datetime
                                current_sys_time = datetime.datetime.now().strftime("%A, %B %d, %Y - %I:%M %p")
                                active_prompt, routing_log = policy_router.route(signals, updated_base)
                                active_prompt = f"[SYSTEM: Current real-world time is {current_sys_time}]\n\n{active_prompt}"
                                
                                active_adapters = [a.strip() for a in routing_log.split("|")] if routing_log != "Default" else []

                                # Push user message with classification
                                push_transcript("user", user_text,
                                              intent=signals.get("intent", ""),
                                              emotion=signals.get("emotion", ""))

                                # Vision context
                                vision_signal = vision.capture_emotion_signal()
                                if "visible" not in vision_signal:
                                    active_prompt += f"\n\n[Live Camera Signal]: {vision_signal}"

                                # Screen Reader — triggered when user asks about screen content
                                screen_keywords = ["screen", "monitor", "display", "looking at", "what do you see", "read this", "debug this", "this error", "what's on", "what is on", "help me with this"]
                                screen_b64 = None
                                if any(kw in user_text.lower() for kw in screen_keywords):
                                    try:
                                        broadcast_sync({"type": "screen_capture_flash", "duration": 350})
                                        screen_b64 = screen_cap.capture(max_width=1280, quality=90)
                                        print("[Vision Layer] 📸 Screen captured in <35ms! User can now freely switch windows.")
                                    except Exception as e:
                                        print(f"[Vision Layer] Screen capture failed: {e}")


                                # RAG
                                rag_context = memory_db.search(user_text)
                                if rag_context:
                                    active_prompt += f"\n\n[RAG Knowledge]: {rag_context}"

                                # ── DIRECT WEB INJECTION ──
                                # Fires whenever web keywords are detected, EVEN if tool keywords also matched.
                                # This handles compound tasks like "search for X and send WhatsApp to Y"
                                web_injected = False
                                if _needs_web_search or signals.get("intent") == "web search":
                                    try:
                                        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core_agent'))
                                        from web_agent import search_and_read_web
                                        web_result = search_and_read_web(user_text)
                                        if "[AetherWeb Silent Search Results]" in web_result:
                                            active_prompt += f"""

IMPORTANT: READ THE FOLLOWING REAL-TIME SEARCH RESULTS AND SUMMARIZE THEM FOR THE USER:

{web_result}

Your job is to take the search results above and present them to the user as a spoken summary. Quote the titles, facts and sources directly. Do not ignore them."""
                                            web_injected = True
                                            print(f"[Direct Web Injection] Scraped live data injected into LLM context.")
                                    except Exception as e:
                                        print(f"[Direct Web Injection Error] {e}")

                                # LLM Response with rolling context memory & Image Swarm Injection
                                history_slice = agent_state["transcript"][-10:-1]
                                reply = llm.generate_response(active_prompt, user_text, chat_history=history_slice, image_b64=screen_b64)
                                
                                # Extract secret tool logs and strip them from the spoken text
                                import re
                                secret_logs = re.search(r'<SECRET_TOOL_LOG>(.*?)</SECRET_TOOL_LOG>', reply, flags=re.DOTALL)
                                if secret_logs:
                                    log_content = secret_logs.group(1).strip()
                                    # Inject silent memory of the tool execution!
                                    push_transcript("system", f"[Background System Task Completed]\n{log_content}")
                                    reply = re.sub(r'<SECRET_TOOL_LOG>.*?</SECRET_TOOL_LOG>', '', reply, flags=re.DOTALL).strip()
                                
                                # ── ACTION SPEECH SUPPRESSION ──
                                # If we just sent a WhatsApp message, don't read the whole thing out.
                                _tool_log_text = log_content if secret_logs else ""
                                if "send_whatsapp_message" in _tool_log_text:
                                    # Check if the user DENIED the action
                                    if "denied" in _tool_log_text.lower() or "DENIED" in _tool_log_text:
                                        reply = "Understood, sir. WhatsApp message cancelled."
                                    else:
                                        print("🔇 [System] Suppressing long speech for WhatsApp task.")
                                        reply = "Message sent successfully, sir."

                                print(f"AETHER: {reply}")

                                # Sanitize for TTS — comprehensive pipeline to strip ALL non-spoken syntax
                                spoken_reply = reply
                                
                                # 1. Strip code blocks entirely
                                spoken_reply = re.sub(r'```[\s\S]*?```', '', spoken_reply)
                                
                                # 2. Strip LaTeX display math blocks: \[ ... \]
                                spoken_reply = re.sub(r'\\\[.*?\\\]', '', spoken_reply, flags=re.DOTALL)
                                
                                # 3. Strip LaTeX inline math: \( ... \)
                                spoken_reply = re.sub(r'\\\(.*?\\\)', '', spoken_reply, flags=re.DOTALL)
                                
                                # 4. Strip any remaining LaTeX commands: \frac{}{}, \alpha, \omega, etc.
                                spoken_reply = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})*', '', spoken_reply)
                                
                                # 5. Strip markdown headers (### Step 1: etc.)
                                spoken_reply = re.sub(r'#{1,6}\s*', '', spoken_reply)
                                
                                # 6. Strip markdown bold/italic
                                spoken_reply = spoken_reply.replace("**", "").replace("*", "").replace("__", "")
                                
                                # 7. Strip numbered list markers (1. 2. 3.)
                                spoken_reply = re.sub(r'^\d+\.\s+', '', spoken_reply, flags=re.MULTILINE)
                                
                                # 8. Strip bullet points
                                spoken_reply = re.sub(r'^[-•]\s+', '', spoken_reply, flags=re.MULTILINE)
                                
                                # 9. Collapse whitespace
                                spoken_reply = re.sub(r'\n+', ' ', spoken_reply)
                                spoken_reply = re.sub(r'\s{2,}', ' ', spoken_reply).strip()

                                # Speak asynchronously (Daemon Thread)
                                update_state(status="speaking", adapters=active_adapters)
                                push_transcript("aether", reply, adapters=active_adapters)
                                
                                def on_speak_complete():
                                    update_state(status="idle", adapters=[])
                                    
                                def on_interrupted(spoken_so_far):
                                    if agent_state["transcript"] and agent_state["transcript"][-1]["role"] == "aether":
                                        agent_state["transcript"][-1]["text"] = f"{spoken_so_far} -- [INTERRUPTED BY USER]"
                                        # Force sync to update GUI
                                        broadcast_sync({"type": "state_update", "state": agent_state})
                                    
                                tts.speak(spoken_reply, on_complete=on_speak_complete, on_interrupt=on_interrupted)
                                
                            threading.Thread(target=process_utterance, args=(audio_float,), daemon=True).start()

                except queue.Empty:
                    continue
    except Exception as e:
        print(f"Audio stream error: {e}")
    finally:
        print("\n[System] Powering down AETHER Engines...")
        vision.close()

# ──────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    connected_clients.append(ws)
    # Send current state on connect
    await ws.send_text(json.dumps({"type": "state_update", "state": agent_state}))
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("action") == "auth_response":
                auth_queue.put(msg.get("allow", False))
                update_state(status="thinking")
            elif msg.get("action") == "toggle_mute":
                new_state = not agent_state.get("mic_muted", False)
                update_state(mic_muted=new_state)
            elif msg.get("action") == "set_mode":
                new_mode = msg.get("mode", "Voice")
                update_state(mode=new_mode)
                print(f"[System] Mode explicitly changed to: {new_mode}")
            elif msg.get("action") == "interrupt":
                print("\n[System] Manual Interruption requested via User Interface.")
                if GLOBAL_TTS:
                    GLOBAL_TTS.stop()
                update_state(status="idle", adapters=[])
            elif msg.get("action") == "shutdown":
                print("\n[System] Shutdown requested via User Interface.")
                if GLOBAL_TTS:
                    goodbyes = [
                        "Shutting down core systems. Goodbye, sir.",
                        "Powering down. Have a good rest of your day.",
                        "System logging off. See you later.",
                        "Goodbye, sir. Aether out."
                    ]
                    goodbye = random.choice(goodbyes)
                    update_state(status="speaking")
                    
                    # We block briefly so the UI sees the state before process dies
                    def exit_process():
                        os._exit(0)
                        
                    GLOBAL_TTS.speak(goodbye, on_complete=exit_process)
                else:
                    os._exit(0)
    except WebSocketDisconnect:
        connected_clients.remove(ws)

@app.get("/api/camera")
async def get_camera_frame():
    """Serve the latest camera frame as base64 JPEG (HTTP fallback)."""
    try:
        # Use the global vision instance via ai_loop thread
        return {"frame": None, "status": "Use WebSocket camera_frame messages for live feed"}
    except Exception:
        return {"frame": None, "error": "Camera not available"}

# ──────────────────────────────────────────────────────────────────────
# Boot
# ──────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    global MAIN_LOOP
    MAIN_LOOP = asyncio.get_running_loop()
    
    import desktop_agent
    import aether_coder
    import skill_factory
    desktop_agent.confirm_action = frontend_confirm
    aether_coder.confirm_action = frontend_confirm
    skill_factory.confirm_action = frontend_confirm
    
    thread = threading.Thread(target=ai_loop, daemon=True)
    thread.start()

if __name__ == "__main__":
    # Default to Edge-TTS for daily-driver mode (Zero-Latency, Low RAM)
    # Set AETHER_TTS_ENGINE="kokoro" to use local high-quality voice
    if "AETHER_TTS_ENGINE" not in os.environ:
        os.environ["AETHER_TTS_ENGINE"] = "edge"
    
    print(f"[System] Starting AETHER with {os.environ['AETHER_TTS_ENGINE'].upper()} TTS Engine...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
